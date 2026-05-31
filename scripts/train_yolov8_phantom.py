#!/usr/bin/env python3
"""Train the PHANTOM YOLOv8 detector after validating dataset labels."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CLASS_NAMES = ["exit", "traffic_cone", "yellow_car"]
RUN_NAME = "phantom_yolov8_exit_cone_yellowcar"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
NO_LABELS_MESSAGE = "只有图片无法训练 YOLO 检测模型，需要标注框与类别标签"


@dataclass
class SplitScan:
    name: str
    root: Path
    images: list[Path] = field(default_factory=list)
    label_files: list[Path] = field(default_factory=list)
    matched_label_files: list[Path] = field(default_factory=list)
    missing_label_images: list[Path] = field(default_factory=list)
    class_ids: set[int] = field(default_factory=set)
    invalid_rows: list[str] = field(default_factory=list)

    @property
    def has_labels(self) -> bool:
        return bool(self.label_files or self.matched_label_files)


class StopTraining(RuntimeError):
    def __init__(self, step: str, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.step = step
        self.exit_code = exit_code


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def resolve_path(path: Path | None, root: Path, default: Path) -> Path:
    selected = default if path is None else path
    if not selected.is_absolute():
        selected = root / selected
    return selected.resolve()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def candidate_label_paths(image_path: Path, split_root: Path) -> list[Path]:
    candidates: list[Path] = [image_path.with_suffix(".txt")]

    try:
        rel = image_path.relative_to(split_root)
    except ValueError:
        rel = Path(image_path.name)

    parts = list(rel.parts)
    if "images" in parts:
        index = parts.index("images")
        label_parts = parts[:]
        label_parts[index] = "labels"
        candidates.append((split_root / Path(*label_parts)).with_suffix(".txt"))

    images_root = split_root / "images"
    try:
        image_rel = image_path.relative_to(images_root)
        candidates.append((split_root / "labels" / image_rel).with_suffix(".txt"))
    except ValueError:
        pass

    candidates.append((split_root / "labels" / rel).with_suffix(".txt"))
    candidates.append(split_root / "labels" / f"{image_path.stem}.txt")

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def parse_label_file(label_file: Path, split: SplitScan) -> None:
    try:
        lines = label_file.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_file.read_text().splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) < 5:
            split.invalid_rows.append(f"{label_file}:{line_number}: expected at least 5 columns")
            continue

        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:5]]
        except ValueError:
            split.invalid_rows.append(f"{label_file}:{line_number}: non-numeric YOLO row")
            continue

        if any(value < 0.0 or value > 1.0 for value in coords):
            split.invalid_rows.append(f"{label_file}:{line_number}: normalized xywh outside [0, 1]")
            continue

        split.class_ids.add(class_id)


def scan_split(name: str, root: Path) -> SplitScan:
    scan = SplitScan(name=name, root=root)
    scan.images = find_images(root)
    scan.label_files = sorted(path for path in root.rglob("*.txt") if path.is_file()) if root.exists() else []

    matched: set[Path] = set()
    for image_path in scan.images:
        label_path = next(
            (candidate for candidate in candidate_label_paths(image_path, root) if candidate.exists()),
            None,
        )
        if label_path is None:
            scan.missing_label_images.append(image_path)
        else:
            matched.add(label_path.resolve())

    all_labels = {path.resolve() for path in scan.label_files} | matched
    scan.matched_label_files = sorted(Path(path) for path in matched)
    scan.label_files = sorted(Path(path) for path in all_labels)

    for label_file in scan.label_files:
        parse_label_file(label_file, scan)

    return scan


def read_yaml_names(path: Path) -> list[str] | dict[int, str] | None:
    if not path.exists():
        return None

    try:
        import yaml
    except ImportError:
        return None

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        parsed: dict[int, str] = {}
        for key, value in names.items():
            try:
                parsed[int(key)] = str(value)
            except (TypeError, ValueError):
                return None
        return parsed
    return None


def normalize_names(names: list[str] | dict[int, str] | None) -> dict[int, str] | None:
    if names is None:
        return None
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return dict(names)


def find_existing_names(workspace: Path, train_dir: Path, eval_dir: Path, config_path: Path) -> tuple[Path | None, dict[int, str] | None]:
    candidates = [
        train_dir.parent / "data.yaml",
        train_dir / "data.yaml",
        eval_dir / "data.yaml",
        config_path,
    ]
    for candidate in candidates:
        names = normalize_names(read_yaml_names(candidate))
        if names:
            return candidate, names
    return None, None


def remap_plan(existing_names: dict[int, str]) -> dict[int, int]:
    expected_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    plan: dict[int, int] = {}
    for old_id, name in existing_names.items():
        if name in expected_ids:
            plan[old_id] = expected_ids[name]
    return plan


def validate_dataset(
    train_scan: SplitScan,
    eval_scan: SplitScan,
    workspace: Path,
    train_dir: Path,
    eval_dir: Path,
    config_path: Path,
) -> None:
    if not train_dir.exists() or not eval_dir.exists():
        missing = [
            str(path)
            for path in (train_dir, eval_dir)
            if not path.exists()
        ]
        raise StopTraining("dataset directory validation", f"Missing dataset directories: {missing}")

    if not train_scan.images or not eval_scan.images:
        raise StopTraining(
            "dataset image validation",
            f"Expected images in both train and eval. train={len(train_scan.images)}, eval={len(eval_scan.images)}",
        )

    if not train_scan.has_labels or not eval_scan.has_labels:
        raise StopTraining("dataset label validation", NO_LABELS_MESSAGE)

    invalid_rows = train_scan.invalid_rows + eval_scan.invalid_rows
    if invalid_rows:
        preview = "\n".join(invalid_rows[:20])
        raise StopTraining("YOLO label row validation", f"Invalid YOLO label rows:\n{preview}")

    if not train_scan.class_ids or not eval_scan.class_ids:
        raise StopTraining("YOLO class id validation", NO_LABELS_MESSAGE)

    used_ids = sorted(train_scan.class_ids | eval_scan.class_ids)
    unexpected_ids = [class_id for class_id in used_ids if class_id not in range(len(CLASS_NAMES))]
    if unexpected_ids:
        raise StopTraining(
            "YOLO class id validation",
            f"Found class IDs outside expected 0..2: {unexpected_ids}. Expected 0: exit, 1: traffic_cone, 2: yellow_car.",
        )

    yaml_path, existing_names = find_existing_names(workspace, train_dir, eval_dir, config_path)
    if existing_names is not None:
        expected = {index: name for index, name in enumerate(CLASS_NAMES)}
        comparable = {index: existing_names.get(index) for index in range(len(CLASS_NAMES))}
        if comparable != expected:
            plan = remap_plan(existing_names)
            raise StopTraining(
                "YOLO class name order validation",
                (
                    f"Class order in {yaml_path} does not match the required order.\n"
                    f"Existing names: {existing_names}\n"
                    "Required names: {0: 'exit', 1: 'traffic_cone', 2: 'yellow_car'}\n"
                    f"Safe remap plan for known names: {plan}\n"
                    "Review the labels and apply the remap to a copy of the label files before training."
                ),
            )


def write_data_yaml(config_path: Path, train_dir: Path, eval_dir: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Auto-generated for PHANTOM YOLOv8 training.",
            f"train: {json.dumps(as_yaml_path(train_dir))}",
            f"val: {json.dumps(as_yaml_path(eval_dir))}",
            "nc: 3",
            "names:",
            "  0: exit",
            "  1: traffic_cone",
            "  2: yellow_car",
            "",
        ]
    )
    config_path.write_text(text, encoding="utf-8")


def memory_error(exc: BaseException) -> bool:
    text = "".join(traceback.format_exception_only(type(exc), exc)).lower()
    markers = [
        "cuda out of memory",
        "out of memory",
        "cublas",
        "cudnn",
        "memoryerror",
    ]
    return any(marker in text for marker in markers)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def metrics_summary(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", None) or {index: name for index, name in enumerate(CLASS_NAMES)}
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}

    p_values = to_list(getattr(box, "p", None))
    r_values = to_list(getattr(box, "r", None))
    map_values = to_list(getattr(box, "maps", None))
    map50_values = to_list(getattr(box, "ap50", None))

    per_class = []
    for class_id, expected_name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "id": class_id,
                "name": str(names.get(class_id, expected_name)),
                "precision": to_float(p_values[class_id]) if class_id < len(p_values) else None,
                "recall": to_float(r_values[class_id]) if class_id < len(r_values) else None,
                "mAP50": to_float(map50_values[class_id]) if class_id < len(map50_values) else None,
                "mAP50-95": to_float(map_values[class_id]) if class_id < len(map_values) else None,
            }
        )

    return {
        "precision": to_float(getattr(box, "mp", None)),
        "recall": to_float(getattr(box, "mr", None)),
        "mAP50": to_float(getattr(box, "map50", None)),
        "mAP50-95": to_float(getattr(box, "map", None)),
        "per_class": per_class,
    }


def count_predictions(results: list[Any]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        for class_value in boxes.cls:
            class_id = int(class_value)
            class_name = str(names.get(class_id, class_id))
            if class_name in counts:
                counts[class_name] += 1
    return counts


def build_batch_candidates(batch: str | None) -> list[int]:
    if batch is None:
        return [-1, 8, 4, 2]
    if batch.strip().lower() in {"auto", "-1"}:
        return [-1, 8, 4, 2]
    return [int(batch)]


def print_failure_report(
    *,
    workspace: Path,
    train_dir: Path,
    eval_dir: Path,
    train_scan: SplitScan | None,
    eval_scan: SplitScan | None,
    commands: list[str],
    step: str,
    error: str,
) -> None:
    ultralytics_version = package_version("ultralytics")
    report = {
        "training_success": False,
        "failed_step": step,
        "current_dir": str(Path.cwd()),
        "workspace": str(workspace),
        "python_version": platform.python_version(),
        "train_dir": str(train_dir),
        "train_dir_exists": train_dir.exists(),
        "eval_dir": str(eval_dir),
        "eval_dir_exists": eval_dir.exists(),
        "train_images": len(train_scan.images) if train_scan else None,
        "eval_images": len(eval_scan.images) if eval_scan else None,
        "label_files_found": bool((train_scan and train_scan.has_labels) or (eval_scan and eval_scan.has_labels)),
        "train_label_files": len(train_scan.label_files) if train_scan else None,
        "eval_label_files": len(eval_scan.label_files) if eval_scan else None,
        "ultralytics_installed": ultralytics_version is not None,
        "ultralytics_version": ultralytics_version,
        "commands": commands,
        "error": error,
        "next_steps": [
            "Add YOLO .txt labels for data/train and data/eval.",
            "Use class IDs 0: exit, 1: traffic_cone, 2: yellow_car.",
            "Re-run the training command after labels are present.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=root)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    train_dir = resolve_path(args.train_dir, workspace, workspace / "data" / "train")
    eval_dir = resolve_path(args.eval_dir, workspace, workspace / "data" / "eval")
    config_path = resolve_path(args.data_yaml, workspace, workspace / "config" / "phantom_yolo_data.yaml")
    project_dir = resolve_path(args.project, workspace, workspace / "runs" / "yolo")
    model_path = resolve_path(args.model, workspace, workspace / "models" / "yolo" / "yolov8n.pt")
    commands = ["python scripts/train_yolov8_phantom.py --epochs 80"]

    train_scan: SplitScan | None = None
    eval_scan: SplitScan | None = None

    try:
        train_scan = scan_split("train", train_dir)
        eval_scan = scan_split("eval", eval_dir)
        validate_dataset(train_scan, eval_scan, workspace, train_dir, eval_dir, config_path)
        write_data_yaml(config_path, train_dir, eval_dir)

        if args.check_only:
            print(f"Dataset check passed. data.yaml: {config_path}")
            return 0

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise StopTraining("ultralytics import", f"ultralytics is not installed: {exc}") from exc

        selected_model = model_path if model_path.exists() else Path("yolov8n.pt")
        batch_candidates = build_batch_candidates(args.batch)
        train_result = None
        last_error: BaseException | None = None

        for batch in batch_candidates:
            commands.append(
                "YOLO.train("
                f"model={selected_model}, data={config_path}, imgsz={args.imgsz}, "
                f"epochs={args.epochs}, batch={batch}, project={project_dir}, name={args.name})"
            )
            try:
                model = YOLO(str(selected_model))
                train_result = model.train(
                    data=str(config_path),
                    imgsz=args.imgsz,
                    epochs=args.epochs,
                    batch=batch,
                    project=str(project_dir),
                    name=args.name,
                    exist_ok=True,
                )
                break
            except Exception as exc:
                last_error = exc
                if batch != batch_candidates[-1] and memory_error(exc):
                    print(f"Training failed with batch={batch}; retrying with a smaller batch.")
                    continue
                raise

        save_dir = getattr(train_result, "save_dir", None)
        run_dir = Path(save_dir) if save_dir is not None else project_dir / args.name
        if not run_dir.exists():
            run_dir = project_dir / args.name

        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"
        if not best_path.exists() or not last_path.exists():
            raise StopTraining(
                "training output validation",
                f"Expected weights were not created. best={best_path.exists()}, last={last_path.exists()}. Last error: {last_error}",
            )

        best_model = YOLO(str(best_path))
        raw_model_names = best_model.names
        if isinstance(raw_model_names, dict):
            model_names = {int(key): str(value) for key, value in raw_model_names.items()}
        else:
            model_names = {index: str(value) for index, value in enumerate(raw_model_names)}
        expected_names = {index: name for index, name in enumerate(CLASS_NAMES)}
        if model_names != expected_names:
            raise StopTraining(
                "trained model class-name validation",
                f"Model names do not match expected names. model={model_names}, expected={expected_names}",
            )

        commands.append(f"YOLO.val(model={best_path}, data={config_path}, imgsz={args.imgsz})")
        metrics = best_model.val(data=str(config_path), imgsz=args.imgsz)

        commands.append(
            "YOLO.predict("
            f"model={best_path}, source={eval_dir}, imgsz={args.imgsz}, "
            f"project={project_dir}, name=predict_eval)"
        )
        prediction_results = best_model.predict(
            source=str(eval_dir),
            imgsz=args.imgsz,
            conf=args.conf,
            project=str(project_dir),
            name="predict_eval",
            save=True,
            exist_ok=True,
        )

        summary = {
            "training_success": True,
            "best_pt": str(best_path),
            "last_pt": str(last_path),
            "metrics": metrics_summary(metrics),
            "eval_prediction_counts": count_predictions(prediction_results),
            "train_log_dir": str(run_dir),
            "predict_eval_dir": str(project_dir / "predict_eval"),
            "data_yaml": str(config_path),
            "commands": commands,
        }
        summary_path = run_dir / "phantom_training_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except StopTraining as exc:
        print_failure_report(
            workspace=workspace,
            train_dir=train_dir,
            eval_dir=eval_dir,
            train_scan=train_scan,
            eval_scan=eval_scan,
            commands=commands,
            step=exc.step,
            error=str(exc),
        )
        return exc.exit_code
    except Exception as exc:
        print_failure_report(
            workspace=workspace,
            train_dir=train_dir,
            eval_dir=eval_dir,
            train_scan=train_scan,
            eval_scan=eval_scan,
            commands=commands,
            step="unexpected exception",
            error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
