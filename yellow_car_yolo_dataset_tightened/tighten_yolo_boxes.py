#!/usr/bin/env python3
"""Create a tightened one-class YOLO dataset from broad yellow-car boxes."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
CLASS_NAME = "yellow_car"


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            max(0, min(self.x0, width - 1)),
            max(0, min(self.y0, height - 1)),
            max(0, min(self.x1, width - 1)),
            max(0, min(self.y1, height - 1)),
        )


@dataclass
class TightenResult:
    original: Box
    tightened: Box
    status: str
    area_ratio: float
    yellow_pixels: int
    reason: str


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def parse_yolo_label(line: str, width: int, height: int) -> Box:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO line: {line!r}")
    class_id = int(float(parts[0]))
    if class_id != CLASS_ID:
        raise ValueError(f"Unexpected class id {class_id}; expected {CLASS_ID}")
    x_center, y_center, box_width, box_height = [float(value) for value in parts[1:]]
    x0 = int(round((x_center - box_width * 0.5) * width))
    y0 = int(round((y_center - box_height * 0.5) * height))
    x1 = int(round((x_center + box_width * 0.5) * width))
    y1 = int(round((y_center + box_height * 0.5) * height))
    return Box(x0, y0, x1, y1).clipped(width, height)


def box_to_yolo(box: Box, width: int, height: int) -> str:
    x_center = ((box.x0 + box.x1) * 0.5) / width
    y_center = ((box.y0 + box.y1) * 0.5) / height
    box_width = box.width / width
    box_height = box.height / height
    values = [x_center, y_center, box_width, box_height]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite YOLO values for box {box}")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Out-of-range YOLO values {values} for box {box}")
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def color_masks(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    yellow = cv2.inRange(hsv, np.array([10, 45, 50]), np.array([45, 255, 255]))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=1)

    # Keep dark body/wheels near yellow panels so chairs/floor are less likely to be included.
    dark = ((gray < 115) & (hsv[:, :, 2] < 150)).astype(np.uint8) * 255
    h, w = roi.shape[:2]
    near_kernel = max(45, int(round(max(h, w) * 0.085)))
    if near_kernel % 2 == 0:
        near_kernel += 1
    near_yellow = cv2.dilate(yellow, np.ones((near_kernel, near_kernel), np.uint8), iterations=1)
    dark_near_yellow = cv2.bitwise_and(dark, near_yellow)

    candidate = cv2.bitwise_or(yellow, dark_near_yellow)
    close_kernel = max(19, int(round(max(h, w) * 0.035)))
    if close_kernel % 2 == 0:
        close_kernel += 1
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    return yellow, dark, candidate


def choose_component(candidate: np.ndarray, yellow: np.ndarray, roi_shape: tuple[int, int]) -> tuple[Box | None, int, str]:
    roi_h, roi_w = roi_shape
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, Box, int] | None = None
    min_area = max(250.0, roi_w * roi_h * 0.015)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        if w < roi_w * 0.08 or h < roi_h * 0.08:
            continue
        aspect = w / max(h, 1)
        if aspect > 5.0 or aspect < 0.25:
            continue
        yellow_pixels = int(np.count_nonzero(yellow[y:y + h, x:x + w]))
        if yellow_pixels < max(80, roi_w * roi_h * 0.002):
            continue

        center_x = x + w * 0.5
        center_y = y + h * 0.5
        center_bias = 1.0 - min(0.8, abs(center_x - roi_w * 0.5) / max(roi_w, 1))
        score = yellow_pixels * 25.0 + area * 0.8 + center_bias * area * 0.15
        box = Box(x, y, x + w, y + h)
        if best is None or score > best[0]:
            best = (score, box, yellow_pixels)

    if best is None:
        return None, 0, "no_valid_component"
    return best[1], best[2], "ok"


def extend_component_to_dark_vehicle(component: Box, yellow: np.ndarray, dark: np.ndarray, roi_shape: tuple[int, int]) -> Box:
    roi_h, roi_w = roi_shape
    x_pad = max(10, int(round(roi_w * 0.08)))
    y_pad = max(10, int(round(roi_h * 0.06)))
    x0 = max(0, component.x0 - x_pad)
    x1 = min(roi_w - 1, component.x1 + x_pad)
    y0 = max(0, component.y0 - y_pad)
    y1 = min(roi_h - 1, component.y1 + y_pad)

    # Scan for dark row support directly above/below the component. This captures wheels and
    # lower body without reconnecting distant background across the whole original box.
    band_width = max(1, x1 - x0)
    row_threshold = max(10, int(round(band_width * 0.055)))
    row_mask = cv2.bitwise_or(dark, yellow)
    row_counts = np.count_nonzero(row_mask[:, x0:x1] > 0, axis=1)
    min_y = max(0, int(component.y0 - roi_h * 0.22))
    max_y = min(roi_h - 1, int(component.y1 + roi_h * 0.34))
    gap_limit = max(8, int(round(roi_h * 0.025)))
    gap = 0
    for y in range(y0 - 1, min_y - 1, -1):
        if row_counts[y] >= row_threshold:
            y0 = y
            gap = 0
        else:
            gap += 1
        if gap > gap_limit:
            break
    gap = 0
    for y in range(y1 + 1, max_y + 1):
        if row_counts[y] >= row_threshold:
            y1 = y
            gap = 0
        else:
            gap += 1
        if gap > gap_limit:
            break

    # Scan columns inside the supported vertical band to retain side wheels/body edges.
    band_height = max(1, y1 - y0)
    col_threshold = max(8, int(round(band_height * 0.05)))
    col_counts = np.count_nonzero(row_mask[y0:y1, :] > 0, axis=0)
    min_x = max(0, int(component.x0 - roi_w * 0.16))
    max_x = min(roi_w - 1, int(component.x1 + roi_w * 0.16))
    gap_limit_x = max(8, int(round(roi_w * 0.02)))
    gap = 0
    for x in range(x0 - 1, min_x - 1, -1):
        if col_counts[x] >= col_threshold:
            x0 = x
            gap = 0
        else:
            gap += 1
        if gap > gap_limit_x:
            break
    gap = 0
    for x in range(x1 + 1, max_x + 1):
        if col_counts[x] >= col_threshold:
            x1 = x
            gap = 0
        else:
            gap += 1
        if gap > gap_limit_x:
            break

    return Box(x0, y0, x1, y1)


def tighten_box(image: np.ndarray, original: Box) -> TightenResult:
    height, width = image.shape[:2]
    original = original.clipped(width, height)
    roi = image[original.y0:original.y1 + 1, original.x0:original.x1 + 1]
    if roi.size == 0 or original.area <= 0:
        return TightenResult(original, original, "fallback_original", 1.0, 0, "empty_roi")

    yellow, dark, candidate = color_masks(roi)
    component, yellow_pixels, reason = choose_component(candidate, yellow, roi.shape[:2])
    if component is None:
        return TightenResult(original, original, "fallback_original", 1.0, yellow_pixels, reason)
    component = extend_component_to_dark_vehicle(component, yellow, dark, roi.shape[:2])

    pad = max(8, int(round(max(width, height) * 0.012)))
    tightened = Box(
        original.x0 + component.x0 - pad,
        original.y0 + component.y0 - pad,
        original.x0 + component.x1 + pad,
        original.y0 + component.y1 + pad,
    ).clipped(width, height)

    # The tightened box may not expand beyond the original broad annotation.
    tightened = Box(
        max(original.x0, tightened.x0),
        max(original.y0, tightened.y0),
        min(original.x1, tightened.x1),
        min(original.y1, tightened.y1),
    ).clipped(width, height)

    if tightened.area <= 0:
        return TightenResult(original, original, "fallback_original", 1.0, yellow_pixels, "empty_tightened_box")

    area_ratio = tightened.area / max(float(original.area), 1.0)
    if area_ratio < 0.22:
        return TightenResult(original, original, "fallback_original", 1.0, yellow_pixels, "tightened_area_too_small")
    if tightened.width < width * 0.12 or tightened.height < height * 0.12:
        return TightenResult(original, original, "fallback_original", 1.0, yellow_pixels, "tightened_box_too_small")

    return TightenResult(original, tightened, "tightened", area_ratio, yellow_pixels, "ok")


def draw_preview(image: np.ndarray, results: Iterable[TightenResult], output_path: Path) -> None:
    preview = image.copy()
    line_width = max(2, image.shape[1] // 360)
    font_scale = max(0.55, image.shape[1] / 1900.0)
    for result in results:
        original = result.original
        tightened = result.tightened
        cv2.rectangle(preview, (original.x0, original.y0), (original.x1, original.y1), (0, 0, 255), line_width)
        cv2.rectangle(preview, (tightened.x0, tightened.y0), (tightened.x1, tightened.y1), (0, 255, 0), line_width)
        label = f"{CLASS_NAME} {result.status} {result.area_ratio:.2f}x"
        cv2.putText(
            preview,
            label,
            (tightened.x0, max(22, tightened.y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    write_image(output_path, preview)


def write_contact_sheet(preview_paths: list[Path], output_path: Path) -> None:
    if not preview_paths:
        return
    thumbs: list[tuple[Path, np.ndarray]] = []
    thumb_w, thumb_h = 360, 260
    for path in preview_paths:
        image = read_image(path)
        h, w = image.shape[:2]
        scale = min(thumb_w / w, thumb_h / h)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append((path, resized))

    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.full((rows * (thumb_h + 34), cols * thumb_w, 3), 255, dtype=np.uint8)
    for index, (path, image) in enumerate(thumbs):
        row = index // cols
        col = index % cols
        x = col * thumb_w + (thumb_w - image.shape[1]) // 2
        y = row * (thumb_h + 34)
        sheet[y:y + image.shape[0], x:x + image.shape[1]] = image
        cv2.putText(
            sheet,
            path.stem.replace("_tightened_preview", "")[-24:],
            (col * thumb_w + 4, row * (thumb_h + 34) + thumb_h + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    write_image(output_path, sheet)


def clean_output(output_dir: Path) -> None:
    for relative in ["images", "labels", "previews"]:
        target = output_dir / relative
        if target.exists():
            shutil.rmtree(target)
    for file_name in ["data.yaml", "tighten_report.txt", "tighten_report.json"]:
        target = output_dir / file_name
        if target.exists():
            target.unlink()


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            f"  {CLASS_ID}: {CLASS_NAME}",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def process_dataset(input_dir: Path, output_dir: Path) -> dict:
    clean_output(output_dir)
    preview_paths: list[Path] = []
    records = []
    fallback_files: list[str] = []
    total_images = 0
    total_labels = 0
    total_boxes = 0
    tightened_boxes = 0

    for split in ["train", "val"]:
        src_image_dir = input_dir / "images" / split
        src_label_dir = input_dir / "labels" / split
        dst_image_dir = output_dir / "images" / split
        dst_label_dir = output_dir / "labels" / split
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in iter_images(src_image_dir):
            total_images += 1
            image = read_image(image_path)
            height, width = image.shape[:2]
            label_path = src_label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise RuntimeError(f"Missing label for {image_path}: {label_path}")

            label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not label_lines:
                raise RuntimeError(f"Empty label file is not allowed: {label_path}")

            results: list[TightenResult] = []
            output_lines: list[str] = []
            for line in label_lines:
                original = parse_yolo_label(line, width, height)
                result = tighten_box(image, original)
                results.append(result)
                output_lines.append(box_to_yolo(result.tightened, width, height))
                total_boxes += 1
                if result.status == "tightened":
                    tightened_boxes += 1
                else:
                    fallback_files.append(str(Path(split) / image_path.name))

            shutil.copy2(image_path, dst_image_dir / image_path.name)
            (dst_label_dir / f"{image_path.stem}.txt").write_text("\n".join(output_lines) + "\n", encoding="utf-8")

            preview_path = output_dir / "previews" / f"{image_path.stem}_tightened_preview.jpg"
            draw_preview(image, results, preview_path)
            preview_paths.append(preview_path)
            total_labels += 1

            records.append(
                {
                    "split": split,
                    "image": str(Path("images") / split / image_path.name),
                    "label": str(Path("labels") / split / f"{image_path.stem}.txt"),
                    "preview": str(preview_path.relative_to(output_dir)),
                    "boxes": [
                        {
                            "original_xyxy": asdict(result.original),
                            "tightened_xyxy": asdict(result.tightened),
                            "status": result.status,
                            "area_ratio": round(result.area_ratio, 4),
                            "yellow_pixels": result.yellow_pixels,
                            "reason": result.reason,
                        }
                        for result in results
                    ],
                }
            )

    write_contact_sheet(preview_paths, output_dir / "previews" / "preview_contact_sheet.jpg")
    write_data_yaml(output_dir)

    suspicious = [
        record["image"]
        for record in records
        for box in record["boxes"]
        if box["status"] != "tightened" or box["area_ratio"] > 0.92 or box["area_ratio"] < 0.30
    ]
    summary = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_images": total_images,
        "total_label_files": total_labels,
        "total_boxes": total_boxes,
        "tightened_boxes": tightened_boxes,
        "fallback_boxes": total_boxes - tightened_boxes,
        "fallback_files": sorted(set(fallback_files)),
        "train_images": len(iter_images(output_dir / "images" / "train")),
        "val_images": len(iter_images(output_dir / "images" / "val")),
        "suggested_manual_review": sorted(set(suspicious)),
    }
    report = {"summary": summary, "records": records}
    (output_dir / "tighten_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "Yellow-car YOLO box tightening report",
        f"input_dir: {summary['input_dir']}",
        f"output_dir: {summary['output_dir']}",
        f"total_images: {summary['total_images']}",
        f"total_label_files: {summary['total_label_files']}",
        f"total_boxes: {summary['total_boxes']}",
        f"tightened_boxes: {summary['tightened_boxes']}",
        f"fallback_boxes: {summary['fallback_boxes']}",
        f"train_images: {summary['train_images']}",
        f"val_images: {summary['val_images']}",
        "fallback_files: " + (", ".join(summary["fallback_files"]) if summary["fallback_files"] else "none"),
        "suggested_manual_review: "
        + (", ".join(summary["suggested_manual_review"]) if summary["suggested_manual_review"] else "none"),
        "",
        "records:",
    ]
    for record in records:
        for index, box in enumerate(record["boxes"]):
            lines.append(
                f"- {record['image']} box={index} status={box['status']} "
                f"ratio={box['area_ratio']} reason={box['reason']} "
                f"orig={box['original_xyxy']} tight={box['tightened_xyxy']}"
            )
    (output_dir / "tighten_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tighten yellow-car YOLO boxes into a new dataset.")
    parser.add_argument("--input-dir", type=Path, default=Path("yellow_car_yolo_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("yellow_car_yolo_dataset_tightened"))
    args = parser.parse_args()

    summary = process_dataset(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
