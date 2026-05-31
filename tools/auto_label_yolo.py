#!/usr/bin/env python3
"""Auto pre-label images for YOLO detection training."""

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
MIN_BOX_PIXELS = 2.0

CLASS_NAMES = {
    0: "traffic_cone",
    1: "yellow_car",
    2: "exit",
}

CLASS_PROMPTS = {
    0: ["traffic cone", "orange traffic cone", "road cone", "cone"],
    1: ["yellow car", "yellow vehicle", "small yellow car", "yellow robot car"],
    2: ["exit", "exit sign", "exit door", "exit gate", "exit area", "doorway", "gate"],
}

PHRASE_TO_CLASS: Dict[str, int] = {
    phrase: class_id
    for class_id, phrases in CLASS_PROMPTS.items()
    for phrase in phrases
}

PALETTE = {
    0: (255, 80, 40),
    1: (245, 210, 35),
    2: (40, 190, 90),
}

CLASS_PRESETS = {
    "all": [0, 1, 2],
    "cone_yellow": [0, 1],
}


@dataclass
class Detection:
    class_id: int
    score: float
    xyxy: Tuple[float, float, float, float]
    label: str = ""


def iter_images(image_dir: Path) -> List[Path]:
    return sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_text(value: str) -> str:
    cleaned = value.lower().strip()
    for char in [".", ",", ";", ":", "_", "-"]:
        cleaned = cleaned.replace(char, " ")
    return " ".join(cleaned.split())


def map_text_to_class(label: str) -> Optional[int]:
    """Map model text labels or YOLO class names into the fixed class ids."""
    normalized = normalize_text(label)
    if normalized in PHRASE_TO_CLASS:
        return PHRASE_TO_CLASS[normalized]

    for phrase, class_id in sorted(PHRASE_TO_CLASS.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase in normalized:
            return class_id

    if "cone" in normalized:
        return 0
    if "yellow" in normalized and ("car" in normalized or "vehicle" in normalized):
        return 1
    if any(token in normalized for token in ["exit", "doorway", "gate"]):
        return 2
    return None


def class_ids_for_preset(preset: str) -> List[int]:
    return list(CLASS_PRESETS[preset])


def allowed_class_ids_for_image(args, image_path: Path) -> List[int]:
    """Apply per-image class gates before writing YOLO labels."""
    allowed = class_ids_for_preset(args.classes)
    keyword = args.yellow_car_filename_keyword
    if keyword and keyword not in image_path.name and 1 in allowed:
        allowed.remove(1)
    return allowed


def clip_xyxy(box: Sequence[float], width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = max(0.0, min(x0, float(width)))
    x1 = max(0.0, min(x1, float(width)))
    y0 = max(0.0, min(y0, float(height)))
    y1 = max(0.0, min(y1, float(height)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 - x0 < MIN_BOX_PIXELS or y1 - y0 < MIN_BOX_PIXELS:
        return None
    return x0, y0, x1, y1


def xyxy_to_yolo(box: Sequence[float], image_width: int, image_height: int) -> Optional[Tuple[float, float, float, float]]:
    """Convert absolute xyxy pixels to normalized YOLO xywh."""
    clipped = clip_xyxy(box, image_width, image_height)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    box_width = (x1 - x0) / float(image_width)
    box_height = (y1 - y0) / float(image_height)
    x_center = ((x0 + x1) * 0.5) / float(image_width)
    y_center = ((y0 + y1) * 0.5) / float(image_height)
    values = (x_center, y_center, box_width, box_height)
    if any(not math.isfinite(value) for value in values):
        return None
    if any(value < 0.0 or value > 1.0 for value in values):
        return None
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    return values


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def nms_per_class(
    detections: Sequence[Detection],
    iou_threshold: float,
    class_ids: Optional[Sequence[int]] = None,
) -> List[Detection]:
    """Apply greedy NMS independently for each fixed class."""
    kept: List[Detection] = []
    for class_id in (class_ids if class_ids is not None else CLASS_NAMES):
        candidates = sorted(
            [det for det in detections if det.class_id == class_id],
            key=lambda det: det.score,
            reverse=True,
        )
        class_kept: List[Detection] = []
        while candidates:
            current = candidates.pop(0)
            class_kept.append(current)
            candidates = [
                det for det in candidates
                if box_iou(current.xyxy, det.xyxy) < iou_threshold
            ]
        kept.extend(class_kept)
    return sorted(kept, key=lambda det: (det.class_id, -det.score))


def validate_detections(
    detections: Iterable[Detection],
    width: int,
    height: int,
    allowed_class_ids: Optional[Sequence[int]] = None,
) -> List[Detection]:
    valid: List[Detection] = []
    allowed = set(allowed_class_ids) if allowed_class_ids is not None else set(CLASS_NAMES)
    for det in detections:
        if det.class_id not in allowed:
            continue
        if not math.isfinite(float(det.score)):
            continue
        clipped = clip_xyxy(det.xyxy, width, height)
        if clipped is None:
            continue
        if xyxy_to_yolo(clipped, width, height) is None:
            continue
        valid.append(Detection(det.class_id, float(det.score), clipped, det.label))
    return valid


def write_yolo_label(label_path: Path, detections: Sequence[Detection], width: int, height: int) -> int:
    lines: List[str] = []
    for det in detections:
        yolo_box = xyxy_to_yolo(det.xyxy, width, height)
        if yolo_box is None:
            continue
        x_center, y_center, box_width, box_height = yolo_box
        lines.append(
            f"{det.class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def draw_visualization(image: Image.Image, detections: Sequence[Detection], output_path: Path) -> None:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, output.width // 90))
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = [int(round(value)) for value in det.xyxy]
        color = PALETTE[det.class_id]
        line_width = max(2, output.width // 360)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
        text = f"{CLASS_NAMES[det.class_id]} {det.score:.2f}"
        bbox = draw.textbbox((x0, y0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_y0 = max(0, y0 - text_h - 6)
        draw.rectangle([x0, label_y0, x0 + text_w + 8, label_y0 + text_h + 6], fill=color)
        draw.text((x0 + 4, label_y0 + 3), text, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def select_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class HfGroundingDinoDetector:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.prompts: Dict[Tuple[int, ...], str] = {}

    def prompt_for(self, class_ids: Sequence[int]) -> str:
        key = tuple(class_ids)
        if key not in self.prompts:
            self.prompts[key] = ". ".join(
                phrase
                for class_id in class_ids
                for phrase in CLASS_PROMPTS[class_id]
            ) + "."
        return self.prompts[key]

    def detect(
        self,
        image: Image.Image,
        box_threshold: float,
        text_threshold: float,
        class_ids: Sequence[int],
    ) -> List[Detection]:
        inputs = self.processor(images=image, text=self.prompt_for(class_ids), return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([(image.height, image.width)], device=self.device)
        try:
            processed = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.get("input_ids"),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            try:
                processed = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.get("input_ids"),
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]
            except TypeError:
                processed = self.processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]

        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        labels = processed.get("text_labels") or processed.get("labels", [])

        detections: List[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            label_text = str(label)
            class_id = map_text_to_class(label_text)
            if class_id is None:
                continue
            box_values = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            score_value = float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)
            detections.append(Detection(class_id, score_value, tuple(box_values), label_text))
        return detections


class UltralyticsYoloDetector:
    def __init__(self, weights_path: Path, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.device = None if device == "auto" else device

    def detect(self, image_path: Path, box_threshold: float, nms_iou: float) -> List[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=box_threshold,
            iou=nms_iou,
            verbose=False,
            device=self.device,
        )
        detections: List[Detection] = []
        names = getattr(self.model, "names", {}) or {}
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                original_id = int(box.cls.item())
                class_name = str(names.get(original_id, original_id))
                class_id = map_text_to_class(class_name)
                if class_id is None and original_id in CLASS_NAMES:
                    class_id = original_id
                if class_id is None:
                    continue
                xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
                score = float(box.conf.item())
                detections.append(Detection(class_id, score, xyxy, class_name))
        return detections


def build_detector(args, device: str):
    if args.use_yolo_model:
        return UltralyticsYoloDetector(args.use_yolo_model, device)
    return HfGroundingDinoDetector(args.model_id, device)


def process_images(args) -> Counter:
    image_dir = args.image_dir
    label_dir = args.label_dir
    vis_dir = args.vis_dir
    images = iter_images(image_dir)
    if args.max_images is not None:
        images = images[:args.max_images]

    label_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    stats["total_images"] = len(images)
    active_class_ids = class_ids_for_preset(args.classes)
    class_counts = Counter({class_id: 0 for class_id in active_class_ids})

    device = select_device(args.device)
    detector = build_detector(args, device)

    for image_path in tqdm(images, desc="auto-label"):
        relative = image_path.relative_to(image_dir)
        label_path = label_dir / relative.with_suffix(".txt")
        if label_path.exists() and not args.overwrite:
            stats["skipped_images"] += 1
            continue

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            stats["skipped_images"] += 1
            print(f"[WARN] Could not read image {image_path}: {exc}")
            continue

        try:
            allowed_class_ids = allowed_class_ids_for_image(args, image_path)
            if args.use_yolo_model:
                raw_detections = detector.detect(image_path, args.box_threshold, args.nms_iou)
            else:
                raw_detections = detector.detect(
                    image,
                    args.box_threshold,
                    args.text_threshold,
                    allowed_class_ids,
                )
        except Exception as exc:
            stats["skipped_images"] += 1
            print(f"[WARN] Detection failed for {image_path}: {exc}")
            continue

        detections = validate_detections(
            raw_detections,
            image.width,
            image.height,
            allowed_class_ids,
        )
        detections = nms_per_class(detections, args.nms_iou, allowed_class_ids)

        if detections or args.save_empty:
            written = write_yolo_label(label_path, detections, image.width, image.height)
            stats["generated_label_files"] += 1
            if written == 0:
                stats["empty_label_images"] += 1
            for det in detections:
                class_counts[det.class_id] += 1

        if vis_dir:
            vis_path = vis_dir / relative
            draw_visualization(image, detections, vis_path)

        stats["processed_images"] += 1

    for class_id, count in class_counts.items():
        stats[f"class_{class_id}_{CLASS_NAMES[class_id]}"] = count
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Auto pre-label images into YOLO detection txt files.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Input image directory.")
    parser.add_argument("--label_dir", type=Path, required=True, help="Output YOLO label directory.")
    parser.add_argument("--vis_dir", type=Path, default=None, help="Optional visualization output directory.")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, help="HuggingFace zero-shot detection model id.")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--save_empty", dest="save_empty", action="store_true", default=True)
    parser.add_argument("--no_save_empty", dest="save_empty", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing label files.")
    parser.add_argument("--max_images", type=int, default=None, help="Only process the first N images.")
    parser.add_argument("--use_yolo_model", type=Path, default=None, help="Optional Ultralytics YOLO weights path.")
    parser.add_argument(
        "--classes",
        choices=sorted(CLASS_PRESETS),
        default="all",
        help="Class preset to write. Use cone_yellow to disable exit.",
    )
    parser.add_argument(
        "--yellow_car_filename_keyword",
        default=None,
        help="If set, keep yellow_car only when the image filename contains this text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.exists():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")

    stats = process_images(args)
    print("\nAuto-label summary")
    print(f"  total images: {stats['total_images']}")
    print(f"  successfully processed images: {stats['processed_images']}")
    print(f"  generated label files: {stats['generated_label_files']}")
    for class_id in class_ids_for_preset(args.classes):
        name = CLASS_NAMES[class_id]
        print(f"  class {class_id} {name}: {stats[f'class_{class_id}_{name}']} boxes")
    print(f"  empty label images: {stats['empty_label_images']}")
    print(f"  skipped images: {stats['skipped_images']}")
    print(f"  label output dir: {args.label_dir.resolve()}")
    if args.vis_dir:
        print(f"  visualization output dir: {args.vis_dir.resolve()}")


if __name__ == "__main__":
    main()
