#!/usr/bin/env python3
"""Build a one-class YOLO dataset from hand-drawn red-box images."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    red_pixels: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def iter_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


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


def red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(hsv, np.array([0, 55, 30]), np.array([7, 255, 255]))
    high_red = cv2.inRange(hsv, np.array([172, 55, 30]), np.array([180, 255, 255]))
    blue, green, red = cv2.split(image)
    red_i = red.astype(np.int16)
    green_i = green.astype(np.int16)
    blue_i = blue.astype(np.int16)
    dominant = (
        (red_i > 75)
        & (red_i > green_i * 1.45 + 20)
        & (red_i > blue_i * 1.35 + 20)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.bitwise_or(low_red, high_red), dominant)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return mask


def find_red_boxes(image: np.ndarray) -> list[Box]:
    height, width = image.shape[:2]
    mask = red_mask(image)
    connected = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=1)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.12 or h < height * 0.12:
            continue
        if w * h < width * height * 0.025:
            continue
        if w / max(h, 1) > 6.0 or h / max(w, 1) > 6.0:
            continue

        roi = mask[y : y + h, x : x + w]
        red_pixels = int(np.count_nonzero(roi))
        if red_pixels < 250:
            continue
        density = red_pixels / float(w * h)
        if density > 0.35:
            continue

        pad = max(2, int(round(max(width, height) * 0.003)))
        x0 = max(0, x + pad)
        y0 = max(0, y + pad)
        x1 = min(width - 1, x + w - pad)
        y1 = min(height - 1, y + h - pad)
        if x1 - x0 < width * 0.10 or y1 - y0 < height * 0.10:
            continue
        boxes.append(Box(x0, y0, x1, y1, red_pixels))

    boxes.sort(key=lambda box: (box.x0, box.y0, box.x1, box.y1))
    return suppress_nested_boxes(boxes)


def suppress_nested_boxes(boxes: list[Box]) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: b.width * b.height, reverse=True):
        area = box.width * box.height
        nested = False
        for other in kept:
            ix0 = max(box.x0, other.x0)
            iy0 = max(box.y0, other.y0)
            ix1 = min(box.x1, other.x1)
            iy1 = min(box.y1, other.y1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if area > 0 and inter / area > 0.85:
                nested = True
                break
        if not nested:
            kept.append(box)
    return sorted(kept, key=lambda b: (b.x0, b.y0))


def yolo_line(box: Box, width: int, height: int) -> str:
    x_center = ((box.x0 + box.x1) * 0.5) / width
    y_center = ((box.y0 + box.y1) * 0.5) / height
    box_width = (box.x1 - box.x0) / width
    box_height = (box.y1 - box.y0) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def inpaint_red_lines(image: np.ndarray) -> np.ndarray:
    mask = cv2.dilate(red_mask(image), np.ones((9, 9), np.uint8), iterations=1)
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def image_similarity(labeled: np.ndarray, raw: np.ndarray) -> float:
    resized_raw = cv2.resize(raw, (labeled.shape[1], labeled.shape[0]), interpolation=cv2.INTER_AREA)
    mask = cv2.dilate(red_mask(labeled), np.ones((19, 19), np.uint8), iterations=1)
    valid = mask == 0
    if float(valid.mean()) < 0.20:
        valid = np.ones(mask.shape, dtype=bool)

    labeled_lab = cv2.cvtColor(labeled, cv2.COLOR_BGR2LAB)
    raw_lab = cv2.cvtColor(resized_raw, cv2.COLOR_BGR2LAB)
    small_labeled = cv2.resize(labeled_lab, (160, 120), interpolation=cv2.INTER_AREA)
    small_raw = cv2.resize(raw_lab, (160, 120), interpolation=cv2.INTER_AREA)
    small_valid = cv2.resize(valid.astype(np.uint8), (160, 120), interpolation=cv2.INTER_NEAREST).astype(bool)
    diff = (small_labeled.astype(np.float32) - small_raw.astype(np.float32))[small_valid]
    if diff.size == 0:
        return float("inf")
    return float(np.mean(diff * diff))


def find_raw_match(
    labeled_image: np.ndarray,
    raw_images: dict[Path, np.ndarray],
    threshold: float,
) -> tuple[Path | None, float]:
    best_path: Path | None = None
    best_score = float("inf")
    for path, image in raw_images.items():
        score = image_similarity(labeled_image, image)
        if score < best_score:
            best_path = path
            best_score = score
    if best_score <= threshold:
        return best_path, best_score
    return None, best_score


def draw_preview(image: np.ndarray, boxes: Iterable[Box], output_path: Path) -> None:
    preview = image.copy()
    for box in boxes:
        cv2.rectangle(preview, (box.x0, box.y0), (box.x1, box.y1), (0, 255, 0), max(2, image.shape[1] // 320))
        cv2.putText(
            preview,
            "yellow_car",
            (box.x0, max(18, box.y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.55, image.shape[1] / 1800.0),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    write_image(output_path, preview)


def copy_or_resize_training_image(source: np.ndarray, target_shape: tuple[int, int], output_path: Path) -> None:
    height, width = target_shape
    if source.shape[:2] != (height, width):
        source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    write_image(output_path, source)


def write_data_yaml(root: Path) -> None:
    content = "\n".join(
        [
            f"path: {root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: yellow_car",
            "",
        ]
    )
    (root / "data.yaml").write_text(content, encoding="utf-8")


def clean_output(root: Path) -> None:
    for relative in ["images", "labels", "previews"]:
        target = root / relative
        if target.exists():
            shutil.rmtree(target)
    for file_name in ["failed_red_box_detection.txt", "preprocess_report.txt", "preprocess_report.json", "data.yaml"]:
        target = root / file_name
        if target.exists():
            target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-threshold", type=float, default=80.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = args.output_dir
    clean_output(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)

    labeled_paths = iter_images(args.labeled_dir)
    raw_images = {path: read_image(path) for path in iter_images(args.raw_dir)}

    records = []
    failed = []
    total_boxes = 0
    used_inpaint = 0
    used_raw = 0

    for labeled_path in labeled_paths:
        labeled_image = read_image(labeled_path)
        boxes = find_red_boxes(labeled_image)
        if not boxes:
            failed.append(labeled_path.name)
            continue

        raw_match, match_score = find_raw_match(labeled_image, raw_images, args.match_threshold)
        if raw_match is not None:
            train_image = raw_images[raw_match]
            image_source = f"raw_match:{raw_match.name}"
            used_raw += 1
        else:
            train_image = inpaint_red_lines(labeled_image)
            image_source = "inpainted_labeled_image"
            used_inpaint += 1

        height, width = labeled_image.shape[:2]
        output_name = labeled_path.name
        records.append(
            {
                "labeled": labeled_path.name,
                "output_image": output_name,
                "raw_match": raw_match.name if raw_match else None,
                "match_score": round(match_score, 3),
                "image_source": image_source,
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
        total_boxes += len(boxes)

    rng = random.Random(args.seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(records) * args.val_ratio))) if records else 0
    val_indices = set(indices[:val_count])

    report_records = []
    for index, record in enumerate(records):
        split = "val" if index in val_indices else "train"
        labeled_image = read_image(args.labeled_dir / record["labeled"])
        raw_match = record["raw_match"]
        if raw_match:
            train_image = raw_images[args.raw_dir / raw_match]
        else:
            train_image = inpaint_red_lines(labeled_image)

        image_path = output_dir / "images" / split / record["output_image"]
        label_path = output_dir / "labels" / split / Path(record["output_image"]).with_suffix(".txt")
        preview_path = output_dir / "previews" / f"{Path(record['output_image']).stem}_preview.jpg"

        copy_or_resize_training_image(train_image, (record["height"], record["width"]), image_path)
        lines = [yolo_line(box, record["width"], record["height"]) for box in record["boxes"]]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        preview_source = read_image(image_path)
        draw_preview(preview_source, record["boxes"], preview_path)

        report_record = {
            key: value for key, value in record.items() if key != "boxes"
        }
        report_record["split"] = split
        report_record["label"] = str(label_path.relative_to(output_dir))
        report_record["image"] = str(image_path.relative_to(output_dir))
        report_record["preview"] = str(preview_path.relative_to(output_dir))
        report_record["boxes_xyxy"] = [
            [box.x0, box.y0, box.x1, box.y1] for box in record["boxes"]
        ]
        report_record["yolo"] = lines
        report_records.append(report_record)

    failed_path = output_dir / "failed_red_box_detection.txt"
    failed_path.write_text(("\n".join(failed) + "\n") if failed else "", encoding="utf-8")
    write_data_yaml(output_dir)

    counts = {
        "red_box_images_read": len(labeled_paths),
        "training_images_generated": len(records),
        "label_files_generated": len(records),
        "yellow_car_boxes_extracted": total_boxes,
        "failed_images": failed,
        "train_images": sum(1 for r in report_records if r["split"] == "train"),
        "val_images": sum(1 for r in report_records if r["split"] == "val"),
        "used_raw_match": used_raw,
        "used_inpaint": used_inpaint,
        "match_threshold": args.match_threshold,
    }
    report = {"counts": counts, "records": report_records}
    (output_dir / "preprocess_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text_lines = [
        "Yellow-car red-box preprocessing report",
        f"red_box_images_read: {counts['red_box_images_read']}",
        f"training_images_generated: {counts['training_images_generated']}",
        f"label_files_generated: {counts['label_files_generated']}",
        f"yellow_car_boxes_extracted: {counts['yellow_car_boxes_extracted']}",
        f"train_images: {counts['train_images']}",
        f"val_images: {counts['val_images']}",
        f"used_raw_match: {counts['used_raw_match']}",
        f"used_inpaint: {counts['used_inpaint']}",
        f"failed_images: {', '.join(failed) if failed else 'none'}",
        "",
        "records:",
    ]
    for item in report_records:
        text_lines.append(
            f"- {item['labeled']} -> {item['image']} boxes={len(item['boxes_xyxy'])} "
            f"source={item['image_source']} split={item['split']} score={item['match_score']}"
        )
    (output_dir / "preprocess_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
