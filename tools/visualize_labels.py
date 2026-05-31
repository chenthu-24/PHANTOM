#!/usr/bin/env python3
"""Visualize YOLO detection labels for quick human inspection."""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "traffic_cone",
    1: "yellow_car",
    2: "exit",
}
PALETTE = {
    0: (255, 80, 40),
    1: (245, 210, 35),
    2: (40, 190, 90),
}


def iter_images(image_dir: Path) -> List[Path]:
    return sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def yolo_to_xyxy(values: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x_center, y_center, box_width, box_height = values
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    x0 = int(round((x_center - box_width * 0.5) * width))
    y0 = int(round((y_center - box_height * 0.5) * height))
    x1 = int(round((x_center + box_width * 0.5) * width))
    y1 = int(round((y_center + box_height * 0.5) * height))
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def read_labels(label_path: Path, width: int, height: int):
    items = []
    if not label_path.exists():
        return items
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f"[WARN] Bad label line {label_path}:{line_number}: {line}")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"[WARN] Bad label values {label_path}:{line_number}: {line}")
            continue
        if class_id not in CLASS_NAMES:
            print(f"[WARN] Unknown class id {class_id} in {label_path}:{line_number}")
            continue
        box = yolo_to_xyxy(coords, width, height)
        if box is None:
            continue
        items.append((class_id, box))
    return items


def draw_labels(image: Image.Image, labels) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, output.width // 90))
    except OSError:
        font = ImageFont.load_default()

    for class_id, box in labels:
        x0, y0, x1, y1 = box
        color = PALETTE[class_id]
        line_width = max(2, output.width // 360)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
        text = CLASS_NAMES[class_id]
        bbox = draw.textbbox((x0, y0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_y0 = max(0, y0 - text_h - 6)
        draw.rectangle([x0, label_y0, x0 + text_w + 8, label_y0 + text_h + 6], fill=color)
        draw.text((x0 + 4, label_y0 + 3), text, fill=(0, 0, 0), font=font)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Draw YOLO labels on images.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--vis_dir", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = iter_images(args.image_dir)
    if args.shuffle:
        random.shuffle(images)
    if args.max_images is not None:
        images = images[:args.max_images]
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_path in tqdm(images, desc="visualize"):
        relative = image_path.relative_to(args.image_dir)
        label_path = args.label_dir / relative.with_suffix(".txt")
        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            print(f"[WARN] Could not read image {image_path}: {exc}")
            continue
        labels = read_labels(label_path, image.width, image.height)
        output = draw_labels(image, labels)
        vis_path = args.vis_dir / relative
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(vis_path)
        written += 1

    print(f"Visualized {written} images into {args.vis_dir.resolve()}")


if __name__ == "__main__":
    main()
