#!/usr/bin/env python3
"""Split images and YOLO labels into a train/val dataset."""

import argparse
import random
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_PRESETS = {
    "all": {
        0: "traffic_cone",
        1: "yellow_car",
        2: "exit",
    },
    "cone_yellow": {
        0: "traffic_cone",
        1: "yellow_car",
    },
}


def iter_images(image_dir: Path) -> List[Path]:
    return sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def path_value(output_dir: Path) -> str:
    if output_dir.is_absolute():
        return output_dir.resolve().as_posix()
    normalized = output_dir.as_posix().lstrip("./")
    return f"./{normalized}"


def write_data_yaml(output_dir: Path, classes: str) -> None:
    lines = [
        f"path: {path_value(output_dir)}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for class_id, class_name in CLASS_PRESETS[classes].items():
        lines.append(f"  {class_id}: {class_name}")
    content = "\n".join(lines) + "\n"
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def copy_pair(image_path: Path, image_dir: Path, label_dir: Path, output_dir: Path, split: str) -> bool:
    relative = image_path.relative_to(image_dir)
    dst_image = output_dir / "images" / split / relative
    dst_label = output_dir / "labels" / split / relative.with_suffix(".txt")
    src_label = label_dir / relative.with_suffix(".txt")

    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dst_image)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
        return True
    dst_label.write_text("", encoding="utf-8")
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Split YOLO images and labels into train/val folders.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--label_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        choices=sorted(CLASS_PRESETS),
        default="all",
        help="Class preset for generated data.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise SystemExit("--val_ratio must be between 0 and 1")
    if not args.image_dir.exists():
        raise SystemExit(f"Image directory does not exist: {args.image_dir}")

    images = iter_images(args.image_dir)
    rng = random.Random(args.seed)
    rng.shuffle(images)
    val_count = int(round(len(images) * args.val_ratio))
    val_images = set(images[:val_count])

    for split in ["train", "val"]:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing_labels = 0
    counts = {"train": 0, "val": 0}
    for image_path in tqdm(images, desc="split"):
        split = "val" if image_path in val_images else "train"
        has_label = copy_pair(image_path, args.image_dir, args.label_dir, args.output_dir, split)
        if not has_label:
            missing_labels += 1
        counts[split] += 1

    write_data_yaml(args.output_dir, args.classes)
    print(f"Dataset written to {args.output_dir.resolve()}")
    print(f"  train images: {counts['train']}")
    print(f"  val images: {counts['val']}")
    print(f"  missing labels written as empty txt: {missing_labels}")
    print(f"  data yaml: {(args.output_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
