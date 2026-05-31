#!/usr/bin/env python3
"""Build a tiny augmented traffic_cone/exit YOLO dataset from two images.

PHASE2_TWO_IMAGE_AUGMENTED_OVERFIT_ONLY:
The generated crops are derived only from the two supplied images. This is a
smoke-test dataset, not a real training corpus.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

from build_phase2_two_image_dataset import ANNOTATIONS, CLASSES, write_labels


EXTRA_CROPS = {
    '1.jpg': [
        {'name': 'exit_wide', 'xyxy': [1750, 700, 3300, 3100]},
        {'name': 'exit_tight', 'xyxy': [1950, 850, 3100, 2450]},
        {'name': 'left_cone', 'xyxy': [800, 1200, 2100, 4100]},
        {'name': 'right_cone', 'xyxy': [3850, 1050, 5712, 4284]},
    ],
    '2.jpg': [
        {'name': 'exit_wide', 'xyxy': [2300, 650, 3500, 2550]},
        {'name': 'exit_tight', 'xyxy': [2480, 800, 3300, 2300]},
        {'name': 'left_cone', 'xyxy': [500, 900, 1750, 3250]},
        {'name': 'right_cone', 'xyxy': [2850, 850, 4650, 4284]},
    ],
}


def intersect_xyxy(a, b):
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def crop_annotations(annotations, crop_box):
    cropped = []
    for item in annotations:
        intersection = intersect_xyxy(item['xyxy'], crop_box)
        if intersection is None:
            continue
        if area(intersection) / max(area(item['xyxy']), 1) < 0.30:
            continue
        x0, y0, _, _ = crop_box
        cropped.append({
            'class_name': item['class_name'],
            'xyxy': [
                intersection[0] - x0,
                intersection[1] - y0,
                intersection[2] - x0,
                intersection[3] - y0,
            ],
        })
    return cropped


def save_sample(image, annotations, stem, split, dataset_dir):
    image_path = dataset_dir / 'images' / split / (stem + '.jpg')
    label_path = dataset_dir / 'labels' / split / (stem + '.txt')
    cv2.imwrite(str(image_path), image)
    height, width = image.shape[:2]
    write_labels(label_path, annotations, width, height)


def build_dataset(raw_dir, dataset_dir):
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    preview_dir = dataset_dir / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    for image_name, annotations in ANNOTATIONS.items():
        source_path = raw_dir / image_name
        image = cv2.imread(str(source_path))
        if image is None:
            raise RuntimeError('Could not read %s' % source_path)

        stem = Path(image_name).stem
        for split in ['train', 'val']:
            dst_image = dataset_dir / 'images' / split / image_name
            dst_label = dataset_dir / 'labels' / split / (stem + '.txt')
            shutil.copy2(source_path, dst_image)
            write_labels(dst_label, annotations, image.shape[1], image.shape[0])
        sample_count += 1

        for crop in EXTRA_CROPS[image_name]:
            x0, y0, x1, y1 = crop['xyxy']
            crop_image = image[y0:y1, x0:x1]
            crop_items = crop_annotations(annotations, crop['xyxy'])
            if not crop_items:
                continue
            crop_stem = '%s_%s' % (stem, crop['name'])
            for split in ['train', 'val']:
                save_sample(crop_image, crop_items, crop_stem, split, dataset_dir)
            sample_count += 1

    data_yaml = dataset_dir / 'data.yaml'
    data_yaml.write_text(
        '\n'.join([
            'path: %s' % dataset_dir.resolve().as_posix(),
            'train: images/train',
            'val: images/val',
            'names:',
            '  0: traffic_cone',
            '  1: exit',
            '',
        ]),
        encoding='utf-8',
    )

    summary = {
        'warning': 'PHASE2_TWO_IMAGE_AUGMENTED_OVERFIT_ONLY',
        'source_images': sorted(ANNOTATIONS.keys()),
        'sample_count_per_split': sample_count,
        'classes': CLASSES,
        'data_yaml': str(data_yaml),
    }
    (dataset_dir / 'dataset_summary.json').write_text(
        json.dumps(summary, indent=2),
        encoding='utf-8',
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, default=Path('datasets/phantom_cone_exit_raw'))
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('datasets/phantom_cone_exit_yolo_aug'),
    )
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.raw_dir, args.dataset_dir), indent=2))


if __name__ == '__main__':
    main()
