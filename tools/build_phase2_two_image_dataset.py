#!/usr/bin/env python3
"""Build a tiny PHANTOM traffic_cone/exit YOLO dataset from two lab images.

PHASE2_TWO_IMAGE_OVERFIT_ONLY:
This dataset is intentionally tiny and is only suitable for an initial local
smoke test. It is not a real validation set and does not prove deployment
performance.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2


CLASSES = {
    'traffic_cone': 0,
    'exit': 1,
}


# Boxes are in absolute xyxy pixels for the copied 5712x4284 source images.
ANNOTATIONS = {
    '1.jpg': [
        {'class_name': 'traffic_cone', 'xyxy': [1080, 1500, 1900, 3980]},
        {'class_name': 'traffic_cone', 'xyxy': [2080, 1320, 2600, 2780]},
        {'class_name': 'traffic_cone', 'xyxy': [4160, 1350, 5710, 4283]},
        {'class_name': 'exit', 'xyxy': [2050, 1000, 2920, 2780]},
    ],
    '2.jpg': [
        {'class_name': 'traffic_cone', 'xyxy': [700, 1160, 1520, 3080]},
        {'class_name': 'traffic_cone', 'xyxy': [2180, 1130, 2585, 2320]},
        {'class_name': 'traffic_cone', 'xyxy': [3100, 1150, 4450, 4283]},
        {'class_name': 'exit', 'xyxy': [2581, 951, 3187, 2150]},
    ],
}


def xyxy_to_yolo(box, width, height):
    x0, y0, x1, y1 = [float(value) for value in box]
    x0 = max(0.0, min(x0, width - 1.0))
    x1 = max(0.0, min(x1, width - 1.0))
    y0 = max(0.0, min(y0, height - 1.0))
    y1 = max(0.0, min(y1, height - 1.0))
    return [
        ((x0 + x1) * 0.5) / width,
        ((y0 + y1) * 0.5) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]


def write_labels(label_path, annotations, width, height):
    lines = []
    for item in annotations:
        class_id = CLASSES[item['class_name']]
        x_center, y_center, box_width, box_height = xyxy_to_yolo(item['xyxy'], width, height)
        lines.append(
            '%d %.6f %.6f %.6f %.6f'
            % (class_id, x_center, y_center, box_width, box_height)
        )
    label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def draw_preview(image, annotations):
    output = image.copy()
    colors = {
        'traffic_cone': (0, 140, 255),
        'exit': (40, 220, 40),
    }
    for item in annotations:
        x0, y0, x1, y1 = [int(value) for value in item['xyxy']]
        color = colors[item['class_name']]
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 10)
        label = item['class_name']
        cv2.rectangle(output, (x0, max(0, y0 - 95)), (x0 + 520, y0), color, -1)
        cv2.putText(
            output,
            label,
            (x0 + 15, y0 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (255, 255, 255),
            6,
            cv2.LINE_AA,
        )
    return output


def build_dataset(raw_dir, dataset_dir):
    image_dirs = [
        dataset_dir / 'images' / 'train',
        dataset_dir / 'images' / 'val',
    ]
    label_dirs = [
        dataset_dir / 'labels' / 'train',
        dataset_dir / 'labels' / 'val',
    ]
    preview_dir = dataset_dir / 'previews'

    for directory in image_dirs + label_dirs + [preview_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for image_name, annotations in ANNOTATIONS.items():
        src = raw_dir / image_name
        image = cv2.imread(str(src))
        if image is None:
            raise RuntimeError('Could not read %s' % src)
        height, width = image.shape[:2]

        for split in ['train', 'val']:
            dst_image = dataset_dir / 'images' / split / image_name
            dst_label = dataset_dir / 'labels' / split / (Path(image_name).stem + '.txt')
            shutil.copy2(src, dst_image)
            write_labels(dst_label, annotations, width, height)

        preview = draw_preview(image, annotations)
        cv2.imwrite(str(preview_dir / image_name), preview)

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
        'warning': 'PHASE2_TWO_IMAGE_OVERFIT_ONLY',
        'classes': CLASSES,
        'images': sorted(ANNOTATIONS.keys()),
        'annotation_count': sum(len(items) for items in ANNOTATIONS.values()),
        'data_yaml': str(data_yaml),
        'preview_dir': str(preview_dir),
    }
    summary_path = dataset_dir / 'dataset_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=Path('datasets/phantom_cone_exit_raw'),
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('datasets/phantom_cone_exit_yolo'),
    )
    args = parser.parse_args()

    summary = build_dataset(args.raw_dir, args.dataset_dir)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
