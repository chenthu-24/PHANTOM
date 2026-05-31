#!/usr/bin/env python3
"""PHASE2_DEMO_HEURISTIC_ONLY traffic cone and exit smoke test.

This script is a local validation helper for PHANTOM phase 2 images. It is not
a trained detector and should not be used as real-platform perception logic.
Replace it with a trained YOLO traffic_cone/exit model before deployment.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    return [
        max(0, min(int(x0), width - 1)),
        max(0, min(int(y0), height - 1)),
        max(0, min(int(x1), width - 1)),
        max(0, min(int(y1), height - 1)),
    ]


def _to_detection(class_name, confidence, box, width, height, source):
    x0, y0, x1, y1 = [float(v) for v in box]
    return {
        'visible': True,
        'class_name': class_name,
        'conf': round(float(confidence), 3),
        'bbox_center_x': round(((x0 + x1) * 0.5) / width, 4),
        'bbox_center_y': round(((y0 + y1) * 0.5) / height, 4),
        'bbox_w': round((x1 - x0) / width, 4),
        'bbox_h': round((y1 - y0) / height, 4),
        'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
        'source': source,
    }


def _contour_boxes(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue
        boxes.append({'area': area, 'box': [x, y, x + w, y + h]})
    return sorted(boxes, key=lambda item: item['area'], reverse=True)


def _merge_overlapping_x_boxes(boxes):
    merged = []
    for box in sorted(boxes, key=lambda item: item[0]):
        did_merge = False
        for index, current in enumerate(merged):
            overlap = min(box[2], current[2]) - max(box[0], current[0])
            min_width = min(box[2] - box[0], current[2] - current[0])
            if overlap > 0.20 * max(min_width, 1):
                merged[index] = [
                    min(box[0], current[0]),
                    min(box[1], current[1]),
                    max(box[2], current[2]),
                    max(box[3], current[3]),
                ]
                did_merge = True
                break
        if not did_merge:
            merged.append(box)
    return merged


def detect_demo_cones_and_exit(image):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_1 = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([18, 255, 255]))
    red_2 = cv2.inRange(hsv, np.array([170, 45, 45]), np.array([180, 255, 255]))
    orange = cv2.inRange(hsv, np.array([5, 35, 40]), np.array([28, 255, 255]))
    red_mask = cv2.bitwise_or(cv2.bitwise_or(red_1, red_2), orange)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))

    yellow_mask = cv2.inRange(hsv, np.array([18, 55, 45]), np.array([42, 255, 255]))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))

    detections = []

    red_boxes = []
    for candidate in _contour_boxes(red_mask, min_area=20000):
        x0, y0, x1, y1 = candidate['box']
        box_w = x1 - x0
        box_h = y1 - y0
        aspect = box_h / max(box_w, 1)
        if box_h < 250 or aspect < 0.8:
            continue

        margin_x = int(0.20 * box_w)
        margin_top = int(0.95 * box_h)
        margin_bottom = int(0.40 * box_h)
        red_boxes.append(_clamp_box(
            [x0 - margin_x, y0 - margin_top, x1 + margin_x, y1 + margin_bottom],
            width,
            height,
        ))

    for box in _merge_overlapping_x_boxes(red_boxes):
        detections.append(_to_detection(
            'traffic_cone',
            0.88,
            box,
            width,
            height,
            'red_cone_color_geometry',
        ))

    for candidate in _contour_boxes(yellow_mask, min_area=80000):
        x0, y0, x1, y1 = candidate['box']
        box_w = x1 - x0
        box_h = y1 - y0
        if box_h < 400 or box_h / max(box_w, 1) < 1.0:
            continue

        box = _clamp_box(
            [
                x0 - int(0.50 * box_w),
                y0 - int(0.45 * box_h),
                x1 + int(0.15 * box_w),
                y1 + int(0.70 * box_h),
            ],
            width,
            height,
        )
        detections.append(_to_detection(
            'traffic_cone',
            0.86,
            box,
            width,
            height,
            'yellow_black_cone_color_geometry',
        ))

    # In this phase-2 lab layout, "exit" is the navigable vertical gap between
    # two blue barrier panels. This is deliberately simple geometry, not a
    # semantic exit detector.
    exit_box = [
        int(width * 0.452),
        int(height * 0.222),
        int(width * 0.558),
        int(height * 0.502),
    ]
    detections.append(_to_detection(
        'exit',
        0.76,
        exit_box,
        width,
        height,
        'barrier_gap_geometry',
    ))

    return detections


def draw_detections(image, detections):
    output = image.copy()
    palette = {
        'traffic_cone': (0, 140, 255),
        'exit': (40, 220, 40),
    }
    for detection in detections:
        x0, y0, x1, y1 = detection['bbox_xyxy']
        color = palette.get(detection['class_name'], (255, 255, 255))
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 12)
        label = '%s %.2f' % (detection['class_name'], detection['conf'])
        font_scale = 3.0
        thickness = 8
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        y_label = max(0, y0 - text_h - baseline - 18)
        cv2.rectangle(output, (x0, y_label), (x0 + text_w + 18, y0), color, -1)
        cv2.putText(
            output,
            label,
            (x0 + 9, y0 - baseline - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('runs/phase2_smoke/cone_exit_demo'),
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit('Could not read image: %s' % args.image)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detections = detect_demo_cones_and_exit(image)
    annotated = draw_detections(image, detections)

    image_out = args.out_dir / args.image.name
    json_out = args.out_dir / 'detections.json'
    cv2.imwrite(str(image_out), annotated)
    json_out.write_text(json.dumps(detections, indent=2, ensure_ascii=False), encoding='utf-8')

    print('PHASE2_DEMO_HEURISTIC_ONLY')
    print('annotated=%s' % image_out)
    print('json=%s' % json_out)
    print(json.dumps(detections, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
