#!/usr/bin/env python3
import ast
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _read(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''


def _decode_data(raw):
    raw = raw.strip()
    try:
        value = ast.literal_eval(raw)
        return value if isinstance(value, str) else json.dumps(value)
    except Exception:
        return raw.strip("'\"")


def _json_payloads(path):
    payloads = []
    for line in _read(path).splitlines():
        line = line.strip()
        if line.startswith('data:'):
            line = _decode_data(line.split(':', 1)[1])
        if not line:
            continue
        try:
            payloads.append(json.loads(line))
        except Exception:
            continue
    return payloads


def _flatten(payload):
    if isinstance(payload, dict) and isinstance(payload.get('data'), list):
        return payload['data']
    if isinstance(payload, dict) and isinstance(payload.get('detections'), list):
        return payload['detections']
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def _put_text(image, text, org, scale=0.55, color=(235, 235, 235), thickness=1):
    if cv2 is None:
        return
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _save(path, image):
    if cv2 is None:
        raise RuntimeError('OpenCV is required for export_debug_images.py')
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError('failed to write %s' % path)


def export_lidar(root):
    front = _json_payloads(root / 'front_free_space.txt')
    planner = _json_payloads(root / 'planner_state.txt')
    payload = (front or planner or [{}])[-1]
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[:] = (24, 28, 34)
    center = (320, 470)
    scale = 180.0

    if cv2 is not None:
        cv2.circle(image, center, int(0.22 * scale), (30, 70, 180), 1)
        cv2.circle(image, center, int(0.35 * scale), (40, 130, 210), 1)
        cv2.circle(image, center, int(0.50 * scale), (60, 160, 80), 1)
        for degrees in (-110, -55, -15, 0, 15, 55, 110):
            angle = math.radians(degrees)
            end = (
                int(center[0] + math.sin(angle) * 2.4 * scale),
                int(center[1] - math.cos(angle) * 2.4 * scale),
            )
            cv2.line(image, center, end, (58, 65, 76), 1)

    sector_distances = {
        'left': float(payload.get('left_min', 0.0) or 0.0),
        'front_left': float(payload.get('left_front_min', 0.0) or 0.0),
        'front': float(payload.get('front_min', 0.0) or 0.0),
        'front_right': float(payload.get('right_front_min', 0.0) or 0.0),
        'right': float(payload.get('right_min', 0.0) or 0.0),
    }
    headings = {'left': 1.25, 'front_left': 0.55, 'front': 0.0, 'front_right': -0.55, 'right': -1.25}
    for name, distance in sector_distances.items():
        distance = min(max(distance, 0.0), 2.4)
        angle = headings[name]
        color = (80, 200, 90)
        if name == 'front' and payload.get('front_blocked_hard'):
            color = (40, 40, 230)
        elif name == 'front' and payload.get('front_blocked_soft'):
            color = (40, 180, 240)
        end = (
            int(center[0] + math.sin(angle) * distance * scale),
            int(center[1] - math.cos(angle) * distance * scale),
        )
        cv2.line(image, center, end, color, 5)
        cv2.circle(image, end, 7, color, -1)

    best_heading = float(payload.get('best_heading', payload.get('selected_heading', 0.0)) or 0.0)
    best_end = (
        int(center[0] + math.sin(best_heading) * 2.0 * scale),
        int(center[1] - math.cos(best_heading) * 2.0 * scale),
    )
    cv2.arrowedLine(image, center, best_end, (255, 210, 80), 3, tipLength=0.08)

    lines = [
        'LiDAR free-space debug',
        'front_min=%.3f m  best_heading=%.3f rad' % (
            float(payload.get('front_min', 0.0) or 0.0),
            best_heading,
        ),
        'blocked_soft=%s  blocked_hard=%s' % (
            bool(payload.get('front_blocked_soft') or payload.get('front_soft')),
            bool(payload.get('front_blocked_hard') or payload.get('front_hard')),
        ),
        'best_sector=%s' % payload.get('best_sector', payload.get('selected_direction', 'unknown')),
    ]
    y = 32
    for line in lines:
        _put_text(image, line, (18, y), 0.6, (235, 235, 235), 1)
        y += 28
    _save(root / 'lidar_debug.png', image)


def export_yolo(root):
    target = root / 'yolo_debug.png'
    for candidate in (root / 'first_yolo_detection.png', root / 'yolo_debug_raw.png'):
        if candidate.exists():
            shutil.copyfile(candidate, target)
            return

    detections = []
    for payload in _json_payloads(root / 'detections.txt'):
        detections.extend(_flatten(payload))
    visible = [item for item in detections if isinstance(item, dict) and item.get('visible', True)]

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (30, 31, 36)
    _put_text(image, 'YOLO debug', (20, 36), 0.8, (245, 245, 245), 2)
    if not visible:
        _put_text(image, 'No YOLO detections in captured /det/detections.', (20, 78), 0.62, (80, 210, 255), 2)
    for item in visible:
        cx = float(item.get('cx', item.get('bbox_center_x', 0.5)) or 0.5)
        cy = float(item.get('cy', item.get('bbox_center_y', 0.5)) or 0.5)
        bw = float(item.get('w', item.get('bbox_w', 0.2)) or 0.2)
        bh = float(item.get('h', item.get('bbox_h', 0.2)) or 0.2)
        x0 = int(max(0, (cx - bw * 0.5) * 640))
        y0 = int(max(0, (cy - bh * 0.5) * 480))
        x1 = int(min(639, (cx + bw * 0.5) * 640))
        y1 = int(min(479, (cy + bh * 0.5) * 480))
        color = (0, 220, 255)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        label = '%s %.2f' % (item.get('class_name', ''), float(item.get('conf', 0.0) or 0.0))
        _put_text(image, label, (x0, max(24, y0 - 8)), 0.55, color, 2)
    _save(target, image)


def export_z_bump(root):
    rear = _json_payloads(root / 'rear_risk.txt')
    planner = _json_payloads(root / 'planner_state.txt')
    z_values = [
        float(item.get('z_bump_score', 0.0) or 0.0)
        for item in (rear or planner)
        if isinstance(item, dict)
    ]
    detected = [
        item for item in (rear or planner)
        if isinstance(item, dict) and bool(item.get('z_bump_detected', False))
    ]
    recover = [
        item for item in planner
        if isinstance(item, dict) and item.get('state') == 'CONE_BASE_RECOVER'
    ]

    image = np.zeros((480, 760, 3), dtype=np.uint8)
    image[:] = (26, 29, 34)
    _put_text(image, 'Rear z-bump / cone-base debug', (20, 36), 0.78, (245, 245, 245), 2)
    _put_text(image, 'depth-image lower ROI approximation, filtered over time', (20, 68), 0.52, (185, 195, 205), 1)

    plot_x0, plot_y0, plot_w, plot_h = 60, 120, 660, 250
    cv2.rectangle(image, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (70, 78, 90), 1)
    cv2.line(
        image,
        (plot_x0, int(plot_y0 + plot_h * (1.0 - 0.65))),
        (plot_x0 + plot_w, int(plot_y0 + plot_h * (1.0 - 0.65))),
        (40, 180, 240),
        1,
    )
    _put_text(image, 'trigger 0.65', (plot_x0 + 8, int(plot_y0 + plot_h * (1.0 - 0.65)) - 6), 0.45, (70, 210, 255), 1)

    if z_values:
        points = []
        for idx, value in enumerate(z_values):
            x = plot_x0 + int((idx / max(1, len(z_values) - 1)) * plot_w)
            y = plot_y0 + plot_h - int(max(0.0, min(1.0, value)) * plot_h)
            points.append((x, y))
        for first, second in zip(points, points[1:]):
            cv2.line(image, first, second, (120, 220, 120), 2)
    else:
        _put_text(image, 'No z_bump_score samples captured.', (90, 240), 0.62, (80, 210, 255), 2)

    latest = (rear or planner or [{}])[-1]
    lines = [
        'samples=%d  detected_count=%d  recover_count=%d' % (len(z_values), len(detected), len(recover)),
        'max_score=%.3f  latest_score=%.3f' % (
            max(z_values) if z_values else 0.0,
            z_values[-1] if z_values else 0.0,
        ),
        'latest_side=%s  latest_reason=%s' % (
            latest.get('z_bump_side', 'none'),
            str(latest.get('z_bump_reason', ''))[:54],
        ),
    ]
    y = 405
    for line in lines:
        _put_text(image, line, (24, y), 0.56, (235, 235, 235), 1)
        y += 26
    _save(root / 'z_bump_debug.png', image)


def main(argv):
    if len(argv) != 2:
        print('usage: export_debug_images.py <artifact_dir>', file=sys.stderr)
        return 2
    if cv2 is None:
        print('OpenCV/cv2 is required to create debug PNGs', file=sys.stderr)
        return 1
    root = Path(argv[1])
    root.mkdir(parents=True, exist_ok=True)
    export_lidar(root)
    export_yolo(root)
    export_z_bump(root)
    print(root / 'lidar_debug.png')
    print(root / 'yolo_debug.png')
    print(root / 'z_bump_debug.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
