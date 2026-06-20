#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _payloads(root, name):
    path = root / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


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


def _put_text(image, text, org, scale=0.65, color=(245, 245, 245), thickness=2):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main(argv):
    if len(argv) != 2:
        print('usage: export_yellow_z_bump_debug.py <artifact_dir>', file=sys.stderr)
        return 2
    root = Path(argv[1])
    detections = []
    for payload in _payloads(root, 'detections.txt'):
        detections.extend(_flatten(payload))
    visible = [
        item for item in detections
        if isinstance(item, dict) and bool(item.get('visible', True))
    ]
    yellow = [
        item for item in visible
        if str(item.get('class_name', '')).strip().lower() == 'yellow_car'
    ]
    rear = _payloads(root, 'rear_risk.txt')
    planner = _payloads(root, 'planner_state.txt')
    summary = {
        'visible_detection_count': len(visible),
        'classes': dict(Counter(str(item.get('class_name', '')) for item in visible)),
        'yellow_car_count': len(yellow),
        'yellow_car_max_conf': max([float(item.get('conf', 0.0) or 0.0) for item in yellow], default=0.0),
        'yellow_car_samples': yellow[:10],
        'z_bump_detected_count': sum(1 for item in rear if item.get('z_bump_detected')),
        'z_bump_score_max': max([float(item.get('z_bump_score', 0.0) or 0.0) for item in rear], default=0.0),
        'z_bump_side_counts': dict(Counter(str(item.get('z_bump_side', 'none')) for item in rear)),
        'cone_base_recover_count': sum(1 for item in planner if item.get('state') == 'CONE_BASE_RECOVER'),
    }
    (root / 'yellow_z_bump_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    if cv2 is not None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (28, 30, 36)
        if yellow:
            _put_text(image, 'YOLO yellow_car detections', (20, 36), 0.8, (0, 255, 255), 2)
            _put_text(
                image,
                'count=%d max_conf=%.3f' % (len(yellow), summary['yellow_car_max_conf']),
                (20, 70),
                0.6,
                (235, 235, 235),
                1,
            )
            for item in yellow[:12]:
                cx = float(item.get('cx', item.get('bbox_center_x', 0.5)) or 0.5)
                cy = float(item.get('cy', item.get('bbox_center_y', 0.5)) or 0.5)
                bw = float(item.get('w', item.get('bbox_w', 0.2)) or 0.2)
                bh = float(item.get('h', item.get('bbox_h', 0.2)) or 0.2)
                x0 = max(0, int((cx - bw / 2.0) * 640))
                y0 = max(0, int((cy - bh / 2.0) * 480))
                x1 = min(639, int((cx + bw / 2.0) * 640))
                y1 = min(479, int((cy + bh / 2.0) * 480))
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 220, 255), 2)
                _put_text(
                    image,
                    'yellow_car %.2f' % float(item.get('conf', 0.0) or 0.0),
                    (x0, max(100, y0 - 8)),
                    0.52,
                    (0, 220, 255),
                    2,
                )
        else:
            _put_text(image, 'YOLO yellow_car debug', (20, 36), 0.8, (245, 245, 245), 2)
            _put_text(image, 'No yellow_car detection captured in this 60s run.', (20, 90), 0.62, (80, 210, 255), 2)
            _put_text(image, 'See detections.txt and yellow_z_bump_summary.json.', (20, 126), 0.52, (210, 210, 210), 1)
        cv2.imwrite(str(root / 'yellow_car_debug.png'), image)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
