import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_detector'))

from phantom_detector.detector_node import normalize_class_name  # noqa: E402


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
EXPECTED_CLASSES = {'traffic_cone', 'yellow_car', 'exit'}


def find_model(root):
    preferred = [
        root / 'models' / 'yolo' / 'phantom_cone_yellow_random200_best.pt',
        root / 'models' / 'yolo' / 'phantom_cone_yellow_yolov8n_best.pt',
        root / 'runs' / 'detect' / 'train_cone_yellow' / 'yolov8n_autolabel_gpu' / 'weights' / 'best.pt',
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(root.rglob('*.pt'), key=lambda item: (('best' not in item.name.lower()), len(str(item))))
    return candidates[0] if candidates else None


def find_images(root, limit):
    search_roots = [
        root / 'data' / 'eval',
        root / 'data' / 'train',
        root / 'datasets',
        root / 'images',
    ]
    images = []
    for base in search_roots:
        if base.exists():
            images.extend(path for path in sorted(base.rglob('*')) if path.suffix.lower() in IMAGE_EXTS)
        if len(images) >= limit:
            break
    if len(images) < limit:
        images.extend(path for path in sorted(root.rglob('*')) if path.suffix.lower() in IMAGE_EXTS and path not in images)
    return images[:limit]


def parse_result(result, image_path):
    names = getattr(result, 'names', {}) or {}
    boxes = getattr(result, 'boxes', None)
    height, width = result.orig_shape[:2]
    detections = []
    if boxes is None:
        return detections
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = normalize_class_name(names.get(class_id, class_id))
        conf = float(box.conf[0])
        x0, y0, x1, y1 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append({
            'image': image_path.name,
            'class_name': class_name,
            'conf': round(conf, 3),
            'cx': round(((x0 + x1) * 0.5) / width, 4),
            'cy': round(((y0 + y1) * 0.5) / height, 4),
            'w': round((x1 - x0) / width, 4),
            'h': round((y1 - y0) / height, 4),
        })
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'artifacts' / 'yolo')
    args = parser.parse_args()

    model_path = args.model or find_model(ROOT)
    images = find_images(ROOT, max(args.limit, 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_json = args.out_dir / 'yolo_dataset_results.json'

    summary = {
        'model_path': str(model_path) if model_path else None,
        'tested_images': [str(path.relative_to(ROOT)) for path in images],
        'image_count': len(images),
        'detections': [],
        'per_image': [],
        'model_class_names': {},
        'normalized_model_class_names': [],
        'expected_classes_missing': [],
        'error': None,
    }

    if model_path is None or not model_path.exists():
        summary['error'] = 'YOLO model .pt file was not found'
        result_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        return 2
    if not images:
        summary['error'] = 'No local dataset images were found'
        result_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        return 2

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
    except Exception as exc:
        summary['error'] = 'Failed to load YOLO model or ultralytics dependency: %s' % exc
        summary['install_command'] = 'python -m pip install ultralytics opencv-python'
        result_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        return 2

    raw_names = getattr(model, 'names', {}) or {}
    summary['model_class_names'] = {str(key): str(value) for key, value in raw_names.items()}
    normalized_names = sorted({normalize_class_name(value) for value in raw_names.values()})
    summary['normalized_model_class_names'] = normalized_names
    summary['expected_classes_missing'] = sorted(EXPECTED_CLASSES - set(normalized_names))

    for index, image_path in enumerate(images):
        results = model.predict(source=str(image_path), imgsz=640, conf=0.25, verbose=False)
        result = results[0]
        detections = parse_result(result, image_path)
        annotated = result.plot()
        annotated_path = args.out_dir / ('annotated_%02d_%s' % (index, image_path.name))
        cv2.imwrite(str(annotated_path), annotated)
        summary['detections'].extend(detections)
        summary['per_image'].append({
            'image': str(image_path.relative_to(ROOT)),
            'detection_count': len(detections),
            'detections': detections,
            'annotated_path': str(annotated_path.relative_to(ROOT)),
        })

    summary['detection_count'] = len(summary['detections'])
    result_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('YOLO_MODEL_PATH=%s' % model_path)
    print('YOLO_TEST_IMAGE_COUNT=%d' % summary['image_count'])
    print('YOLO_DETECTION_COUNT=%d' % summary['detection_count'])
    print('YOLO_RESULT_JSON=%s' % result_json.relative_to(ROOT))
    print('YOLO_ANNOTATED_DIR=%s' % args.out_dir.relative_to(ROOT))
    if summary['expected_classes_missing']:
        print('YOLO_CLASS_WARNING missing expected normalized names: %s' % ', '.join(summary['expected_classes_missing']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
