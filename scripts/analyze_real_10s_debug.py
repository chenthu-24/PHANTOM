#!/usr/bin/env python3
import ast
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


def _read(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'count': 0, 'avg': None, 'max': None, 'min': None}
    return {
        'count': len(values),
        'avg': round(statistics.fmean(values), 5),
        'max': round(max(values), 5),
        'min': round(min(values), 5),
    }


def _twists(text):
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if 'linear_x' in payload or 'angular_z' in payload:
            samples.append({
                'linear_x': float(payload.get('linear_x', 0.0)),
                'angular_z': float(payload.get('angular_z', 0.0)),
                'record_time': payload.get('record_time'),
            })
    if samples:
        return samples
    for block in re.split(r'\n---\s*\n', text):
        linear = re.search(r'linear:\s*.*?\n\s*x:\s*([-+0-9.eE]+)', block, re.S)
        angular = re.search(r'angular:\s*.*?\n\s*z:\s*([-+0-9.eE]+)', block, re.S)
        if linear or angular:
            samples.append({
                'linear_x': float(linear.group(1)) if linear else 0.0,
                'angular_z': float(angular.group(1)) if angular else 0.0,
                'record_time': None,
            })
    return samples


def _odometry(text):
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if 'x' in payload or 'linear_x' in payload:
            samples.append({
                'x': payload.get('x'),
                'y': payload.get('y'),
                'linear_x': payload.get('linear_x'),
                'angular_z': payload.get('angular_z'),
            })
    if samples:
        return samples
    for block in re.split(r'\n---\s*\n', text):
        pose = re.search(
            r'pose:\s*.*?pose:\s*.*?position:\s*.*?\n\s*x:\s*([-+0-9.eE]+)\s*\n\s*y:\s*([-+0-9.eE]+)',
            block,
            re.S,
        )
        twist = re.search(
            r'twist:\s*.*?twist:\s*.*?linear:\s*.*?\n\s*x:\s*([-+0-9.eE]+).*?angular:\s*.*?\n\s*z:\s*([-+0-9.eE]+)',
            block,
            re.S,
        )
        if pose or twist:
            samples.append({
                'x': float(pose.group(1)) if pose else None,
                'y': float(pose.group(2)) if pose else None,
                'linear_x': float(twist.group(1)) if twist else None,
                'angular_z': float(twist.group(2)) if twist else None,
            })
    return samples


def _odom_summary(samples):
    positions = [(item['x'], item['y']) for item in samples if item.get('x') is not None and item.get('y') is not None]
    displacement = None
    if len(positions) >= 2:
        x0, y0 = positions[0]
        x1, y1 = positions[-1]
        displacement = round(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 5)
    return {
        'sample_count': len(samples),
        'linear_x': _stats([item.get('linear_x') for item in samples]),
        'angular_z': _stats([item.get('angular_z') for item in samples]),
        'displacement_xy': displacement,
    }


def _decode_data_value(raw):
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, str):
            return value
        return json.dumps(value)
    except Exception:
        return raw.strip("'\"")


def _json_payloads(text):
    payloads = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('data:'):
            line = _decode_data_value(line.split(':', 1)[1])
        if not line:
            continue
        candidates = [line]
        match = re.search(r'(\{.*\}|\[.*\])', line)
        if match:
            candidates.insert(0, match.group(1))
        for candidate in candidates:
            try:
                payloads.append(json.loads(candidate))
                break
            except Exception:
                continue
    return payloads


def _flatten_detections(payload):
    if isinstance(payload, dict) and isinstance(payload.get('data'), list):
        return payload['data']
    if isinstance(payload, dict) and isinstance(payload.get('detections'), list):
        return payload['detections']
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def _num(payload, key):
    try:
        value = float(payload.get(key))
    except Exception:
        return None
    return value


def _dominant(values):
    values = [v for v in values if v not in (None, '')]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _max_spin_run_seconds(samples):
    best = 0.0
    start_time = None
    last_time = None
    fallback_count = 0
    best_count = 0
    for item in samples:
        spinning = abs(item.get('angular_z', 0.0)) > 0.25 and abs(item.get('linear_x', 0.0)) < 0.035
        timestamp = item.get('record_time')
        try:
            timestamp = float(timestamp)
        except Exception:
            timestamp = None
        if spinning:
            if start_time is None:
                start_time = timestamp
                fallback_count = 1
            else:
                fallback_count += 1
            last_time = timestamp
            if start_time is not None and last_time is not None:
                best = max(best, last_time - start_time)
            best_count = max(best_count, fallback_count)
        else:
            start_time = None
            last_time = None
            fallback_count = 0
    if best <= 0.0 and best_count:
        best = best_count / 20.0
    return round(best, 3)


def _last_samples_stopped(samples, count=15):
    tail = samples[-count:]
    if len(tail) < max(3, min(count, 5)):
        return False
    return all(abs(item.get('linear_x', 0.0)) < 0.01 and abs(item.get('angular_z', 0.0)) < 0.01 for item in tail)


def analyze(root):
    root = Path(root)
    raw = _twists(_read(root / 'cmd_vel_raw.txt'))
    safe = _twists(_read(root / 'controller_cmd_vel.txt'))
    front = _json_payloads(_read(root / 'front_free_space.txt'))
    rear = _json_payloads(_read(root / 'rear_risk.txt'))
    detections_payloads = _json_payloads(_read(root / 'detections.txt'))
    planner = _json_payloads(_read(root / 'planner_state.txt'))
    safety = _json_payloads(_read(root / 'safety_decision.txt'))
    odom = _odometry(_read(root / 'odom.txt'))
    odom_raw = _odometry(_read(root / 'odom_raw.txt'))
    try:
        stop_confirmed = bool(json.loads(_read(root / 'stop_confirmed.json') or '{}').get('success', False))
    except Exception:
        stop_confirmed = False

    all_detections = []
    for payload in detections_payloads:
        all_detections.extend(_flatten_detections(payload))
    visible_detections = [
        item for item in all_detections
        if isinstance(item, dict) and bool(item.get('visible', True))
    ]

    front_soft_count = sum(1 for item in front if item.get('front_blocked_soft') or item.get('front_soft'))
    front_hard_count = sum(1 for item in front if item.get('front_blocked_hard') or item.get('front_hard'))
    if not front and planner:
        front_soft_count = sum(1 for item in planner if item.get('front_blocked_soft') or item.get('front_soft'))
        front_hard_count = sum(1 for item in planner if item.get('front_blocked_hard') or item.get('front_hard'))

    traffic_cone_count = sum(
        1 for item in visible_detections
        if isinstance(item, dict) and str(item.get('class_name', '')).strip().lower() == 'traffic_cone'
    )
    z_source = rear or planner
    z_bump_detected_count = sum(1 for item in z_source if bool(item.get('z_bump_detected', False)))
    cone_recover_count = sum(1 for item in planner if item.get('state') == 'CONE_BASE_RECOVER')
    max_spin_s = _max_spin_run_seconds(safe)
    normal_forward = sum(1 for item in safe if item.get('linear_x', 0.0) > 0.05) >= 20
    avoidance_behavior = (
        front_soft_count > 0
        or front_hard_count > 0
        or sum(1 for item in safe if abs(item.get('angular_z', 0.0)) > 0.20 and item.get('linear_x', 0.0) > 0.02) >= 20
    )

    summary = {
        'root': str(root),
        'cmd_vel_raw': {
            'linear_x': _stats([item['linear_x'] for item in raw]),
            'angular_z': _stats([item['angular_z'] for item in raw]),
        },
        'controller_cmd_vel': {
            'linear_x': _stats([item['linear_x'] for item in safe]),
            'angular_z': _stats([item['angular_z'] for item in safe]),
        },
        'front_min': _stats([_num(item, 'front_min') for item in front] or [_num(item, 'front_min') for item in planner]),
        'front_blocked_soft_count': front_soft_count,
        'front_blocked_hard_count': front_hard_count,
        'best_heading': _stats([_num(item, 'best_heading') for item in front] or [_num(item, 'best_heading') for item in planner]),
        'rear_pressure': _stats([_num(item, 'rear_pressure') for item in rear] or [_num(item, 'rear_pressure') for item in planner]),
        'yolo_detection_count': len(visible_detections),
        'traffic_cone_detection_count': traffic_cone_count,
        'yolo_classes': dict(Counter(str(item.get('class_name', '')) for item in visible_detections)),
        'z_bump_detected_count': z_bump_detected_count,
        'z_bump_score': _stats([_num(item, 'z_bump_score') for item in z_source]),
        'z_bump_depth_jump_m': _stats([_num(item, 'z_bump_depth_jump_m') for item in rear]),
        'cone_base_recover_count': cone_recover_count,
        'long_spin_seconds': max_spin_s,
        'long_time_spin_detected': max_spin_s >= 5.0,
        'normal_forward_detected': bool(normal_forward),
        'avoidance_behavior_detected': bool(avoidance_behavior),
        'stop_success': bool(_last_samples_stopped(safe) or stop_confirmed),
        'stop_success_source': 'controller_cmd_tail' if _last_samples_stopped(safe) else (
            'stop_confirmed_json' if stop_confirmed else 'missing'
        ),
        'dominant_planner_state': _dominant([item.get('state') for item in planner]),
        'dominant_safety_reason': _dominant([item.get('reason') for item in safety]),
        'odom': _odom_summary(odom),
        'odom_raw': _odom_summary(odom_raw),
        'sample_counts': {
            'cmd_vel_raw': len(raw),
            'controller_cmd_vel': len(safe),
            'front_free_space': len(front),
            'rear_risk': len(rear),
            'detections_messages': len(detections_payloads),
            'planner_state': len(planner),
            'safety_decision': len(safety),
            'odom': len(odom),
            'odom_raw': len(odom_raw),
        },
    }
    return summary


def _fmt_stats(label, data):
    return '%s avg/max/min: %s / %s / %s (n=%s)' % (
        label, data.get('avg'), data.get('max'), data.get('min'), data.get('count')
    )


def write_outputs(root, summary):
    root = Path(root)
    (root / 'analysis_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    lines = [
        _fmt_stats('/cmd_vel_raw.linear.x', summary['cmd_vel_raw']['linear_x']),
        _fmt_stats('/cmd_vel_raw.angular.z', summary['cmd_vel_raw']['angular_z']),
        _fmt_stats('/controller/cmd_vel.linear.x', summary['controller_cmd_vel']['linear_x']),
        _fmt_stats('/controller/cmd_vel.angular.z', summary['controller_cmd_vel']['angular_z']),
        _fmt_stats('front_min', summary['front_min']),
        'front_blocked_soft/hard count: %s / %s' % (
            summary['front_blocked_soft_count'],
            summary['front_blocked_hard_count'],
        ),
        _fmt_stats('best_heading trend', summary['best_heading']),
        _fmt_stats('rear_pressure', summary['rear_pressure']),
        'YOLO detection count: %s classes=%s' % (
            summary['yolo_detection_count'],
            summary['yolo_classes'],
        ),
        'traffic_cone detection count: %s' % summary['traffic_cone_detection_count'],
        'z_bump_detected count: %s' % summary['z_bump_detected_count'],
        _fmt_stats('z_bump_score', summary['z_bump_score']),
        _fmt_stats('z_bump_depth_jump_m', summary['z_bump_depth_jump_m']),
        'CONE_BASE_RECOVER count: %s' % summary['cone_base_recover_count'],
        'long_time_spin_detected: %s (max_spin_seconds=%s)' % (
            summary['long_time_spin_detected'],
            summary['long_spin_seconds'],
        ),
        'normal_forward_detected: %s' % summary['normal_forward_detected'],
        'avoidance_behavior_detected: %s' % summary['avoidance_behavior_detected'],
        'stop_success: %s (source=%s)' % (summary['stop_success'], summary['stop_success_source']),
        'dominant planner state: %s' % summary['dominant_planner_state'],
        'dominant safety reason: %s' % summary['dominant_safety_reason'],
        _fmt_stats('/odom.twist.linear.x', summary['odom']['linear_x']),
        _fmt_stats('/odom.twist.angular.z', summary['odom']['angular_z']),
        '/odom displacement_xy: %s' % summary['odom']['displacement_xy'],
        _fmt_stats('/odom_raw.twist.linear.x', summary['odom_raw']['linear_x']),
        _fmt_stats('/odom_raw.twist.angular.z', summary['odom_raw']['angular_z']),
        '/odom_raw displacement_xy: %s' % summary['odom_raw']['displacement_xy'],
        'sample counts: %s' % summary['sample_counts'],
    ]
    (root / 'analysis_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv):
    if len(argv) != 2:
        print('usage: analyze_real_10s_debug.py <artifact_dir>', file=sys.stderr)
        return 2
    root = Path(argv[1])
    summary = analyze(root)
    write_outputs(root, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
