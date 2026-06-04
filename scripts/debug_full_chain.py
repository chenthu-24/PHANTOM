import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_planner_controller'))
sys.path.insert(0, str(ROOT / 'src' / 'phantom_safety_shield'))

from phantom_planner_controller.planner_controller_node import compute_planner_command  # noqa: E402
from phantom_safety_shield.safety_shield_node import apply_safety_shield  # noqa: E402


def load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    return None


def scenario_by_name(payload, name):
    for item in (payload or {}).get('scenarios', []):
        if item.get('scenario') == name:
            return item
    return None


def main():
    yolo = load_json(ROOT / 'artifacts' / 'yolo' / 'yolo_dataset_results.json') or {'detections': [], 'error': 'missing yolo artifact'}
    lidar = load_json(ROOT / 'artifacts' / 'lidar' / 'lidar_sim_results.json')
    rear = load_json(ROOT / 'artifacts' / 'rear' / 'rear_depth_sim_results.json')

    open_front = scenario_by_name(lidar, 'A_front_open')['front_free_space']
    left_open = scenario_by_name(lidar, 'B_front_block_left_open')['front_free_space']
    dead_end = scenario_by_name(lidar, 'D_dead_end')['front_free_space'].copy()
    hard_front = dead_end.copy()
    hard_front.update({'front_min': 0.18, 'front_blocked_soft': True, 'front_blocked_hard': True})
    rear_open = scenario_by_name(rear, 'rear_open')['rear_risk']
    rear_near = scenario_by_name(rear, 'rear_near')['rear_risk']
    rear_hard = scenario_by_name(rear, 'rear_very_near')['rear_risk']

    yolo_detections = yolo.get('detections', [])
    threat_detection = next(
        (item for item in yolo_detections if item.get('class_name') in ('yellow_car', 'traffic_cone')),
        {'class_name': 'yellow_car', 'conf': 0.7, 'cx': 0.5, 'cy': 0.5, 'w': 0.2, 'h': 0.2, 'depth': 0.9},
    )

    cases = [
        ('Case 1 CRUISE', open_front, rear_open, [], 'CRUISE'),
        ('Case 2 ESCAPE', open_front, {**rear_open, 'rear_pressure': 0.72, 'threat_visible': True, 'threat_class': 'yellow_car', 'threat_conf': 0.7}, [threat_detection], 'ESCAPE'),
        ('Case 3 AVOID_FRONT', left_open, rear_open, [], 'AVOID_FRONT'),
        ('Case 4 RECOVER', hard_front, rear_open, [], 'RECOVER'),
        ('Case 5 STOP', {**hard_front, 'left_front_min': 0.18, 'right_front_min': 0.18, 'left_min': 0.18, 'right_min': 0.18}, rear_hard, [], 'STOP'),
    ]
    results = []
    for name, front, rear_risk, detections, expected_state in cases:
        planned = compute_planner_command(front, rear_risk, detections)
        safe = apply_safety_shield(planned['cmd_vel_raw'], front, rear_risk)
        state = planned['state']
        passed = state == expected_state
        if name.startswith('Case 4'):
            passed = passed and planned['cmd_vel_raw']['linear_x'] < 0.0 and safe['linear_x'] < 0.0
        if name.startswith('Case 5'):
            passed = passed and safe['linear_x'] == 0.0
        results.append({
            'case': name,
            'front_input': front,
            'rear_input': rear_risk,
            'detections_input': detections,
            'planner_state': state,
            'cmd_vel_raw': planned['cmd_vel_raw'],
            'cmd_vel_safe': safe,
            'passed': bool(passed),
            'reason': 'expected %s, got %s' % (expected_state, state),
        })

    out_dir = ROOT / 'artifacts' / 'chain'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'full_chain_results.json'
    payload = {
        'yolo_fallback_used': not bool(yolo_detections),
        'yolo_source_error': yolo.get('error'),
        'cases': results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'result_json': str(path.relative_to(ROOT)), **payload}, indent=2))
    return 0 if all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
