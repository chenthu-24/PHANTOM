import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_planner_controller'))
sys.path.insert(0, str(ROOT / 'src' / 'phantom_safety_shield'))

from phantom_planner_controller.planner_controller_node import (  # noqa: E402
    _smoothed_angular_z,
    compute_planner_command,
)
from phantom_safety_shield.safety_shield_node import apply_safety_shield  # noqa: E402


PARAMS = {
    'max_probe_angle_deg': 18.0,
    'max_probe_angular_z': 0.30,
    'probe_angle_step_deg': 6.0,
    'turn_rate_limit': 0.08,
    'angular_smoothing_alpha': 0.30,
    'max_angular_z_near_obstacle': 0.30,
    'min_safe_forward_speed': 0.025,
    'obstacle_slowdown_distance': 0.60,
    'max_wz': 0.55,
}


def front_payload(left_open=True):
    return {
        'valid': True,
        'front_unknown': False,
        'front_min': 0.32,
        'front_blocked_soft': True,
        'front_blocked_hard': False,
        'front_path_safe': True,
        'best_heading': 0.0,
        'best_gap_heading': 0.0,
        'best_gap_width_m': 0.0,
        'min_exit_corridor_width_m': 0.39,
        'left_front_min': 0.85 if left_open else 0.28,
        'left_min': 0.85 if left_open else 0.28,
        'right_front_min': 0.28 if left_open else 0.85,
        'right_min': 0.28 if left_open else 0.85,
        'sector_scores': {
            'front': -0.05,
            'front_left': 0.78 if left_open else 0.30,
            'left': 0.78 if left_open else 0.30,
            'front_right': 0.30 if left_open else 0.78,
            'right': 0.30 if left_open else 0.78,
        },
    }


def main():
    rear = {'valid': True, 'reverse_allowed': True, 'rear_blocked_hard': False}
    raw_commands = []
    safe_commands = []
    smoothed = []
    previous = 0.0
    alternating = [True, False] * 8
    for left_open in alternating:
        front = front_payload(left_open)
        planned = compute_planner_command(front, rear, [], PARAMS)
        safe = apply_safety_shield(planned['cmd_vel_raw'], front, rear, PARAMS)
        previous = _smoothed_angular_z(
            previous,
            planned['cmd_vel_raw']['angular_z'],
            PARAMS['angular_smoothing_alpha'],
            PARAMS['turn_rate_limit'],
            PARAMS['max_angular_z_near_obstacle'],
        )
        raw_commands.append(planned['cmd_vel_raw']['angular_z'])
        safe_commands.append(safe['angular_z'])
        smoothed.append(round(previous, 4))

    raw_abs_max = max(abs(value) for value in raw_commands)
    safe_abs_max = max(abs(value) for value in safe_commands)
    smooth_abs_max = max(abs(value) for value in smoothed)
    smooth_max_delta = max(abs(smoothed[i] - smoothed[i - 1]) for i in range(1, len(smoothed)))
    sign_flips = sum(
        1
        for i in range(1, len(smoothed))
        if abs(smoothed[i]) > 0.05
        and abs(smoothed[i - 1]) > 0.05
        and smoothed[i] * smoothed[i - 1] < 0.0
    )

    passed = (
        raw_abs_max <= PARAMS['max_angular_z_near_obstacle'] + 1e-6
        and safe_abs_max <= PARAMS['max_angular_z_near_obstacle'] + 1e-6
        and smooth_abs_max <= PARAMS['max_angular_z_near_obstacle'] + 1e-6
        and smooth_max_delta <= PARAMS['turn_rate_limit'] + 1e-6
        and sign_flips <= 2
    )
    payload = {
        'passed': bool(passed),
        'raw_angular_z': raw_commands,
        'safe_angular_z': safe_commands,
        'smoothed_angular_z': smoothed,
        'raw_abs_max': round(raw_abs_max, 4),
        'safe_abs_max': round(safe_abs_max, 4),
        'smoothed_abs_max': round(smooth_abs_max, 4),
        'smoothed_max_delta': round(smooth_max_delta, 4),
        'smoothed_sign_flips': sign_flips,
        'params': PARAMS,
    }
    out_dir = ROOT / 'artifacts' / 'probe_smoothing'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'probe_smoothing_results.json'
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'result_json': str(path.relative_to(ROOT)), **payload}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
