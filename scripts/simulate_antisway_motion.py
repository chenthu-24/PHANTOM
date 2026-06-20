import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_planner_controller'))
sys.path.insert(0, str(ROOT / 'src' / 'phantom_safety_shield'))

from phantom_planner_controller.planner_controller_node import (  # noqa: E402
    PlannerControllerNode,
    _deg_to_rad,
)
from phantom_safety_shield.safety_shield_node import apply_safety_shield  # noqa: E402


def make_planner():
    planner = object.__new__(PlannerControllerNode)
    planner.cruise_vx = 0.16
    planner.escape_vx = 0.28
    planner.avoid_front_vx = 0.06
    planner.recover_reverse_vx = -0.06
    planner.recover_wz = 0.30
    planner.gap_escape_vx = 0.045
    planner.gap_escape_min_wz = 0.16
    planner.max_forward_vx = 0.32
    planner.max_reverse_vx = -0.08
    planner.max_wz = 0.55
    planner.k_heading = 1.15
    planner.front_hard_stop_m = 0.22
    planner.front_soft_stop_m = 0.40
    planner.front_slowdown_m = 0.72
    planner.max_probe_angle_deg = 22.0
    planner.max_probe_angle_rad = _deg_to_rad(planner.max_probe_angle_deg)
    planner.max_probe_angular_z = 0.34
    planner.probe_angle_step_deg = 8.0
    planner.probe_angle_step_rad = _deg_to_rad(planner.probe_angle_step_deg)
    planner.turn_rate_limit = 0.10
    planner.angular_smoothing_alpha = 0.35
    planner.direction_switch_margin = 0.24
    planner.min_direction_hold_time = 1.20
    planner.max_angular_z_near_obstacle = 0.34
    planner.min_safe_forward_speed = 0.025
    planner.obstacle_slowdown_distance = 0.72
    planner.forward_corridor_clearance_m = 0.62
    planner.center_forward_override_m = 0.80
    planner.balanced_side_score_margin = 0.12
    planner.min_side_clearance_m = 0.195
    planner.min_exit_corridor_width_m = 0.39
    planner.inflation_radius_m = 0.265
    planner.min_obstacle_clearance_m = 0.405
    planner.rear_pressure_escape_enter = 0.55
    planner.rear_pressure_escape_exit = 0.35
    planner.rear_pressure_cruise_max = 0.45
    planner.min_escape_duration_s = 1.50
    planner.escape_clear_duration_s = 1.20
    planner.detection_timeout_sec = 0.8
    planner.direction_lock_duration_s = 1.20
    planner.switch_margin = 0.24
    planner.keep_bonus = 0.18
    planner.switch_penalty = 0.28
    planner.oscillation_penalty = 0.30
    planner.oscillation_window_s = 2.0
    planner.stuck_enter_threshold = 0.75
    planner.stuck_enter_duration_s = 0.80
    planner.cone_base_recover_enabled = True
    planner.cone_base_recover_score_enter = 0.65
    planner.cone_base_recover_cooldown_s = 3.0
    planner.last_cone_recover_exit_time = -999.0
    planner.last_direction = 'front'
    planner.last_direction_switch_time = -999.0
    planner.prev_direction_scores = {}
    planner.direction_switch_times = __import__('collections').deque()
    planner.angular_flip_times = __import__('collections').deque()
    planner.last_angular_sign = 0
    planner.state = 'CRUISE'
    planner.state_enter_time = 0.0
    planner.escape_clear_since = None
    planner.last_threat_time = -999.0
    planner.last_traffic_cone_time = -999.0
    planner.front_rear_soft_since = None
    planner.stuck_score = 0.0
    planner.stuck_started_at = None
    planner.last_smoothed_wz = 0.0
    planner.recover_mode = 'turn'
    planner.cone_recover_mode = 'turn'
    return planner


def make_context(
    front_min,
    left_front,
    right_front,
    left_side,
    right_side,
    path_safe,
    gap_allowed,
    side_corridor_clear=False,
    side_corridor_heading=0.0,
):
    front = {
        'valid': True,
        'front_unknown': False,
        'front_min': front_min,
        'front_blocked_soft': (front_min < 0.40) or ((not path_safe) and not side_corridor_clear),
        'front_blocked_hard': front_min < 0.22,
        'front_path_safe': path_safe,
        'side_corridor_clear': side_corridor_clear,
        'side_corridor_heading': side_corridor_heading,
        'side_corridor_left_m': min(left_front, left_side),
        'side_corridor_right_m': min(right_front, right_side),
        'gap_escape_allowed': gap_allowed,
        'best_heading': 0.0,
        'best_gap_heading': 0.0,
        'best_gap_width_m': 0.52 if gap_allowed else 0.32,
        'min_exit_corridor_width_m': 0.39,
        'left_front_min': left_front,
        'right_front_min': right_front,
        'left_min': left_side,
        'right_min': right_side,
        'sector_scores': {
            'front': max(-0.2, min(front_min / 2.0, 1.0)),
            'front_left': max(-0.2, min(left_front / 2.0, 1.0)),
            'left': max(-0.2, min(left_side / 2.0, 1.0)),
            'front_right': max(-0.2, min(right_front / 2.0, 1.0)),
            'right': max(-0.2, min(right_side / 2.0, 1.0)),
        },
    }
    return {
        'front': front,
        'rear': {'valid': True, 'reverse_allowed': False},
        'front_valid': True,
        'front_unknown': False,
        'front_timeout': False,
        'front_min': front_min,
        'front_hard': front_min < 0.22,
        'front_soft': (front_min < 0.40) or ((not path_safe) and not side_corridor_clear),
        'rear_valid': True,
        'rear_recent': True,
        'rear_pressure': 0.0,
        'rear_center_min': 2.0,
        'rear_soft': False,
        'rear_hard': False,
        'reverse_allowed': False,
        'traffic_cone_recent': False,
        'z_bump_detected': False,
        'z_bump_score': 0.0,
        'z_bump_side': 'none',
        'z_bump_reason': '',
        'gap_escape_allowed': gap_allowed,
        'gap_heading': 0.0,
        'gap_width_m': 0.52 if gap_allowed else 0.32,
        'front_path_safe': path_safe,
        'side_corridor_clear': side_corridor_clear,
        'side_corridor_heading': side_corridor_heading,
        'side_corridor_left_m': min(left_front, left_side),
        'side_corridor_right_m': min(right_front, right_side),
        'forward_corridor_clear': side_corridor_clear or (front_min >= 0.62 and (path_safe or gap_allowed)) or front_min >= 0.80,
        'min_exit_corridor_width_m': 0.39,
        'min_side_clearance_m': 0.195,
        'inflation_radius_m': 0.265,
        'min_obstacle_clearance_m': 0.405,
        'dead_end_score': 0.0,
        'corner_trap_score': 0.0,
    }


def run_sequence(name, contexts):
    planner = make_planner()
    rows = []
    x = 0.0
    theta = 0.0
    for step, context in enumerate(contexts):
        now = step * 0.1
        selected = planner._choose_direction(context, now)
        desired = planner._desired_state(context, selected, now)
        planner._set_state(desired, now)
        cmd = planner._command_for_state(context, selected)
        cmd = planner._smooth_cmd_angular(cmd, context)
        planner._record_angular_sign(now, cmd.angular.z)
        planner.last_cmd = cmd
        safe = apply_safety_shield(
            {'linear_x': cmd.linear.x, 'angular_z': cmd.angular.z},
            context['front'],
            context['rear'],
            {
                'gap_escape_max_vx': 0.055,
                'front_soft_stop_m': 0.40,
                'obstacle_slowdown_distance': 0.72,
                'max_angular_z_near_obstacle': 0.34,
                'center_forward_override_m': 0.80,
                'side_corridor_max_vx': 0.14,
            },
        )
        theta += safe['angular_z'] * 0.1
        x += safe['linear_x'] * math.cos(theta) * 0.1
        rows.append({
            'step': step,
            'state': planner.state,
            'direction': selected['direction'],
            'selection_reason': selected.get('selection_reason'),
            'front_min': round(context['front_min'], 3),
            'left_front_min': round(context['front']['left_front_min'], 3),
            'right_front_min': round(context['front']['right_front_min'], 3),
            'forward_corridor_clear': context['forward_corridor_clear'],
            'side_corridor_clear': context.get('side_corridor_clear', False),
            'vx': round(safe['linear_x'], 4),
            'wz': round(safe['angular_z'], 4),
            'x': round(x, 4),
        })
    side_dirs = [row['direction'] for row in rows if row['direction'] in ('left', 'right')]
    side_switches = sum(1 for a, b in zip(side_dirs, side_dirs[1:]) if a != b)
    spin_steps = sum(1 for row in rows if abs(row['wz']) >= 0.30 and row['vx'] <= 0.01)
    forward_steps = sum(1 for row in rows if row['vx'] > 0.0)
    return {
        'case': name,
        'passed': side_switches <= 1 and spin_steps <= 3 and forward_steps >= int(len(rows) * 0.70),
        'summary': {
            'steps': len(rows),
            'side_switches': side_switches,
            'spin_steps': spin_steps,
            'forward_steps': forward_steps,
            'final_x_m': rows[-1]['x'] if rows else 0.0,
            'directions': {key: sum(1 for row in rows if row['direction'] == key) for key in ('front', 'left', 'right')},
            'selection_reasons': {
                key: sum(1 for row in rows if row['selection_reason'] == key)
                for key in sorted(set(row['selection_reason'] for row in rows))
            },
        },
        'first_rows': rows[:8],
        'last_rows': rows[-8:],
    }


def main():
    balanced = []
    for i in range(60):
        wobble = 0.015 if i % 2 == 0 else -0.015
        balanced.append(make_context(0.68, 0.46 + wobble, 0.46 - wobble, 0.72, 0.72, False, True))

    blocked = []
    for i in range(60):
        wobble = 0.012 if i % 2 == 0 else -0.012
        blocked.append(make_context(0.35, 0.52 + wobble, 0.52 - wobble, 0.74, 0.74, False, True))

    hard_trap = [make_context(0.18, 0.24, 0.25, 0.28, 0.29, False, False) for _ in range(30)]
    center_override = []
    for i in range(60):
        wobble = 0.025 if i % 2 == 0 else -0.025
        center_override.append(make_context(0.92, 0.35 + wobble, 0.58 - wobble, 0.62, 0.70, False, False))
    side_corridor = []
    for i in range(60):
        wobble = 0.018 if i % 2 == 0 else -0.018
        left = 0.34 + wobble
        right = 0.33 - wobble
        heading = max(-0.16, min(0.16, 0.65 * (left - right)))
        side_corridor.append(make_context(1.40, left, right, left + 0.03, right + 0.03, False, False, True, heading))
    results = [
        run_sequence('balanced_front_obstacles_forward_gap', balanced),
        run_sequence('medium_front_obstacles_hold_one_side', blocked),
        run_sequence('center_clear_path_soft_block_forward_override', center_override),
        run_sequence('side_obstacles_front_clear_corridor_forward', side_corridor),
        run_sequence('hard_front_trap_no_forward_push', hard_trap),
    ]
    results[4]['passed'] = results[4]['summary']['forward_steps'] == 0
    out_dir = ROOT / 'artifacts' / 'antisway_sim'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'antisway_results.json'
    out_path.write_text(json.dumps({'passed': all(item['passed'] for item in results), 'cases': results}, indent=2), encoding='utf-8')
    print(json.dumps({'result_json': str(out_path.relative_to(ROOT)), 'passed': all(item['passed'] for item in results), 'cases': results}, indent=2))
    return 0 if all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
