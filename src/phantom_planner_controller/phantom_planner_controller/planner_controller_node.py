import json
import math
import time
from collections import deque

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # Allows local non-ROS chain tests to import planner helpers.
    rclpy = None
    Node = object

    class _Vector:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:  # pragma: no cover - only used when ROS messages are unavailable.
        def __init__(self):
            self.linear = _Vector()
            self.angular = _Vector()

    class String:  # pragma: no cover
        def __init__(self):
            self.data = ''

try:
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover - keeps py_compile usable outside ROS.
    Odometry = None


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def _now_seconds(node):
    return node.get_clock().now().nanoseconds * 1e-9


def _as_float(payload, key, default):
    try:
        value = float(payload.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _normalize_detections_for_planner(payload):
    if isinstance(payload, dict) and isinstance(payload.get('detections'), list):
        return payload['detections']
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def _threat_visible_for_planner(detections, confidence=0.25):
    for detection in _normalize_detections_for_planner(detections):
        visible = bool(detection.get('visible', True))
        class_name = str(detection.get('class_name', '')).strip().lower()
        conf = _as_float(detection, 'conf', 0.0)
        if visible and class_name in ('traffic_cone', 'yellow_car') and conf >= confidence:
            return True
    return False


def _front_scale_value(front_min, hard=0.22, soft=0.35, slowdown=0.50):
    if front_min <= 0.0:
        return 0.0
    if front_min < hard:
        return 0.0
    if front_min < soft:
        return 0.25
    if front_min < slowdown:
        return 0.55
    return 1.0


def _deg_to_rad(value):
    return math.radians(float(value))


def _smoothed_angular_z(previous, target, alpha, rate_limit, limit):
    limit = abs(float(limit))
    target = _clamp(target, -limit, limit)
    alpha = _clamp(alpha, 0.0, 1.0)
    blended = previous + alpha * (target - previous)
    rate_limit = abs(float(rate_limit))
    if rate_limit > 0.0:
        blended = _clamp(blended, previous - rate_limit, previous + rate_limit)
    return _clamp(blended, -limit, limit)


def compute_planner_command(front, rear=None, detections=None, params=None):
    """Stateless local planner equivalent for debug chain injection tests."""
    values = {
        'cruise_vx': 0.16,
        'escape_vx': 0.28,
        'avoid_front_vx': 0.06,
        'recover_reverse_vx': -0.06,
        'recover_wz': 0.30,
        'gap_escape_vx': 0.06,
        'gap_escape_min_wz': 0.12,
        'max_forward_vx': 0.32,
        'max_reverse_vx': -0.08,
        'max_wz': 0.55,
        'k_heading': 1.15,
        'front_hard_stop_m': 0.22,
        'front_soft_stop_m': 0.34,
        'front_slowdown_m': 0.72,
        'max_probe_angle_deg': 22.0,
        'max_probe_angular_z': 0.34,
        'probe_angle_step_deg': 8.0,
        'turn_rate_limit': 0.10,
        'angular_smoothing_alpha': 0.35,
        'direction_switch_margin': 0.24,
        'min_direction_hold_time': 1.20,
        'oscillation_penalty': 0.35,
        'max_angular_z_near_obstacle': 0.34,
        'min_safe_forward_speed': 0.025,
        'obstacle_slowdown_distance': 0.72,
        'forward_corridor_clearance_m': 0.62,
        'center_forward_override_m': 0.80,
        'balanced_side_score_margin': 0.12,
        'robot_width_m': 0.25,
        'robot_length_m': 0.30,
        'safety_margin_m': 0.05,
        'min_exit_corridor_width_m': 0.35,
        'min_side_clearance_m': 0.175,
        'inflation_radius_m': 0.245,
        'cone_base_radius_m': 0.15,
        'obstacle_extra_margin_m': 0.03,
        'min_obstacle_clearance_m': 0.355,
        'rear_pressure_escape_enter': 0.55,
        'threat_confidence': 0.25,
        'cone_base_recover_score_enter': 0.65,
    }
    if params:
        values.update(params)
    front = front or {}
    rear = rear or {}
    detections = detections or []
    front_unknown = bool(front.get('front_unknown', False))
    front_valid = bool(front.get('valid', False)) and not front_unknown
    front_min = _as_float(front, 'front_min', _as_float(front, 'front_clearance', 0.0))
    front_path_safe = bool(front.get('front_path_safe', True))
    side_corridor_clear = bool(front.get('side_corridor_clear', False))
    min_exit_width = _as_float(front, 'min_exit_corridor_width_m', values['min_exit_corridor_width_m'])
    gap_width = _as_float(front, 'best_gap_width_m', 0.0)
    gap_escape_allowed = (
        bool(front.get('gap_escape_allowed', False))
        and gap_width >= min_exit_width
    )
    front_soft = bool(front.get('front_blocked_soft', False)) or (
        front_valid and 0.0 < front_min < values['front_soft_stop_m']
    ) or (front_valid and not front_path_safe and not side_corridor_clear and not gap_escape_allowed)
    front_hard = bool(front.get('front_blocked_hard', False)) or (
        front_valid and 0.0 < front_min < values['front_hard_stop_m']
    )
    rear_valid = bool(rear.get('valid', False))
    rear_pressure = _clamp(_as_float(rear, 'rear_pressure', 0.0), 0.0, 1.0)
    rear_hard = bool(rear.get('rear_blocked_hard', False))
    reverse_allowed = bool(rear.get('reverse_allowed', False)) and rear_valid and not rear_hard
    threat_visible = _threat_visible_for_planner(detections, values['threat_confidence']) or (
        bool(rear.get('threat_visible', False))
        and str(rear.get('threat_class', '')).strip().lower() in ('traffic_cone', 'yellow_car')
        and _as_float(rear, 'threat_conf', 0.0) >= values['threat_confidence']
    )
    traffic_cone_visible = any(
        bool(item.get('visible', True))
        and str(item.get('class_name', '')).strip().lower() == 'traffic_cone'
        and _as_float(item, 'conf', 0.0) >= values['threat_confidence']
        for item in _normalize_detections_for_planner(detections)
    ) or (
        bool(rear.get('threat_visible', False))
        and str(rear.get('threat_class', '')).strip().lower() == 'traffic_cone'
        and _as_float(rear, 'threat_conf', 0.0) >= values['threat_confidence']
    )
    z_bump_detected = bool(rear.get('z_bump_detected', False))
    z_bump_score = _clamp(_as_float(rear, 'z_bump_score', 0.0), 0.0, 1.0)
    max_probe_angle = _deg_to_rad(values['max_probe_angle_deg'])
    probe_step = _deg_to_rad(values['probe_angle_step_deg'])
    max_near_wz = min(values['max_wz'], values['max_probe_angular_z'], values['max_angular_z_near_obstacle'])
    best_heading = _clamp(_as_float(front, 'best_heading', 0.0), -max_probe_angle, max_probe_angle)
    if side_corridor_clear:
        best_heading = _clamp(_as_float(front, 'side_corridor_heading', best_heading), -0.16, 0.16)
    gap_heading = _clamp(_as_float(front, 'best_gap_heading', best_heading), -max_probe_angle, max_probe_angle)
    left_open = max(_as_float(front, 'left_front_min', 0.0), _as_float(front, 'left_min', 0.0))
    right_open = max(_as_float(front, 'right_front_min', 0.0), _as_float(front, 'right_min', 0.0))
    if abs(best_heading) < 0.05 and front_soft:
        best_heading = probe_step if left_open >= right_open else -probe_step
    turn_direction = 1.0 if (best_heading >= 0.0 or left_open >= right_open) else -1.0

    emergency_front_stop = front_hard and not (
        gap_escape_allowed or reverse_allowed or max(left_open, right_open) >= values['front_soft_stop_m']
    )

    if not front_valid:
        state = 'STOP'
        vx = 0.0
        wz = 0.0
    elif emergency_front_stop:
        state = 'STOP'
        vx = 0.0
        wz = 0.0
    elif traffic_cone_visible and z_bump_detected and z_bump_score >= values['cone_base_recover_score_enter']:
        state = 'CONE_BASE_RECOVER'
        vx = 0.0
        wz = 0.0
    elif front_hard:
        if gap_escape_allowed:
            state = 'GAP_ESCAPE'
            vx = values['gap_escape_vx']
            direction = 1.0 if gap_heading >= 0.0 else -1.0
            wz = direction * _clamp(abs(values['k_heading'] * gap_heading), values['gap_escape_min_wz'], max_near_wz)
        elif reverse_allowed:
            state = 'RECOVER'
            vx = values['recover_reverse_vx']
            wz = 0.0
        elif max(left_open, right_open) >= values['front_soft_stop_m'] and not (rear_hard and rear_valid):
            state = 'RECOVER'
            vx = 0.0
            wz = turn_direction * values['recover_wz']
        else:
            state = 'STOP'
            vx = 0.0
            wz = 0.0
    elif front_soft:
        state = 'AVOID_FRONT'
        vx = min(values['avoid_front_vx'], 0.04 if front_min < values['obstacle_slowdown_distance'] else values['avoid_front_vx'])
        if not front_hard:
            vx = max(vx, values['min_safe_forward_speed'])
        wz = turn_direction * _clamp(abs(values['k_heading'] * best_heading), 0.10, max_near_wz)
    elif side_corridor_clear:
        state = 'CRUISE'
        wz = _clamp(values['k_heading'] * best_heading, -max_near_wz, max_near_wz)
        vx = values['cruise_vx'] * _front_scale_value(
            front_min,
            values['front_hard_stop_m'],
            values['front_soft_stop_m'],
            values['front_slowdown_m'],
        ) * max(0.55, 1.0 - abs(wz) / max(values['max_wz'], 0.01))
    elif rear_pressure >= values['rear_pressure_escape_enter'] or threat_visible:
        state = 'ESCAPE'
        wz = _clamp(values['k_heading'] * best_heading, -values['max_wz'], values['max_wz'])
        turn_scale = max(0.30, 1.0 - abs(wz) / max(values['max_wz'], 0.01))
        escape_boost = 1.0 + 0.35 * rear_pressure
        vx = values['escape_vx'] * _front_scale_value(
            front_min,
            values['front_hard_stop_m'],
            values['front_soft_stop_m'],
            values['front_slowdown_m'],
        ) * turn_scale * escape_boost
    else:
        state = 'CRUISE'
        wz = _clamp(values['k_heading'] * best_heading, -values['max_wz'], values['max_wz'])
        turn_scale = max(0.30, 1.0 - abs(wz) / max(values['max_wz'], 0.01))
        vx = values['cruise_vx'] * _front_scale_value(
            front_min,
            values['front_hard_stop_m'],
            values['front_soft_stop_m'],
            values['front_slowdown_m'],
        ) * turn_scale

    return {
        'state': state,
        'cmd_vel_raw': {
            'linear_x': round(_clamp(vx, values['max_reverse_vx'], values['max_forward_vx']), 4),
            'angular_z': round(_clamp(wz, -values['max_wz'], values['max_wz']), 4),
        },
    }


class PlannerControllerNode(Node):
    def __init__(self):
        super().__init__('planner_controller_node')

        self.declare_parameter('front_free_space_topic', '/nav/front_free_space')
        self.declare_parameter('rear_risk_topic', '/nav/rear_risk')
        self.declare_parameter('detections_topic', '/det/detections')
        self.declare_parameter('odom_topic', '/odom_raw')
        self.declare_parameter('odom_fallback_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_raw')
        self.declare_parameter('planner_state_topic', '/debug/planner_state')
        self.declare_parameter('stuck_status_topic', '/planner/stuck_status')
        self.declare_parameter('control_frequency_hz', 10.0)
        self.declare_parameter('front_timeout_sec', 0.5)
        self.declare_parameter('rear_timeout_sec', 0.8)
        self.declare_parameter('detection_timeout_sec', 0.8)
        self.declare_parameter('cruise_vx', 0.16)
        self.declare_parameter('escape_vx', 0.28)
        self.declare_parameter('avoid_front_vx', 0.06)
        self.declare_parameter('recover_reverse_vx', -0.06)
        self.declare_parameter('recover_wz', 0.30)
        self.declare_parameter('gap_escape_vx', 0.06)
        self.declare_parameter('gap_escape_min_wz', 0.12)
        self.declare_parameter('max_forward_vx', 0.32)
        self.declare_parameter('max_reverse_vx', -0.08)
        self.declare_parameter('max_wz', 0.55)
        self.declare_parameter('k_heading', 1.15)
        self.declare_parameter('front_hard_stop_m', 0.22)
        self.declare_parameter('front_soft_stop_m', 0.34)
        self.declare_parameter('front_slowdown_m', 0.72)
        self.declare_parameter('max_probe_angle_deg', 22.0)
        self.declare_parameter('max_probe_angular_z', 0.34)
        self.declare_parameter('probe_angle_step_deg', 8.0)
        self.declare_parameter('turn_rate_limit', 0.10)
        self.declare_parameter('angular_smoothing_alpha', 0.35)
        self.declare_parameter('direction_switch_margin', 0.24)
        self.declare_parameter('min_direction_hold_time', 1.20)
        self.declare_parameter('max_angular_z_near_obstacle', 0.34)
        self.declare_parameter('min_safe_forward_speed', 0.025)
        self.declare_parameter('obstacle_slowdown_distance', 0.72)
        self.declare_parameter('forward_corridor_clearance_m', 0.62)
        self.declare_parameter('center_forward_override_m', 0.80)
        self.declare_parameter('balanced_side_score_margin', 0.12)
        self.declare_parameter('robot_width_m', 0.25)
        self.declare_parameter('robot_length_m', 0.30)
        self.declare_parameter('safety_margin_m', 0.05)
        self.declare_parameter('min_side_clearance_m', 0.175)
        self.declare_parameter('inflation_radius_m', 0.245)
        self.declare_parameter('min_exit_corridor_width_m', 0.35)
        self.declare_parameter('cone_base_radius_m', 0.15)
        self.declare_parameter('obstacle_extra_margin_m', 0.03)
        self.declare_parameter('min_obstacle_clearance_m', 0.355)
        self.declare_parameter('rear_pressure_escape_enter', 0.55)
        self.declare_parameter('rear_pressure_escape_exit', 0.35)
        self.declare_parameter('rear_pressure_cruise_max', 0.45)
        self.declare_parameter('min_escape_duration_s', 1.50)
        self.declare_parameter('escape_clear_duration_s', 1.20)
        self.declare_parameter('threat_confidence', 0.25)
        self.declare_parameter('direction_lock_duration_s', 1.20)
        self.declare_parameter('switch_margin', 0.24)
        self.declare_parameter('keep_bonus', 0.18)
        self.declare_parameter('switch_penalty', 0.28)
        self.declare_parameter('oscillation_penalty', 0.30)
        self.declare_parameter('oscillation_window_s', 2.0)
        self.declare_parameter('stuck_enter_threshold', 0.75)
        self.declare_parameter('stuck_enter_duration_s', 0.80)
        self.declare_parameter('front_rear_soft_blocked_duration_s', 1.00)
        self.declare_parameter('cone_base_recover_enabled', True)
        self.declare_parameter('cone_base_recover_score_enter', 0.65)
        self.declare_parameter('cone_base_recover_score_exit', 0.35)
        self.declare_parameter('cone_base_recover_cooldown_s', 3.00)
        self.declare_parameter('cone_base_recover_stop_s', 0.20)
        self.declare_parameter('cone_base_recover_reverse_s', 0.60)
        self.declare_parameter('cone_base_recover_rotate_s', 0.80)
        self.declare_parameter('cone_base_recover_timeout_s', 2.20)

        self.front_topic = str(self.get_parameter('front_free_space_topic').value)
        self.rear_topic = str(self.get_parameter('rear_risk_topic').value)
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.odom_fallback_topic = str(self.get_parameter('odom_fallback_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.planner_state_topic = str(self.get_parameter('planner_state_topic').value)
        self.stuck_status_topic = str(self.get_parameter('stuck_status_topic').value)
        self.control_frequency_hz = float(self.get_parameter('control_frequency_hz').value)
        self.front_timeout_sec = float(self.get_parameter('front_timeout_sec').value)
        self.rear_timeout_sec = float(self.get_parameter('rear_timeout_sec').value)
        self.detection_timeout_sec = float(self.get_parameter('detection_timeout_sec').value)
        self.cruise_vx = float(self.get_parameter('cruise_vx').value)
        self.escape_vx = float(self.get_parameter('escape_vx').value)
        self.avoid_front_vx = float(self.get_parameter('avoid_front_vx').value)
        self.recover_reverse_vx = float(self.get_parameter('recover_reverse_vx').value)
        self.recover_wz = float(self.get_parameter('recover_wz').value)
        self.gap_escape_vx = float(self.get_parameter('gap_escape_vx').value)
        self.gap_escape_min_wz = float(self.get_parameter('gap_escape_min_wz').value)
        self.max_forward_vx = float(self.get_parameter('max_forward_vx').value)
        self.max_reverse_vx = float(self.get_parameter('max_reverse_vx').value)
        self.max_wz = float(self.get_parameter('max_wz').value)
        self.k_heading = float(self.get_parameter('k_heading').value)
        self.front_hard_stop_m = float(self.get_parameter('front_hard_stop_m').value)
        self.front_soft_stop_m = float(self.get_parameter('front_soft_stop_m').value)
        self.front_slowdown_m = float(self.get_parameter('front_slowdown_m').value)
        self.max_probe_angle_deg = float(self.get_parameter('max_probe_angle_deg').value)
        self.max_probe_angle_rad = _deg_to_rad(self.max_probe_angle_deg)
        self.max_probe_angular_z = float(self.get_parameter('max_probe_angular_z').value)
        self.probe_angle_step_deg = float(self.get_parameter('probe_angle_step_deg').value)
        self.probe_angle_step_rad = _deg_to_rad(self.probe_angle_step_deg)
        self.turn_rate_limit = float(self.get_parameter('turn_rate_limit').value)
        self.angular_smoothing_alpha = float(self.get_parameter('angular_smoothing_alpha').value)
        self.direction_switch_margin = float(self.get_parameter('direction_switch_margin').value)
        self.min_direction_hold_time = float(self.get_parameter('min_direction_hold_time').value)
        self.max_angular_z_near_obstacle = float(self.get_parameter('max_angular_z_near_obstacle').value)
        self.min_safe_forward_speed = float(self.get_parameter('min_safe_forward_speed').value)
        self.obstacle_slowdown_distance = float(self.get_parameter('obstacle_slowdown_distance').value)
        self.forward_corridor_clearance_m = float(self.get_parameter('forward_corridor_clearance_m').value)
        self.center_forward_override_m = float(self.get_parameter('center_forward_override_m').value)
        self.balanced_side_score_margin = float(self.get_parameter('balanced_side_score_margin').value)
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.robot_length_m = float(self.get_parameter('robot_length_m').value)
        self.safety_margin_m = float(self.get_parameter('safety_margin_m').value)
        self.min_side_clearance_m = float(self.get_parameter('min_side_clearance_m').value)
        self.inflation_radius_m = float(self.get_parameter('inflation_radius_m').value)
        self.min_exit_corridor_width_m = float(self.get_parameter('min_exit_corridor_width_m').value)
        self.cone_base_radius_m = float(self.get_parameter('cone_base_radius_m').value)
        self.obstacle_extra_margin_m = float(self.get_parameter('obstacle_extra_margin_m').value)
        self.min_obstacle_clearance_m = float(self.get_parameter('min_obstacle_clearance_m').value)
        self.rear_pressure_escape_enter = float(self.get_parameter('rear_pressure_escape_enter').value)
        self.rear_pressure_escape_exit = float(self.get_parameter('rear_pressure_escape_exit').value)
        self.rear_pressure_cruise_max = float(self.get_parameter('rear_pressure_cruise_max').value)
        self.min_escape_duration_s = float(self.get_parameter('min_escape_duration_s').value)
        self.escape_clear_duration_s = float(self.get_parameter('escape_clear_duration_s').value)
        self.threat_confidence = float(self.get_parameter('threat_confidence').value)
        self.direction_lock_duration_s = float(self.get_parameter('direction_lock_duration_s').value)
        self.switch_margin = float(self.get_parameter('switch_margin').value)
        self.keep_bonus = float(self.get_parameter('keep_bonus').value)
        self.switch_penalty = float(self.get_parameter('switch_penalty').value)
        self.oscillation_penalty = float(self.get_parameter('oscillation_penalty').value)
        self.oscillation_window_s = float(self.get_parameter('oscillation_window_s').value)
        self.stuck_enter_threshold = float(self.get_parameter('stuck_enter_threshold').value)
        self.stuck_enter_duration_s = float(self.get_parameter('stuck_enter_duration_s').value)
        self.front_rear_soft_blocked_duration_s = float(
            self.get_parameter('front_rear_soft_blocked_duration_s').value
        )
        self.cone_base_recover_enabled = bool(self.get_parameter('cone_base_recover_enabled').value)
        self.cone_base_recover_score_enter = float(self.get_parameter('cone_base_recover_score_enter').value)
        self.cone_base_recover_score_exit = float(self.get_parameter('cone_base_recover_score_exit').value)
        self.cone_base_recover_cooldown_s = float(self.get_parameter('cone_base_recover_cooldown_s').value)
        self.cone_base_recover_stop_s = float(self.get_parameter('cone_base_recover_stop_s').value)
        self.cone_base_recover_reverse_s = float(self.get_parameter('cone_base_recover_reverse_s').value)
        self.cone_base_recover_rotate_s = float(self.get_parameter('cone_base_recover_rotate_s').value)
        self.cone_base_recover_timeout_s = float(self.get_parameter('cone_base_recover_timeout_s').value)
        self.switch_margin = max(self.switch_margin, self.direction_switch_margin)
        self.direction_lock_duration_s = max(self.direction_lock_duration_s, self.min_direction_hold_time)
        self.recover_wz = min(self.recover_wz, self.max_probe_angular_z)
        self.gap_escape_min_wz = min(self.gap_escape_min_wz, self.max_probe_angular_z)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.state_pub = self.create_publisher(String, self.planner_state_topic, 10)
        self.stuck_pub = self.create_publisher(String, self.stuck_status_topic, 10)
        self.front_sub = self.create_subscription(String, self.front_topic, self.front_callback, 10)
        self.rear_sub = self.create_subscription(String, self.rear_topic, self.rear_callback, 10)
        self.det_sub = self.create_subscription(String, self.detections_topic, self.detections_callback, 10)
        self.odom_subs = []
        if Odometry is not None:
            self.odom_subs.append(self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10))
            if self.odom_fallback_topic and self.odom_fallback_topic != self.odom_topic:
                self.odom_subs.append(
                    self.create_subscription(Odometry, self.odom_fallback_topic, self.odom_callback, 10)
                )

        now = _now_seconds(self)
        self.front = None
        self.rear = None
        self.detections = []
        self.last_front_time = -999.0
        self.last_rear_time = -999.0
        self.last_detection_time = -999.0
        self.last_threat_time = -999.0
        self.last_traffic_cone_time = -999.0
        self.state = 'STOP'
        self.state_enter_time = now
        self.escape_clear_since = None
        self.recover_start_time = -999.0
        self.recover_mode = 'turn'
        self.recover_direction = 1.0
        self.cone_recover_start_time = -999.0
        self.cone_recover_mode = 'turn'
        self.cone_recover_direction = 1.0
        self.cone_recover_phase = 'idle'
        self.last_cone_recover_exit_time = -999.0
        self.last_direction = 'front'
        self.last_direction_switch_time = -999.0
        self.prev_direction_scores = {}
        self.direction_switch_times = deque()
        self.angular_flip_times = deque()
        self.last_angular_sign = 0
        self.front_rear_soft_since = None
        self.low_progress_since = None
        self.zero_cmd_since = None
        self.stuck_started_at = None
        self.stuck_score = 0.0
        self.last_cmd = Twist()
        self.last_smoothed_wz = 0.0
        self.odom_history = deque()
        self.last_debug_log_time = -999.0

        period = 1.0 / max(self.control_frequency_hz, 1.0)
        self.timer = self.create_timer(period, self.control_step)

        self.get_logger().info(
            'planner_controller_node subscribed to %s, %s, %s, %s/%s; publishing %s'
            % (
                self.front_topic,
                self.rear_topic,
                self.detections_topic,
                self.odom_topic,
                self.odom_fallback_topic,
                self.cmd_vel_topic,
            )
        )

    def front_callback(self, message):
        payload = self._parse_json(message.data, 'front_free_space')
        if payload is None:
            return
        self.front = payload
        self.last_front_time = _now_seconds(self)

    def rear_callback(self, message):
        payload = self._parse_json(message.data, 'rear_risk')
        if payload is None:
            return
        self.rear = payload
        self.last_rear_time = _now_seconds(self)
        if self._rear_payload_has_threat(payload):
            self.last_threat_time = self.last_rear_time
        if self._rear_payload_has_traffic_cone(payload):
            self.last_traffic_cone_time = self.last_rear_time

    def detections_callback(self, message):
        payload = self._parse_json(message.data, 'detections')
        if payload is None:
            return
        self.detections = self._normalize_detections(payload)
        self.last_detection_time = _now_seconds(self)
        if any(self._is_threat_detection(item) for item in self.detections):
            self.last_threat_time = self.last_detection_time
        if any(self._is_traffic_cone_detection(item) for item in self.detections):
            self.last_traffic_cone_time = self.last_detection_time

    def odom_callback(self, message):
        now = _now_seconds(self)
        pose = message.pose.pose.position
        twist = message.twist.twist
        self.odom_history.append((now, float(pose.x), float(pose.y), float(twist.linear.x)))
        while self.odom_history and now - self.odom_history[0][0] > 3.0:
            self.odom_history.popleft()

    def control_step(self):
        now = _now_seconds(self)
        context = self._build_context(now)
        selected = self._choose_direction(context, now)
        self._update_soft_block_timer(context, now)
        self._update_stuck_score(context, now)

        if context['front_timeout'] or not context['front_valid']:
            self._set_state('STOP', now)
            cmd = Twist()
        elif self.state == 'CONE_BASE_RECOVER':
            cmd = self._cone_base_recover_cmd(context, selected, now)
        elif self.state == 'RECOVER':
            cmd = self._recover_cmd(context, selected, now)
        else:
            desired_state = self._desired_state(context, selected, now)
            if desired_state == 'CONE_BASE_RECOVER':
                self._enter_cone_base_recover(context, selected, now)
                cmd = self._cone_base_recover_cmd(context, selected, now)
            elif desired_state == 'RECOVER':
                self._enter_recover(context, selected, now)
                cmd = self._recover_cmd(context, selected, now)
            else:
                self._set_state(desired_state, now)
                cmd = self._command_for_state(context, selected)

        cmd = self._smooth_cmd_angular(cmd, context)
        self._record_angular_sign(now, cmd.angular.z)
        self.last_cmd = cmd
        self.cmd_pub.publish(cmd)
        self._publish_state(context, selected, now)
        self._publish_stuck_status(context, now)

    def _desired_state(self, context, selected, now):
        front_emergency_stop = context['front_hard'] and not (
            context['gap_escape_allowed'] or self._safe_turn_available(context) or context['reverse_allowed']
        )
        if front_emergency_stop:
            return 'STOP'
        if self._cone_base_recover_should_enter(context, now):
            return 'CONE_BASE_RECOVER'
        if context['front_hard'] and context['gap_escape_allowed']:
            return 'GAP_ESCAPE'
        if context['front_hard']:
            return 'RECOVER' if self._safe_turn_available(context) or context['reverse_allowed'] else 'STOP'
        if self._front_rear_soft_blocked(now):
            return 'RECOVER'
        if self._oscillation_detected(now) and not context['forward_corridor_clear']:
            return 'RECOVER'
        if self._stuck_ready(now):
            return 'RECOVER'

        threat_recent = now - self.last_threat_time <= self.detection_timeout_sec
        if context['front_soft'] and not context['forward_corridor_clear']:
            return 'AVOID_FRONT'

        escape_trigger = context['rear_pressure'] >= self.rear_pressure_escape_enter or threat_recent
        if escape_trigger:
            self.escape_clear_since = None
            return 'ESCAPE'

        if self.state == 'ESCAPE':
            if now - self.state_enter_time < self.min_escape_duration_s:
                return 'ESCAPE'
            if context['rear_pressure'] < self.rear_pressure_escape_exit and not threat_recent:
                if self.escape_clear_since is None:
                    self.escape_clear_since = now
                    return 'ESCAPE'
                if now - self.escape_clear_since < self.escape_clear_duration_s:
                    return 'ESCAPE'
                return 'CRUISE'
            self.escape_clear_since = None
            return 'ESCAPE'

        if context['rear_pressure'] >= self.rear_pressure_cruise_max:
            return 'ESCAPE'
        return 'CRUISE'

    def _command_for_state(self, context, selected):
        cmd = Twist()
        heading = selected['heading']
        if self.state == 'STOP':
            return cmd
        if self.state == 'CRUISE':
            wz = _clamp(self.k_heading * heading, -self.max_wz, self.max_wz)
            vx = self.cruise_vx * self._front_scale(context['front_min']) * self._turn_scale(wz)
        elif self.state == 'ESCAPE':
            wz = _clamp(self.k_heading * heading, -self.max_wz, self.max_wz)
            escape_boost = 1.0 + 0.35 * _clamp(context['rear_pressure'], 0.0, 1.0)
            vx = self.escape_vx * self._front_scale(context['front_min']) * self._turn_scale(wz) * escape_boost
        elif self.state == 'AVOID_FRONT':
            direction = self._turn_direction(context, selected)
            near_limit = self._angular_limit_for_context(context)
            wz = direction * _clamp(abs(self.k_heading * selected['heading']), 0.10, near_limit)
            vx = self.avoid_front_vx
            if context['front_min'] < self.front_soft_stop_m:
                vx = min(vx, 0.035)
            elif context['front_min'] < self.forward_corridor_clearance_m:
                vx = min(vx, 0.055)
            elif context['front_min'] < self.obstacle_slowdown_distance:
                vx = min(vx, 0.075)
            if not context['front_hard']:
                vx = max(vx, self.min_safe_forward_speed)
        elif self.state == 'GAP_ESCAPE':
            heading = context['gap_heading']
            direction = 1.0 if heading >= 0.0 else -1.0
            near_limit = self._angular_limit_for_context(context)
            wz = direction * _clamp(abs(self.k_heading * heading), self.gap_escape_min_wz, near_limit)
            vx = self.gap_escape_vx
        else:
            return cmd

        if context['front_hard'] and self.state != 'GAP_ESCAPE':
            vx = 0.0
        cmd.linear.x = _clamp(vx, self.max_reverse_vx, self.max_forward_vx)
        cmd.angular.z = _clamp(wz, -self.max_wz, self.max_wz)
        return cmd

    def _smooth_cmd_angular(self, cmd, context):
        if abs(float(cmd.angular.z)) <= 1e-6:
            self.last_smoothed_wz = 0.0
            cmd.angular.z = 0.0
            return cmd
        limit = self._angular_limit_for_context(context)
        cmd.angular.z = _smoothed_angular_z(
            self.last_smoothed_wz,
            float(cmd.angular.z),
            self.angular_smoothing_alpha,
            self.turn_rate_limit,
            limit,
        )
        self.last_smoothed_wz = float(cmd.angular.z)
        return cmd

    def _angular_limit_for_context(self, context):
        limit = self.max_wz
        if (
            context.get('front_soft', False)
            or context.get('front_hard', False)
            or 0.0 < context.get('front_min', 0.0) < self.obstacle_slowdown_distance
            or self.state in ('AVOID_FRONT', 'GAP_ESCAPE', 'RECOVER', 'CONE_BASE_RECOVER')
        ):
            limit = min(limit, self.max_angular_z_near_obstacle, self.max_probe_angular_z)
        return max(0.08, abs(float(limit)))

    def _recover_cmd(self, context, selected, now):
        cmd = Twist()
        elapsed = now - self.recover_start_time

        if self.recover_mode == 'reverse' and 0.20 <= elapsed < 0.80 and not context['reverse_allowed']:
            self.get_logger().warn('rear_risk no longer allows reverse; switching RECOVER to turn-only')
            self.recover_mode = 'turn'
            self.recover_start_time = now
            elapsed = 0.0

        if self.recover_mode == 'reverse':
            if elapsed < 0.20:
                return cmd
            if elapsed < 0.80:
                cmd.linear.x = _clamp(self.recover_reverse_vx, self.max_reverse_vx, -0.04)
                return cmd
            if elapsed < 1.55:
                cmd.angular.z = _clamp(self.recover_direction * self.recover_wz, -self.max_probe_angular_z, self.max_probe_angular_z)
                return cmd
        else:
            if elapsed < 0.20:
                return cmd
            if elapsed < 1.05:
                cmd.angular.z = _clamp(self.recover_direction * self.recover_wz, -self.max_probe_angular_z, self.max_probe_angular_z)
                return cmd

        next_state = 'AVOID_FRONT' if context['front_soft'] else 'CRUISE'
        self._set_state(next_state, now)
        return self._command_for_state(context, selected)

    def _enter_recover(self, context, selected, now):
        self._set_state('RECOVER', now)
        self.recover_start_time = now
        self.recover_direction = self._turn_direction(context, selected)
        self.recover_mode = 'reverse' if context['reverse_allowed'] else 'turn'
        self.direction_switch_times.clear()
        self.angular_flip_times.clear()
        self.get_logger().warn('entering RECOVER mode=%s direction=%.0f' % (self.recover_mode, self.recover_direction))

    def _cone_base_recover_should_enter(self, context, now):
        if not self.cone_base_recover_enabled:
            return False
        if now - self.last_cone_recover_exit_time < self.cone_base_recover_cooldown_s:
            return False
        if not context['traffic_cone_recent']:
            return False
        if not context['z_bump_detected']:
            return False
        return context['z_bump_score'] >= self.cone_base_recover_score_enter

    def _enter_cone_base_recover(self, context, selected, now):
        self._set_state('CONE_BASE_RECOVER', now)
        self.cone_recover_start_time = now
        self.cone_recover_mode = 'reverse' if context['reverse_allowed'] else 'turn'
        self.cone_recover_direction = self._turn_direction(context, selected)
        self.cone_recover_phase = 'stop'
        self.direction_switch_times.clear()
        self.angular_flip_times.clear()
        self.get_logger().warn(
            'entering CONE_BASE_RECOVER mode=%s direction=%.0f z_score=%.2f side=%s'
            % (
                self.cone_recover_mode,
                self.cone_recover_direction,
                context['z_bump_score'],
                context['z_bump_side'],
            )
        )

    def _cone_base_recover_cmd(self, context, selected, now):
        cmd = Twist()
        elapsed = now - self.cone_recover_start_time
        if elapsed >= self.cone_base_recover_timeout_s:
            return self._exit_cone_base_recover(context, selected, now, 'timeout')

        if (
            self.cone_recover_mode == 'reverse'
            and self.cone_base_recover_stop_s <= elapsed < self.cone_base_recover_stop_s + self.cone_base_recover_reverse_s
            and not context['reverse_allowed']
        ):
            self.get_logger().warn('rear_risk no longer allows reverse; CONE_BASE_RECOVER switching to turn-only')
            self.cone_recover_mode = 'turn'
            self.cone_recover_start_time = now
            elapsed = 0.0

        if self.cone_recover_mode == 'reverse':
            reverse_end = self.cone_base_recover_stop_s + self.cone_base_recover_reverse_s
            rotate_end = reverse_end + self.cone_base_recover_rotate_s
            if elapsed < self.cone_base_recover_stop_s:
                self.cone_recover_phase = 'stop'
                return cmd
            if elapsed < reverse_end:
                self.cone_recover_phase = 'reverse'
                cmd.linear.x = _clamp(self.recover_reverse_vx, self.max_reverse_vx, -0.04)
                return cmd
            if elapsed < rotate_end:
                self.cone_recover_phase = 'rotate'
                cmd.angular.z = _clamp(self.cone_recover_direction * self.recover_wz, -self.max_probe_angular_z, self.max_probe_angular_z)
                return cmd
        else:
            rotate_end = self.cone_base_recover_stop_s + self.cone_base_recover_rotate_s
            if elapsed < self.cone_base_recover_stop_s:
                self.cone_recover_phase = 'stop'
                return cmd
            if elapsed < rotate_end:
                self.cone_recover_phase = 'rotate'
                cmd.angular.z = _clamp(self.cone_recover_direction * self.recover_wz, -self.max_probe_angular_z, self.max_probe_angular_z)
                return cmd

        return self._exit_cone_base_recover(context, selected, now, 'exit')

    def _exit_cone_base_recover(self, context, selected, now, phase):
        self.cone_recover_phase = phase
        self.last_cone_recover_exit_time = now
        next_state = 'AVOID_FRONT' if context['front_soft'] else 'CRUISE'
        if context['z_bump_score'] > self.cone_base_recover_score_exit and context['traffic_cone_recent']:
            next_state = 'ESCAPE'
        self._set_state(next_state, now)
        return self._command_for_state(context, selected)

    def _build_context(self, now):
        front_recent = self.front is not None and now - self.last_front_time <= self.front_timeout_sec
        front_payload = self.front or {}
        front_min = _as_float(front_payload, 'front_min', _as_float(front_payload, 'front_clearance', 0.0))
        front_unknown = bool(front_payload.get('front_unknown', False))
        front_valid = bool(front_payload.get('valid', True)) and front_recent and not front_unknown
        front_hard = bool(front_payload.get('front_blocked_hard', False)) or (
            front_valid and 0.0 < front_min < self.front_hard_stop_m
        )
        front_path_safe = bool(front_payload.get('front_path_safe', True))
        side_corridor_clear = bool(front_payload.get('side_corridor_clear', False))

        rear_recent = self.rear is not None and now - self.last_rear_time <= self.rear_timeout_sec
        rear_payload = self.rear or {}
        rear_valid = bool(rear_payload.get('valid', False)) and rear_recent
        rear_pressure = _as_float(rear_payload, 'rear_pressure', 0.0) if rear_recent else 0.0
        rear_center_min = _as_float(rear_payload, 'rear_center_min', 0.0)
        rear_soft = bool(rear_payload.get('rear_blocked_soft', False)) or (
            rear_valid and 0.0 < rear_center_min < 0.55
        )
        rear_hard = bool(rear_payload.get('rear_blocked_hard', False)) or (
            rear_valid and 0.0 < rear_center_min < 0.30
        )
        reverse_allowed = bool(rear_payload.get('reverse_allowed', False)) and rear_valid and not rear_hard
        z_bump_detected = bool(rear_payload.get('z_bump_detected', False)) and rear_recent
        z_bump_score = _clamp(_as_float(rear_payload, 'z_bump_score', 0.0), 0.0, 1.0) if rear_recent else 0.0
        traffic_cone_recent = now - self.last_traffic_cone_time <= self.detection_timeout_sec
        gap_width = _as_float(front_payload, 'best_gap_width_m', 0.0)
        min_exit_width = _as_float(front_payload, 'min_exit_corridor_width_m', self.min_exit_corridor_width_m)
        gap_escape_allowed = bool(front_payload.get('gap_escape_allowed', False)) and gap_width >= min_exit_width
        front_soft = bool(front_payload.get('front_blocked_soft', False)) or (
            front_valid and 0.0 < front_min < self.front_soft_stop_m
        ) or (front_valid and not front_path_safe and not side_corridor_clear and not gap_escape_allowed)
        forward_corridor_clear = (
            front_valid
            and not front_hard
            and (
                side_corridor_clear
                or
                (
                    front_min >= self.forward_corridor_clearance_m
                    and (front_path_safe or gap_escape_allowed)
                )
                or front_min >= self.center_forward_override_m
            )
        )

        return {
            'front': front_payload,
            'rear': rear_payload,
            'front_valid': front_valid,
            'front_unknown': front_unknown,
            'front_timeout': not front_recent,
            'front_min': front_min,
            'front_hard': front_hard,
            'front_soft': front_soft,
            'rear_valid': rear_valid,
            'rear_recent': rear_recent,
            'rear_pressure': _clamp(rear_pressure, 0.0, 1.0),
            'rear_center_min': rear_center_min,
            'rear_soft': rear_soft,
            'rear_hard': rear_hard,
            'reverse_allowed': reverse_allowed,
            'traffic_cone_recent': traffic_cone_recent,
            'z_bump_detected': z_bump_detected,
            'z_bump_score': z_bump_score,
            'z_bump_side': str(rear_payload.get('z_bump_side', 'none')),
            'z_bump_reason': str(rear_payload.get('z_bump_reason', '')),
            'gap_escape_allowed': gap_escape_allowed,
            'gap_heading': _clamp(
                _as_float(front_payload, 'best_gap_heading', _as_float(front_payload, 'best_heading', 0.0)),
                -self.max_probe_angle_rad,
                self.max_probe_angle_rad,
            ),
            'gap_width_m': gap_width,
            'front_path_safe': front_path_safe,
            'side_corridor_clear': side_corridor_clear,
            'side_corridor_heading': _as_float(front_payload, 'side_corridor_heading', 0.0),
            'side_corridor_left_m': _as_float(front_payload, 'side_corridor_left_m', 0.0),
            'side_corridor_right_m': _as_float(front_payload, 'side_corridor_right_m', 0.0),
            'forward_corridor_clear': forward_corridor_clear,
            'min_exit_corridor_width_m': min_exit_width,
            'min_side_clearance_m': _as_float(front_payload, 'min_side_clearance_m', self.min_side_clearance_m),
            'inflation_radius_m': _as_float(front_payload, 'inflation_radius_m', self.inflation_radius_m),
            'min_obstacle_clearance_m': _as_float(front_payload, 'min_obstacle_clearance_m', self.min_obstacle_clearance_m),
            'dead_end_score': _as_float(front_payload, 'dead_end_score', 0.0),
            'corner_trap_score': _as_float(front_payload, 'corner_trap_score', 0.0),
        }

    def _choose_direction(self, context, now):
        front = context['front']
        best_heading = _clamp(_as_float(front, 'best_heading', 0.0), -self.max_probe_angle_rad, self.max_probe_angle_rad)
        raw_scores = self._direction_scores(context)
        raw_best = max(raw_scores, key=raw_scores.get)

        if context.get('side_corridor_clear', False):
            if self.last_direction != 'front':
                self.last_direction_switch_time = now
            self.last_direction = 'front'
            self.prev_direction_scores = raw_scores
            heading = _clamp(
                context.get('side_corridor_heading', best_heading),
                -min(self.max_probe_angle_rad, 0.16),
                min(self.max_probe_angle_rad, 0.16),
            )
            return {
                'direction': 'front',
                'heading': heading,
                'score': raw_scores.get('front', 0.0),
                'scores': raw_scores,
                'raw_scores': raw_scores,
                'selection_reason': 'side_corridor',
                'direction_hold_active': False,
                'score_margin_hold': False,
                'switch_penalty_active': False,
                'score_delta': 0.0,
                'side_score_delta': abs(raw_scores.get('left', 0.0) - raw_scores.get('right', 0.0)),
                'raw_best_direction': raw_best,
            }

        if (
            context['front_valid']
            and not context['front_hard']
            and (
                (
                    not context['front_soft']
                    and context['front_min'] >= self.front_slowdown_m
                )
                or context['forward_corridor_clear']
            )
        ):
            if self.last_direction != 'front':
                self.last_direction_switch_time = now
            self.last_direction = 'front'
            self.prev_direction_scores = raw_scores
            return {
                'direction': 'front',
                'heading': self._heading_for_direction('front', best_heading),
                'score': raw_scores.get('front', 0.0),
                'scores': raw_scores,
                'raw_scores': raw_scores,
                'selection_reason': 'forward_corridor',
                'direction_hold_active': False,
                'score_margin_hold': False,
                'switch_penalty_active': False,
                'score_delta': 0.0,
                'raw_best_direction': raw_best,
            }

        final_scores = {}
        oscillating = self._oscillation_detected(now)
        for name, score in raw_scores.items():
            gain = score - self.prev_direction_scores.get(name, score)
            adjusted = score + 0.25 * _clamp(gain, -0.4, 0.4)
            if name == self.last_direction:
                adjusted += self.keep_bonus
            else:
                adjusted -= self.switch_penalty
            if oscillating and name != self.last_direction:
                adjusted -= self.oscillation_penalty
            if not self._direction_safe(name, context):
                adjusted -= 1.0
            final_scores[name] = adjusted

        current = self.last_direction if self.last_direction in final_scores else 'front'
        candidate = max(final_scores, key=final_scores.get)
        best = candidate
        current_safe = self._direction_safe(current, context)
        score_delta = final_scores[candidate] - final_scores.get(current, final_scores[candidate])
        side_score_delta = abs(raw_scores.get('left', 0.0) - raw_scores.get('right', 0.0))
        direction_hold_active = False
        score_margin_hold = False
        selection_reason = 'best_score'
        if current_safe and now - self.last_direction_switch_time < self.direction_lock_duration_s:
            if score_delta <= self.switch_margin:
                best = current
                direction_hold_active = True
                selection_reason = 'direction_hold'
        elif (
            current_safe
            and {candidate, current} == {'left', 'right'}
            and side_score_delta <= self.balanced_side_score_margin
        ):
            best = current
            score_margin_hold = True
            selection_reason = 'balanced_side_margin'
        elif score_delta <= self.switch_margin and current_safe:
            best = current
            score_margin_hold = True
            selection_reason = 'score_margin'
        elif candidate != raw_best:
            selection_reason = 'switch_penalty'

        if best != self.last_direction:
            if {best, self.last_direction} == {'left', 'right'}:
                self.direction_switch_times.append(now)
            self.last_direction_switch_time = now
            self.last_direction = best
        self._prune_times(self.direction_switch_times, now, self.oscillation_window_s)
        self.prev_direction_scores = raw_scores

        heading = self._heading_for_direction(best, best_heading)
        return {
            'direction': best,
            'heading': heading,
            'score': final_scores[best],
            'scores': final_scores,
            'raw_scores': raw_scores,
            'selection_reason': selection_reason,
            'direction_hold_active': direction_hold_active,
            'score_margin_hold': score_margin_hold,
            'switch_penalty_active': candidate != raw_best,
            'score_delta': score_delta,
            'side_score_delta': side_score_delta,
            'raw_best_direction': raw_best,
        }

    def _direction_scores(self, context):
        front = context['front']
        sector_scores = front.get('sector_scores', {}) if isinstance(front.get('sector_scores', {}), dict) else {}

        def derived(distance):
            return _clamp(distance / 2.0, 0.0, 1.0)

        left_score = max(
            float(sector_scores.get('front_left', derived(_as_float(front, 'left_front_min', 0.0)))),
            float(sector_scores.get('left', derived(_as_float(front, 'left_min', 0.0)))),
        )
        right_score = max(
            float(sector_scores.get('front_right', derived(_as_float(front, 'right_front_min', 0.0)))),
            float(sector_scores.get('right', derived(_as_float(front, 'right_min', 0.0)))),
        )
        front_score = float(sector_scores.get('front', derived(context['front_min'])))
        return {'left': left_score, 'front': front_score, 'right': right_score}

    def _direction_safe(self, direction, context):
        front = context['front']
        side_clearance = max(context.get('min_side_clearance_m', self.min_side_clearance_m), self.front_hard_stop_m)
        if direction == 'front':
            return (
                (not context['front_soft'] and context.get('front_path_safe', True))
                or context.get('forward_corridor_clear', False)
                or context.get('side_corridor_clear', False)
            )
        if direction == 'left':
            return (
                _as_float(front, 'left_front_min', 0.0) >= side_clearance
                and _as_float(front, 'left_min', 0.0) >= side_clearance
            )
        if direction == 'right':
            return (
                _as_float(front, 'right_front_min', 0.0) >= side_clearance
                and _as_float(front, 'right_min', 0.0) >= side_clearance
            )
        return False

    def _heading_for_direction(self, direction, best_heading):
        if direction == 'front':
            return _clamp(best_heading, -self.max_probe_angle_rad, self.max_probe_angle_rad)
        if direction == 'left':
            return _clamp(abs(best_heading), self.probe_angle_step_rad, self.max_probe_angle_rad)
        return -_clamp(abs(best_heading), self.probe_angle_step_rad, self.max_probe_angle_rad)

    def _front_scale(self, front_min):
        if front_min <= 0.0:
            return 0.0
        if front_min < self.front_hard_stop_m:
            return 0.0
        if front_min < self.front_soft_stop_m:
            return 0.25
        if front_min < self.front_slowdown_m:
            return 0.55
        return 1.0

    def _turn_scale(self, wz):
        return max(0.30, 1.0 - abs(float(wz)) / max(self.max_wz, 0.01))

    def _turn_direction(self, context, selected):
        if selected['direction'] == 'left':
            return 1.0
        if selected['direction'] == 'right':
            return -1.0
        front = context['front']
        left_score = _as_float(front, 'left_front_min', 0.0) + _as_float(front, 'left_min', 0.0)
        right_score = _as_float(front, 'right_front_min', 0.0) + _as_float(front, 'right_min', 0.0)
        if abs(left_score - right_score) > 0.05:
            return 1.0 if left_score > right_score else -1.0
        return 1.0 if selected['heading'] >= 0.0 else -1.0

    def _safe_turn_available(self, context):
        front = context['front']
        side_clearance = max(context.get('min_side_clearance_m', self.min_side_clearance_m), self.front_soft_stop_m)
        return max(
            _as_float(front, 'left_front_min', 0.0),
            _as_float(front, 'right_front_min', 0.0),
            _as_float(front, 'left_min', 0.0),
            _as_float(front, 'right_min', 0.0),
        ) >= side_clearance

    def _update_soft_block_timer(self, context, now):
        if context['front_soft'] and context['rear_soft']:
            if self.front_rear_soft_since is None:
                self.front_rear_soft_since = now
        else:
            self.front_rear_soft_since = None

    def _front_rear_soft_blocked(self, now):
        return (
            self.front_rear_soft_since is not None
            and now - self.front_rear_soft_since >= self.front_rear_soft_blocked_duration_s
        )

    def _update_stuck_score(self, context, now):
        expected_vx = abs(float(self.last_cmd.linear.x))
        odom_speed = self._latest_odom_speed()
        low_progress = 0.0
        if expected_vx > 0.08 and odom_speed is not None and abs(odom_speed) < 0.02:
            if self.low_progress_since is None:
                self.low_progress_since = now
            low_progress = _clamp((now - self.low_progress_since) / 0.8, 0.0, 1.0)
        else:
            self.low_progress_since = None

        zero_cmd = 0.0
        if self.state not in ('STOP', 'RECOVER') and expected_vx < 0.01 and abs(self.last_cmd.angular.z) < 0.01:
            if self.zero_cmd_since is None:
                self.zero_cmd_since = now
            zero_cmd = _clamp((now - self.zero_cmd_since) / 0.8, 0.0, 1.0)
        else:
            self.zero_cmd_since = None

        oscillation = _clamp(len(self.angular_flip_times) / 3.0, 0.0, 1.0)
        trap = _clamp(max(context['dead_end_score'], context['corner_trap_score']), 0.0, 1.0)
        self.stuck_score = _clamp(
            0.35 * low_progress + 0.25 * zero_cmd + 0.20 * oscillation + 0.20 * trap,
            0.0,
            1.0,
        )
        if self.stuck_score >= self.stuck_enter_threshold:
            if self.stuck_started_at is None:
                self.stuck_started_at = now
        else:
            self.stuck_started_at = None

    def _latest_odom_speed(self):
        if not self.odom_history:
            return None
        return self.odom_history[-1][3]

    def _stuck_ready(self, now):
        return self.stuck_started_at is not None and now - self.stuck_started_at >= self.stuck_enter_duration_s

    def _record_angular_sign(self, now, angular_z):
        sign = 0
        if abs(angular_z) > 0.08:
            sign = 1 if angular_z > 0.0 else -1
        if sign and self.last_angular_sign and sign != self.last_angular_sign:
            self.angular_flip_times.append(now)
        if sign:
            self.last_angular_sign = sign
        self._prune_times(self.angular_flip_times, now, self.oscillation_window_s)

    def _oscillation_detected(self, now):
        self._prune_times(self.angular_flip_times, now, self.oscillation_window_s)
        self._prune_times(self.direction_switch_times, now, self.oscillation_window_s)
        return len(self.angular_flip_times) >= 3 or len(self.direction_switch_times) >= 3

    @staticmethod
    def _prune_times(times, now, window):
        while times and now - times[0] > window:
            times.popleft()

    def _set_state(self, state, now):
        if state == self.state:
            return
        self.state = state
        self.state_enter_time = now
        if state != 'ESCAPE':
            self.escape_clear_since = None

    def _publish_state(self, context, selected, now):
        payload = {
            'stamp': round(now, 6),
            'state': self.state,
            'motion_state': self._motion_state_label(selected),
            'reason': self._state_reason(context),
            'selected_direction': selected['direction'],
            'raw_best_direction': selected.get('raw_best_direction', selected['direction']),
            'selection_reason': selected.get('selection_reason', 'unknown'),
            'direction_hold_active': bool(selected.get('direction_hold_active', False)),
            'score_margin_hold': bool(selected.get('score_margin_hold', False)),
            'switch_penalty_active': bool(selected.get('switch_penalty_active', False)),
            'score_delta': round(float(selected.get('score_delta', 0.0)), 3),
            'selected_heading': round(selected['heading'], 3),
            'best_heading': round(selected['heading'], 3),
            'front_min': round(context['front_min'], 3),
            'left_front_min': round(_as_float(context['front'], 'left_front_min', 0.0), 3),
            'right_front_min': round(_as_float(context['front'], 'right_front_min', 0.0), 3),
            'left_min': round(_as_float(context['front'], 'left_min', 0.0), 3),
            'right_min': round(_as_float(context['front'], 'right_min', 0.0), 3),
            'front_valid': context['front_valid'],
            'front_soft': context['front_soft'],
            'front_hard': context['front_hard'],
            'front_blocked_soft': context['front_soft'],
            'front_blocked_hard': context['front_hard'],
            'front_unknown': context['front_unknown'],
            'forward_corridor_clear': context['forward_corridor_clear'],
            'side_corridor_clear': context.get('side_corridor_clear', False),
            'side_corridor_left_m': round(context.get('side_corridor_left_m', 0.0), 3),
            'side_corridor_right_m': round(context.get('side_corridor_right_m', 0.0), 3),
            'rear_valid': context['rear_valid'],
            'rear_pressure': round(context['rear_pressure'], 3),
            'threat_visible': self._threat_recent(now),
            'traffic_cone_recent': context['traffic_cone_recent'],
            'reverse_allowed': context['reverse_allowed'],
            'z_bump_detected': context['z_bump_detected'],
            'z_bump_score': round(context['z_bump_score'], 3),
            'z_bump_side': context['z_bump_side'],
            'z_bump_reason': context['z_bump_reason'],
            'recover_phase': self.cone_recover_phase if self.state == 'CONE_BASE_RECOVER' else (
                self.recover_mode if self.state == 'RECOVER' else 'idle'
            ),
            'gap_escape_allowed': context['gap_escape_allowed'],
            'gap_width_m': round(context['gap_width_m'], 3),
            'front_path_safe': context['front_path_safe'],
            'min_exit_corridor_width_m': round(context['min_exit_corridor_width_m'], 3),
            'min_side_clearance_m': round(context['min_side_clearance_m'], 3),
            'inflation_radius_m': round(context['inflation_radius_m'], 3),
            'min_obstacle_clearance_m': round(context['min_obstacle_clearance_m'], 3),
            'stuck_score': round(self.stuck_score, 3),
            'cmd_raw_vx': round(float(self.last_cmd.linear.x), 4),
            'cmd_raw_wz': round(float(self.last_cmd.angular.z), 4),
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.state_pub.publish(message)
        self._log_control_debug(payload, now)

    def _motion_state_label(self, selected):
        if self.state in ('STOP',):
            return 'STOP'
        if self.state == 'RECOVER':
            return 'BACKUP' if self.recover_mode == 'reverse' else 'RECOVERY'
        if self.state == 'CONE_BASE_RECOVER':
            return 'BACKUP' if self.cone_recover_mode == 'reverse' else 'RECOVERY'
        direction = selected.get('direction', 'front')
        if direction == 'left':
            return 'AVOID_LEFT'
        if direction == 'right':
            return 'AVOID_RIGHT'
        return 'FORWARD'

    def _log_control_debug(self, payload, now):
        if now - self.last_debug_log_time < 1.0:
            return
        self.last_debug_log_time = now
        self.get_logger().info(
            'avoid_debug state=%s motion=%s front=%.3f left_front=%.3f right_front=%.3f '
            'dir=%s reason=%s hold=%s margin=%s penalty=%s vx=%.4f wz=%.4f'
            % (
                payload['state'],
                payload['motion_state'],
                payload['front_min'],
                payload['left_front_min'],
                payload['right_front_min'],
                payload['selected_direction'],
                payload['selection_reason'],
                payload['direction_hold_active'],
                payload['score_margin_hold'],
                payload['switch_penalty_active'],
                payload['cmd_raw_vx'],
                payload['cmd_raw_wz'],
            )
        )

    def _state_reason(self, context):
        if context['front_timeout']:
            return 'front_timeout'
        if not context['front_valid']:
            return 'front_invalid'
        if self.state == 'GAP_ESCAPE':
            return 'front_hard_gap_escape'
        if self.state == 'CONE_BASE_RECOVER':
            return 'cone_base_z_bump_recover'
        if self.state == 'RECOVER':
            return 'recover'
        if self.state == 'STOP':
            return 'stop'
        if self.state == 'AVOID_FRONT':
            return 'front_soft_avoid'
        if self.state == 'ESCAPE':
            if context['rear_pressure'] >= self.rear_pressure_escape_enter:
                return 'rear_pressure_escape'
            if self._threat_recent(_now_seconds(self)):
                return 'threat_escape'
            return 'escape_hold'
        return 'cruise'

    def _threat_recent(self, now):
        return now - self.last_threat_time <= self.detection_timeout_sec

    def _publish_stuck_status(self, context, now):
        payload = {
            'stamp': round(now, 6),
            'state': self.state,
            'stuck_score': round(self.stuck_score, 3),
            'oscillation_flips': len(self.angular_flip_times),
            'direction_switches': len(self.direction_switch_times),
            'dead_end_score': round(context['dead_end_score'], 3),
            'corner_trap_score': round(context['corner_trap_score'], 3),
            'recover_active': self.state in ('RECOVER', 'CONE_BASE_RECOVER'),
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.stuck_pub.publish(message)

    def _rear_payload_has_threat(self, payload):
        return (
            bool(payload.get('threat_visible', False))
            and str(payload.get('threat_class', '')).strip().lower() in ('traffic_cone', 'yellow_car')
            and _as_float(payload, 'threat_conf', 0.0) >= self.threat_confidence
        )

    def _rear_payload_has_traffic_cone(self, payload):
        return (
            bool(payload.get('threat_visible', False))
            and str(payload.get('threat_class', '')).strip().lower() == 'traffic_cone'
            and _as_float(payload, 'threat_conf', 0.0) >= self.threat_confidence
        )

    def _is_threat_detection(self, detection):
        if not isinstance(detection, dict):
            return False
        visible = bool(detection.get('visible', True))
        class_name = str(detection.get('class_name', '')).strip().lower()
        confidence = _as_float(detection, 'conf', 0.0)
        return visible and class_name in ('traffic_cone', 'yellow_car') and confidence >= self.threat_confidence

    def _is_traffic_cone_detection(self, detection):
        if not isinstance(detection, dict):
            return False
        visible = bool(detection.get('visible', True))
        class_name = str(detection.get('class_name', '')).strip().lower()
        confidence = _as_float(detection, 'conf', 0.0)
        return visible and class_name == 'traffic_cone' and confidence >= self.threat_confidence

    @staticmethod
    def _normalize_detections(payload):
        if isinstance(payload, dict) and isinstance(payload.get('detections'), list):
            return payload['detections']
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        return []

    def _parse_json(self, data, label):
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('invalid %s JSON: %s' % (label, exc))
            return None

    def stop_robot(self):
        try:
            self.cmd_pub.publish(Twist())
        except BaseException:
            pass

    def destroy_node(self):
        try:
            for _ in range(5):
                self.stop_robot()
                time.sleep(0.03)
        finally:
            try:
                super().destroy_node()
            except BaseException:
                pass


def main(args=None):
    if rclpy is None:
        raise RuntimeError('rclpy is required to run planner_controller_node as a ROS2 node')
    rclpy.init(args=args)
    node = PlannerControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error('planner_controller_node crashed: %s' % exc)
        raise
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
