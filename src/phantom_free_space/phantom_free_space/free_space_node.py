import json
import math

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
except ImportError:  # Allows local non-ROS algorithm tests to import this module.
    rclpy = None
    Node = object
    LaserScan = object

    class String:  # pragma: no cover - only used when ROS messages are unavailable.
        def __init__(self):
            self.data = ''


SECTOR_SPECS = {
    'front': (-15.0, 15.0, 0.0),
    'front_left': (15.0, 55.0, 0.55),
    'left': (55.0, 110.0, 1.25),
    'front_right': (-55.0, -15.0, -0.55),
    'right': (-110.0, -55.0, -1.25),
}


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (float(percentile) / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _stamp_seconds(message, node):
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    if stamp is not None:
        seconds = float(getattr(stamp, 'sec', 0)) + float(getattr(stamp, 'nanosec', 0)) * 1e-9
        if seconds > 0.0:
            return seconds
    return node.get_clock().now().nanoseconds * 1e-9


def _stamp_seconds_without_node(message, fallback=0.0):
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    if stamp is not None:
        seconds = float(getattr(stamp, 'sec', 0)) + float(getattr(stamp, 'nanosec', 0)) * 1e-9
        if seconds > 0.0:
            return seconds
    return float(fallback)


def _valid_range(raw_value, range_min, range_max):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value <= 0.0:
        return None
    if range_min > 0.0 and value < range_min:
        return None
    if range_max > range_min and value > range_max:
        return None
    return value


def _free_space_params(params=None):
    values = {
        'lidar_angle_sign': 1.0,
        'lidar_angle_offset_rad': 0.0,
        'front_min_valid_ratio': 0.05,
        'front_hard_stop_m': 0.22,
        'front_soft_stop_m': 0.35,
        'front_slowdown_m': 0.50,
        'score_distance_cap_m': 2.0,
        'gap_min_width_m': 0.25,
        'gap_clearance_min_m': 0.35,
        'gap_search_min_deg': -90.0,
        'gap_search_max_deg': 90.0,
        'range_min': 0.05,
        'range_max': 8.0,
        'angle_min': -math.pi,
        'angle_increment': math.radians(1.0),
        'stamp': 0.0,
    }
    if params:
        values.update(params)
    return values


def _scan_angle_to_robot_angle(scan_angle, params):
    return _normalize_angle(
        float(scan_angle) * float(params['lidar_angle_sign'])
        + float(params['lidar_angle_offset_rad'])
    )


def _sector_stats(values, expected_count, heading, range_max, params):
    expected = max(int(expected_count), 1)
    valid_ratio = _clamp(len(values) / float(expected), 0.0, 1.0)
    if not values:
        min_distance = range_max
        p50 = range_max
        p70 = range_max
    else:
        min_distance = min(values)
        p50 = _percentile(values, 50)
        p70 = _percentile(values, 70)

    def norm(distance):
        return _clamp(distance / max(params['score_distance_cap_m'], 0.1), 0.0, 1.0)

    obstacle_penalty = 0.0
    if min_distance < params['front_hard_stop_m']:
        obstacle_penalty += 1.00
    if min_distance < params['front_soft_stop_m']:
        obstacle_penalty += 0.45
    if valid_ratio < 0.30:
        obstacle_penalty += 0.30

    free_width_score = valid_ratio
    score = (
        0.45 * norm(p70)
        + 0.25 * norm(p50)
        + 0.20 * valid_ratio
        + 0.10 * free_width_score
        - obstacle_penalty
    )
    return {
        'min': float(min_distance),
        'p50': float(p50),
        'p70': float(p70),
        'valid_ratio': float(valid_ratio),
        'score': float(score),
        'heading': float(heading),
    }


def _dead_end_score(stats):
    front_risk = _clamp((0.65 - stats['front']['min']) / 0.43, 0.0, 1.0)
    side_min = min(
        stats['front_left']['min'],
        stats['front_right']['min'],
        stats['left']['min'],
        stats['right']['min'],
    )
    side_risk = _clamp((0.70 - side_min) / 0.48, 0.0, 1.0)
    return _clamp(0.65 * front_risk + 0.35 * side_risk, 0.0, 1.0)


def _corner_trap_score(stats):
    left_pressure = _clamp((0.55 - min(stats['front_left']['min'], stats['left']['min'])) / 0.33, 0.0, 1.0)
    right_pressure = _clamp((0.55 - min(stats['front_right']['min'], stats['right']['min'])) / 0.33, 0.0, 1.0)
    front_pressure = _clamp((0.55 - stats['front']['min']) / 0.33, 0.0, 1.0)
    return _clamp(front_pressure * max(left_pressure, right_pressure), 0.0, 1.0)


def _detect_gap_candidates(ranges, angle_min, angle_increment, range_min, range_max, params):
    search_low = math.radians(float(params['gap_search_min_deg']))
    search_high = math.radians(float(params['gap_search_max_deg']))
    max_jump = abs(angle_increment) * 1.5
    runs = []
    current = []

    def close_run():
        nonlocal current
        if current:
            runs.append(current)
            current = []

    for index, raw_range in enumerate(ranges):
        angle = _scan_angle_to_robot_angle(angle_min + index * angle_increment, params)
        if not (search_low <= angle <= search_high):
            close_run()
            continue
        value = _valid_range(raw_range, range_min, range_max)
        if value is None or value < params['gap_clearance_min_m']:
            close_run()
            continue
        if current and abs(angle - current[-1][0]) > max_jump:
            close_run()
        current.append((angle, value))
    close_run()

    candidates = []
    for run in runs:
        if len(run) < 2:
            continue
        low_angle = run[0][0]
        high_angle = run[-1][0]
        span = abs(high_angle - low_angle)
        distances = [value for _, value in run]
        distance = _percentile(distances, 50)
        width = 2.0 * distance * math.sin(span * 0.5)
        candidates.append({
            'heading': round((low_angle + high_angle) * 0.5, 3),
            'width_m': round(width, 3),
            'distance_m': round(distance, 3),
            'angle_low_deg': round(math.degrees(low_angle), 1),
            'angle_high_deg': round(math.degrees(high_angle), 1),
        })
    candidates.sort(key=lambda item: item['width_m'], reverse=True)
    return candidates[:5]


def compute_front_free_space_from_scan(scan_msg_or_ranges, params=None):
    """Compute /nav/front_free_space JSON payload from a LaserScan-like object or ranges list."""
    params = _free_space_params(params)
    ranges = getattr(scan_msg_or_ranges, 'ranges', scan_msg_or_ranges)
    ranges = list(ranges or [])
    range_min = float(getattr(scan_msg_or_ranges, 'range_min', params['range_min']))
    range_max = float(getattr(scan_msg_or_ranges, 'range_max', params['range_max']))
    angle_min = float(getattr(scan_msg_or_ranges, 'angle_min', params['angle_min']))
    angle_increment = float(getattr(scan_msg_or_ranges, 'angle_increment', params['angle_increment']))
    range_min = range_min if range_min > 0.0 else params['range_min']
    range_max = range_max if range_max > range_min else params['range_max']

    def invalid_payload():
        stamp = round(_stamp_seconds_without_node(scan_msg_or_ranges, params['stamp']), 6)
        return {
            'stamp': stamp,
            'valid': False,
            'front_min': round(range_max, 3),
            'front_p50': round(range_max, 3),
            'front_p70': round(range_max, 3),
            'left_front_min': round(range_max, 3),
            'right_front_min': round(range_max, 3),
            'left_min': round(range_max, 3),
            'right_min': round(range_max, 3),
            'best_heading': 0.0,
            'best_score': 0.0,
            'front_blocked_soft': False,
            'front_blocked_hard': False,
            'front_unknown': True,
            'front_valid_ratio': 0.0,
            'lidar_angle_offset_rad': round(float(params['lidar_angle_offset_rad']), 6),
            'gap_escape_allowed': False,
            'best_gap_heading': 0.0,
            'best_gap_width_m': 0.0,
            'best_gap_distance_m': 0.0,
            'gap_candidates': [],
            'dead_end_score': 0.0,
            'corner_trap_score': 0.0,
            'sector_scores': {},
            'sector_valid_ratio': {},
            'best_sector': 'front',
        }

    if not ranges or not math.isfinite(angle_increment) or angle_increment == 0.0:
        return invalid_payload()

    gap_candidates = _detect_gap_candidates(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        params,
    )
    best_gap = gap_candidates[0] if gap_candidates else {
        'heading': 0.0,
        'width_m': 0.0,
        'distance_m': 0.0,
        'angle_low_deg': 0.0,
        'angle_high_deg': 0.0,
    }

    sectors = {
        name: {
            'values': [],
            'expected': 0,
            'low': math.radians(low),
            'high': math.radians(high),
            'heading': heading,
        }
        for name, (low, high, heading) in SECTOR_SPECS.items()
    }
    for index, raw_range in enumerate(ranges):
        angle = _scan_angle_to_robot_angle(angle_min + index * angle_increment, params)
        for sector in sectors.values():
            if sector['low'] <= angle <= sector['high']:
                sector['expected'] += 1
                value = _valid_range(raw_range, range_min, range_max)
                if value is not None:
                    sector['values'].append(value)
                break

    stats = {
        name: _sector_stats(data['values'], data['expected'], data['heading'], range_max, params)
        for name, data in sectors.items()
    }
    valid_points = sum(len(data['values']) for data in sectors.values())
    valid = valid_points > 0
    best_name = max(stats, key=lambda name: stats[name]['score'])
    best = stats[best_name]
    front = stats['front']
    front_unknown = front['valid_ratio'] < params['front_min_valid_ratio']
    dead_end_score = _dead_end_score(stats)
    corner_trap_score = _corner_trap_score(stats)
    return {
        'stamp': round(_stamp_seconds_without_node(scan_msg_or_ranges, params['stamp']), 6),
        'valid': bool(valid),
        'front_min': round(front['min'], 3),
        'front_p50': round(front['p50'], 3),
        'front_p70': round(front['p70'], 3),
        'left_front_min': round(stats['front_left']['min'], 3),
        'right_front_min': round(stats['front_right']['min'], 3),
        'left_min': round(stats['left']['min'], 3),
        'right_min': round(stats['right']['min'], 3),
        'best_heading': round(best['heading'], 3),
        'best_score': round(best['score'], 3),
        'front_blocked_soft': bool(valid and not front_unknown and front['min'] < params['front_soft_stop_m']),
        'front_blocked_hard': bool(valid and not front_unknown and front['min'] < params['front_hard_stop_m']),
        'front_unknown': bool(front_unknown),
        'front_valid_ratio': round(front['valid_ratio'], 3),
        'gap_escape_allowed': bool(valid and best_gap['width_m'] >= params['gap_min_width_m']),
        'best_gap_heading': round(best_gap['heading'], 3),
        'best_gap_width_m': round(best_gap['width_m'], 3),
        'best_gap_distance_m': round(best_gap['distance_m'], 3),
        'gap_candidates': gap_candidates,
        'lidar_angle_offset_rad': round(float(params['lidar_angle_offset_rad']), 6),
        'dead_end_score': round(dead_end_score, 3),
        'corner_trap_score': round(corner_trap_score, 3),
        'sector_scores': {
            name: round(stats[name]['score'], 3)
            for name in ('left', 'front_left', 'front', 'front_right', 'right')
        },
        'sector_valid_ratio': {
            name: round(stats[name]['valid_ratio'], 3)
            for name in ('left', 'front_left', 'front', 'front_right', 'right')
        },
        'best_sector': best_name,
    }


class FreeSpaceNode(Node):
    def __init__(self):
        super().__init__('free_space_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('front_free_space_topic', '/nav/front_free_space')
        self.declare_parameter('features_topic', '/nav/local_obstacle_features')
        self.declare_parameter('publish_legacy_features', True)
        self.declare_parameter('lidar_angle_sign', 1.0)
        self.declare_parameter('lidar_angle_offset_rad', 0.0)
        self.declare_parameter('front_min_valid_ratio', 0.05)
        self.declare_parameter('front_hard_stop_m', 0.22)
        self.declare_parameter('front_soft_stop_m', 0.35)
        self.declare_parameter('front_slowdown_m', 0.50)
        self.declare_parameter('score_distance_cap_m', 2.0)
        self.declare_parameter('gap_min_width_m', 0.25)
        self.declare_parameter('gap_clearance_min_m', 0.35)
        self.declare_parameter('gap_search_min_deg', -90.0)
        self.declare_parameter('gap_search_max_deg', 90.0)
        self.declare_parameter('debug_log_period_sec', 0.0)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.front_free_space_topic = str(self.get_parameter('front_free_space_topic').value)
        self.features_topic = str(self.get_parameter('features_topic').value)
        self.publish_legacy_features = bool(self.get_parameter('publish_legacy_features').value)
        self.lidar_angle_sign = float(self.get_parameter('lidar_angle_sign').value)
        self.lidar_angle_offset_rad = float(self.get_parameter('lidar_angle_offset_rad').value)
        self.front_min_valid_ratio = float(self.get_parameter('front_min_valid_ratio').value)
        self.front_hard_stop_m = float(self.get_parameter('front_hard_stop_m').value)
        self.front_soft_stop_m = float(self.get_parameter('front_soft_stop_m').value)
        self.front_slowdown_m = float(self.get_parameter('front_slowdown_m').value)
        self.score_distance_cap_m = float(self.get_parameter('score_distance_cap_m').value)
        self.gap_min_width_m = float(self.get_parameter('gap_min_width_m').value)
        self.gap_clearance_min_m = float(self.get_parameter('gap_clearance_min_m').value)
        self.gap_search_min_deg = float(self.get_parameter('gap_search_min_deg').value)
        self.gap_search_max_deg = float(self.get_parameter('gap_search_max_deg').value)
        self.debug_log_period_sec = float(self.get_parameter('debug_log_period_sec').value)

        self.last_debug_log_time = -999.0
        self.front_pub = self.create_publisher(String, self.front_free_space_topic, 10)
        self.legacy_pub = None
        if self.publish_legacy_features and self.features_topic != self.front_free_space_topic:
            self.legacy_pub = self.create_publisher(String, self.features_topic, 10)

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)

        self.get_logger().info(
            'free_space_node subscribed to %s, publishing %s'
            % (self.scan_topic, self.front_free_space_topic)
        )
        if self.legacy_pub is not None:
            self.get_logger().info('free_space_node also publishing legacy %s' % self.features_topic)

    def scan_callback(self, scan):
        payload = self.build_front_free_space(scan)
        self._publish_json(self.front_pub, payload)
        if self.legacy_pub is not None:
            self._publish_json(self.legacy_pub, self._legacy_payload(payload))
        self._debug_log(payload)

    def build_front_free_space(self, scan):
        return compute_front_free_space_from_scan(scan, {
            'lidar_angle_sign': self.lidar_angle_sign,
            'lidar_angle_offset_rad': self.lidar_angle_offset_rad,
            'front_min_valid_ratio': self.front_min_valid_ratio,
            'front_hard_stop_m': self.front_hard_stop_m,
            'front_soft_stop_m': self.front_soft_stop_m,
            'front_slowdown_m': self.front_slowdown_m,
            'score_distance_cap_m': self.score_distance_cap_m,
            'gap_min_width_m': self.gap_min_width_m,
            'gap_clearance_min_m': self.gap_clearance_min_m,
            'gap_search_min_deg': self.gap_search_min_deg,
            'gap_search_max_deg': self.gap_search_max_deg,
            'stamp': _stamp_seconds(scan, self),
        })

    def _invalid_payload(self, scan, range_max):
        stamp = round(_stamp_seconds(scan, self), 6)
        return {
            'stamp': stamp,
            'valid': False,
            'front_min': round(range_max, 3),
            'front_p50': round(range_max, 3),
            'front_p70': round(range_max, 3),
            'left_front_min': round(range_max, 3),
            'right_front_min': round(range_max, 3),
            'left_min': round(range_max, 3),
            'right_min': round(range_max, 3),
            'best_heading': 0.0,
            'best_score': 0.0,
            'front_blocked_soft': False,
            'front_blocked_hard': False,
            'front_unknown': True,
            'front_valid_ratio': 0.0,
            'lidar_angle_offset_rad': round(float(self.lidar_angle_offset_rad), 6),
            'gap_escape_allowed': False,
            'best_gap_heading': 0.0,
            'best_gap_width_m': 0.0,
            'best_gap_distance_m': 0.0,
            'gap_candidates': [],
            'dead_end_score': 0.0,
            'corner_trap_score': 0.0,
            'sector_scores': {},
            'sector_valid_ratio': {},
            'best_sector': 'front',
        }

    def _sector_stats(self, values, expected_count, heading, range_max):
        expected = max(int(expected_count), 1)
        valid_ratio = _clamp(len(values) / float(expected), 0.0, 1.0)
        if not values:
            min_distance = range_max
            p50 = range_max
            p70 = range_max
        else:
            min_distance = min(values)
            p50 = _percentile(values, 50)
            p70 = _percentile(values, 70)

        def norm(distance):
            return _clamp(distance / max(self.score_distance_cap_m, 0.1), 0.0, 1.0)

        obstacle_penalty = 0.0
        if min_distance < self.front_hard_stop_m:
            obstacle_penalty += 1.00
        if min_distance < self.front_soft_stop_m:
            obstacle_penalty += 0.45
        if valid_ratio < 0.30:
            obstacle_penalty += 0.30

        free_width_score = valid_ratio
        score = (
            0.45 * norm(p70)
            + 0.25 * norm(p50)
            + 0.20 * valid_ratio
            + 0.10 * free_width_score
            - obstacle_penalty
        )
        return {
            'min': float(min_distance),
            'p50': float(p50),
            'p70': float(p70),
            'valid_ratio': float(valid_ratio),
            'score': float(score),
            'heading': float(heading),
        }

    def _dead_end_score(self, stats):
        front_risk = _clamp((0.65 - stats['front']['min']) / 0.43, 0.0, 1.0)
        side_min = min(
            stats['front_left']['min'],
            stats['front_right']['min'],
            stats['left']['min'],
            stats['right']['min'],
        )
        side_risk = _clamp((0.70 - side_min) / 0.48, 0.0, 1.0)
        return _clamp(0.65 * front_risk + 0.35 * side_risk, 0.0, 1.0)

    def _corner_trap_score(self, stats):
        left_pressure = _clamp((0.55 - min(stats['front_left']['min'], stats['left']['min'])) / 0.33, 0.0, 1.0)
        right_pressure = _clamp((0.55 - min(stats['front_right']['min'], stats['right']['min'])) / 0.33, 0.0, 1.0)
        front_pressure = _clamp((0.55 - stats['front']['min']) / 0.33, 0.0, 1.0)
        return _clamp(front_pressure * max(left_pressure, right_pressure), 0.0, 1.0)

    def _legacy_payload(self, payload):
        return {
            'stamp': payload['stamp'],
            'valid': payload['valid'],
            'front_clearance': payload['front_min'],
            'left_front_clearance': payload['left_front_min'],
            'right_front_clearance': payload['right_front_min'],
            'left_clearance': payload['left_min'],
            'right_clearance': payload['right_min'],
            'local_free_heading': payload['best_heading'],
            'front_blocked': payload['front_blocked_soft'],
            'front_blocked_soft': payload['front_blocked_soft'],
            'front_blocked_hard': payload['front_blocked_hard'],
            'front_unknown': payload.get('front_unknown', False),
            'front_valid_ratio': payload.get('front_valid_ratio', 0.0),
            'lidar_angle_offset_rad': payload.get('lidar_angle_offset_rad', 0.0),
            'gap_escape_allowed': payload.get('gap_escape_allowed', False),
            'best_gap_heading': payload.get('best_gap_heading', 0.0),
            'best_gap_width_m': payload.get('best_gap_width_m', 0.0),
            'dead_end_score': payload['dead_end_score'],
            'corner_trap_score': payload['corner_trap_score'],
            'escape_corridor_score': _clamp(payload['best_score'], 0.0, 1.0),
        }

    def _debug_log(self, payload):
        if self.debug_log_period_sec <= 0.0:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_debug_log_time < self.debug_log_period_sec:
            return
        self.last_debug_log_time = now
        self.get_logger().info(
            'front_free_space valid=%s unknown=%s front=%.2f best=%s heading=%.2f score=%.2f offset=%.3f'
            % (
                payload['valid'],
                payload.get('front_unknown', False),
                payload['front_min'],
                payload.get('best_sector', 'front'),
                payload['best_heading'],
                payload['best_score'],
                payload.get('lidar_angle_offset_rad', 0.0),
            )
        )

    @staticmethod
    def _publish_json(publisher, payload):
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        publisher.publish(message)


def main(args=None):
    if rclpy is None:
        raise RuntimeError('rclpy is required to run free_space_node as a ROS2 node')
    rclpy.init(args=args)
    node = FreeSpaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
