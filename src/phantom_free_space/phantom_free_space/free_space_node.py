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
        'front_soft_stop_m': 0.34,
        'front_slowdown_m': 0.72,
        'front_path_soft_check_m': 0.48,
        'score_distance_cap_m': 2.0,
        'gap_min_width_m': 0.25,
        'gap_clearance_min_m': 0.35,
        'gap_search_min_deg': -90.0,
        'gap_search_max_deg': 90.0,
        'gap_eval_distance_m': 1.10,
        'robot_width_m': 0.25,
        'robot_length_m': 0.30,
        'safety_margin_m': 0.05,
        'cone_base_radius_m': 0.15,
        'obstacle_extra_margin_m': 0.03,
        'cone_cluster_max_width_m': 0.25,
        'cone_cluster_min_points': 4,
        'side_corridor_forward_min_m': 0.34,
        'side_corridor_obstacle_m': 0.75,
        'side_corridor_balance_deadband_m': 0.04,
        'side_corridor_heading_limit_rad': 0.16,
        'range_min': 0.05,
        'range_max': 8.0,
        'angle_min': -math.pi,
        'angle_increment': math.radians(1.0),
        'stamp': 0.0,
    }
    if params:
        values.update(params)
    half_width = max(float(values['robot_width_m']) * 0.5, 0.01)
    half_length = max(float(values['robot_length_m']) * 0.5, 0.01)
    margin = max(float(values['safety_margin_m']), 0.0)
    min_side_clearance = half_width + margin
    inflation_radius = math.hypot(half_width, half_length) + margin
    min_exit_width = float(values.get('min_exit_corridor_width_m', 0.0) or 0.0)
    if min_exit_width <= 0.0:
        min_exit_width = float(values['robot_width_m']) + 2.0 * margin
    cone_base_radius = max(float(values['cone_base_radius_m']), 0.0)
    obstacle_extra = max(float(values['obstacle_extra_margin_m']), 0.0)
    min_obstacle_clearance = float(values.get('min_obstacle_clearance_m', 0.0) or 0.0)
    if min_obstacle_clearance <= 0.0:
        min_obstacle_clearance = min_side_clearance + cone_base_radius + obstacle_extra
    values.update({
        'half_width_m': half_width,
        'half_length_m': half_length,
        'min_side_clearance_m': min_side_clearance,
        'inflation_radius_m': inflation_radius,
        'min_exit_corridor_width_m': min_exit_width,
        'min_obstacle_clearance_m': min_obstacle_clearance,
    })
    values['front_hard_stop_m'] = max(float(values['front_hard_stop_m']), half_length + margin)
    values['front_soft_stop_m'] = max(float(values['front_soft_stop_m']), half_length + 2.0 * margin)
    values['front_path_soft_check_m'] = max(float(values['front_path_soft_check_m']), values['front_soft_stop_m'])
    values['gap_min_width_m'] = max(float(values['gap_min_width_m']), min_exit_width)
    values['gap_clearance_min_m'] = max(float(values['gap_clearance_min_m']), half_length + margin)
    values['side_corridor_forward_min_m'] = max(
        float(values['side_corridor_forward_min_m']),
        values['front_soft_stop_m'],
    )
    values['side_corridor_obstacle_m'] = max(
        float(values['side_corridor_obstacle_m']),
        values['min_side_clearance_m'] + values['safety_margin_m'],
    )
    values['side_corridor_balance_deadband_m'] = max(float(values['side_corridor_balance_deadband_m']), 0.0)
    values['side_corridor_heading_limit_rad'] = max(float(values['side_corridor_heading_limit_rad']), 0.0)
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


def _collect_scan_points(ranges, angle_min, angle_increment, range_min, range_max, params):
    points = []
    for index, raw_range in enumerate(ranges):
        angle = _scan_angle_to_robot_angle(angle_min + index * angle_increment, params)
        value = _valid_range(raw_range, range_min, range_max)
        if value is not None:
            points.append((angle, value))
    points.sort(key=lambda item: item[0])
    return points


def _scan_point_clusters(points, angle_increment, params):
    if not points:
        return []
    max_angle_gap = max(abs(angle_increment) * 1.8, math.radians(2.0))
    max_xy_gap = 0.16
    clusters = []
    current = [points[0]]

    def xy(point):
        angle, radius = point
        return radius * math.cos(angle), radius * math.sin(angle)

    def close_current():
        if current:
            clusters.append(list(current))

    for point in points[1:]:
        prev = current[-1]
        px, py = xy(prev)
        x, y = xy(point)
        if abs(point[0] - prev[0]) <= max_angle_gap and math.hypot(x - px, y - py) <= max_xy_gap:
            current.append(point)
        else:
            close_current()
            current = [point]
    close_current()

    enriched = []
    for cluster in clusters:
        xs = [radius * math.cos(angle) for angle, radius in cluster]
        ys = [radius * math.sin(angle) for angle, radius in cluster]
        width = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if len(cluster) > 1 else 0.0
        enriched.append({'points': cluster, 'width_m': width})
    return enriched


def _continuous_path_safe(heading, distance, scan_points, clusters, params):
    distance = max(float(distance), params['front_soft_stop_m'])
    half_angle = math.atan2(params['min_side_clearance_m'], max(distance, 0.05))
    base_front = distance + params['half_length_m'] + params['safety_margin_m']
    base_rear = -params['half_length_m'] - params['safety_margin_m']
    side_limit = params['min_side_clearance_m']
    cone_extra = params['cone_base_radius_m'] + params['obstacle_extra_margin_m']
    cone_cluster_max = params['cone_cluster_max_width_m']
    cone_cluster_min_points = max(int(params.get('cone_cluster_min_points', 4)), 1)
    min_left = range_max_left = None
    min_right = range_max_right = None

    cluster_by_point = {}
    for cluster in clusters:
        for point in cluster['points']:
            cluster_by_point[id(point)] = cluster

    unsafe_reasons = []
    for point in scan_points:
        angle, radius = point
        delta = _normalize_angle(angle - heading)
        if abs(delta) > max(half_angle, math.radians(1.0)):
            continue
        x = radius * math.cos(delta)
        y = radius * math.sin(delta)
        if y >= 0.0:
            min_left = abs(y) if min_left is None else min(min_left, abs(y))
            range_max_left = radius if range_max_left is None else min(range_max_left, radius)
        else:
            min_right = abs(y) if min_right is None else min(min_right, abs(y))
            range_max_right = radius if range_max_right is None else min(range_max_right, radius)

        cluster = cluster_by_point.get(id(point), {})
        is_cone_like = (
            len(cluster.get('points', ())) >= cone_cluster_min_points
            and float(cluster.get('width_m', 999.0)) <= cone_cluster_max
        )
        extra = cone_extra if is_cone_like else 0.0
        if base_rear - extra <= x <= base_front + extra and abs(y) <= side_limit + extra:
            unsafe_reasons.append('cone_base' if is_cone_like else 'footprint')
            break

    return {
        'safe': not unsafe_reasons,
        'required_half_angle': half_angle,
        'min_left_lateral_m': min_left,
        'min_right_lateral_m': min_right,
        'left_range_m': range_max_left,
        'right_range_m': range_max_right,
        'reason': unsafe_reasons[0] if unsafe_reasons else '',
    }


def _detect_gap_candidates(ranges, angle_min, angle_increment, range_min, range_max, params):
    search_low = math.radians(float(params['gap_search_min_deg']))
    search_high = math.radians(float(params['gap_search_max_deg']))
    max_jump = abs(angle_increment) * 1.5
    runs = []
    current = []
    scan_points = _collect_scan_points(ranges, angle_min, angle_increment, range_min, range_max, params)
    clusters = _scan_point_clusters(scan_points, angle_increment, params)

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
        heading = (low_angle + high_angle) * 0.5
        width = 2.0 * distance * math.sin(span * 0.5)
        eval_distance = min(distance, float(params['gap_eval_distance_m']))
        required_half_angle = math.atan2(params['min_side_clearance_m'], max(eval_distance, 0.05))
        left_clearance = max(0.0, distance * math.sin(max(0.0, high_angle - heading)))
        right_clearance = max(0.0, distance * math.sin(max(0.0, heading - low_angle)))
        path_check = _continuous_path_safe(heading, eval_distance, scan_points, clusters, params)
        safe = (
            width >= params['min_exit_corridor_width_m']
            and span * 0.5 >= required_half_angle
            and min(left_clearance, right_clearance) >= params['min_side_clearance_m']
            and path_check['safe']
        )
        clearance_surplus = min(left_clearance, right_clearance) - params['min_side_clearance_m']
        balance_penalty = abs(left_clearance - right_clearance)
        score = 2.0 * clearance_surplus - 0.35 * abs(heading) - 0.15 * balance_penalty + 0.05 * min(distance, 3.0)
        candidates.append({
            'heading': round(heading, 3),
            'width_m': round(width, 3),
            'distance_m': round(distance, 3),
            'angle_low_deg': round(math.degrees(low_angle), 1),
            'angle_high_deg': round(math.degrees(high_angle), 1),
            'required_half_angle_deg': round(math.degrees(required_half_angle), 1),
            'left_clearance_m': round(left_clearance, 3),
            'right_clearance_m': round(right_clearance, 3),
            'centerline_error_m': round((left_clearance - right_clearance) * 0.5, 3),
            'footprint_safe': bool(safe),
            'unsafe_reason': path_check['reason'],
            'score': round(score if safe else score - 10.0, 3),
        })
    candidates.sort(key=lambda item: item['score'], reverse=True)
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
            'centerline_bias_heading': 0.0,
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
            'front_path_safe': False,
            'side_corridor_clear': False,
            'side_corridor_heading': 0.0,
            'side_corridor_left_m': round(range_max, 3),
            'side_corridor_right_m': round(range_max, 3),
            'front_path_required_half_angle_deg': 0.0,
            'robot_width_m': round(float(params['robot_width_m']), 3),
            'robot_length_m': round(float(params['robot_length_m']), 3),
            'safety_margin_m': round(float(params['safety_margin_m']), 3),
            'min_side_clearance_m': round(float(params['min_side_clearance_m']), 3),
            'inflation_radius_m': round(float(params['inflation_radius_m']), 3),
            'min_exit_corridor_width_m': round(float(params['min_exit_corridor_width_m']), 3),
            'cone_base_radius_m': round(float(params['cone_base_radius_m']), 3),
            'obstacle_extra_margin_m': round(float(params['obstacle_extra_margin_m']), 3),
            'min_obstacle_clearance_m': round(float(params['min_obstacle_clearance_m']), 3),
            'dead_end_score': 0.0,
            'corner_trap_score': 0.0,
            'sector_scores': {},
            'sector_valid_ratio': {},
            'best_sector': 'front',
        }

    if not ranges or not math.isfinite(angle_increment) or angle_increment == 0.0:
        return invalid_payload()

    scan_points = _collect_scan_points(ranges, angle_min, angle_increment, range_min, range_max, params)
    clusters = _scan_point_clusters(scan_points, angle_increment, params)
    front_path_check = _continuous_path_safe(
        0.0,
        float(params['front_path_soft_check_m']),
        scan_points,
        clusters,
        params,
    )
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
    left_side = min(stats['front_left']['min'], stats['left']['min'])
    right_side = min(stats['front_right']['min'], stats['right']['min'])
    side_corridor_clear = bool(
        valid
        and not front_unknown
        and front['min'] >= params['side_corridor_forward_min_m']
        and left_side >= params['min_side_clearance_m']
        and right_side >= params['min_side_clearance_m']
        and left_side <= params['side_corridor_obstacle_m']
        and right_side <= params['side_corridor_obstacle_m']
    )
    side_corridor_heading = 0.0
    if side_corridor_clear:
        side_delta = left_side - right_side
        if abs(side_delta) >= params['side_corridor_balance_deadband_m']:
            side_corridor_heading = _clamp(
                0.65 * side_delta,
                -params['side_corridor_heading_limit_rad'],
                params['side_corridor_heading_limit_rad'],
            )
    gap_escape_allowed = bool(
        valid
        and best_gap.get('footprint_safe', False)
        and best_gap['width_m'] >= params['min_exit_corridor_width_m']
    )
    front_blocked_soft = bool(
        valid
        and not front_unknown
        and (
            front['min'] < params['front_soft_stop_m']
            or (not front_path_check['safe'] and not side_corridor_clear and not gap_escape_allowed)
        )
    )
    front_blocked_hard = bool(valid and not front_unknown and front['min'] < params['front_hard_stop_m'])
    recommended_heading = best['heading']
    centerline_bias_heading = 0.0
    if side_corridor_clear:
        centerline_bias_heading = side_corridor_heading
        recommended_heading = side_corridor_heading
    elif valid and front_path_check['safe'] and front['min'] >= params['front_soft_stop_m']:
        side_delta = left_side - right_side
        if abs(side_delta) >= params['safety_margin_m'] * 0.5:
            centerline_bias_heading = _clamp(0.85 * side_delta, -0.25, 0.25)
            recommended_heading = centerline_bias_heading
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
        'best_heading': round(recommended_heading, 3),
        'centerline_bias_heading': round(centerline_bias_heading, 3),
        'best_score': round(best['score'], 3),
        'front_blocked_soft': front_blocked_soft,
        'front_blocked_hard': front_blocked_hard,
        'front_unknown': bool(front_unknown),
        'front_valid_ratio': round(front['valid_ratio'], 3),
        'gap_escape_allowed': gap_escape_allowed,
        'best_gap_heading': round(best_gap['heading'], 3),
        'best_gap_width_m': round(best_gap['width_m'], 3),
        'best_gap_distance_m': round(best_gap['distance_m'], 3),
        'gap_candidates': gap_candidates,
        'front_path_safe': bool(valid and front_path_check['safe']),
        'side_corridor_clear': side_corridor_clear,
        'side_corridor_heading': round(side_corridor_heading, 3),
        'side_corridor_left_m': round(left_side, 3),
        'side_corridor_right_m': round(right_side, 3),
        'front_path_unsafe_reason': front_path_check['reason'],
        'front_path_required_half_angle_deg': round(math.degrees(front_path_check['required_half_angle']), 1),
        'robot_width_m': round(float(params['robot_width_m']), 3),
        'robot_length_m': round(float(params['robot_length_m']), 3),
        'safety_margin_m': round(float(params['safety_margin_m']), 3),
        'min_side_clearance_m': round(float(params['min_side_clearance_m']), 3),
        'inflation_radius_m': round(float(params['inflation_radius_m']), 3),
        'min_exit_corridor_width_m': round(float(params['min_exit_corridor_width_m']), 3),
        'cone_base_radius_m': round(float(params['cone_base_radius_m']), 3),
        'obstacle_extra_margin_m': round(float(params['obstacle_extra_margin_m']), 3),
        'min_obstacle_clearance_m': round(float(params['min_obstacle_clearance_m']), 3),
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
        self.declare_parameter('front_soft_stop_m', 0.34)
        self.declare_parameter('front_slowdown_m', 0.72)
        self.declare_parameter('front_path_soft_check_m', 0.48)
        self.declare_parameter('score_distance_cap_m', 2.0)
        self.declare_parameter('gap_min_width_m', 0.25)
        self.declare_parameter('gap_clearance_min_m', 0.35)
        self.declare_parameter('gap_search_min_deg', -90.0)
        self.declare_parameter('gap_search_max_deg', 90.0)
        self.declare_parameter('gap_eval_distance_m', 1.10)
        self.declare_parameter('robot_width_m', 0.25)
        self.declare_parameter('robot_length_m', 0.30)
        self.declare_parameter('safety_margin_m', 0.05)
        self.declare_parameter('min_side_clearance_m', 0.0)
        self.declare_parameter('inflation_radius_m', 0.0)
        self.declare_parameter('min_exit_corridor_width_m', 0.0)
        self.declare_parameter('cone_base_radius_m', 0.15)
        self.declare_parameter('obstacle_extra_margin_m', 0.03)
        self.declare_parameter('cone_cluster_max_width_m', 0.25)
        self.declare_parameter('cone_cluster_min_points', 4)
        self.declare_parameter('min_obstacle_clearance_m', 0.0)
        self.declare_parameter('side_corridor_forward_min_m', 0.34)
        self.declare_parameter('side_corridor_obstacle_m', 0.75)
        self.declare_parameter('side_corridor_balance_deadband_m', 0.04)
        self.declare_parameter('side_corridor_heading_limit_rad', 0.16)
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
        self.front_path_soft_check_m = float(self.get_parameter('front_path_soft_check_m').value)
        self.score_distance_cap_m = float(self.get_parameter('score_distance_cap_m').value)
        self.gap_min_width_m = float(self.get_parameter('gap_min_width_m').value)
        self.gap_clearance_min_m = float(self.get_parameter('gap_clearance_min_m').value)
        self.gap_search_min_deg = float(self.get_parameter('gap_search_min_deg').value)
        self.gap_search_max_deg = float(self.get_parameter('gap_search_max_deg').value)
        self.gap_eval_distance_m = float(self.get_parameter('gap_eval_distance_m').value)
        self.robot_width_m = float(self.get_parameter('robot_width_m').value)
        self.robot_length_m = float(self.get_parameter('robot_length_m').value)
        self.safety_margin_m = float(self.get_parameter('safety_margin_m').value)
        self.min_side_clearance_m = float(self.get_parameter('min_side_clearance_m').value)
        self.inflation_radius_m = float(self.get_parameter('inflation_radius_m').value)
        self.min_exit_corridor_width_m = float(self.get_parameter('min_exit_corridor_width_m').value)
        self.cone_base_radius_m = float(self.get_parameter('cone_base_radius_m').value)
        self.obstacle_extra_margin_m = float(self.get_parameter('obstacle_extra_margin_m').value)
        self.cone_cluster_max_width_m = float(self.get_parameter('cone_cluster_max_width_m').value)
        self.cone_cluster_min_points = int(self.get_parameter('cone_cluster_min_points').value)
        self.min_obstacle_clearance_m = float(self.get_parameter('min_obstacle_clearance_m').value)
        self.side_corridor_forward_min_m = float(self.get_parameter('side_corridor_forward_min_m').value)
        self.side_corridor_obstacle_m = float(self.get_parameter('side_corridor_obstacle_m').value)
        self.side_corridor_balance_deadband_m = float(self.get_parameter('side_corridor_balance_deadband_m').value)
        self.side_corridor_heading_limit_rad = float(self.get_parameter('side_corridor_heading_limit_rad').value)
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
            'front_path_soft_check_m': self.front_path_soft_check_m,
            'score_distance_cap_m': self.score_distance_cap_m,
            'gap_min_width_m': self.gap_min_width_m,
            'gap_clearance_min_m': self.gap_clearance_min_m,
            'gap_search_min_deg': self.gap_search_min_deg,
            'gap_search_max_deg': self.gap_search_max_deg,
            'gap_eval_distance_m': self.gap_eval_distance_m,
            'robot_width_m': self.robot_width_m,
            'robot_length_m': self.robot_length_m,
            'safety_margin_m': self.safety_margin_m,
            'min_side_clearance_m': self.min_side_clearance_m,
            'inflation_radius_m': self.inflation_radius_m,
            'min_exit_corridor_width_m': self.min_exit_corridor_width_m,
            'cone_base_radius_m': self.cone_base_radius_m,
            'obstacle_extra_margin_m': self.obstacle_extra_margin_m,
            'cone_cluster_max_width_m': self.cone_cluster_max_width_m,
            'cone_cluster_min_points': self.cone_cluster_min_points,
            'min_obstacle_clearance_m': self.min_obstacle_clearance_m,
            'side_corridor_forward_min_m': self.side_corridor_forward_min_m,
            'side_corridor_obstacle_m': self.side_corridor_obstacle_m,
            'side_corridor_balance_deadband_m': self.side_corridor_balance_deadband_m,
            'side_corridor_heading_limit_rad': self.side_corridor_heading_limit_rad,
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
            'front_path_safe': payload.get('front_path_safe', False),
            'side_corridor_clear': payload.get('side_corridor_clear', False),
            'side_corridor_heading': payload.get('side_corridor_heading', 0.0),
            'side_corridor_left_m': payload.get('side_corridor_left_m', 0.0),
            'side_corridor_right_m': payload.get('side_corridor_right_m', 0.0),
            'min_side_clearance_m': payload.get('min_side_clearance_m', 0.0),
            'inflation_radius_m': payload.get('inflation_radius_m', 0.0),
            'min_exit_corridor_width_m': payload.get('min_exit_corridor_width_m', 0.0),
            'min_obstacle_clearance_m': payload.get('min_obstacle_clearance_m', 0.0),
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
