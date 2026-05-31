import math


FRONT_KEYS = {
    'stamp',
    'valid',
    'front_min',
    'front_p50',
    'front_p70',
    'left_front_min',
    'right_front_min',
    'left_min',
    'right_min',
    'best_heading',
    'best_score',
    'front_blocked_soft',
    'front_blocked_hard',
    'dead_end_score',
    'corner_trap_score',
}

REAR_KEYS = {
    'stamp',
    'valid',
    'rear_min',
    'rear_center_min',
    'rear_left_min',
    'rear_right_min',
    'rear_clearance_score',
    'reverse_allowed',
    'rear_pressure',
    'rear_blocked_soft',
    'rear_blocked_hard',
    'threat_visible',
    'threat_class',
    'threat_conf',
    'threat_depth',
}


def clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def percentile(values, pct):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def front_from_scan_mock(ranges, angle_min_deg=-110.0, angle_step_deg=1.0):
    sectors = {
        'front': (-15.0, 15.0, 0.0),
        'front_left': (15.0, 55.0, 0.55),
        'left': (55.0, 110.0, 1.25),
        'front_right': (-55.0, -15.0, -0.55),
        'right': (-110.0, -55.0, -1.25),
    }
    values = {name: [] for name in sectors}
    for index, raw in enumerate(ranges):
        angle = angle_min_deg + index * angle_step_deg
        if raw is None or not math.isfinite(raw) or raw <= 0.0:
            continue
        for name, (lo, hi, _) in sectors.items():
            if lo <= angle <= hi:
                values[name].append(float(raw))
                break

    stats = {}
    for name, (_, _, heading) in sectors.items():
        sector_values = values[name]
        if not sector_values:
            stats[name] = {'min': 8.0, 'p50': 8.0, 'p70': 8.0, 'heading': heading, 'score': -0.3}
            continue
        p50 = percentile(sector_values, 50)
        p70 = percentile(sector_values, 70)
        min_value = min(sector_values)
        penalty = 1.0 if min_value < 0.22 else 0.45 if min_value < 0.35 else 0.0
        stats[name] = {
            'min': min_value,
            'p50': p50,
            'p70': p70,
            'heading': heading,
            'score': 0.45 * clamp(p70 / 2.0, 0.0, 1.0) + 0.25 * clamp(p50 / 2.0, 0.0, 1.0) - penalty,
        }
    best_name = max(stats, key=lambda name: stats[name]['score'])
    front = stats['front']
    return {
        'stamp': 1.0,
        'valid': any(values.values()),
        'front_min': front['min'],
        'front_p50': front['p50'],
        'front_p70': front['p70'],
        'left_front_min': stats['front_left']['min'],
        'right_front_min': stats['front_right']['min'],
        'left_min': stats['left']['min'],
        'right_min': stats['right']['min'],
        'best_heading': stats[best_name]['heading'],
        'best_score': stats[best_name]['score'],
        'front_blocked_soft': front['min'] < 0.35,
        'front_blocked_hard': front['min'] < 0.22,
        'dead_end_score': 0.0,
        'corner_trap_score': 0.0,
    }


def rear_from_depth_mock(depth):
    height = len(depth)
    width = len(depth[0])
    y0 = int(round(0.35 * height))
    y1 = int(round(0.90 * height))

    def roi(x0, x1):
        values = []
        for row in depth[y0:y1]:
            for value in row[x0:x1]:
                if math.isfinite(value) and value > 0.0:
                    values.append(float(value))
        return values

    left = roi(0, int(round(0.35 * width)))
    center = roi(int(round(0.35 * width)), int(round(0.65 * width)))
    right = roi(int(round(0.65 * width)), width)
    if not center:
        center_near = 0.0
    else:
        center_near = percentile(center, 20)
    left_near = percentile(left, 20) if left else 0.0
    right_near = percentile(right, 20) if right else 0.0
    valid = center_near > 0.0
    pressure = clamp((1.20 - center_near) / (1.20 - 0.35), 0.0, 1.0) if valid else 0.0
    return {
        'stamp': 1.0,
        'valid': valid,
        'rear_min': min(value for value in (center_near, left_near, right_near) if value > 0.0) if valid else 0.0,
        'rear_center_min': center_near,
        'rear_left_min': left_near,
        'rear_right_min': right_near,
        'rear_clearance_score': clamp((center_near - 0.30) / (1.20 - 0.30), 0.0, 1.0) if valid else 0.0,
        'reverse_allowed': valid and center_near > 0.55 and left_near >= 0.30 and right_near >= 0.30,
        'rear_pressure': pressure,
        'rear_blocked_soft': valid and center_near < 0.55,
        'rear_blocked_hard': valid and center_near < 0.30,
        'threat_visible': False,
        'threat_class': '',
        'threat_conf': 0.0,
        'threat_depth': None,
    }


def planner_decision_mock(front, rear, detections):
    threat = any(item['class_name'] in ('yellow_car', 'traffic_cone') and item['conf'] >= 0.25 for item in detections)
    if not front['valid']:
        return 'STOP', 0.0, 0.0
    if front['front_blocked_hard']:
        return 'RECOVER', 0.0, 0.55
    if front['front_blocked_soft']:
        return 'AVOID_FRONT', 0.04, math.copysign(0.45, front['best_heading'] or 1.0)
    if rear['rear_pressure'] >= 0.55 or threat:
        return 'ESCAPE', 0.28, clamp(1.15 * front['best_heading'], -0.75, 0.75)
    return 'CRUISE', 0.16, clamp(1.15 * front['best_heading'], -0.75, 0.75)


def safety_mock(raw_vx, raw_wz, front, rear):
    vx = raw_vx
    wz = raw_wz
    front_min = front['front_min']
    if vx > 0.0 and front_min < 0.22:
        vx = 0.0
        wz = clamp(wz, -0.55, 0.55)
    elif vx > 0.0 and front_min < 0.35:
        vx = min(vx, 0.05)
        wz = clamp(wz, -0.65, 0.65)
    elif vx > 0.0 and front_min < 0.50:
        vx = min(vx, 0.12)
    if vx < 0.0 and not rear['valid']:
        vx = 0.0
    elif vx < 0.0 and rear['rear_center_min'] < 0.30:
        vx = 0.0
    elif vx < 0.0 and rear['rear_center_min'] < 0.55:
        vx = max(vx, -0.03)
    return clamp(vx, -0.08, 0.32), clamp(wz, -0.75, 0.75)


def main():
    ranges = [1.0] * 221
    ranges[110] = float('nan')
    ranges[111] = float('inf')
    ranges[112] = 0.0
    front = front_from_scan_mock(ranges)
    assert FRONT_KEYS <= set(front)
    assert front['valid'] is True
    assert front['front_blocked_soft'] is False

    clear_depth = [[1.0 for _ in range(10)] for _ in range(10)]
    rear = rear_from_depth_mock(clear_depth)
    assert REAR_KEYS <= set(rear)
    assert rear['valid'] is True
    assert rear['reverse_allowed'] is True

    state, vx, _ = planner_decision_mock(front, rear, [])
    assert state == 'CRUISE'
    assert 0.14 <= vx <= 0.18

    pressure_rear = dict(rear)
    pressure_rear['rear_pressure'] = 0.7
    state, vx, _ = planner_decision_mock(front, pressure_rear, [])
    assert state == 'ESCAPE'
    assert vx > 0.20

    soft_front = dict(front)
    soft_front['front_min'] = 0.30
    soft_front['front_blocked_soft'] = True
    state, vx, wz = planner_decision_mock(soft_front, rear, [])
    assert state == 'AVOID_FRONT'
    assert 0.03 <= vx <= 0.08
    assert abs(wz) >= 0.35

    safe_vx, safe_wz = safety_mock(0.20, 0.80, soft_front, rear)
    assert safe_vx == 0.05
    assert abs(safe_wz) <= 0.65

    hard_front = dict(front)
    hard_front['front_min'] = 0.20
    safe_vx, safe_wz = safety_mock(0.20, 0.80, hard_front, rear)
    assert safe_vx == 0.0
    assert abs(safe_wz) <= 0.55

    invalid_rear = dict(rear)
    invalid_rear['valid'] = False
    safe_vx, _ = safety_mock(-0.06, 0.0, front, invalid_rear)
    assert safe_vx == 0.0
    print('mock_strategy_contract_check: OK')


if __name__ == '__main__':
    main()
