import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_free_space'))

from phantom_free_space.free_space_node import compute_front_free_space_from_scan  # noqa: E402


ROBOT_WIDTH = 0.25
ROBOT_LENGTH = 0.30
SAFETY_MARGIN = 0.07
CONE_BASE_RADIUS = 0.15
OBSTACLE_EXTRA_MARGIN = 0.06
MIN_SIDE_CLEARANCE = ROBOT_WIDTH * 0.5 + SAFETY_MARGIN
INFLATION_RADIUS = math.hypot(ROBOT_WIDTH * 0.5, ROBOT_LENGTH * 0.5) + SAFETY_MARGIN
MIN_EXIT_CORRIDOR_WIDTH = ROBOT_WIDTH + 2.0 * SAFETY_MARGIN
MIN_OBSTACLE_CLEARANCE = MIN_SIDE_CLEARANCE + CONE_BASE_RADIUS + OBSTACLE_EXTRA_MARGIN


def clamp(value, low, high):
    return max(low, min(float(value), high))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def ray_segment_distance(origin, angle, segment):
    direction = (math.cos(angle), math.sin(angle))
    q = segment[0]
    s = sub(segment[1], segment[0])
    denom = cross(direction, s)
    if abs(denom) < 1e-9:
        return None
    qp = sub(q, origin)
    t = cross(qp, s) / denom
    u = cross(qp, direction) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def ray_circle_distance(origin, angle, circle):
    cx, cy, radius = circle
    direction = (math.cos(angle), math.sin(angle))
    oc = (origin[0] - cx, origin[1] - cy)
    b = 2.0 * dot(oc, direction)
    c = dot(oc, oc) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    values = [(-b - root) * 0.5, (-b + root) * 0.5]
    values = [value for value in values if value >= 0.0]
    return min(values) if values else None


def make_wall_exit_env(exit_width=0.43, cones=None):
    half = exit_width * 0.5
    walls = [
        ((-0.45, -0.75), (1.00, -0.75)),
        ((-0.45, 0.75), (1.00, 0.75)),
        ((-0.45, -0.75), (-0.45, 0.75)),
        ((1.00, -0.75), (1.00, -half)),
        ((1.00, half), (1.00, 0.75)),
        ((1.00, -half), (2.55, -half)),
        ((1.00, half), (2.55, half)),
    ]
    return {'walls': walls, 'cones': cones or [], 'exit_width': exit_width}


def make_scan(env, pose, samples=720, max_range=8.0):
    x, y, theta = pose
    ranges = []
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / samples
    inflated_cones = [
        (cx, cy, CONE_BASE_RADIUS + OBSTACLE_EXTRA_MARGIN)
        for cx, cy in env.get('cones', [])
    ]
    for index in range(samples):
        local_angle = angle_min + index * angle_increment
        angle = theta + local_angle
        best = max_range
        for segment in env['walls']:
            hit = ray_segment_distance((x, y), angle, segment)
            if hit is not None and hit < best:
                best = hit
        for circle in inflated_cones:
            hit = ray_circle_distance((x, y), angle, circle)
            if hit is not None and hit < best:
                best = hit
        ranges.append(best)
    return SimpleNamespace(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.05,
        range_max=max_range,
    )


def footprint_vertices(pose):
    x, y, theta = pose
    c = math.cos(theta)
    s = math.sin(theta)
    local = [
        (ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5),
        (ROBOT_LENGTH * 0.5, -ROBOT_WIDTH * 0.5),
        (-ROBOT_LENGTH * 0.5, -ROBOT_WIDTH * 0.5),
        (-ROBOT_LENGTH * 0.5, ROBOT_WIDTH * 0.5),
    ]
    return [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]


def point_in_poly(point, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > point[1]) != (yj > point[1])) and (
            point[0] < (xj - xi) * (point[1] - yi) / max(yj - yi, 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def segments_intersect(a, b, c, d):
    def orient(p, q, r):
        return cross(sub(q, p), sub(r, p))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 * o2 < 0.0 and o3 * o4 < 0.0


def point_segment_distance(point, a, b):
    ab = sub(b, a)
    denom = max(dot(ab, ab), 1e-12)
    t = clamp(dot(sub(point, a), ab) / denom, 0.0, 1.0)
    closest = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


def segment_distance(a, b, c, d):
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def poly_segment_distance(poly, segment):
    edges = list(zip(poly, poly[1:] + poly[:1]))
    if point_in_poly(segment[0], poly) or point_in_poly(segment[1], poly):
        return 0.0
    return min(segment_distance(a, b, segment[0], segment[1]) for a, b in edges)


def point_rect_distance(point, pose):
    x, y, theta = pose
    c = math.cos(-theta)
    s = math.sin(-theta)
    dx = point[0] - x
    dy = point[1] - y
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    ox = max(abs(lx) - ROBOT_LENGTH * 0.5, 0.0)
    oy = max(abs(ly) - ROBOT_WIDTH * 0.5, 0.0)
    return math.hypot(ox, oy)


def check_path(env, path, require_centered=False):
    min_wall_clearance = 999.0
    min_cone_clearance = 999.0
    max_center_error = 0.0
    wall_collision = False
    cone_collision = False
    clearance_ok = True
    for pose in path:
        poly = footprint_vertices(pose)
        for wall in env['walls']:
            distance = poly_segment_distance(poly, wall)
            min_wall_clearance = min(min_wall_clearance, distance)
            if distance <= 1e-5:
                wall_collision = True
            if 1.0 <= pose[0] <= 2.45 and distance < SAFETY_MARGIN - 1e-6:
                clearance_ok = False
        for cone in env.get('cones', []):
            distance = point_rect_distance(cone, pose)
            clearance = distance - (CONE_BASE_RADIUS + OBSTACLE_EXTRA_MARGIN)
            min_cone_clearance = min(min_cone_clearance, clearance)
            if clearance <= 0.0:
                cone_collision = True
        if require_centered and pose[0] >= 1.05:
            max_center_error = max(max_center_error, abs(pose[1]))
    centered_ok = (not require_centered) or max_center_error <= 0.065
    return {
        'safe': not wall_collision and not cone_collision and clearance_ok and centered_ok,
        'wall_collision': wall_collision,
        'cone_base_collision': cone_collision,
        'clearance_ok': clearance_ok,
        'centered_ok': centered_ok,
        'min_wall_clearance_m': round(min_wall_clearance, 4),
        'min_cone_base_clearance_m': round(min_cone_clearance, 4) if min_cone_clearance < 900 else None,
        'max_exit_center_error_m': round(max_center_error, 4),
    }


def simulate_guided_exit(env, start_y=0.0, start_x=0.0):
    pose = [start_x, start_y, 0.0]
    path = [tuple(pose)]
    payloads = []
    for _ in range(95):
        scan = make_scan(env, pose)
        payload = compute_front_free_space_from_scan(scan)
        payloads.append(payload)
        heading = payload.get('best_gap_heading', payload.get('best_heading', 0.0))
        if not payload.get('gap_escape_allowed', False):
            heading = payload.get('best_heading', 0.0)
        pose[2] += clamp(1.35 * heading - 0.55 * pose[2], -0.18, 0.18)
        pose[0] += 0.035 * math.cos(pose[2])
        pose[1] += 0.035 * math.sin(pose[2])
        path.append(tuple(pose))
        if pose[0] > 2.35:
            break
    return path, payloads


def straight_path(y, x0=0.0, x1=2.35, steps=80):
    return [(x0 + (x1 - x0) * i / (steps - 1), y, 0.0) for i in range(steps)]


def curved_path_near_cone():
    path = []
    for i in range(85):
        t = i / 84.0
        x = 0.05 + 2.1 * t
        y = -0.22 + 0.34 * math.sin(t * math.pi * 0.95)
        if i == 0:
            theta = 0.0
        else:
            px, py, _ = path[-1]
            theta = math.atan2(y - py, x - px)
        path.append((x, y, theta))
    return path


def safe_cone_path():
    path = []
    for i in range(90):
        t = i / 89.0
        x = 0.05 + 2.15 * t
        y = -0.12 + 0.03 * math.sin(t * math.pi)
        if i == 0:
            theta = 0.0
        else:
            px, py, _ = path[-1]
            theta = math.atan2(y - py, x - px)
        path.append((x, y, theta))
    return path


def run_cases():
    narrow_env = make_wall_exit_env(0.43)
    offset_env = make_wall_exit_env(0.62)
    cone_env = make_wall_exit_env(0.70, cones=[(1.30, 0.30), (1.78, 0.30)])

    narrow_path, narrow_payloads = simulate_guided_exit(narrow_env, 0.0)
    offset_path, offset_payloads = simulate_guided_exit(offset_env, 0.09, start_x=0.85)
    wall_hug_path = straight_path(0.43 * 0.5 - ROBOT_WIDTH * 0.5 - 0.005)
    unsafe_cone_path = curved_path_near_cone()
    final_safe_path = safe_cone_path()

    narrow_check = check_path(narrow_env, narrow_path, require_centered=True)
    offset_check = check_path(offset_env, offset_path, require_centered=True)
    wall_hug_check = check_path(narrow_env, wall_hug_path, require_centered=False)
    cone_unsafe_check = check_path(cone_env, unsafe_cone_path, require_centered=False)
    final_safe_check = check_path(cone_env, final_safe_path, require_centered=False)

    offset_first = offset_payloads[0]
    cone_scan_payload = compute_front_free_space_from_scan(make_scan(cone_env, (0.75, 0.0, 0.0)))

    cases = [
        {
            'case': 'narrow_exit_centered_pass',
            'passed': bool(narrow_check['safe'] and narrow_payloads[0]['gap_escape_allowed']),
            'details': narrow_check,
            'planner_evidence': {
                'initial_gap_allowed': narrow_payloads[0]['gap_escape_allowed'],
                'initial_gap_width_m': narrow_payloads[0]['best_gap_width_m'],
                'initial_gap_heading': narrow_payloads[0]['best_gap_heading'],
            },
        },
        {
            'case': 'offset_start_auto_centerline_correction',
            'passed': bool(offset_check['safe'] and offset_first['best_gap_heading'] < -0.02),
            'details': offset_check,
            'planner_evidence': {
                'initial_gap_heading': offset_first['best_gap_heading'],
                'initial_centerline_error_m': round(0.09, 3),
            },
        },
        {
            'case': 'old_wall_hugging_exit_route_is_unsafe',
            'passed': bool(not wall_hug_check['safe']),
            'details': wall_hug_check,
        },
        {
            'case': 'old_cone_base_cutting_route_is_unsafe',
            'passed': bool(not cone_unsafe_check['safe'] and not cone_scan_payload['front_path_safe']),
            'details': cone_unsafe_check,
            'planner_evidence': {
                'front_path_safe_near_cone': cone_scan_payload['front_path_safe'],
                'front_path_unsafe_reason': cone_scan_payload.get('front_path_unsafe_reason', ''),
                'min_obstacle_clearance_m': cone_scan_payload['min_obstacle_clearance_m'],
            },
        },
        {
            'case': 'final_planned_route_footprint_clear',
            'passed': bool(final_safe_check['safe']),
            'details': final_safe_check,
        },
    ]
    paths = {
        'narrow_centered': narrow_path,
        'offset_corrected': offset_path,
        'old_wall_hug': wall_hug_path,
        'old_cone_cut': unsafe_cone_path,
        'final_safe_cones': final_safe_path,
    }
    return cases, paths, {'narrow': narrow_env, 'offset': offset_env, 'cone': cone_env}


def plot_results(paths, envs, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    fig, ax = plt.subplots(figsize=(11, 6))
    env = envs['cone']
    for wall in envs['narrow']['walls']:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='black', linewidth=2)
    for wall in env['walls']:
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color='0.65', linewidth=1, linestyle='--')
    for cone in env.get('cones', []):
        ax.add_patch(Circle(cone, CONE_BASE_RADIUS, color='#f4a261', alpha=0.45, label='cone base'))
        ax.add_patch(Circle(cone, CONE_BASE_RADIUS + OBSTACLE_EXTRA_MARGIN, fill=False, color='#e76f51', linewidth=1.8))
    colors = {
        'narrow_centered': '#2a9d8f',
        'offset_corrected': '#264653',
        'old_wall_hug': '#e63946',
        'old_cone_cut': '#d62828',
        'final_safe_cones': '#457b9d',
    }
    labels = {
        'narrow_centered': 'centered narrow exit',
        'offset_corrected': 'offset corrected',
        'old_wall_hug': 'unsafe wall-hug route',
        'old_cone_cut': 'unsafe cone-base route',
        'final_safe_cones': 'final safe route',
    }
    for name, path in paths.items():
        xs = [pose[0] for pose in path]
        ys = [pose[1] for pose in path]
        ax.plot(xs, ys, color=colors[name], linewidth=2.0, label=labels[name])
        for pose in path[::max(1, len(path) // 6)]:
            ax.add_patch(Polygon(footprint_vertices(pose), closed=True, fill=False, edgecolor=colors[name], linewidth=0.9))
    half = envs['narrow']['exit_width'] * 0.5
    ax.fill_between([1.0, 2.55], half - SAFETY_MARGIN, half, color='#8ecae6', alpha=0.22)
    ax.fill_between([1.0, 2.55], -half, -half + SAFETY_MARGIN, color='#8ecae6', alpha=0.22)
    ax.axhline(0.0, color='0.35', linewidth=0.8, linestyle=':')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.55, 2.65)
    ax.set_ylim(-0.88, 0.88)
    ax.set_xlabel('x forward (m)')
    ax.set_ylabel('y lateral (m)')
    ax.set_title('Footprint-safe exit and cone-base simulation')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    out_dir = ROOT / 'artifacts' / 'footprint_safety'
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, paths, envs = run_cases()
    image_path = out_dir / 'trajectory.png'
    plot_results(paths, envs, image_path)
    payload = {
        'parameters': {
            'robot_width_m': ROBOT_WIDTH,
            'robot_length_m': ROBOT_LENGTH,
            'safety_margin_m': SAFETY_MARGIN,
            'min_side_clearance_m': round(MIN_SIDE_CLEARANCE, 4),
            'inflation_radius_m': round(INFLATION_RADIUS, 4),
            'min_exit_corridor_width_m': round(MIN_EXIT_CORRIDOR_WIDTH, 4),
            'cone_base_radius_m': CONE_BASE_RADIUS,
            'obstacle_extra_margin_m': OBSTACLE_EXTRA_MARGIN,
            'min_obstacle_clearance_m': round(MIN_OBSTACLE_CLEARANCE, 4),
        },
        'cases': cases,
        'trajectory_png': str(image_path.relative_to(ROOT)),
    }
    result_path = out_dir / 'footprint_safety_results.json'
    result_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'result_json': str(result_path.relative_to(ROOT)), **payload}, indent=2))
    return 0 if all(case['passed'] for case in cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
