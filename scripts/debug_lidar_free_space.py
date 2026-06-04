import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_free_space'))

from phantom_free_space.free_space_node import compute_front_free_space_from_scan  # noqa: E402


def make_scan(default=1.5, sectors=None, invalid=False):
    ranges = [float(default)] * 360
    if invalid:
        ranges = [math.nan if i % 3 == 0 else math.inf if i % 3 == 1 else 0.0 for i in range(360)]
    sectors = sectors or {}
    for low_deg, high_deg, value in sectors.values():
        for index in range(360):
            angle = -180 + index
            if low_deg <= angle <= high_deg:
                ranges[index] = value
    return SimpleNamespace(
        ranges=ranges,
        angle_min=-math.pi,
        angle_increment=math.radians(1.0),
        range_min=0.05,
        range_max=8.0,
    )


def passed(name, payload):
    if name == 'A_front_open':
        return payload['valid'] and not payload['front_blocked_soft'] and not payload['front_blocked_hard'] and abs(payload['best_heading']) <= 0.1
    if name == 'B_front_block_left_open':
        return payload['valid'] and payload['front_blocked_soft'] and payload['best_heading'] > 0.0
    if name == 'C_front_block_right_open':
        return payload['valid'] and payload['front_blocked_soft'] and payload['best_heading'] < 0.0
    if name == 'D_dead_end':
        return payload['valid'] and payload['front_blocked_soft'] and payload['dead_end_score'] >= 0.7
    if name == 'E_invalid_data':
        return not payload['valid']
    return False


def main():
    out_dir = ROOT / 'artifacts' / 'lidar'
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {
        'A_front_open': make_scan(1.0, {'front': (-15, 15, 1.5)}),
        'B_front_block_left_open': make_scan(0.5, {'front': (-15, 15, 0.25), 'left': (15, 110, 1.2), 'right': (-110, -15, 0.5)}),
        'C_front_block_right_open': make_scan(0.5, {'front': (-15, 15, 0.25), 'left': (15, 110, 0.5), 'right': (-110, -15, 1.2)}),
        'D_dead_end': make_scan(0.3),
        'E_invalid_data': make_scan(invalid=True),
    }
    results = []
    for name, scan in scenarios.items():
        payload = compute_front_free_space_from_scan(scan)
        results.append({'scenario': name, 'passed': passed(name, payload), 'front_free_space': payload})
    path = out_dir / 'lidar_sim_results.json'
    path.write_text(json.dumps({'scenarios': results}, indent=2), encoding='utf-8')
    print(json.dumps({'result_json': str(path.relative_to(ROOT)), 'scenarios': results}, indent=2))
    return 0 if all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
