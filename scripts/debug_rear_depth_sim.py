import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'phantom_detector'))

from phantom_detector.rear_perception_node import compute_rear_risk_from_depth  # noqa: E402


def depth_image(center, left=None, right=None):
    left = center if left is None else left
    right = center if right is None else right
    image = np.full((80, 120), center, dtype=np.float32)
    image[:, :42] = left
    image[:, 42:78] = center
    image[:, 78:] = right
    return image


def check(name, payload):
    if name == 'rear_open':
        return payload['valid'] and payload['reverse_allowed'] and payload['rear_pressure'] < 0.25
    if name == 'rear_near':
        return payload['valid'] and payload['rear_blocked_soft'] and not payload['reverse_allowed']
    if name == 'rear_very_near':
        return payload['valid'] and payload['rear_blocked_hard']
    if name == 'rear_invalid':
        return not payload['valid']
    return False


def main():
    out_dir = ROOT / 'artifacts' / 'rear'
    out_dir.mkdir(parents=True, exist_ok=True)
    invalid = np.full((80, 120), math.nan, dtype=np.float32)
    invalid[:, ::3] = 0.0
    invalid[:, 1::3] = math.inf
    scenarios = {
        'rear_open': depth_image(1.2),
        'rear_near': depth_image(0.45),
        'rear_very_near': depth_image(0.25),
        'rear_invalid': invalid,
    }
    results = []
    for name, depth in scenarios.items():
        payload = compute_rear_risk_from_depth(depth)
        results.append({'scenario': name, 'passed': check(name, payload), 'rear_risk': payload})
    path = out_dir / 'rear_depth_sim_results.json'
    path.write_text(json.dumps({'scenarios': results}, indent=2), encoding='utf-8')
    print(json.dumps({'result_json': str(path.relative_to(ROOT)), 'scenarios': results}, indent=2))
    return 0 if all(item['passed'] for item in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
