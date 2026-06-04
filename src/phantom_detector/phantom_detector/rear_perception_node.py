import json
import math
from collections import deque

import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:  # Allows local non-ROS algorithm tests to import this module.
    rclpy = None
    Node = object
    qos_profile_sensor_data = 10
    Image = object

    class String:  # pragma: no cover - only used when ROS messages are unavailable.
        def __init__(self):
            self.data = ''


DEPTH_TOPIC_CANDIDATES = [
    '/camera/depth/image_rect_raw',
    '/camera/depth/image_raw',
    '/depth/image_raw',
]


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def _stamp_seconds(message, node):
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    if stamp is not None:
        seconds = float(getattr(stamp, 'sec', 0)) + float(getattr(stamp, 'nanosec', 0)) * 1e-9
        if seconds > 0.0:
            return seconds
    return node.get_clock().now().nanoseconds * 1e-9


def _as_float(value, default=None):
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _rear_params(params=None):
    values = {
        'rear_hard_stop_m': 0.30,
        'rear_soft_stop_m': 0.55,
        'rear_pressure_far_m': 1.20,
        'rear_pressure_near_m': 0.35,
        'max_depth_m': 8.0,
        'threat_confidence': 0.25,
        'z_bump_depth_jump_m': 0.06,
        'z_bump_cone_threshold_scale': 0.75,
        'stamp': 0.0,
    }
    if params:
        values.update(params)
    return values


def _region_stats_from_depth(region, max_depth_m):
    if region.size == 0:
        return {'near': 0.0, 'median': 0.0, 'valid_ratio': 0.0}
    mask = np.isfinite(region) & (region > 0.0) & (region <= max_depth_m)
    valid_ratio = float(np.count_nonzero(mask)) / float(region.size)
    if not np.any(mask):
        return {'near': 0.0, 'median': 0.0, 'valid_ratio': valid_ratio}
    values = region[mask].astype(np.float32)
    return {
        'near': float(np.percentile(values, 20)),
        'median': float(np.median(values)),
        'valid_ratio': valid_ratio,
    }


def _z_bump_defaults():
    return {
        'z_bump_detected': False,
        'z_bump_score': 0.0,
        'z_bump_side': 'none',
        'z_bump_reason': '',
    }


def _normalize_detections_payload(payload):
    if isinstance(payload, dict) and isinstance(payload.get('detections'), list):
        return payload['detections']
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def _is_threat_detection_payload(detection, threat_confidence=0.25):
    class_name = str(detection.get('class_name', '')).strip().lower()
    conf = _as_float(detection.get('conf'), 0.0)
    visible = bool(detection.get('visible', True))
    return visible and class_name in ('traffic_cone', 'yellow_car') and conf >= threat_confidence


def _has_traffic_cone_detection(detections, threat_confidence=0.25):
    for detection in _normalize_detections_payload(detections):
        class_name = str(detection.get('class_name', '')).strip().lower()
        conf = _as_float(detection.get('conf'), 0.0)
        visible = bool(detection.get('visible', True))
        if visible and class_name == 'traffic_cone' and conf >= threat_confidence:
            return True
    return False


def _cell_medians(region, max_depth_m, rows=6, cols=5):
    if region.size == 0:
        return []
    height, width = region.shape
    medians = []
    for row_idx in range(rows):
        y0 = int(round(row_idx * height / rows))
        y1 = int(round((row_idx + 1) * height / rows))
        for col_idx in range(cols):
            x0 = int(round(col_idx * width / cols))
            x1 = int(round((col_idx + 1) * width / cols))
            cell = region[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
            mask = np.isfinite(cell) & (cell > 0.0) & (cell <= max_depth_m)
            if np.count_nonzero(mask) < max(4, int(cell.size * 0.08)):
                medians.append(None)
            else:
                medians.append(float(np.median(cell[mask].astype(np.float32))))
    return medians


def _depth_jump_from_cells(region, max_depth_m):
    medians = _cell_medians(region, max_depth_m)
    if not medians:
        return {'jump_m': 0.0, 'valid_ratio': 0.0}
    rows = 6
    cols = 5
    jumps = []
    valid_cells = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            value = medians[row_idx * cols + col_idx]
            if value is None:
                continue
            valid_cells += 1
            if row_idx + 1 < rows:
                below = medians[(row_idx + 1) * cols + col_idx]
                if below is not None:
                    jumps.append(abs(below - value))
            if col_idx + 1 < cols:
                right = medians[row_idx * cols + col_idx + 1]
                if right is not None:
                    jumps.append(abs(right - value))
    valid_ratio = valid_cells / float(rows * cols)
    if not jumps:
        return {'jump_m': 0.0, 'valid_ratio': valid_ratio}
    return {'jump_m': float(np.percentile(np.asarray(jumps, dtype=np.float32), 90)), 'valid_ratio': valid_ratio}


def _compute_z_bump_from_depth(depth, params, detections=None):
    """Approximate vertical bump risk from lower rear depth-image discontinuities.

    Camera optical z is depth, not body-frame height. When no point cloud is
    available, this uses local lower-ROI depth discontinuities as an approximate
    ground/low-obstacle height-change cue and leaves the filtered trigger to the
    ROS node.
    """
    height, width = depth.shape
    y0 = int(round(0.55 * height))
    y1 = max(y0 + 1, int(round(0.95 * height)))
    y1 = min(y1, height)
    regions = {
        'left': (
            int(round(0.20 * width)),
            int(round(0.45 * width)),
        ),
        'center': (
            int(round(0.35 * width)),
            int(round(0.65 * width)),
        ),
        'right': (
            int(round(0.55 * width)),
            int(round(0.80 * width)),
        ),
    }
    threshold = float(params['z_bump_depth_jump_m'])
    if _has_traffic_cone_detection(detections or [], params['threat_confidence']):
        threshold *= float(params['z_bump_cone_threshold_scale'])

    best = {'side': 'none', 'jump_m': 0.0, 'valid_ratio': 0.0, 'score': 0.0}
    for side, (x0, x1) in regions.items():
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        stats = _depth_jump_from_cells(depth[y0:y1, x0:x1], params['max_depth_m'])
        jump_m = stats['jump_m']
        valid_ratio = stats['valid_ratio']
        if valid_ratio < 0.25:
            score = 0.0
        else:
            score = _clamp((jump_m - threshold * 0.55) / max(threshold * 1.6, 0.01), 0.0, 1.0)
        if score > best['score']:
            best = {'side': side, 'jump_m': jump_m, 'valid_ratio': valid_ratio, 'score': score}

    reason = ''
    if best['score'] > 0.0:
        reason = 'depth_roi_jump %.3fm side=%s valid=%.2f' % (
            best['jump_m'],
            best['side'],
            best['valid_ratio'],
        )
    return {
        'z_bump_detected': False,
        'z_bump_score': round(float(best['score']), 3),
        'z_bump_side': best['side'] if best['score'] > 0.0 else 'none',
        'z_bump_reason': reason,
        'z_bump_depth_jump_m': round(float(best['jump_m']), 4),
    }


def _best_threat_from_detections(detections, params):
    threats = [
        detection for detection in _normalize_detections_payload(detections)
        if _is_threat_detection_payload(detection, params['threat_confidence'])
    ]
    if not threats:
        return {'visible': False, 'class_name': '', 'conf': 0.0, 'depth': None, 'pressure': 0.0}
    best = max(threats, key=lambda item: float(item.get('conf', 0.0)))
    depth = _as_float(best.get('depth'), None)
    if depth is not None and depth < 0.80:
        pressure = 1.0
    elif depth is not None and depth < 1.20:
        pressure = 0.7
    elif depth is not None:
        pressure = 0.25
    else:
        width = _as_float(best.get('w'), _as_float(best.get('bbox_w'), 0.0))
        height = _as_float(best.get('h'), _as_float(best.get('bbox_h'), 0.0))
        pressure = max(0.4, _clamp(((width or 0.0) * (height or 0.0) - 0.03) / 0.20, 0.0, 1.0))
    return {
        'visible': True,
        'class_name': str(best.get('class_name', '')),
        'conf': float(best.get('conf', 0.0)),
        'depth': None if depth is None else round(depth, 3),
        'pressure': float(pressure),
    }


def _invalid_rear_payload(stamp, params, detections=None):
    threat = _best_threat_from_detections(detections or [], params)
    payload = {
        'stamp': round(float(stamp), 6),
        'valid': False,
        'rear_min': 0.0,
        'rear_center_min': 0.0,
        'rear_left_min': 0.0,
        'rear_right_min': 0.0,
        'rear_clearance_score': 0.0,
        'reverse_allowed': False,
        'rear_pressure': round(threat['pressure'], 3),
        'rear_blocked_soft': False,
        'rear_blocked_hard': False,
        'threat_visible': bool(threat['visible']),
        'threat_class': threat['class_name'],
        'threat_conf': round(threat['conf'], 3),
        'threat_depth': threat['depth'],
    }
    payload.update(_z_bump_defaults())
    return payload


def compute_rear_risk_from_depth(depth_array, params=None, detections=None):
    """Compute /nav/rear_risk JSON payload from a depth image in meters."""
    params = _rear_params(params)
    stamp = params['stamp']
    try:
        depth = np.asarray(depth_array, dtype=np.float32)
    except (TypeError, ValueError):
        return _invalid_rear_payload(stamp, params, detections)
    if depth.ndim != 2 or depth.size == 0:
        return _invalid_rear_payload(stamp, params, detections)

    height, width = depth.shape
    y0 = int(round(0.35 * height))
    y1 = max(y0 + 1, int(round(0.90 * height)))
    y1 = min(y1, height)
    roi = depth[y0:y1, :]
    left = roi[:, :max(1, int(round(0.35 * width)))]
    center = roi[:, int(round(0.35 * width)):max(int(round(0.35 * width)) + 1, int(round(0.65 * width)))]
    right = roi[:, int(round(0.65 * width)):]

    left_stats = _region_stats_from_depth(left, params['max_depth_m'])
    center_stats = _region_stats_from_depth(center, params['max_depth_m'])
    right_stats = _region_stats_from_depth(right, params['max_depth_m'])
    all_values = [value for value in (left_stats['near'], center_stats['near'], right_stats['near']) if value > 0.0]
    if center_stats['near'] <= 0.0 or not all_values:
        return _invalid_rear_payload(stamp, params, detections)

    rear_min = min(all_values)
    rear_center_near = center_stats['near']
    rear_left_near = left_stats['near']
    rear_right_near = right_stats['near']
    rear_blocked_hard = rear_center_near < params['rear_hard_stop_m']
    rear_blocked_soft = rear_center_near < params['rear_soft_stop_m']
    left_hard = 0.0 < rear_left_near < params['rear_hard_stop_m']
    right_hard = 0.0 < rear_right_near < params['rear_hard_stop_m']
    reverse_allowed = rear_center_near > params['rear_soft_stop_m'] and not left_hard and not right_hard
    depth_pressure = _clamp(
        (params['rear_pressure_far_m'] - rear_center_near)
        / max(params['rear_pressure_far_m'] - params['rear_pressure_near_m'], 0.01),
        0.0,
        1.0,
    )
    threat = _best_threat_from_detections(detections or [], params)
    rear_pressure = max(depth_pressure, threat['pressure'])
    clearance_score = _clamp(
        (rear_center_near - params['rear_hard_stop_m'])
        / max(params['rear_pressure_far_m'] - params['rear_hard_stop_m'], 0.01),
        0.0,
        1.0,
    )
    payload = {
        'stamp': round(float(stamp), 6),
        'valid': True,
        'rear_min': round(rear_min, 3),
        'rear_center_min': round(rear_center_near, 3),
        'rear_left_min': round(rear_left_near, 3),
        'rear_right_min': round(rear_right_near, 3),
        'rear_clearance_score': round(clearance_score, 3),
        'reverse_allowed': bool(reverse_allowed),
        'rear_pressure': round(rear_pressure, 3),
        'rear_blocked_soft': bool(rear_blocked_soft),
        'rear_blocked_hard': bool(rear_blocked_hard),
        'threat_visible': bool(threat['visible']),
        'threat_class': threat['class_name'],
        'threat_conf': round(threat['conf'], 3),
        'threat_depth': threat['depth'],
    }
    payload.update(_compute_z_bump_from_depth(depth, params, detections))
    return payload


class RearPerceptionNode(Node):
    def __init__(self):
        super().__init__('rear_perception_node')

        self.declare_parameter('depth_topic', DEPTH_TOPIC_CANDIDATES[0])
        self.declare_parameter('rear_risk_topic', '/nav/rear_risk')
        self.declare_parameter('detections_topic', '/det/detections')
        self.declare_parameter('use_detections', True)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('depth_timeout_sec', 0.6)
        self.declare_parameter('rear_hard_stop_m', 0.30)
        self.declare_parameter('rear_soft_stop_m', 0.55)
        self.declare_parameter('rear_pressure_far_m', 1.20)
        self.declare_parameter('rear_pressure_near_m', 0.35)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('threat_confidence', 0.25)
        self.declare_parameter('z_bump_depth_jump_m', 0.06)
        self.declare_parameter('z_bump_trigger_score', 0.65)
        self.declare_parameter('z_bump_consecutive_frames', 3)
        self.declare_parameter('z_bump_cone_threshold_scale', 0.75)

        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.rear_risk_topic = str(self.get_parameter('rear_risk_topic').value)
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.use_detections = bool(self.get_parameter('use_detections').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.depth_timeout_sec = float(self.get_parameter('depth_timeout_sec').value)
        self.rear_hard_stop_m = float(self.get_parameter('rear_hard_stop_m').value)
        self.rear_soft_stop_m = float(self.get_parameter('rear_soft_stop_m').value)
        self.rear_pressure_far_m = float(self.get_parameter('rear_pressure_far_m').value)
        self.rear_pressure_near_m = float(self.get_parameter('rear_pressure_near_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.threat_confidence = float(self.get_parameter('threat_confidence').value)
        self.z_bump_depth_jump_m = float(self.get_parameter('z_bump_depth_jump_m').value)
        self.z_bump_trigger_score = float(self.get_parameter('z_bump_trigger_score').value)
        self.z_bump_consecutive_frames = int(self.get_parameter('z_bump_consecutive_frames').value)
        self.z_bump_cone_threshold_scale = float(self.get_parameter('z_bump_cone_threshold_scale').value)

        self.latest_depth_payload = None
        self.latest_depth_time = -999.0
        self.latest_detections = []
        self.last_detections_time = -999.0
        self.reported_topics = False
        self.z_bump_history = deque(maxlen=max(1, self.z_bump_consecutive_frames))

        self.risk_pub = self.create_publisher(String, self.rear_risk_topic, 10)
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.det_sub = None
        if self.use_detections:
            self.det_sub = self.create_subscription(String, self.detections_topic, self.detections_callback, 10)

        period = 1.0 / max(self.publish_rate_hz, 0.5)
        self.timer = self.create_timer(period, self.timer_publish)
        self.topic_timer = self.create_timer(2.0, self._log_depth_topics_once)

        self.get_logger().info(
            'rear_perception_node subscribed to depth %s and detections %s, publishing %s'
            % (self.depth_topic, self.detections_topic if self.use_detections else 'disabled', self.rear_risk_topic)
        )

    def depth_callback(self, image_msg):
        stamp = _stamp_seconds(image_msg, self)
        try:
            depth = self._depth_image_to_meters(image_msg)
        except (TypeError, ValueError) as exc:
            self.get_logger().warn('invalid rear depth image on %s: %s' % (self.depth_topic, exc))
            payload = self._invalid_payload(stamp)
        else:
            payload = self._build_depth_payload(depth, stamp)

        self.latest_depth_payload = payload
        self.latest_depth_time = self.get_clock().now().nanoseconds * 1e-9
        self._publish_payload(payload)

    def detections_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('invalid detection JSON for rear fusion: %s' % exc)
            return
        self.latest_detections = self._normalize_detections(payload)
        self.last_detections_time = self.get_clock().now().nanoseconds * 1e-9

    def timer_publish(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.latest_depth_payload is None or now - self.latest_depth_time > self.depth_timeout_sec:
            self._publish_payload(self._invalid_payload(now))
            return
        self._publish_payload(self.latest_depth_payload)

    def _build_depth_payload(self, depth, stamp):
        payload = compute_rear_risk_from_depth(depth, {
            'stamp': stamp,
            'rear_hard_stop_m': self.rear_hard_stop_m,
            'rear_soft_stop_m': self.rear_soft_stop_m,
            'rear_pressure_far_m': self.rear_pressure_far_m,
            'rear_pressure_near_m': self.rear_pressure_near_m,
            'max_depth_m': self.max_depth_m,
            'threat_confidence': self.threat_confidence,
            'z_bump_depth_jump_m': self.z_bump_depth_jump_m,
            'z_bump_cone_threshold_scale': self.z_bump_cone_threshold_scale,
        }, self.latest_detections)
        return self._apply_z_bump_filter(payload)

    def _region_stats(self, region):
        if region.size == 0:
            return {'near': 0.0, 'median': 0.0, 'valid_ratio': 0.0}
        mask = np.isfinite(region) & (region > 0.0) & (region <= self.max_depth_m)
        valid_ratio = float(np.count_nonzero(mask)) / float(region.size)
        if not np.any(mask):
            return {'near': 0.0, 'median': 0.0, 'valid_ratio': valid_ratio}
        values = region[mask].astype(np.float32)
        return {
            'near': float(np.percentile(values, 20)),
            'median': float(np.median(values)),
            'valid_ratio': valid_ratio,
        }

    def _invalid_payload(self, stamp):
        payload = _invalid_rear_payload(stamp, _rear_params({
            'rear_hard_stop_m': self.rear_hard_stop_m,
            'rear_soft_stop_m': self.rear_soft_stop_m,
            'rear_pressure_far_m': self.rear_pressure_far_m,
            'rear_pressure_near_m': self.rear_pressure_near_m,
            'max_depth_m': self.max_depth_m,
            'threat_confidence': self.threat_confidence,
        }), self.latest_detections)
        payload.update(_z_bump_defaults())
        self.z_bump_history.append(False)
        return payload

    def _apply_z_bump_filter(self, payload):
        score = _as_float(payload.get('z_bump_score'), 0.0)
        threshold = self.z_bump_trigger_score
        if self._recent_traffic_cone():
            threshold *= 0.88
        hit = bool(score is not None and score >= threshold)
        self.z_bump_history.append(hit)
        detected = (
            len(self.z_bump_history) >= max(1, self.z_bump_consecutive_frames)
            and all(self.z_bump_history)
        )
        payload['z_bump_detected'] = bool(detected)
        if detected and not payload.get('z_bump_reason'):
            payload['z_bump_reason'] = 'filtered_depth_roi_jump'
        if not detected and not hit:
            payload['z_bump_side'] = 'none' if score < 0.10 else payload.get('z_bump_side', 'none')
        payload['z_bump_filter_hits'] = int(sum(1 for item in self.z_bump_history if item))
        payload['z_bump_filter_window'] = len(self.z_bump_history)
        return payload

    def _recent_traffic_cone(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_detections_time > 0.8:
            return False
        return _has_traffic_cone_detection(self.latest_detections, self.threat_confidence)

    def _best_threat(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_detections_time > 0.8:
            return {'visible': False, 'class_name': '', 'conf': 0.0, 'depth': None, 'pressure': 0.0}
        threats = [
            detection for detection in self.latest_detections
            if self._is_threat_detection(detection)
        ]
        if not threats:
            return {'visible': False, 'class_name': '', 'conf': 0.0, 'depth': None, 'pressure': 0.0}
        best = max(threats, key=lambda item: float(item.get('conf', 0.0)))
        depth = _as_float(best.get('depth'), None)
        if depth is not None and depth < 0.80:
            pressure = 1.0
        elif depth is not None and depth < 1.20:
            pressure = 0.7
        elif depth is not None:
            pressure = 0.25
        else:
            area = _as_float(best.get('w'), _as_float(best.get('bbox_w'), 0.0))
            height = _as_float(best.get('h'), _as_float(best.get('bbox_h'), 0.0))
            pressure = max(0.4, _clamp(((area or 0.0) * (height or 0.0) - 0.03) / 0.20, 0.0, 1.0))
        return {
            'visible': True,
            'class_name': str(best.get('class_name', '')),
            'conf': float(best.get('conf', 0.0)),
            'depth': None if depth is None else round(depth, 3),
            'pressure': float(pressure),
        }

    def _is_threat_detection(self, detection):
        class_name = str(detection.get('class_name', '')).strip().lower()
        conf = _as_float(detection.get('conf'), 0.0)
        visible = bool(detection.get('visible', True))
        return visible and class_name in ('traffic_cone', 'yellow_car') and conf >= self.threat_confidence

    @staticmethod
    def _normalize_detections(payload):
        if isinstance(payload, dict) and isinstance(payload.get('detections'), list):
            return payload['detections']
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        return []

    @staticmethod
    def _depth_image_to_meters(image_msg):
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = str(image_msg.encoding).upper()
        if height <= 0 or width <= 0:
            raise ValueError('empty depth dimensions')
        if encoding in ('16UC1', 'MONO16'):
            row_values = max(int(image_msg.step) // 2, width)
            array = np.frombuffer(image_msg.data, dtype=np.uint16).reshape((height, row_values))[:, :width]
            return array.astype(np.float32) * 0.001
        if encoding == '32FC1':
            row_values = max(int(image_msg.step) // 4, width)
            array = np.frombuffer(image_msg.data, dtype=np.float32).reshape((height, row_values))[:, :width]
            return array.astype(np.float32)
        raise ValueError('unsupported encoding %s' % image_msg.encoding)

    def _publish_payload(self, payload):
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.risk_pub.publish(message)

    def _log_depth_topics_once(self):
        if self.reported_topics:
            return
        self.reported_topics = True
        visible = []
        for name, types in self.get_topic_names_and_types():
            haystack = ' '.join([name] + list(types)).lower()
            if 'depth' in haystack or 'camera' in haystack:
                visible.append('%s [%s]' % (name, ','.join(types)))
        if visible:
            self.get_logger().info('visible camera/depth topics: %s' % '; '.join(sorted(visible)))
        else:
            self.get_logger().warn(
                'no camera/depth topics visible locally; rear_risk will publish valid=false fallback'
            )


def main(args=None):
    if rclpy is None:
        raise RuntimeError('rclpy is required to run rear_perception_node as a ROS2 node')
    rclpy.init(args=args)
    node = RearPerceptionNode()
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
