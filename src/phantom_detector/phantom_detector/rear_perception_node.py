import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


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

        self.latest_depth_payload = None
        self.latest_depth_time = -999.0
        self.latest_detections = []
        self.last_detections_time = -999.0
        self.reported_topics = False

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
        height, width = depth.shape
        if height <= 0 or width <= 0:
            return self._invalid_payload(stamp)

        y0 = int(round(0.35 * height))
        y1 = max(y0 + 1, int(round(0.90 * height)))
        y1 = min(y1, height)
        roi = depth[y0:y1, :]
        left = roi[:, :max(1, int(round(0.35 * width)))]
        center = roi[:, int(round(0.35 * width)):max(int(round(0.35 * width)) + 1, int(round(0.65 * width)))]
        right = roi[:, int(round(0.65 * width)):]

        left_stats = self._region_stats(left)
        center_stats = self._region_stats(center)
        right_stats = self._region_stats(right)
        all_values = [value for value in (left_stats['near'], center_stats['near'], right_stats['near']) if value > 0.0]

        if center_stats['near'] <= 0.0 or not all_values:
            return self._invalid_payload(stamp)

        rear_min = min(all_values)
        rear_center_near = center_stats['near']
        rear_left_near = left_stats['near']
        rear_right_near = right_stats['near']
        rear_blocked_hard = rear_center_near < self.rear_hard_stop_m
        rear_blocked_soft = rear_center_near < self.rear_soft_stop_m
        left_hard = 0.0 < rear_left_near < self.rear_hard_stop_m
        right_hard = 0.0 < rear_right_near < self.rear_hard_stop_m
        reverse_allowed = (
            rear_center_near > self.rear_soft_stop_m
            and not left_hard
            and not right_hard
        )
        depth_pressure = _clamp(
            (self.rear_pressure_far_m - rear_center_near)
            / max(self.rear_pressure_far_m - self.rear_pressure_near_m, 0.01),
            0.0,
            1.0,
        )
        threat = self._best_threat()
        rear_pressure = max(depth_pressure, threat['pressure'])
        clearance_score = _clamp(
            (rear_center_near - self.rear_hard_stop_m)
            / max(self.rear_pressure_far_m - self.rear_hard_stop_m, 0.01),
            0.0,
            1.0,
        )

        return {
            'stamp': round(stamp, 6),
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
        threat = self._best_threat()
        return {
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
