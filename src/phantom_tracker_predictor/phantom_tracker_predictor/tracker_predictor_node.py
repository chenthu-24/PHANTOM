import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def _empty_state():
    return {
        'visible': False,
        'bearing': 0.0,
        'range_est': 0.0,
        'velocity_est': 0.0,
        'closing_rate': 0.0,
        'predicted_sector_0_5s': 0.0,
    }


class TrackerPredictorNode(Node):
    def __init__(self):
        super().__init__('tracker_predictor_node')

        self.declare_parameter('yellow_topic', '/det/yellow_boxes')
        self.declare_parameter('state_topic', '/track/yellow_state')
        self.declare_parameter('horizontal_fov_rad', 1.0472)
        self.declare_parameter('range_scale', 0.45)
        self.declare_parameter('min_range_est', 0.25)
        self.declare_parameter('max_range_est', 6.0)

        self.yellow_topic = self.get_parameter('yellow_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.horizontal_fov_rad = float(self.get_parameter('horizontal_fov_rad').value)
        self.range_scale = float(self.get_parameter('range_scale').value)
        self.min_range_est = float(self.get_parameter('min_range_est').value)
        self.max_range_est = float(self.get_parameter('max_range_est').value)

        self.previous_time = None
        self.previous_center_x = None
        self.previous_range = None
        self.previous_bearing = None

        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.yellow_sub = self.create_subscription(
            String,
            self.yellow_topic,
            self.yellow_callback,
            10,
        )

        self.get_logger().info(
            'tracker_predictor_node subscribed to %s, publishing %s'
            % (self.yellow_topic, self.state_topic)
        )

    def yellow_callback(self, message):
        try:
            detection = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('invalid yellow detection JSON: %s' % exc)
            return

        state = self._track_detection(detection)
        output = String()
        output.data = json.dumps(state, sort_keys=True)
        self.state_pub.publish(output)

    def _track_detection(self, detection):
        if not bool(detection.get('visible', False)):
            self._reset_track()
            return _empty_state()

        try:
            center_x = float(detection['bbox_center_x'])
            bbox_w = float(detection['bbox_w'])
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn('invalid detection fields: %s' % exc)
            self._reset_track()
            return _empty_state()

        now = self.get_clock().now().nanoseconds * 1e-9
        bearing = (center_x - 0.5) * self.horizontal_fov_rad
        range_est = _clamp(
            self.range_scale / max(bbox_w, 0.03),
            self.min_range_est,
            self.max_range_est,
        )

        velocity_est = 0.0
        closing_rate = 0.0
        if self.previous_time is not None:
            dt = max(now - self.previous_time, 1e-3)
            velocity_est = (center_x - self.previous_center_x) * self.horizontal_fov_rad / dt
            closing_rate = (self.previous_range - range_est) / dt

        predicted_sector = bearing + 0.5 * velocity_est

        self.previous_time = now
        self.previous_center_x = center_x
        self.previous_range = range_est
        self.previous_bearing = bearing

        return {
            'visible': True,
            'bearing': round(bearing, 3),
            'range_est': round(range_est, 3),
            'velocity_est': round(velocity_est, 3),
            'closing_rate': round(closing_rate, 3),
            'predicted_sector_0_5s': round(predicted_sector, 3),
        }

    def _reset_track(self):
        self.previous_time = None
        self.previous_center_x = None
        self.previous_range = None
        self.previous_bearing = None


def main(args=None):
    rclpy.init(args=args)
    node = TrackerPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
