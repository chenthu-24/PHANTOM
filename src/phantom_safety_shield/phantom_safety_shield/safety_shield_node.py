import json
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


class SafetyShieldNode(Node):
    def __init__(self):
        super().__init__('safety_shield_node')

        self.declare_parameter('raw_cmd_topic', '/cmd_vel_raw')
        self.declare_parameter('features_topic', '/nav/local_obstacle_features')
        self.declare_parameter('safe_cmd_topic', '/cmd_vel')
        # Upper-layer command smoothing only. These limits keep sudden planner changes
        # from scraping nearby obstacles.
        self.declare_parameter('max_linear_accel_mps2', 0.25)
        self.declare_parameter('max_angular_accel_rps2', 1.2)
        self.declare_parameter('max_forward_speed_mps', 0.35)
        self.declare_parameter('max_reverse_speed_mps', 0.10)
        self.declare_parameter('max_angular_speed_rps', 1.0)
        self.declare_parameter('front_stop_clearance_m', 0.30)
        self.declare_parameter('side_stop_clearance_m', 0.24)
        self.declare_parameter('rear_stop_clearance_m', 0.22)
        self.declare_parameter('near_obstacle_slow_scale', 0.45)

        self.raw_cmd_topic = self.get_parameter('raw_cmd_topic').value
        self.features_topic = self.get_parameter('features_topic').value
        self.safe_cmd_topic = self.get_parameter('safe_cmd_topic').value
        self.max_linear_accel_mps2 = float(self.get_parameter('max_linear_accel_mps2').value)
        self.max_angular_accel_rps2 = float(self.get_parameter('max_angular_accel_rps2').value)
        self.max_forward_speed_mps = float(self.get_parameter('max_forward_speed_mps').value)
        self.max_reverse_speed_mps = float(self.get_parameter('max_reverse_speed_mps').value)
        self.max_angular_speed_rps = float(self.get_parameter('max_angular_speed_rps').value)
        self.front_stop_clearance_m = float(self.get_parameter('front_stop_clearance_m').value)
        self.side_stop_clearance_m = float(self.get_parameter('side_stop_clearance_m').value)
        self.rear_stop_clearance_m = float(self.get_parameter('rear_stop_clearance_m').value)
        self.near_obstacle_slow_scale = float(self.get_parameter('near_obstacle_slow_scale').value)

        self.latest_raw_cmd = None
        self.latest_features = None
        self.last_safe_linear_x = 0.0
        self.last_safe_angular_z = 0.0
        self.last_publish_time = time.monotonic()

        self.cmd_pub = self.create_publisher(Twist, self.safe_cmd_topic, 10)
        self.raw_cmd_sub = self.create_subscription(
            Twist,
            self.raw_cmd_topic,
            self.raw_cmd_callback,
            10,
        )
        self.features_sub = self.create_subscription(
            String,
            self.features_topic,
            self.features_callback,
            10,
        )

        self.get_logger().info(
            'Safety shield subscribed to %s and %s, publishing %s'
            % (self.raw_cmd_topic, self.features_topic, self.safe_cmd_topic)
        )

    def raw_cmd_callback(self, raw_cmd):
        self.latest_raw_cmd = raw_cmd
        self.publish_safe_cmd()

    def features_callback(self, message):
        try:
            self.latest_features = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('Invalid obstacle features JSON: %s' % exc)
            return

        if self.latest_raw_cmd is not None:
            self.publish_safe_cmd()

    def publish_safe_cmd(self):
        safe_cmd = Twist()

        if self.latest_raw_cmd is None or self.latest_features is None:
            self.cmd_pub.publish(safe_cmd)
            return

        try:
            front_clearance = float(self.latest_features['front_clearance'])
            dead_end_score = float(self.latest_features['dead_end_score'])
            left_front_clearance = float(self.latest_features.get('left_front_clearance', 99.0))
            right_front_clearance = float(self.latest_features.get('right_front_clearance', 99.0))
            rear_clearance = float(self.latest_features.get('rear_clearance', 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn('Invalid obstacle feature fields: %s' % exc)
            self.cmd_pub.publish(safe_cmd)
            return

        linear_x = float(self.latest_raw_cmd.linear.x)
        angular_z = float(self.latest_raw_cmd.angular.z)

        if linear_x > 0.0 and front_clearance < self.front_stop_clearance_m:
            linear_x = 0.0
        elif linear_x > 0.0 and min(left_front_clearance, right_front_clearance) < self.side_stop_clearance_m:
            linear_x *= self.near_obstacle_slow_scale
        elif linear_x < 0.0 and rear_clearance < self.rear_stop_clearance_m:
            linear_x = 0.0

        # Do not continue turning into a blocked front corner; let planner/recover pick another side.
        if left_front_clearance < self.side_stop_clearance_m and angular_z > 0.0:
            angular_z = 0.0
            linear_x = min(linear_x, 0.03)
        if right_front_clearance < self.side_stop_clearance_m and angular_z < 0.0:
            angular_z = 0.0
            linear_x = min(linear_x, 0.03)

        if dead_end_score > 0.8:
            linear_x = min(linear_x, 0.05)

        linear_x = _clamp(linear_x, -self.max_reverse_speed_mps, self.max_forward_speed_mps)
        angular_z = _clamp(angular_z, -self.max_angular_speed_rps, self.max_angular_speed_rps)
        linear_x, angular_z = self._smooth_cmd(linear_x, angular_z)

        safe_cmd.linear.x = linear_x
        safe_cmd.angular.z = angular_z
        self.cmd_pub.publish(safe_cmd)

    def _smooth_cmd(self, linear_x, angular_z):
        now = time.monotonic()
        dt = _clamp(now - self.last_publish_time, 0.001, 0.2)
        self.last_publish_time = now

        linear_step = self.max_linear_accel_mps2 * dt
        angular_step = self.max_angular_accel_rps2 * dt
        if abs(linear_x) > abs(self.last_safe_linear_x):
            linear_x = _clamp(linear_x, self.last_safe_linear_x - linear_step,
                              self.last_safe_linear_x + linear_step)
        if abs(angular_z) > abs(self.last_safe_angular_z):
            angular_z = _clamp(angular_z, self.last_safe_angular_z - angular_step,
                               self.last_safe_angular_z + angular_step)

        self.last_safe_linear_x = linear_x
        self.last_safe_angular_z = angular_z
        return linear_x, angular_z


def main(args=None):
    rclpy.init(args=args)
    node = SafetyShieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
