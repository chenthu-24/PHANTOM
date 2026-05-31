import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(float(value), upper))


def _number(payload, key, default=0.0):
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


class ModeManagerNode(Node):
    def __init__(self):
        super().__init__('mode_manager_node')

        self.declare_parameter('features_topic', '/tactics/features')
        self.declare_parameter('mode_topic', '/tactics/mode')
        self.declare_parameter('recover_stuck_threshold', 0.8)
        self.declare_parameter('recover_stuck_hold_sec', 0.8)

        self.features_topic = self.get_parameter('features_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value
        self.recover_stuck_threshold = float(self.get_parameter('recover_stuck_threshold').value)
        self.recover_stuck_hold_sec = float(self.get_parameter('recover_stuck_hold_sec').value)
        self.stuck_started_at = None

        self.mode_pub = self.create_publisher(String, self.mode_topic, 10)
        self.features_sub = self.create_subscription(
            String,
            self.features_topic,
            self.features_callback,
            10,
        )

        self.get_logger().info(
            'Mode manager subscribed to %s, publishing %s'
            % (self.features_topic, self.mode_topic)
        )

    def features_callback(self, message):
        try:
            features = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('Invalid tactics features JSON: %s' % exc)
            return

        stuck_score = _number(features, 'stuck_score')
        exit_opportunity = _number(features, 'exit_opportunity')
        intercept_margin = _number(features, 'intercept_margin')
        yellow_threat = _number(features, 'yellow_threat')
        dead_end_score = _number(features, 'dead_end_score')

        now = self.get_clock().now().nanoseconds * 1e-9
        if stuck_score > self.recover_stuck_threshold:
            if self.stuck_started_at is None:
                self.stuck_started_at = now
        else:
            self.stuck_started_at = None

        # RECOVER is entered only after stuck_score stays high, so CRUISE/REPOSITION
        # does not oscillate into recover on a single noisy obstacle frame.
        stuck_sustained = (
            self.stuck_started_at is not None
            and now - self.stuck_started_at >= self.recover_stuck_hold_sec
        )

        if stuck_sustained:
            mode = 'RECOVER'
            confidence = stuck_score
        elif exit_opportunity > 0.65 and intercept_margin > 0.05:
            mode = 'ESCAPE'
            confidence = min(exit_opportunity, 0.5 + intercept_margin)
        elif yellow_threat > 0.65:
            mode = 'HIDE'
            confidence = yellow_threat
        elif dead_end_score > 0.4:
            mode = 'REPOSITION'
            confidence = dead_end_score
        else:
            mode = 'CRUISE'
            confidence = 1.0 - max(stuck_score, yellow_threat, dead_end_score)

        payload = {
            'mode': mode,
            'confidence': round(_clamp(confidence), 3),
        }
        output = String()
        output.data = json.dumps(payload, sort_keys=True)
        self.mode_pub.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
