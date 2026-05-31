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


class SubgoalGeneratorNode(Node):
    def __init__(self):
        super().__init__('subgoal_generator_node')

        self.declare_parameter('features_topic', '/tactics/features')
        self.declare_parameter('mode_topic', '/tactics/mode')
        self.declare_parameter('subgoal_topic', '/tactics/subgoal_pose')
        self.declare_parameter('publish_rate_hz', 5.0)

        self.features_topic = self.get_parameter('features_topic').value
        self.mode_topic = self.get_parameter('mode_topic').value
        self.subgoal_topic = self.get_parameter('subgoal_topic').value
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self.latest_features = {}
        self.latest_mode = {'mode': 'CRUISE', 'confidence': 1.0}

        self.subgoal_pub = self.create_publisher(String, self.subgoal_topic, 10)
        self.features_sub = self.create_subscription(
            String,
            self.features_topic,
            self.features_callback,
            10,
        )
        self.mode_sub = self.create_subscription(
            String,
            self.mode_topic,
            self.mode_callback,
            10,
        )

        timer_period = 1.0 / max(publish_rate_hz, 0.1)
        self.timer = self.create_timer(timer_period, self.publish_subgoal)

        self.get_logger().info(
            'Subgoal generator subscribed to %s and %s, publishing %s'
            % (self.features_topic, self.mode_topic, self.subgoal_topic)
        )

    def features_callback(self, message):
        parsed = self._parse_json(message.data, 'tactics features')
        if parsed is not None:
            self.latest_features = parsed

    def mode_callback(self, message):
        parsed = self._parse_json(message.data, 'tactics mode')
        if parsed is not None:
            self.latest_mode = parsed

    def publish_subgoal(self):
        mode = str(self.latest_mode.get('mode', 'CRUISE'))
        features = self.latest_features
        escape_corridor_score = _clamp(_number(features, 'escape_corridor_score', 0.0))
        obstacle_risk = _clamp(_number(features, 'obstacle_risk', 0.0))
        dead_end_score = _clamp(_number(features, 'dead_end_score', 0.0))
        yellow_threat = _clamp(_number(features, 'yellow_threat', 0.0))

        if mode == 'ESCAPE':
            payload = {
                'mode': mode,
                'type': 'exit_direction',
                'x': round(1.0 + 0.6 * escape_corridor_score, 3),
                'y': 0.0,
            }
        elif mode == 'HIDE':
            lateral = 0.75 if yellow_threat >= 0.5 else 0.55
            payload = {
                'mode': mode,
                'type': 'hide_behind_obstacle',
                'x': round(0.25 + 0.25 * (1.0 - obstacle_risk), 3),
                'y': round(lateral, 3),
            }
        elif mode == 'REPOSITION':
            payload = {
                'mode': mode,
                'type': 'open_area',
                'x': round(0.65 + 0.25 * (1.0 - dead_end_score), 3),
                'y': -0.65 if obstacle_risk > 0.55 else 0.65,
            }
        elif mode == 'RECOVER':
            payload = {
                'mode': mode,
                'type': 'reverse_rotate',
                'x': -0.25,
                'y': 0.55,
            }
        else:
            payload = {
                'mode': 'CRUISE',
                'type': 'forward_cruise',
                'x': 1.0,
                'y': 0.0,
            }

        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.subgoal_pub.publish(message)

    def _parse_json(self, data, topic_name):
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('Invalid %s JSON: %s' % (topic_name, exc))
            return None


def main(args=None):
    rclpy.init(args=args)
    node = SubgoalGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
