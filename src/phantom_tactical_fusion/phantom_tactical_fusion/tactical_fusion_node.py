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


class TacticalFusionNode(Node):
    def __init__(self):
        super().__init__('tactical_fusion_node')

        self.declare_parameter('yellow_state_topic', '/track/yellow_state')
        self.declare_parameter('exit_topic', '/det/exit_boxes')
        self.declare_parameter('obstacle_features_topic', '/nav/local_obstacle_features')
        self.declare_parameter('tactics_features_topic', '/tactics/features')
        self.declare_parameter('publish_rate_hz', 10.0)

        self.yellow_state_topic = self.get_parameter('yellow_state_topic').value
        self.exit_topic = self.get_parameter('exit_topic').value
        self.obstacle_features_topic = self.get_parameter('obstacle_features_topic').value
        self.tactics_features_topic = self.get_parameter('tactics_features_topic').value
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self.latest_yellow_state = {'visible': False}
        self.latest_exit_box = {'visible': False}
        self.latest_obstacle_features = None

        self.features_pub = self.create_publisher(String, self.tactics_features_topic, 10)
        self.yellow_sub = self.create_subscription(
            String,
            self.yellow_state_topic,
            self.yellow_callback,
            10,
        )
        self.exit_sub = self.create_subscription(
            String,
            self.exit_topic,
            self.exit_callback,
            10,
        )
        self.obstacle_sub = self.create_subscription(
            String,
            self.obstacle_features_topic,
            self.obstacle_callback,
            10,
        )

        timer_period = 1.0 / max(publish_rate_hz, 0.1)
        self.timer = self.create_timer(timer_period, self.publish_features)

        self.get_logger().info(
            'Tactical fusion subscribed to %s, %s, %s; publishing %s'
            % (
                self.yellow_state_topic,
                self.exit_topic,
                self.obstacle_features_topic,
                self.tactics_features_topic,
            )
        )

    def yellow_callback(self, message):
        parsed = self._parse_json(message.data, 'yellow_state')
        if parsed is not None:
            self.latest_yellow_state = parsed

    def exit_callback(self, message):
        parsed = self._parse_json(message.data, 'exit_boxes')
        if parsed is not None:
            self.latest_exit_box = parsed

    def obstacle_callback(self, message):
        parsed = self._parse_json(message.data, 'local_obstacle_features')
        if parsed is not None:
            self.latest_obstacle_features = parsed

    def publish_features(self):
        if self.latest_obstacle_features is None:
            return

        nav = self.latest_obstacle_features
        yellow = self.latest_yellow_state
        exit_box = self.latest_exit_box

        front_clearance = _number(nav, 'front_clearance', 3.0)
        left_clearance = _number(nav, 'left_clearance', 3.0)
        right_clearance = _number(nav, 'right_clearance', 3.0)
        left_front_clearance = _number(nav, 'left_front_clearance', left_clearance)
        right_front_clearance = _number(nav, 'right_front_clearance', right_clearance)
        free_heading_switch_count = _number(nav, 'free_heading_switch_count', 0.0)
        dead_end_score = _clamp(_number(nav, 'dead_end_score', 0.0))
        escape_corridor_score = _clamp(_number(nav, 'escape_corridor_score', 0.0))

        front_risk = _clamp((0.9 - front_clearance) / 0.9)
        side_risk = _clamp((0.7 - min(left_clearance, right_clearance)) / 0.7)
        front_corner_risk = _clamp((0.45 - min(left_front_clearance, right_front_clearance)) / 0.45)
        obstacle_risk = _clamp(0.6 * front_risk + 0.2 * side_risk + 0.2 * dead_end_score)

        yellow_visible = bool(yellow.get('visible', False))
        if yellow_visible:
            range_est = _number(yellow, 'range_est', 6.0)
            bearing = abs(_number(yellow, 'bearing', 0.0))
            closing_rate = _number(yellow, 'closing_rate', 0.0)
            range_risk = _clamp((3.0 - range_est) / 2.75)
            bearing_risk = _clamp(1.0 - bearing / 0.7)
            closing_risk = _clamp((closing_rate + 0.2) / 1.2)
            yellow_threat = _clamp(0.55 * range_risk + 0.25 * bearing_risk + 0.20 * closing_risk)
        else:
            yellow_threat = 0.0

        exit_visible = bool(exit_box.get('visible', False))
        exit_conf = _number(exit_box, 'conf', 0.0) if exit_visible else 0.0
        exit_opportunity = _clamp(0.35 * escape_corridor_score + 0.65 * exit_conf)

        intercept_margin = _clamp(
            exit_opportunity - yellow_threat - 0.25 * obstacle_risk,
            -1.0,
            1.0,
        )
        # Extends the existing stuck_score with real free-space oscillation and corner proximity
        # fields published by free_space_node; no simulated or fabricated evidence is introduced.
        oscillation_risk = _clamp(free_heading_switch_count / 6.0)
        stuck_score = _clamp(
            max(dead_end_score * 0.75, front_risk * 0.85)
            + obstacle_risk * 0.15
            + front_corner_risk * 0.15
            + oscillation_risk * 0.10
        )
        occlusion_score = _clamp((0.0 if yellow_visible else 0.45) * obstacle_risk + 0.2 * dead_end_score)

        features = {
            'yellow_threat': round(yellow_threat, 3),
            'exit_opportunity': round(exit_opportunity, 3),
            'intercept_margin': round(intercept_margin, 3),
            'obstacle_risk': round(obstacle_risk, 3),
            'dead_end_score': round(dead_end_score, 3),
            'stuck_score': round(stuck_score, 3),
            'occlusion_score': round(occlusion_score, 3),
            'escape_corridor_score': round(escape_corridor_score, 3),
        }

        message = String()
        message.data = json.dumps(features, sort_keys=True)
        self.features_pub.publish(message)

    def _parse_json(self, data, topic_name):
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('Invalid %s JSON: %s' % (topic_name, exc))
            return None


def main(args=None):
    rclpy.init(args=args)
    node = TacticalFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
