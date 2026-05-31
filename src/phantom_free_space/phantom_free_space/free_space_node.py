import json
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def _sanitize_range(value, range_min, range_max):
    if math.isinf(value) and value > 0.0:
        return range_max
    if not math.isfinite(value):
        return range_min
    return max(range_min, min(float(value), range_max))


class FreeSpaceNode(Node):
    def __init__(self):
        super().__init__('free_space_node')

        self.declare_parameter('scan_topic', '/perception/scan_or_depth')
        self.declare_parameter('features_topic', '/nav/local_obstacle_features')
        # Vehicle envelope parameters. obstacle_inflation_m should cover half body width,
        # wheel overhang, control error, and a small safety margin around LiDAR points.
        self.declare_parameter('obstacle_inflation_m', 0.22)
        self.declare_parameter('front_stop_clearance_m', 0.30)
        self.declare_parameter('side_stop_clearance_m', 0.24)
        self.declare_parameter('front_sector_half_angle_rad', 0.28)
        self.declare_parameter('front_side_sector_min_rad', 0.28)
        self.declare_parameter('front_side_sector_max_rad', 0.95)
        self.declare_parameter('direction_lock_time_sec', 0.45)
        self.declare_parameter('direction_switch_margin', 0.18)
        self.declare_parameter('heading_change_penalty', 0.08)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.features_topic = self.get_parameter('features_topic').value
        self.obstacle_inflation_m = float(self.get_parameter('obstacle_inflation_m').value)
        self.front_stop_clearance_m = float(self.get_parameter('front_stop_clearance_m').value)
        self.side_stop_clearance_m = float(self.get_parameter('side_stop_clearance_m').value)
        self.front_sector_half_angle_rad = float(self.get_parameter('front_sector_half_angle_rad').value)
        self.front_side_sector_min_rad = float(self.get_parameter('front_side_sector_min_rad').value)
        self.front_side_sector_max_rad = float(self.get_parameter('front_side_sector_max_rad').value)
        self.direction_lock_time_sec = float(self.get_parameter('direction_lock_time_sec').value)
        self.direction_switch_margin = float(self.get_parameter('direction_switch_margin').value)
        self.heading_change_penalty = float(self.get_parameter('heading_change_penalty').value)

        self.held_free_heading = 0.0
        self.last_heading_switch_time = -999.0
        self.free_heading_switch_count = 0

        self.features_pub = self.create_publisher(String, self.features_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )

        self.get_logger().info(
            'free_space_node subscribed to %s, publishing %s'
            % (self.scan_topic, self.features_topic)
        )

    def scan_callback(self, scan):
        if not scan.ranges:
            self.get_logger().warn('received empty LaserScan ranges')
            return

        range_min = scan.range_min if scan.range_min > 0.0 else 0.05
        range_max = scan.range_max if scan.range_max > range_min else 10.0

        angle_range_pairs = []
        for index, raw_range in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            distance = _sanitize_range(raw_range, range_min, range_max)
            angle_range_pairs.append((angle, distance))

        front_raw_clearance = self._sector_min(
            angle_range_pairs, -self.front_sector_half_angle_rad,
            self.front_sector_half_angle_rad, range_max)
        left_front_raw_clearance = self._sector_min(
            angle_range_pairs, self.front_side_sector_min_rad,
            self.front_side_sector_max_rad, range_max)
        right_front_raw_clearance = self._sector_min(
            angle_range_pairs, -self.front_side_sector_max_rad,
            -self.front_side_sector_min_rad, range_max)
        left_raw_clearance = self._sector_min(angle_range_pairs, 0.25, 1.2, range_max)
        right_raw_clearance = self._sector_min(angle_range_pairs, -1.2, -0.25, range_max)
        rear_raw_clearance = self._rear_min(angle_range_pairs, range_max)

        # Inflated clearances are the values downstream safety logic should use.
        # Raw clearances are kept in the payload for tuning with real robot logs.
        front_clearance = self._inflate_clearance(front_raw_clearance)
        left_front_clearance = self._inflate_clearance(left_front_raw_clearance)
        right_front_clearance = self._inflate_clearance(right_front_raw_clearance)
        left_clearance = self._inflate_clearance(left_raw_clearance)
        right_clearance = self._inflate_clearance(right_raw_clearance)
        rear_clearance = self._inflate_clearance(rear_raw_clearance)

        local_free_heading, best_range = self._select_free_heading(angle_range_pairs)

        dead_end_score = 0.0
        if front_clearance < 0.8 and left_clearance < 0.8 and right_clearance < 0.8:
            dead_end_score = 1.0
        elif front_clearance < 0.8:
            dead_end_score = 0.5

        corner_trap_score = 0.0
        if front_clearance < 0.7 and (left_clearance < 0.6 or right_clearance < 0.6):
            corner_trap_score = 0.8

        features = {
            'raw_front_clearance': round(front_raw_clearance, 3),
            'raw_left_front_clearance': round(left_front_raw_clearance, 3),
            'raw_right_front_clearance': round(right_front_raw_clearance, 3),
            'front_clearance': round(front_clearance, 3),
            'left_front_clearance': round(left_front_clearance, 3),
            'right_front_clearance': round(right_front_clearance, 3),
            'left_clearance': round(left_clearance, 3),
            'right_clearance': round(right_clearance, 3),
            'rear_clearance': round(rear_clearance, 3),
            'local_free_heading': round(local_free_heading, 3),
            'front_blocked': front_clearance < self.front_stop_clearance_m,
            'left_front_blocked': left_front_clearance < self.side_stop_clearance_m,
            'right_front_blocked': right_front_clearance < self.side_stop_clearance_m,
            'obstacle_inflation_m': round(self.obstacle_inflation_m, 3),
            'free_heading_switch_count': int(self.free_heading_switch_count),
            'dead_end_score': round(dead_end_score, 3),
            'corner_trap_score': round(corner_trap_score, 3),
            'escape_corridor_score': round(min(self._inflate_clearance(best_range) / 3.0, 1.0), 3),
        }

        message = String()
        message.data = json.dumps(features, sort_keys=True)
        self.features_pub.publish(message)

    @staticmethod
    def _sector_min(angle_range_pairs, angle_min, angle_max, default_value):
        sector_ranges = [
            distance
            for angle, distance in angle_range_pairs
            if angle_min <= angle <= angle_max
        ]
        if not sector_ranges:
            return default_value
        return min(sector_ranges)

    @staticmethod
    def _rear_min(angle_range_pairs, default_value):
        sector_ranges = [
            distance
            for angle, distance in angle_range_pairs
            if abs(angle) > 2.6
        ]
        if not sector_ranges:
            return default_value
        return min(sector_ranges)

    def _inflate_clearance(self, distance):
        return max(0.0, float(distance) - max(0.0, self.obstacle_inflation_m))

    def _select_free_heading(self, angle_range_pairs):
        now = self.get_clock().now().nanoseconds * 1e-9
        candidates = []
        for angle, distance in angle_range_pairs:
            inflated = self._inflate_clearance(distance)
            score = inflated - self.heading_change_penalty * abs(self._angle_delta(angle, self.held_free_heading))
            candidates.append((angle, distance, score))

        best_angle, best_range, best_score = max(candidates, key=lambda item: item[2])
        held = min(candidates, key=lambda item: abs(self._angle_delta(item[0], self.held_free_heading)))
        held_angle, held_range, held_score = held

        # Direction lock reduces left/right free-space chatter without hiding a clear safety win.
        if now - self.last_heading_switch_time < self.direction_lock_time_sec:
            if held_score + self.direction_switch_margin >= best_score:
                return held_angle, held_range

        if abs(self._angle_delta(best_angle, self.held_free_heading)) > 0.35:
            self.free_heading_switch_count += 1
            self.last_heading_switch_time = now
        self.held_free_heading = best_angle
        return best_angle, best_range

    @staticmethod
    def _angle_delta(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))


def main(args=None):
    rclpy.init(args=args)
    node = FreeSpaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
