import json
import math
import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

try:
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover - keeps static checks usable on non-ROS hosts.
    Odometry = None


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def _now_seconds(node):
    return node.get_clock().now().nanoseconds * 1e-9


class PlannerControllerNode(Node):
    def __init__(self):
        super().__init__('planner_controller_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('detection_topics', ['/det/detections', '/det/yellow_boxes'])
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('cmd_vel_topic', '/controller/cmd_vel')
        self.declare_parameter('stuck_status_topic', '/planner/stuck_status')
        self.declare_parameter('artifacts_dir', '/home/ubuntu/phantom_ws/artifacts')
        self.declare_parameter('safe_distance', 0.55)
        self.declare_parameter('emergency_stop_distance', 0.20)
        # obstacle_inflation_m accounts for body width, wheel overhang, control error, and margin.
        self.declare_parameter('obstacle_inflation_m', 0.22)
        self.declare_parameter('front_stop_clearance_m', 0.30)
        self.declare_parameter('side_stop_clearance_m', 0.24)
        self.declare_parameter('rear_stop_clearance_m', 0.22)
        self.declare_parameter('front_angle_offset_rad', 0.0)
        self.declare_parameter('lidar_timeout_sec', 0.5)
        self.declare_parameter('cruise_max_linear', 0.18)
        self.declare_parameter('escape_max_linear', 0.35)
        self.declare_parameter('max_angular', 0.8)
        self.declare_parameter('escape_min_duration', 1.5)
        self.declare_parameter('threat_clear_timeout', 1.2)
        self.declare_parameter('threat_confidence', 0.25)
        self.declare_parameter('direction_lock_time_sec', 0.55)
        self.declare_parameter('direction_switch_score_margin', 0.35)
        self.declare_parameter('heading_change_penalty', 0.18)
        self.declare_parameter('direction_switch_penalty', 0.35)
        self.declare_parameter('oscillation_penalty', 0.15)
        self.declare_parameter('oscillation_window_sec', 3.0)
        self.declare_parameter('stuck_score_threshold', 0.78)
        self.declare_parameter('stuck_score_hold_sec', 0.8)
        self.declare_parameter('stuck_window_sec', 3.0)
        self.declare_parameter('min_progress_m', 0.04)
        self.declare_parameter('low_speed_error_threshold', 0.08)
        self.declare_parameter('recover_stop_time_sec', 0.25)
        self.declare_parameter('recover_backup_time_sec', 0.65)
        self.declare_parameter('recover_turn_time_sec', 0.9)
        self.declare_parameter('recover_forward_time_sec', 0.55)
        self.declare_parameter('recover_backup_linear', -0.06)
        self.declare_parameter('recover_forward_linear', 0.06)
        self.declare_parameter('recover_angular', 0.55)
        self.declare_parameter('probe_front_threshold_m', 0.10)
        self.declare_parameter('probe_tie_margin', 0.25)
        self.declare_parameter('min_probe_duration', 0.6)
        self.declare_parameter('max_probe_duration', 1.8)
        self.declare_parameter('probe_linear', 0.0)
        self.declare_parameter('probe_angular', 0.6)
        self.declare_parameter('depth_timeout_sec', 0.8)
        self.declare_parameter('rear_valid_ratio_min', 0.35)
        self.declare_parameter('rear_center_depth_min', 0.45)
        self.declare_parameter('rear_min_depth_min', 0.30)
        self.declare_parameter('recover_post_backup_stop_time_sec', 0.25)
        self.declare_parameter('recover_forward_clearance_m', 0.35)
        self.declare_parameter('max_linear_accel_mps2', 0.25)
        self.declare_parameter('max_angular_accel_rps2', 1.2)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.detection_topics = list(self.get_parameter('detection_topics').value)
        self.depth_topic = self.get_parameter('depth_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.stuck_status_topic = self.get_parameter('stuck_status_topic').value
        self.artifacts_dir = os.path.expanduser(str(self.get_parameter('artifacts_dir').value))
        self.safe_distance = float(self.get_parameter('safe_distance').value)
        self.emergency_stop_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.obstacle_inflation_m = float(self.get_parameter('obstacle_inflation_m').value)
        self.front_stop_clearance_m = float(self.get_parameter('front_stop_clearance_m').value)
        self.side_stop_clearance_m = float(self.get_parameter('side_stop_clearance_m').value)
        self.rear_stop_clearance_m = float(self.get_parameter('rear_stop_clearance_m').value)
        self.front_angle_offset_rad = float(self.get_parameter('front_angle_offset_rad').value)
        self.lidar_timeout_sec = float(self.get_parameter('lidar_timeout_sec').value)
        self.cruise_max_linear = float(self.get_parameter('cruise_max_linear').value)
        self.escape_max_linear = float(self.get_parameter('escape_max_linear').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.escape_min_duration = float(self.get_parameter('escape_min_duration').value)
        self.threat_clear_timeout = float(self.get_parameter('threat_clear_timeout').value)
        self.threat_confidence = float(self.get_parameter('threat_confidence').value)
        self.direction_lock_time_sec = float(self.get_parameter('direction_lock_time_sec').value)
        self.direction_switch_score_margin = float(self.get_parameter('direction_switch_score_margin').value)
        self.heading_change_penalty = float(self.get_parameter('heading_change_penalty').value)
        self.direction_switch_penalty = float(self.get_parameter('direction_switch_penalty').value)
        self.oscillation_penalty = float(self.get_parameter('oscillation_penalty').value)
        self.oscillation_window_sec = float(self.get_parameter('oscillation_window_sec').value)
        self.stuck_score_threshold = float(self.get_parameter('stuck_score_threshold').value)
        self.stuck_score_hold_sec = float(self.get_parameter('stuck_score_hold_sec').value)
        self.stuck_window_sec = float(self.get_parameter('stuck_window_sec').value)
        self.min_progress_m = float(self.get_parameter('min_progress_m').value)
        self.low_speed_error_threshold = float(self.get_parameter('low_speed_error_threshold').value)
        self.recover_stop_time_sec = float(self.get_parameter('recover_stop_time_sec').value)
        self.recover_backup_time_sec = float(self.get_parameter('recover_backup_time_sec').value)
        self.recover_turn_time_sec = float(self.get_parameter('recover_turn_time_sec').value)
        self.recover_forward_time_sec = float(self.get_parameter('recover_forward_time_sec').value)
        self.recover_backup_linear = float(self.get_parameter('recover_backup_linear').value)
        self.recover_forward_linear = float(self.get_parameter('recover_forward_linear').value)
        self.recover_angular = float(self.get_parameter('recover_angular').value)
        self.probe_front_threshold_m = float(self.get_parameter('probe_front_threshold_m').value)
        self.probe_tie_margin = float(self.get_parameter('probe_tie_margin').value)
        self.min_probe_duration = float(self.get_parameter('min_probe_duration').value)
        self.max_probe_duration = float(self.get_parameter('max_probe_duration').value)
        self.probe_linear = float(self.get_parameter('probe_linear').value)
        self.probe_angular = float(self.get_parameter('probe_angular').value)
        self.depth_timeout_sec = float(self.get_parameter('depth_timeout_sec').value)
        self.rear_valid_ratio_min = float(self.get_parameter('rear_valid_ratio_min').value)
        self.rear_center_depth_min = float(self.get_parameter('rear_center_depth_min').value)
        self.rear_min_depth_min = float(self.get_parameter('rear_min_depth_min').value)
        self.recover_post_backup_stop_time_sec = float(
            self.get_parameter('recover_post_backup_stop_time_sec').value)
        self.recover_forward_clearance_m = float(self.get_parameter('recover_forward_clearance_m').value)
        self.max_linear_accel_mps2 = float(self.get_parameter('max_linear_accel_mps2').value)
        self.max_angular_accel_rps2 = float(self.get_parameter('max_angular_accel_rps2').value)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.stuck_status_pub = self.create_publisher(String, self.stuck_status_topic, 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = None
        if Odometry is not None:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.odom_topic,
                self.odom_callback,
                10,
            )
        self.detection_subs = [
            self.create_subscription(String, topic, self.detection_callback, 10)
            for topic in self.detection_topics
        ]
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.latest_scan = None
        self.latest_sectors = None
        self.last_scan_time = 0.0
        self.last_threat_time = -999.0
        self.escape_until = 0.0
        self.state = 'CRUISE'
        self.held_direction = 0.0
        self.held_sector_name = 'front'
        self.last_direction_switch_time = -999.0
        self.direction_switch_times = deque()
        self.last_selected_heading = 0.0
        self.turning_since = None
        self.recover_until = 0.0
        self.recover_start = 0.0
        self.recover_direction = 1.0
        self.recover_sequence_duration = (
            self.recover_stop_time_sec + self.recover_backup_time_sec
            + self.recover_turn_time_sec + self.recover_forward_time_sec
        )
        self.stuck_score = 0.0
        self.stuck_started_at = None
        self.probe_start = 0.0
        self.probe_direction = 1.0
        self.recover_phase_start = 0.0
        self.latest_rear_depth = None
        self.last_depth_time = -999.0
        self.reported_depth_topics = False
        self.reported_no_depth = False
        self.cmd_history = deque()
        self.odom_history = deque()
        self.last_cmd = Twist()
        self.last_cmd_time = _now_seconds(self)
        self.saved_lidar_image = False

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.timer = self.create_timer(0.1, self.control_step)
        self.depth_topic_timer = self.create_timer(1.5, self._log_depth_topics_once)

        self.get_logger().info(
            'planner_controller_node subscribed to %s, %s, and depth %s, publishing %s'
            % (self.scan_topic, self.detection_topics, self.depth_topic, self.cmd_vel_topic)
        )

    def scan_callback(self, scan):
        sectors = self._compute_sectors(scan)
        now = _now_seconds(self)
        if sectors is None:
            self.get_logger().warn('invalid LiDAR scan; stopping')
            self.stop_robot()
            return

        self.latest_scan = scan
        self.latest_sectors = sectors
        self.last_scan_time = now

        if not self.saved_lidar_image:
            selected = self._select_direction(sectors)
            self._save_lidar_visualization(sectors, selected)
            self.saved_lidar_image = True

    def detection_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('invalid detection JSON: %s' % exc)
            return

        if self._payload_has_threat(payload):
            now = _now_seconds(self)
            self.last_threat_time = now
            self.escape_until = max(self.escape_until, now + self.escape_min_duration)

    def odom_callback(self, message):
        now = _now_seconds(self)
        pose = message.pose.pose.position
        twist = message.twist.twist
        self.odom_history.append((now, float(pose.x), float(pose.y), float(twist.linear.x)))
        while self.odom_history and now - self.odom_history[0][0] > self.stuck_window_sec:
            self.odom_history.popleft()

    def depth_callback(self, image_msg):
        features = self._rear_depth_features(image_msg)
        if features is None:
            return
        self.latest_rear_depth = features
        self.last_depth_time = _now_seconds(self)

    def control_step(self):
        now = _now_seconds(self)
        if self.latest_sectors is None or now - self.last_scan_time > self.lidar_timeout_sec:
            self.state = 'CRUISE'
            self.stop_robot()
            return

        if self.state in (
                'PROBE_LEFT', 'PROBE_RIGHT', 'CAMERA_SAFE_BACKUP_CHECK',
                'RECOVER_BACKUP', 'RECOVER_TURN', 'RECOVER_PROBE_FORWARD'):
            cmd = self._recover_state_cmd(now)
            self._publish_cmd(cmd)
            return
        if self.recover_until > now:
            cmd = self._recover_cmd(now)
            self._publish_cmd(cmd)
            return
        if self.state == 'RECOVER':
            self._finish_recover()

        threat_recent = now - self.last_threat_time <= self.threat_clear_timeout
        if threat_recent or now < self.escape_until:
            self.state = 'ESCAPE'
        else:
            self.state = 'CRUISE'

        selected = self._select_direction(self.latest_sectors)
        if self._should_enter_probe(self.latest_sectors):
            self._enter_probe(now)
            cmd = self._recover_state_cmd(now)
            self._publish_cmd(cmd)
            return

        cmd = self._make_cmd(self.latest_sectors, selected, self.state)
        self._update_stuck_score(now, selected, cmd)

        if self._should_enter_recover(now):
            self._enter_recover(now, selected['heading'])
            cmd = self._recover_state_cmd(now)
        elif abs(cmd.angular.z) > 0.25 and cmd.linear.x < 0.04:
            if self.turning_since is None:
                self.turning_since = now
            elif now - self.turning_since > 2.5:
                self._enter_recover(now, selected['heading'])
                cmd = self._recover_state_cmd(now)
        else:
            self.turning_since = None

        self._publish_cmd(cmd)

    def _compute_sectors(self, scan):
        if not scan.ranges or not math.isfinite(scan.angle_increment) or scan.angle_increment == 0.0:
            return None

        range_min = scan.range_min if scan.range_min > 0.0 else 0.05
        range_max = scan.range_max if scan.range_max > range_min else 6.0
        sectors = {
            'left': {'bounds': (0.85, 1.55), 'heading': 1.10, 'values': []},
            'front_left': {'bounds': (0.28, 0.85), 'heading': 0.52, 'values': []},
            'front': {'bounds': (-0.28, 0.28), 'heading': 0.0, 'values': []},
            'front_right': {'bounds': (-0.85, -0.28), 'heading': -0.52, 'values': []},
            'right': {'bounds': (-1.55, -0.85), 'heading': -1.10, 'values': []},
            'rear': {'bounds': (2.55, math.pi), 'heading': math.pi, 'values': []},
            'rear_neg': {'bounds': (-math.pi, -2.55), 'heading': -math.pi, 'values': []},
        }

        counts = {name: 0 for name in sectors}
        for index, raw_range in enumerate(scan.ranges):
            angle = self._normalize_angle(
                scan.angle_min + index * scan.angle_increment - self.front_angle_offset_rad
            )

            for name, sector in sectors.items():
                low, high = sector['bounds']
                if low <= angle <= high:
                    counts[name] += 1
                    distance = self._sector_distance(raw_range, range_min, range_max)
                    if distance is not None:
                        sector['values'].append(distance)
                    break

        valid_total = 0
        for name, sector in sectors.items():
            values = sector.pop('values')
            expected = max(counts[name], 1)
            if not values:
                sector.update({
                    'min': 0.0,
                    'median': 0.0,
                    'p70': 0.0,
                    'width': 0.0,
                    'score': -10.0,
                })
                continue
            valid_total += len(values)
            arr = np.asarray(values, dtype=np.float32)
            width = float(len(values)) / float(expected)
            raw_nearest = float(np.min(arr))
            raw_median = float(np.median(arr))
            raw_p70 = float(np.percentile(arr, 70))
            nearest = self._inflated_distance(raw_nearest)
            median = self._inflated_distance(raw_median)
            p70 = self._inflated_distance(raw_p70)
            score = 0.55 * p70 + 0.25 * median + 0.45 * width
            if nearest < self.emergency_stop_distance:
                score -= 4.0
            elif nearest < self.safe_distance:
                score -= 1.2 * (self.safe_distance - nearest)
            sector.update({
                'min': nearest,
                'median': median,
                'p70': p70,
                'raw_min': raw_nearest,
                'raw_median': raw_median,
                'raw_p70': raw_p70,
                'width': width,
                'score': score,
            })

        if valid_total < 8:
            return None
        if sectors['rear']['min'] <= 0.0 or (
                sectors['rear_neg']['min'] > 0.0 and sectors['rear_neg']['min'] < sectors['rear']['min']):
            sectors['rear'].update({
                'min': sectors['rear_neg']['min'],
                'median': sectors['rear_neg']['median'],
                'p70': sectors['rear_neg']['p70'],
                'raw_min': sectors['rear_neg'].get('raw_min', 0.0),
                'raw_median': sectors['rear_neg'].get('raw_median', 0.0),
                'raw_p70': sectors['rear_neg'].get('raw_p70', 0.0),
                'width': max(sectors['rear']['width'], sectors['rear_neg']['width']),
                'score': max(sectors['rear']['score'], sectors['rear_neg']['score']),
            })
        sectors.pop('rear_neg', None)
        return sectors

    def _inflated_distance(self, distance):
        return max(0.0, float(distance) - max(0.0, self.obstacle_inflation_m))

    @staticmethod
    def _sector_distance(value, range_min, range_max):
        if value is None:
            return None
        value = float(value)
        if math.isinf(value) and value > 0.0:
            return range_max
        if not math.isfinite(value):
            return None
        if value <= max(range_min, 0.03):
            return None
        return min(value, range_max)

    def _select_direction(self, sectors):
        selectable = ['left', 'front_left', 'front', 'front_right', 'right']
        front = sectors['front']
        if front['min'] >= self.safe_distance and front['width'] > 0.45:
            target_name = 'front'
        else:
            target_name = max(selectable, key=lambda name: self._direction_score(sectors, name))

        held_name = self._nearest_sector_name(self.held_direction)
        held_score = self._direction_score(sectors, held_name)
        target_score = self._direction_score(sectors, target_name)
        now = _now_seconds(self)
        held_safe = self._sector_turn_safe(sectors, held_name)
        target_safe = self._sector_turn_safe(sectors, target_name)
        locked = now - self.last_direction_switch_time < self.direction_lock_time_sec
        if locked and held_safe and (not target_safe or
                                     held_score + self.direction_switch_score_margin >= target_score):
            target_name = held_name
        elif held_name != target_name and held_safe and held_score + 0.25 >= target_score:
            target_name = held_name

        if target_name != self.held_sector_name:
            self.direction_switch_times.append(now)
            self.last_direction_switch_time = now
        while self.direction_switch_times and now - self.direction_switch_times[0] > self.oscillation_window_sec:
            self.direction_switch_times.popleft()

        self.held_direction = float(sectors[target_name]['heading'])
        self.held_sector_name = target_name
        self.last_selected_heading = self.held_direction
        selected = dict(sectors[target_name])
        selected['name'] = target_name
        return selected

    def _direction_score(self, sectors, name):
        sector = sectors[name]
        score = float(sector['score'])
        score -= self.heading_change_penalty * abs(self._angle_delta(sector['heading'], self.held_direction))
        if self._turn_side(sector['heading']) != self._turn_side(self.held_direction):
            score -= self.direction_switch_penalty
        score -= self.oscillation_penalty * len(self.direction_switch_times)
        return score

    def _sector_turn_safe(self, sectors, name):
        if name in ('left', 'front_left'):
            return sectors['front_left']['min'] >= self.side_stop_clearance_m
        if name in ('right', 'front_right'):
            return sectors['front_right']['min'] >= self.side_stop_clearance_m
        return sectors['front']['min'] >= self.front_stop_clearance_m

    @staticmethod
    def _turn_side(heading):
        if heading > 0.05:
            return 1
        if heading < -0.05:
            return -1
        return 0

    def _nearest_sector_name(self, heading):
        names = ['left', 'front_left', 'front', 'front_right', 'right']
        return min(names, key=lambda name: abs(self.latest_sectors[name]['heading'] - heading))

    def _make_cmd(self, sectors, selected, state):
        cmd = Twist()
        front = sectors['front']
        front_left = sectors['front_left']
        front_right = sectors['front_right']
        max_linear = self.escape_max_linear if state == 'ESCAPE' else self.cruise_max_linear

        if front['min'] < self.emergency_stop_distance or front['min'] < self.front_stop_clearance_m:
            cmd.linear.x = 0.0
            cmd.angular.z = _clamp(0.65 * math.copysign(1.0, selected['heading'] or 1.0),
                                   -self.max_angular, self.max_angular)
            cmd.angular.z = self._apply_corner_turn_limit(cmd.angular.z, front_left, front_right)
            return cmd

        if front['min'] < self.safe_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = _clamp(0.65 * math.copysign(1.0, selected['heading'] or 1.0),
                                   -self.max_angular, self.max_angular)
            cmd.angular.z = self._apply_corner_turn_limit(cmd.angular.z, front_left, front_right)
            return cmd

        heading = selected['heading']
        angular = _clamp(1.05 * heading, -self.max_angular, self.max_angular)
        angular = self._apply_corner_turn_limit(angular, front_left, front_right)
        clearance = max(front['min'], selected['min'])
        speed_scale = _clamp((clearance - self.emergency_stop_distance) /
                             max(self.safe_distance - self.emergency_stop_distance, 0.05),
                             0.0, 1.0)
        turn_scale = _clamp(1.0 - 0.55 * abs(angular) / max(self.max_angular, 0.1), 0.35, 1.0)
        if min(front_left['min'], front_right['min']) < self.side_stop_clearance_m:
            turn_scale = min(turn_scale, 0.45)

        base = 0.12 if state == 'CRUISE' else 0.24
        linear = min(max_linear, base + 0.18 * speed_scale)

        cmd.linear.x = _clamp(linear * turn_scale, 0.0, max_linear)
        cmd.angular.z = angular
        return cmd

    def _apply_corner_turn_limit(self, angular_z, front_left, front_right):
        # Prevent side scraping by stopping turns into a front-corner obstacle.
        if front_left['min'] < self.side_stop_clearance_m and angular_z > 0.0:
            return 0.0
        if front_right['min'] < self.side_stop_clearance_m and angular_z < 0.0:
            return 0.0
        return angular_z

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _angle_delta(a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _should_enter_probe(self, sectors):
        if sectors['front']['min'] >= self.probe_front_threshold_m:
            return False
        left_score = sectors['left']['score'] + sectors['front_left']['score']
        right_score = sectors['right']['score'] + sectors['front_right']['score']
        return abs(left_score - right_score) <= self.probe_tie_margin

    def _enter_probe(self, now):
        self.state = 'PROBE_LEFT' if self._choose_recover_direction(self.held_direction) > 0.0 else 'PROBE_RIGHT'
        self.probe_direction = 1.0 if self.state == 'PROBE_LEFT' else -1.0
        self.probe_start = now
        self.turning_since = None
        self.stuck_started_at = None
        self.get_logger().warn('entering %s: front corridor tied and blocked' % self.state)

    def _front_cluster_unsafe(self):
        if self.latest_sectors is None:
            return True
        return (
            self.latest_sectors['front']['min'] < self.front_stop_clearance_m
            or self.latest_sectors['front_left']['min'] < self.side_stop_clearance_m
            or self.latest_sectors['front_right']['min'] < self.side_stop_clearance_m
        )

    def _recover_state_cmd(self, now):
        cmd = Twist()
        if self.state in ('PROBE_LEFT', 'PROBE_RIGHT'):
            elapsed = now - self.probe_start
            if elapsed >= self.min_probe_duration and not self._front_cluster_unsafe():
                self._finish_recover()
                return cmd
            if elapsed >= self.max_probe_duration:
                if self._front_cluster_unsafe():
                    self.state = 'CAMERA_SAFE_BACKUP_CHECK'
                    return self._camera_safe_backup_check(now)
                self._finish_recover()
                return cmd
            cmd.linear.x = _clamp(self.probe_linear, 0.0, 0.02)
            cmd.angular.z = _clamp(self.probe_direction * self.probe_angular, -self.max_angular, self.max_angular)
            return cmd

        if self.state == 'CAMERA_SAFE_BACKUP_CHECK':
            return self._camera_safe_backup_check(now)

        if self.state == 'RECOVER_BACKUP':
            elapsed = now - self.recover_phase_start
            stop_end = self.recover_stop_time_sec
            backup_end = stop_end + self.recover_backup_time_sec
            pause_end = backup_end + self.recover_post_backup_stop_time_sec
            if elapsed < stop_end:
                return cmd
            if elapsed < backup_end:
                if self._rear_depth_allows_backup(now):
                    cmd.linear.x = _clamp(self.recover_backup_linear, -0.07, -0.04)
                return cmd
            if elapsed < pause_end:
                return cmd
            self._enter_recover_turn(now)
            return self._recover_state_cmd(now)

        if self.state == 'RECOVER_TURN':
            elapsed = now - self.recover_phase_start
            if elapsed >= self.recover_turn_time_sec:
                self.state = 'RECOVER_PROBE_FORWARD'
                self.recover_phase_start = now
                return cmd
            cmd.angular.z = _clamp(self.recover_angular * self.recover_direction,
                                   -self.max_angular, self.max_angular)
            return cmd

        if self.state == 'RECOVER_PROBE_FORWARD':
            elapsed = now - self.recover_phase_start
            if elapsed >= self.recover_forward_time_sec:
                self._finish_recover()
                return cmd
            if self.latest_sectors and self.latest_sectors['front']['min'] > self.recover_forward_clearance_m:
                selected = self._select_direction(self.latest_sectors)
                if selected['score'] > -1.0:
                    cmd.linear.x = _clamp(self.recover_forward_linear, 0.04, 0.07)
            return cmd

        return cmd

    def _camera_safe_backup_check(self, now):
        cmd = Twist()
        if self._rear_depth_allows_backup(now):
            self.state = 'RECOVER_BACKUP'
            self.recover_phase_start = now
            self.recover_direction = self._choose_recover_direction(self.held_direction)
            self.direction_switch_times.clear()
            self.get_logger().warn('rear depth safe; entering RECOVER_BACKUP')
            return cmd

        self._report_no_depth_if_needed(now)
        self._enter_recover_turn(now)
        return cmd

    def _enter_recover_turn(self, now):
        self.state = 'RECOVER_TURN'
        self.recover_phase_start = now
        self.recover_direction = self._choose_recover_direction(self.held_direction)
        self.direction_switch_times.clear()
        self.last_direction_switch_time = -999.0
        self.get_logger().warn('entering RECOVER_TURN')

    def _rear_depth_allows_backup(self, now):
        if self.latest_rear_depth is None or now - self.last_depth_time > self.depth_timeout_sec:
            return False
        depth = self.latest_rear_depth
        return (
            depth['rear_valid_ratio'] > self.rear_valid_ratio_min
            and depth['rear_center_depth'] > self.rear_center_depth_min
            and depth['rear_min_depth'] > self.rear_min_depth_min
            and not depth['rear_blocked']
        )

    def _rear_depth_features(self, image_msg):
        try:
            depth = self._depth_image_to_meters(image_msg)
        except (ValueError, TypeError) as exc:
            self.get_logger().warn('invalid depth image on %s: %s' % (self.depth_topic, exc))
            return None
        if depth.size == 0:
            return None

        height, width = depth.shape
        center = depth[:, width // 3: 2 * width // 3]
        left = depth[:, :width // 3]
        right = depth[:, 2 * width // 3:]
        valid = np.isfinite(depth) & (depth > 0.05) & (depth < 8.0)
        if not np.any(valid):
            return {
                'rear_min_depth': 0.0,
                'rear_center_depth': 0.0,
                'rear_left_depth': 0.0,
                'rear_right_depth': 0.0,
                'rear_valid_ratio': 0.0,
                'rear_blocked': True,
            }

        def region_median(region):
            mask = np.isfinite(region) & (region > 0.05) & (region < 8.0)
            if not np.any(mask):
                return 0.0
            return float(np.median(region[mask]))

        valid_values = depth[valid]
        rear_min = float(np.percentile(valid_values, 10))
        center_depth = region_median(center)
        left_depth = region_median(left)
        right_depth = region_median(right)
        valid_ratio = float(np.count_nonzero(valid)) / float(depth.size)
        rear_blocked = (
            valid_ratio <= self.rear_valid_ratio_min
            or center_depth <= self.rear_center_depth_min
            or rear_min <= self.rear_min_depth_min
        )
        return {
            'rear_min_depth': rear_min,
            'rear_center_depth': center_depth,
            'rear_left_depth': left_depth,
            'rear_right_depth': right_depth,
            'rear_valid_ratio': valid_ratio,
            'rear_blocked': rear_blocked,
        }

    @staticmethod
    def _depth_image_to_meters(image_msg):
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = str(image_msg.encoding).upper()
        if height <= 0 or width <= 0:
            raise ValueError('empty depth dimensions')
        if encoding in ('16UC1', 'MONO16'):
            row_words = max(int(image_msg.step) // 2, width)
            arr = np.frombuffer(image_msg.data, dtype=np.uint16).reshape((height, row_words))[:, :width]
            arr = arr.astype(np.float32)
            return arr * 0.001
        if encoding in ('32FC1',):
            row_floats = max(int(image_msg.step) // 4, width)
            return np.frombuffer(image_msg.data, dtype=np.float32).reshape((height, row_floats))[:, :width].astype(np.float32)
        raise ValueError('unsupported encoding %s' % image_msg.encoding)

    def _log_depth_topics_once(self):
        if self.reported_depth_topics:
            return
        self.reported_depth_topics = True
        topics = []
        for name, types in self.get_topic_names_and_types():
            haystack = ' '.join([name] + list(types)).lower()
            if 'image' in haystack or 'depth' in haystack or 'camera' in haystack:
                topics.append('%s [%s]' % (name, ','.join(types)))
        if topics:
            self.get_logger().info('available image/depth topics: %s' % '; '.join(sorted(topics)))
        else:
            self.get_logger().warn(
                'no image/depth topics currently visible; CAMERA_SAFE_BACKUP_CHECK will not back up blindly')

    def _report_no_depth_if_needed(self, now):
        if self.reported_no_depth:
            return
        if self.latest_rear_depth is None:
            self.get_logger().warn(
                'no valid rear depth received on %s; skipping backup and using turn-only recovery'
                % self.depth_topic)
        elif now - self.last_depth_time > self.depth_timeout_sec:
            self.get_logger().warn(
                'rear depth on %s is stale; skipping backup and using turn-only recovery'
                % self.depth_topic)
        else:
            self.get_logger().warn(
                'rear depth is blocked/invalid %s; skipping backup and using turn-only recovery'
                % json.dumps(self.latest_rear_depth, sort_keys=True))
        self.reported_no_depth = True

    def _enter_recover(self, now, heading):
        self.state = 'RECOVER_BACKUP' if self._rear_depth_allows_backup(now) else 'RECOVER_TURN'
        self.recover_start = now
        self.recover_until = now + self.recover_sequence_duration
        self.recover_direction = self._choose_recover_direction(heading)
        self.turning_since = None
        self.stuck_started_at = None
        self.recover_phase_start = now
        self.direction_switch_times.clear()
        if self.state == 'RECOVER_TURN':
            self._report_no_depth_if_needed(now)
        self.get_logger().warn('entering %s maneuver' % self.state)

    def _recover_cmd(self, now):
        elapsed = now - self.recover_start
        cmd = Twist()
        stop_end = self.recover_stop_time_sec
        backup_end = stop_end + self.recover_backup_time_sec
        turn_end = backup_end + self.recover_turn_time_sec
        if elapsed < stop_end:
            return cmd
        if elapsed < backup_end:
            if self._rear_depth_allows_backup(now):
                cmd.linear.x = self.recover_backup_linear
            return cmd
        if elapsed < turn_end:
            angular = _clamp(self.recover_angular * self.recover_direction,
                             -self.max_angular, self.max_angular)
            if self.latest_sectors:
                angular = self._apply_corner_turn_limit(
                    angular,
                    self.latest_sectors['front_left'],
                    self.latest_sectors['front_right'],
                )
            cmd.angular.z = angular
            return cmd
        if self.latest_sectors and self.latest_sectors['front']['min'] >= self.front_stop_clearance_m:
            cmd.linear.x = self.recover_forward_linear
        return cmd

    def _finish_recover(self):
        self.state = 'CRUISE'
        self.recover_until = 0.0
        self.turning_since = None
        self.direction_switch_times.clear()
        self.cmd_history.clear()
        self.stuck_score = 0.0

    def _choose_recover_direction(self, fallback_heading):
        if self.latest_sectors is None:
            return math.copysign(1.0, fallback_heading or self.held_direction or 1.0)
        left_score = self.latest_sectors['left']['score'] + self.latest_sectors['front_left']['score']
        right_score = self.latest_sectors['right']['score'] + self.latest_sectors['front_right']['score']
        if self.latest_sectors['front_left']['min'] < self.side_stop_clearance_m:
            left_score -= 5.0
        if self.latest_sectors['front_right']['min'] < self.side_stop_clearance_m:
            right_score -= 5.0
        if abs(left_score - right_score) > 0.1:
            return 1.0 if left_score > right_score else -1.0
        return math.copysign(1.0, fallback_heading or self.held_direction or 1.0)

    def _update_stuck_score(self, now, selected, cmd):
        self.cmd_history.append((now, float(cmd.linear.x), float(cmd.angular.z), float(selected['heading'])))
        while self.cmd_history and now - self.cmd_history[0][0] > self.stuck_window_sec:
            self.cmd_history.popleft()

        obstacle_score = self._obstacle_stuck_score()
        reversal_score = self._angular_reversal_score()
        switch_score = _clamp(len(self.direction_switch_times) / 4.0, 0.0, 1.0)
        progress_score = self._odom_progress_score()
        speed_score = self._low_speed_score(cmd)

        # Additive extension of the older turn-in-place trigger: no single weak signal forces RECOVER.
        self.stuck_score = _clamp(
            0.32 * obstacle_score + 0.24 * reversal_score + 0.18 * switch_score
            + 0.16 * progress_score + 0.10 * speed_score,
            0.0,
            1.0,
        )
        self._publish_stuck_status(now, obstacle_score, reversal_score, switch_score,
                                   progress_score, speed_score)

    def _obstacle_stuck_score(self):
        if self.latest_sectors is None:
            return 0.0
        front = self.latest_sectors['front']['min']
        side = min(self.latest_sectors['front_left']['min'], self.latest_sectors['front_right']['min'])
        front_risk = _clamp((self.safe_distance - front) / max(self.safe_distance, 0.05), 0.0, 1.0)
        side_risk = _clamp((self.side_stop_clearance_m - side) / max(self.side_stop_clearance_m, 0.05),
                           0.0, 1.0)
        return _clamp(0.7 * front_risk + 0.3 * side_risk, 0.0, 1.0)

    def _angular_reversal_score(self):
        signs = []
        for _, _, angular, _ in self.cmd_history:
            if abs(angular) > 0.18:
                signs.append(1 if angular > 0.0 else -1)
        reversals = sum(1 for prev, cur in zip(signs, signs[1:]) if prev != cur)
        return _clamp(reversals / 3.0, 0.0, 1.0)

    def _odom_progress_score(self):
        if len(self.odom_history) < 2:
            return 0.0
        start = self.odom_history[0]
        end = self.odom_history[-1]
        distance = math.hypot(end[1] - start[1], end[2] - start[2])
        commanded_motion = any(abs(item[1]) > 0.04 or abs(item[2]) > 0.25 for item in self.cmd_history)
        if not commanded_motion:
            return 0.0
        return _clamp((self.min_progress_m - distance) / max(self.min_progress_m, 0.01), 0.0, 1.0)

    def _low_speed_score(self, cmd):
        if not self.odom_history:
            return 0.0
        measured = abs(self.odom_history[-1][3])
        expected = abs(float(cmd.linear.x))
        if expected < self.low_speed_error_threshold:
            return 0.0
        return _clamp((expected - measured) / max(expected, 0.01), 0.0, 1.0)

    def _should_enter_recover(self, now):
        if self.stuck_score < self.stuck_score_threshold:
            self.stuck_started_at = None
            return False
        if self.stuck_started_at is None:
            self.stuck_started_at = now
            return False
        return now - self.stuck_started_at >= self.stuck_score_hold_sec

    def _publish_stuck_status(self, now, obstacle, reversal, switch, progress, speed):
        payload = {
            'stamp': round(now, 3),
            'state': self.state,
            'stuck_score': round(self.stuck_score, 3),
            'obstacle_score': round(obstacle, 3),
            'reversal_score': round(reversal, 3),
            'direction_switch_score': round(switch, 3),
            'progress_score': round(progress, 3),
            'speed_score': round(speed, 3),
            'recover_active': self.state == 'RECOVER' or self.state.startswith('RECOVER'),
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.stuck_status_pub.publish(message)

    def _payload_has_threat(self, payload):
        if isinstance(payload, dict) and 'detections' in payload:
            return any(self._detection_is_threat(item) for item in payload.get('detections', []))
        if isinstance(payload, list):
            return any(self._detection_is_threat(item) for item in payload)
        return self._detection_is_threat(payload)

    def _detection_is_threat(self, detection):
        if not isinstance(detection, dict) or not bool(detection.get('visible', False)):
            return False
        class_name = str(detection.get('class_name', '')).strip().lower()
        confidence = float(detection.get('conf', 0.0))
        return class_name in ('traffic_cone', 'yellow_car') and confidence >= self.threat_confidence

    def _publish_cmd(self, cmd):
        # Upper-level smoothing only; wheel-speed feedback is handled below this ROS2 layer.
        now = _now_seconds(self)
        dt = _clamp(now - self.last_cmd_time, 0.001, 0.2)
        self.last_cmd_time = now

        linear = _clamp(float(cmd.linear.x), -abs(self.recover_backup_linear), self.escape_max_linear)
        angular = _clamp(float(cmd.angular.z), -self.max_angular, self.max_angular)
        linear_step = self.max_linear_accel_mps2 * dt
        angular_step = self.max_angular_accel_rps2 * dt

        if abs(linear) > abs(self.last_cmd.linear.x):
            linear = _clamp(linear, self.last_cmd.linear.x - linear_step,
                            self.last_cmd.linear.x + linear_step)
        if abs(angular) > abs(self.last_cmd.angular.z):
            angular = _clamp(angular, self.last_cmd.angular.z - angular_step,
                             self.last_cmd.angular.z + angular_step)

        smoothed = Twist()
        smoothed.linear.x = linear
        smoothed.angular.z = angular
        self.last_cmd = smoothed
        self.cmd_pub.publish(smoothed)

    def _save_lidar_visualization(self, sectors, selected):
        image = np.full((420, 520, 3), 245, dtype=np.uint8)
        origin = (260, 360)
        max_radius = 260.0
        scale = max_radius / 2.5
        colors = {
            'left': (80, 170, 80),
            'front_left': (100, 190, 100),
            'front': (80, 190, 220),
            'front_right': (100, 160, 230),
            'right': (90, 130, 220),
            'rear': (170, 170, 170),
        }
        for name, sector in sectors.items():
            radius = int(_clamp(sector['p70'] * scale, 20, max_radius))
            low, high = sector['bounds']
            points = [origin]
            for angle in np.linspace(low, high, 18):
                x = int(origin[0] + radius * math.sin(angle))
                y = int(origin[1] - radius * math.cos(angle))
                points.append((x, y))
            cv2.fillPoly(image, [np.asarray(points, dtype=np.int32)], colors[name])
            mid_x = int(origin[0] + (radius + 18) * math.sin(sector['heading']))
            mid_y = int(origin[1] - (radius + 18) * math.cos(sector['heading']))
            cv2.putText(image, name, (mid_x - 42, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (30, 30, 30), 1, cv2.LINE_AA)

        arrow_len = 145
        end = (
            int(origin[0] + arrow_len * math.sin(selected['heading'])),
            int(origin[1] - arrow_len * math.cos(selected['heading'])),
        )
        cv2.arrowedLine(image, origin, end, (0, 0, 220), 4, tipLength=0.18)
        cv2.circle(image, origin, 6, (30, 30, 30), -1)
        cv2.putText(image, 'selected: %s' % selected['name'], (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(image, 'front min %.2fm safe %.2fm' % (sectors['front']['min'], self.safe_distance),
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)
        path = os.path.join(self.artifacts_dir, 'first_lidar_freespace.png')
        cv2.imwrite(path, image)
        self.get_logger().info('saved first LiDAR free-space visualization: %s' % path)

    def stop_robot(self):
        try:
            if rclpy.ok():
                self.last_cmd = Twist()
                self.cmd_pub.publish(Twist())
        except BaseException:
            pass

    def destroy_node(self):
        try:
            for _ in range(3):
                self.stop_robot()
                time.sleep(0.03)
        except BaseException:
            pass
        try:
            super().destroy_node()
        except BaseException:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PlannerControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error('planner_controller_node crashed: %s' % exc)
        raise
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
