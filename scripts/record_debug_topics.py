#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class DebugRecorder(Node):
    def __init__(self, root):
        super().__init__('phantom_debug_topic_recorder')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.create_subscription(LaserScan, '/scan', lambda msg: self._scan('scan.txt', msg), 10)
        self.create_subscription(String, '/nav/front_free_space', lambda msg: self._string('front_free_space.txt', msg), 10)
        self.create_subscription(String, '/nav/rear_risk', lambda msg: self._string('rear_risk.txt', msg), 10)
        self.create_subscription(String, '/det/detections', lambda msg: self._string('detections.txt', msg), 10)
        self.create_subscription(String, '/debug/planner_state', lambda msg: self._string('planner_state.txt', msg), 10)
        self.create_subscription(String, '/debug/safety_decision', lambda msg: self._string('safety_decision.txt', msg), 10)
        self.create_subscription(Twist, '/cmd_vel_raw', lambda msg: self._twist('cmd_vel_raw.txt', msg), 10)
        self.create_subscription(Twist, '/phantom/disabled_cmd_vel', lambda msg: self._twist('disabled_cmd_vel.txt', msg), 10)
        self.create_subscription(Twist, '/controller/cmd_vel', lambda msg: self._twist('controller_cmd_vel.txt', msg), 10)
        self.create_subscription(Odometry, '/odom', lambda msg: self._odom('odom.txt', msg), 10)
        self.create_subscription(Odometry, '/odom_raw', lambda msg: self._odom('odom_raw.txt', msg), 10)

    def _file(self, name):
        if name not in self.files:
            self.files[name] = (self.root / name).open('a', encoding='utf-8')
        return self.files[name]

    def _write(self, name, payload):
        if not isinstance(payload, dict):
            payload = {'data': payload}
        payload.setdefault('record_time', round(time.time(), 6))
        handle = self._file(name)
        handle.write(json.dumps(payload, sort_keys=True) + '\n')
        handle.flush()

    def _string(self, name, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {'data': msg.data}
        self._write(name, payload)

    def _twist(self, name, msg):
        self._write(name, {
            'linear_x': float(msg.linear.x),
            'linear_y': float(msg.linear.y),
            'angular_z': float(msg.angular.z),
        })

    def _odom(self, name, msg):
        self._write(name, {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'linear_x': float(msg.twist.twist.linear.x),
            'linear_y': float(msg.twist.twist.linear.y),
            'angular_z': float(msg.twist.twist.angular.z),
        })

    def _scan(self, name, msg):
        values = [float(v) for v in msg.ranges if v > 0.0]
        self._write(name, {
            'count': len(msg.ranges),
            'valid_count': len(values),
            'min_range': min(values) if values else None,
            'angle_min': float(msg.angle_min),
            'angle_increment': float(msg.angle_increment),
            'range_min': float(msg.range_min),
            'range_max': float(msg.range_max),
        })

    def close(self):
        for handle in self.files.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('artifact_dir')
    parser.add_argument('--duration', type=float, default=12.0)
    args = parser.parse_args()
    rclpy.init()
    node = DebugRecorder(args.artifact_dir)
    deadline = time.monotonic() + max(args.duration, 0.0)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
