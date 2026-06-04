#!/usr/bin/env python3
import argparse
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class TimedRelay(Node):
    def __init__(self, source, dest):
        super().__init__('phantom_timed_cmd_vel_relay')
        self.latest = Twist()
        self.pub = self.create_publisher(Twist, dest, 10)
        self.sub = self.create_subscription(Twist, source, self._callback, 10)

    def _callback(self, msg):
        self.latest = msg
        self.pub.publish(msg)

    def publish_zero(self):
        zero = Twist()
        try:
            self.pub.publish(zero)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/phantom/disabled_cmd_vel')
    parser.add_argument('--dest', default='/controller/cmd_vel')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = TimedRelay(args.source, args.dest)
    end_time = time.monotonic() + max(args.duration, 0.0)
    try:
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(node, timeout_sec=0.05)
        for _ in range(2):
            node.publish_zero()
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.2)
    finally:
        try:
            node.publish_zero()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
