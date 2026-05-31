import math
import random

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


# ===== PHANTOM Sensor Bridge =====
# FAKE 模式：本地模拟数据
# REAL 模式：接入 JetAuto 硬件
#
# 修改位置：
# - 相机输入：/camera/rgb/image_raw
# - 雷达输入：/scan
#
# 输出接口保持不变：
# /perception/image_raw
# /perception/scan_or_depth
#
# 下游节点无需修改
mode = "FAKE"


class SensorBridgeNode(Node):
    """Bridge fake or JetAuto sensor inputs to stable perception topics."""

    def __init__(self):
        super().__init__('sensor_bridge_node')

        self.declare_parameter('mode', mode)
        self.declare_parameter('image_topic', '/perception/image_raw')
        self.declare_parameter('scan_topic', '/perception/scan_or_depth')
        self.declare_parameter('real_image_topic', '/camera/color/image_raw')
        self.declare_parameter('real_scan_topic', '/scan')
        self.declare_parameter('camera_frame_id', 'camera_link')
        self.declare_parameter('scan_frame_id', 'base_link')
        self.declare_parameter('publish_rate_hz', 10.0)

        self.mode = str(self.get_parameter('mode').value).strip().upper()
        self.image_topic = self.get_parameter('image_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.real_image_topic = self.get_parameter('real_image_topic').value
        self.real_scan_topic = self.get_parameter('real_scan_topic').value
        self.camera_frame_id = self.get_parameter('camera_frame_id').value
        self.scan_frame_id = self.get_parameter('scan_frame_id').value
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        if self.mode not in ('FAKE', 'REAL'):
            self.get_logger().warn('unsupported mode %s, fallback to FAKE' % self.mode)
            self.mode = 'FAKE'

        self.image_width = 640
        self.image_height = 480
        self.tick = 0
        self.rng = random.Random(42)
        self.timer_period = 1.0 / max(publish_rate_hz, 0.1)

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)

        self.get_logger().info('sensor_bridge_node started')
        if self.mode == 'REAL':
            self.image_sub = self.create_subscription(
                Image,
                self.real_image_topic,
                self.real_image_callback,
                10,
            )
            self.scan_sub = self.create_subscription(
                LaserScan,
                self.real_scan_topic,
                self.real_scan_callback,
                10,
            )
            self.get_logger().info(
                'REAL mode: forwarding %s -> %s, %s -> %s'
                % (
                    self.real_image_topic,
                    self.image_topic,
                    self.real_scan_topic,
                    self.scan_topic,
                )
            )
        else:
            self.timer = self.create_timer(self.timer_period, self.publish_fake_data)
            self.get_logger().info('FAKE mode: publishing fake camera and lidar')

    def real_image_callback(self, image):
        self.image_pub.publish(image)

    def real_scan_callback(self, scan):
        self.scan_pub.publish(scan)

    def publish_fake_data(self):
        stamp = self.get_clock().now().to_msg()
        sim_time = self.tick * self.timer_period

        self.publish_fake_camera(stamp, sim_time)
        self.publish_fake_lidar(stamp, sim_time)

        self.tick += 1

    def publish_fake_camera(self, stamp, sim_time):
        frame = np.full(
            (self.image_height, self.image_width, 3),
            (38, 38, 38),
            dtype=np.uint8,
        )

        car_center_x = int(self.image_width * 0.5 + 150.0 * math.sin(sim_time * 0.7))
        car_center_y = int(self.image_height * 0.63)
        car_w = 120
        car_h = 70
        cv2.rectangle(
            frame,
            (car_center_x - car_w // 2, car_center_y - car_h // 2),
            (car_center_x + car_w // 2, car_center_y + car_h // 2),
            (0, 255, 255),
            thickness=-1,
        )

        cv2.rectangle(
            frame,
            (int(self.image_width * 0.72), int(self.image_height * 0.10)),
            (int(self.image_width * 0.92), int(self.image_height * 0.33)),
            (0, 220, 0),
            thickness=-1,
        )

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = self.camera_frame_id
        image.height = self.image_height
        image.width = self.image_width
        image.encoding = 'bgr8'
        image.is_bigendian = 0
        image.step = self.image_width * 3
        image.data = frame.tobytes()
        self.image_pub.publish(image)

    def publish_fake_lidar(self, stamp, sim_time):
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.scan_frame_id
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = math.pi / 180.0
        scan.time_increment = self.timer_period / 361.0
        scan.scan_time = self.timer_period
        scan.range_min = 0.05
        scan.range_max = 5.0

        ranges = []
        for index in range(361):
            angle = scan.angle_min + index * scan.angle_increment
            distance = self._field_distance(angle, sim_time)
            noise = self.rng.uniform(-0.03, 0.03)
            ranges.append(max(scan.range_min, min(distance + noise, scan.range_max)))

        scan.ranges = ranges
        scan.intensities = []
        self.scan_pub.publish(scan)

    @staticmethod
    def _field_distance(angle, sim_time):
        angle_deg = math.degrees(angle)
        front_obstacle = 1.0 + 0.12 * math.sin(sim_time * 0.8)

        if abs(angle_deg) <= 12.0:
            return front_obstacle
        if 60.0 <= angle_deg <= 120.0:
            return 2.5
        if -120.0 <= angle_deg <= -60.0:
            return 1.2
        if abs(angle_deg) >= 150.0:
            return 3.0
        if angle_deg > 0.0:
            return 2.0
        return 1.5


def main(args=None):
    rclpy.init(args=args)
    node = SensorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
