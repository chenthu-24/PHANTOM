import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class LocalMockInputsNode(Node):
    def __init__(self):
        super().__init__('local_mock_inputs_node')
        self.declare_parameter('scan_topic', '/mock/scan')
        self.declare_parameter('depth_topic', '/mock/depth/image_raw')
        self.declare_parameter('image_topic', '/mock/usb_cam/image_raw')
        self.declare_parameter('odom_topic', '/mock/odom')
        self.scan_pub = self.create_publisher(LaserScan, str(self.get_parameter('scan_topic').value), 10)
        self.depth_pub = self.create_publisher(Image, str(self.get_parameter('depth_topic').value), 10)
        self.image_pub = self.create_publisher(Image, str(self.get_parameter('image_topic').value), 10)
        self.odom_pub = self.create_publisher(Odometry, str(self.get_parameter('odom_topic').value), 10)
        self.timer = self.create_timer(0.1, self.publish_inputs)
        self.get_logger().info('publishing isolated local mock scan/depth/rgb/odom inputs')

    def publish_inputs(self):
        now = self.get_clock().now().to_msg()
        scan = LaserScan()
        scan.header.stamp = now
        scan.header.frame_id = 'mock_lidar'
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = math.radians(1.0)
        scan.range_min = 0.05
        scan.range_max = 8.0
        scan.ranges = [1.2] * 360
        for index in range(165, 196):
            scan.ranges[index] = 1.5
        self.scan_pub.publish(scan)

        depth = Image()
        depth.header.stamp = now
        depth.header.frame_id = 'mock_rear_depth'
        depth.height = 80
        depth.width = 120
        depth.encoding = '32FC1'
        depth.is_bigendian = 0
        depth.step = depth.width * 4
        depth.data = (b'\x9a\x99\x99?' * depth.height * depth.width)  # float32 1.2 m, little endian.
        self.depth_pub.publish(depth)

        image = Image()
        image.header.stamp = now
        image.header.frame_id = 'mock_rear_rgb'
        image.height = 120
        image.width = 160
        image.encoding = 'bgr8'
        image.is_bigendian = 0
        image.step = image.width * 3
        data = bytearray(image.height * image.step)
        for y in range(35, 90):
            for x in range(60, 100):
                offset = y * image.step + x * 3
                data[offset:offset + 3] = bytes((0, 230, 230))
        image.data = bytes(data)
        self.image_pub.publish(image)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'mock_odom'
        odom.child_frame_id = 'mock_base'
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = LocalMockInputsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
