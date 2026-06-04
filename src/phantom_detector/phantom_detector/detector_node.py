import json
import math
import os

import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:  # Allows local non-ROS YOLO tests to import class mapping helpers.
    rclpy = None
    Node = object
    qos_profile_sensor_data = 10
    Image = object

    class String:  # pragma: no cover - only used when ROS messages are unavailable.
        def __init__(self):
            self.data = ''

try:
    import cv2
except ImportError:  # pragma: no cover - runtime fallback for minimal ROS installs.
    cv2 = None

try:
    from cv_bridge import CvBridge
except ImportError:  # pragma: no cover
    CvBridge = None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _stamp_seconds(message, node):
    stamp = getattr(getattr(message, 'header', None), 'stamp', None)
    if stamp is not None:
        seconds = float(getattr(stamp, 'sec', 0)) + float(getattr(stamp, 'nanosec', 0)) * 1e-9
        if seconds > 0.0:
            return seconds
    return node.get_clock().now().nanoseconds * 1e-9


def _empty_detection(class_name='', stamp=0.0):
    return {
        'visible': False,
        'class_name': class_name,
        'conf': 0.0,
        'cx': 0.0,
        'cy': 0.0,
        'w': 0.0,
        'h': 0.0,
        'depth': None,
        'stamp': round(float(stamp), 6),
        'bbox_center_x': 0.0,
        'bbox_center_y': 0.0,
        'bbox_w': 0.0,
        'bbox_h': 0.0,
    }


CLASS_NAME_ALIASES = {
    'cone': 'traffic_cone',
    'traffic cone': 'traffic_cone',
    'traffic_cone': 'traffic_cone',
    'yellow car': 'yellow_car',
    'yellow-car': 'yellow_car',
    'yellow_car': 'yellow_car',
    'car_yellow': 'yellow_car',
    'exit': 'exit',
}


def normalize_class_name(class_name):
    key = str(class_name).strip().lower().replace('-', ' ').replace('_', ' ')
    key = ' '.join(key.split())
    return CLASS_NAME_ALIASES.get(key, str(class_name).strip())


class DetectorNode(Node):
    """Rear RGB detector with YOLO, color-debug, and empty fallback modes."""

    def __init__(self):
        super().__init__('detector_node')

        self.declare_parameter('mode', 'YOLO')
        self.declare_parameter(
            'model_path',
            '/home/ubuntu/phantom_ws/models/yolo/phantom_cone_yellow_random200_best.pt',
        )
        self.declare_parameter('image_topic', '/usb_cam/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('subscribe_depth', True)
        self.declare_parameter('yellow_topic', '/det/yellow_boxes')
        self.declare_parameter('exit_topic', '/det/exit_boxes')
        self.declare_parameter('detections_topic', '/det/detections')
        self.declare_parameter('artifacts_dir', '/home/ubuntu/phantom_ws/artifacts')
        self.declare_parameter('horizontal_fov_rad', 1.0472)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show', False)
        self.declare_parameter('min_area', 250.0)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('yellow_classes', ['yellow_car', 'traffic_cone'])
        self.declare_parameter('exit_classes', ['exit', 'door', 'traffic light', 'stop sign'])

        self.mode = str(self.get_parameter('mode').value).strip().upper()
        if self.mode == 'ROS':
            self.mode = 'YOLO'
        if self.mode not in ('YOLO', 'COLOR_DEBUG'):
            self.get_logger().warn('unsupported detector mode %s, fallback to COLOR_DEBUG' % self.mode)
            self.mode = 'COLOR_DEBUG'

        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.subscribe_depth = bool(self.get_parameter('subscribe_depth').value)
        self.yellow_topic = str(self.get_parameter('yellow_topic').value)
        self.exit_topic = str(self.get_parameter('exit_topic').value)
        self.detections_topic = str(self.get_parameter('detections_topic').value)
        self.artifacts_dir = os.path.expanduser(str(self.get_parameter('artifacts_dir').value))
        self.horizontal_fov_rad = float(self.get_parameter('horizontal_fov_rad').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.show = _as_bool(self.get_parameter('show').value)
        self.min_area = float(self.get_parameter('min_area').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.yellow_classes = set(self.get_parameter('yellow_classes').value)
        self.exit_classes = set(self.get_parameter('exit_classes').value)

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.bridge = CvBridge() if CvBridge is not None else None
        self.model = self._load_model() if self.mode == 'YOLO' else None
        self.saved_detection_image = False
        self.latest_depth = None
        self.latest_depth_time = -999.0

        self.yellow_pub = self.create_publisher(String, self.yellow_topic, 10)
        self.exit_pub = self.create_publisher(String, self.exit_topic, 10)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.depth_sub = None
        if self.subscribe_depth:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_callback,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            'detector_node started in %s mode: image %s -> %s, depth %s'
            % (self.mode, self.image_topic, self.detections_topic, self.depth_topic if self.subscribe_depth else 'disabled')
        )

    def depth_callback(self, image_msg):
        try:
            self.latest_depth = self._depth_image_to_meters(image_msg)
        except (TypeError, ValueError) as exc:
            self.get_logger().warn('detector depth conversion failed on %s: %s' % (self.depth_topic, exc))
            self.latest_depth = None
            return
        self.latest_depth_time = self.get_clock().now().nanoseconds * 1e-9

    def image_callback(self, image_msg):
        stamp = _stamp_seconds(image_msg, self)
        frame = self._ros_image_to_bgr(image_msg)
        if frame is None:
            self._publish_all([], _empty_detection('yellow_car', stamp), _empty_detection('exit', stamp))
            return

        if self.mode == 'YOLO':
            detections = self._detect_yolo(frame, stamp)
        else:
            detections = self._detect_color_debug(frame, stamp)
        detections = self._attach_depths(detections)

        yellow = self._select_detection(detections, self.yellow_classes)
        exit_box = self._select_detection(detections, self.exit_classes)
        if not exit_box['visible']:
            exit_box = _empty_detection('exit', stamp)
            exit_box['bearing'] = 0.0
        else:
            exit_box['class_name'] = 'exit'
            exit_box['bearing'] = self._bearing_from_center(exit_box['cx'])

        if not self.saved_detection_image:
            self._save_detection_visualization(frame, detections)
            self.saved_detection_image = True

        self._publish_all(detections, yellow, exit_box)

    def _ros_image_to_bgr(self, image_msg):
        if self.bridge is not None:
            try:
                return self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            except Exception as exc:
                self.get_logger().warn('cv_bridge conversion failed: %s' % exc)

        try:
            raw = np.frombuffer(image_msg.data, dtype=np.uint8)
            height = int(image_msg.height)
            width = int(image_msg.width)
            encoding = image_msg.encoding.lower()
            if encoding == 'bgr8':
                return np.ascontiguousarray(raw.reshape((height, width, 3)))
            if encoding == 'rgb8':
                frame = raw.reshape((height, width, 3))
                return np.ascontiguousarray(frame[:, :, ::-1])
            if encoding in ('bgra8', 'rgba8'):
                frame = raw.reshape((height, width, 4))[:, :, :3]
                if encoding == 'rgba8':
                    frame = frame[:, :, ::-1]
                return np.ascontiguousarray(frame)
            if encoding == 'mono8' and cv2 is not None:
                frame = raw.reshape((height, width))
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except ValueError as exc:
            self.get_logger().warn('invalid image buffer: %s' % exc)
            return None

        self.get_logger().warn('unsupported image encoding: %s' % image_msg.encoding)
        return None

    def _detect_color_debug(self, frame, stamp):
        if cv2 is None:
            self.get_logger().warn('OpenCV unavailable; COLOR_DEBUG detector will publish empty detections')
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(
            hsv,
            np.array([20, 80, 80], dtype=np.uint8),
            np.array([35, 255, 255], dtype=np.uint8),
        )
        green_mask = cv2.inRange(
            hsv,
            np.array([45, 70, 70], dtype=np.uint8),
            np.array([90, 255, 255], dtype=np.uint8),
        )
        detections = []
        yellow = self._mask_to_detection(yellow_mask, frame.shape, 'yellow_car', 0.90, stamp)
        if yellow['visible']:
            detections.append(yellow)
        exit_box = self._mask_to_detection(green_mask, frame.shape, 'exit', 0.90, stamp)
        if exit_box['visible']:
            detections.append(exit_box)
        return detections

    def _detect_yolo(self, frame, stamp):
        if self.model is None:
            return []
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                show=self.show,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().warn('YOLO predict failed: %s' % exc)
            return []
        if not results:
            return []
        return self._parse_yolo_result(results[0], stamp)

    def _load_model(self):
        model_path = self._resolve_model_path()
        if not model_path:
            self.get_logger().warn('YOLO model not found, publishing empty fallback detections: %s' % self.model_path)
            return None
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.get_logger().warn('ultralytics is not installed, publishing empty fallback detections: %s' % exc)
            return None
        try:
            model = YOLO(model_path)
        except Exception as exc:
            self.get_logger().warn('failed to load YOLO model %s: %s' % (model_path, exc))
            return None
        self.model_path = model_path
        self.get_logger().info('loaded YOLO model: %s' % self.model_path)
        return model

    def _resolve_model_path(self):
        candidates = [self.model_path]
        model_name = os.path.basename(self.model_path)
        candidates.extend([
            os.path.join('/home/jetauto/phantom_ws/models/yolo', model_name),
            os.path.join('/home/ubuntu/phantom_ws/models/yolo', model_name),
            os.path.join(os.getcwd(), 'models', 'yolo', model_name),
            os.path.join(os.getcwd(), 'models', 'yolo', 'phantom_cone_yellow_random200_best.pt'),
        ])
        for path in candidates:
            path = os.path.expanduser(str(path))
            if os.path.exists(path):
                return path
        return None

    def _mask_to_detection(self, mask, frame_shape, class_name, confidence, stamp):
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _empty_detection(class_name, stamp)
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return _empty_detection(class_name, stamp)
        x, y, w, h = cv2.boundingRect(contour)
        image_height, image_width = frame_shape[:2]
        return self._detection_from_xyxy(
            class_name,
            confidence,
            x,
            y,
            x + w,
            y + h,
            image_width,
            image_height,
            stamp,
        )

    def _parse_yolo_result(self, result, stamp):
        detections = []
        names = getattr(result, 'names', {}) or {}
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return detections
        image_height, image_width = result.orig_shape[:2]
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = normalize_class_name(names.get(class_id, class_id))
            confidence = float(box.conf[0])
            x0, y0, x1, y1 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                self._detection_from_xyxy(
                    class_name,
                    confidence,
                    x0,
                    y0,
                    x1,
                    y1,
                    image_width,
                    image_height,
                    stamp,
                )
            )
        return detections

    @staticmethod
    def _detection_from_xyxy(class_name, confidence, x0, y0, x1, y1, image_width, image_height, stamp):
        cx = ((x0 + x1) * 0.5) / image_width
        cy = ((y0 + y1) * 0.5) / image_height
        width = (x1 - x0) / image_width
        height = (y1 - y0) / image_height
        return {
            'visible': True,
            'class_name': class_name,
            'conf': round(float(confidence), 3),
            'cx': round(cx, 4),
            'cy': round(cy, 4),
            'w': round(width, 4),
            'h': round(height, 4),
            'depth': None,
            'stamp': round(float(stamp), 6),
            'bbox_center_x': round(cx, 4),
            'bbox_center_y': round(cy, 4),
            'bbox_w': round(width, 4),
            'bbox_h': round(height, 4),
            'x0': round(x0 / image_width, 4),
            'y0': round(y0 / image_height, 4),
            'x1': round(x1 / image_width, 4),
            'y1': round(y1 / image_height, 4),
        }

    def _attach_depths(self, detections):
        now = self.get_clock().now().nanoseconds * 1e-9
        depth_available = self.latest_depth is not None and now - self.latest_depth_time <= 0.8
        for detection in detections:
            detection['depth'] = self._bbox_depth(detection) if depth_available else None
        return detections

    def _bbox_depth(self, detection):
        depth = self.latest_depth
        if depth is None or depth.size == 0:
            return None
        height, width = depth.shape
        cx = float(detection.get('cx', detection.get('bbox_center_x', 0.0)))
        cy = float(detection.get('cy', detection.get('bbox_center_y', 0.0)))
        bw = float(detection.get('w', detection.get('bbox_w', 0.0)))
        bh = float(detection.get('h', detection.get('bbox_h', 0.0)))
        x0 = int(max(0, round((cx - 0.2 * bw) * width)))
        x1 = int(min(width, round((cx + 0.2 * bw) * width)))
        y0 = int(max(0, round((cy - 0.2 * bh) * height)))
        y1 = int(min(height, round((cy + 0.2 * bh) * height)))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = depth[y0:y1, x0:x1]
        mask = np.isfinite(roi) & (roi > 0.0) & (roi <= self.max_depth_m)
        if not np.any(mask):
            return None
        return round(float(np.median(roi[mask])), 3)

    @staticmethod
    def _depth_image_to_meters(image_msg):
        height = int(image_msg.height)
        width = int(image_msg.width)
        encoding = str(image_msg.encoding).upper()
        if height <= 0 or width <= 0:
            raise ValueError('empty depth dimensions')
        if encoding in ('16UC1', 'MONO16'):
            row_values = max(int(image_msg.step) // 2, width)
            array = np.frombuffer(image_msg.data, dtype=np.uint16).reshape((height, row_values))[:, :width]
            return array.astype(np.float32) * 0.001
        if encoding == '32FC1':
            row_values = max(int(image_msg.step) // 4, width)
            array = np.frombuffer(image_msg.data, dtype=np.float32).reshape((height, row_values))[:, :width]
            return array.astype(np.float32)
        raise ValueError('unsupported encoding %s' % image_msg.encoding)

    def _select_detection(self, detections, class_names):
        matches = [
            detection for detection in detections
            if detection.get('visible', True) and normalize_class_name(detection.get('class_name')) in class_names
        ]
        if not matches:
            default_class = 'yellow_car' if class_names == self.yellow_classes else ''
            return _empty_detection(default_class)
        return max(matches, key=lambda detection: float(detection.get('conf', 0.0)))

    def _bearing_from_center(self, center_x):
        if center_x <= 0.0:
            return 0.0
        return round((float(center_x) - 0.5) * self.horizontal_fov_rad, 4)

    def _save_detection_visualization(self, frame, detections):
        if cv2 is None:
            return
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        for detection in detections:
            if not detection.get('visible', False):
                continue
            x0 = int(float(detection.get('x0', 0.0)) * width)
            y0 = int(float(detection.get('y0', 0.0)) * height)
            x1 = int(float(detection.get('x1', 0.0)) * width)
            y1 = int(float(detection.get('y1', 0.0)) * height)
            color = (0, 220, 255) if detection.get('class_name') in ('traffic_cone', 'yellow_car') else (0, 180, 0)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            label = '%s %.2f' % (detection.get('class_name', ''), float(detection.get('conf', 0.0)))
            cv2.putText(annotated, label, (max(0, x0), max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if not detections:
            cv2.putText(annotated, 'no detections', (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2, cv2.LINE_AA)
        path = os.path.join(self.artifacts_dir, 'first_yolo_detection.png')
        if cv2.imwrite(path, annotated):
            self.get_logger().info('saved first YOLO detection visualization: %s' % path)

    def _publish_all(self, detections, yellow, exit_box):
        self._publish_json(self.detections_pub, detections)
        self._publish_json(self.yellow_pub, yellow)
        self._publish_json(self.exit_pub, exit_box)

    @staticmethod
    def _publish_json(publisher, payload):
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        publisher.publish(message)


def main(args=None):
    if rclpy is None:
        raise RuntimeError('rclpy is required to run detector_node as a ROS2 node')
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
