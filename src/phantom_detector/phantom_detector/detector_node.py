import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _empty_detection(class_name=''):
    return {
        'visible': False,
        'class_name': class_name,
        'conf': 0.0,
        'bbox_center_x': 0.0,
        'bbox_center_y': 0.0,
        'bbox_w': 0.0,
        'bbox_h': 0.0,
    }


class DetectorNode(Node):
    """Detect rear threats from ROS Image messages.

    mode=COLOR_DEBUG uses HSV thresholds for stable local fake-data testing.
    mode=YOLO loads the trained YOLO model and publishes JSON detections.
    """

    def __init__(self):
        super().__init__('detector_node')

        self.declare_parameter('mode', 'YOLO')
        self.declare_parameter(
            'model_path',
            '/home/ubuntu/phantom_ws/models/yolo/phantom_cone_yellow_random200_best.pt',
        )
        self.declare_parameter('image_topic', '/usb_cam/image_raw')
        self.declare_parameter('yellow_topic', '/det/yellow_boxes')
        self.declare_parameter('exit_topic', '/det/exit_boxes')
        self.declare_parameter('detections_topic', '/det/detections')
        self.declare_parameter('artifacts_dir', '/home/ubuntu/phantom_ws/artifacts')
        self.declare_parameter('horizontal_fov_rad', 1.0472)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('show', False)
        self.declare_parameter('min_area', 250.0)
        self.declare_parameter('yellow_classes', ['yellow_car', 'traffic_cone'])
        self.declare_parameter('exit_classes', ['door', 'traffic light', 'stop sign'])

        self.mode = str(self.get_parameter('mode').value).strip().upper()
        if self.mode == 'ROS':
            self.mode = 'YOLO'
        if self.mode not in ('YOLO', 'COLOR_DEBUG'):
            self.get_logger().warn('unsupported detector mode %s, fallback to COLOR_DEBUG' % self.mode)
            self.mode = 'COLOR_DEBUG'

        self.model_path = os.path.expanduser(str(self.get_parameter('model_path').value))
        self.image_topic = self.get_parameter('image_topic').value
        self.yellow_topic = self.get_parameter('yellow_topic').value
        self.exit_topic = self.get_parameter('exit_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.artifacts_dir = os.path.expanduser(str(self.get_parameter('artifacts_dir').value))
        self.horizontal_fov_rad = float(self.get_parameter('horizontal_fov_rad').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.conf = float(self.get_parameter('conf').value)
        self.show = _as_bool(self.get_parameter('show').value)
        self.min_area = float(self.get_parameter('min_area').value)
        self.yellow_classes = set(self.get_parameter('yellow_classes').value)
        self.exit_classes = set(self.get_parameter('exit_classes').value)
        self.saved_detection_image = False
        os.makedirs(self.artifacts_dir, exist_ok=True)

        self.bridge = CvBridge() if CvBridge is not None else None
        self.model = self._load_model() if self.mode == 'YOLO' else None

        self.yellow_pub = self.create_publisher(String, self.yellow_topic, 10)
        self.exit_pub = self.create_publisher(String, self.exit_topic, 10)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            'detector_node started in %s mode: %s -> %s, %s'
            % (self.mode, self.image_topic, self.yellow_topic, self.detections_topic)
        )

    def image_callback(self, image_msg):
        frame = self._ros_image_to_bgr(image_msg)
        if frame is None:
            self._publish_all([], _empty_detection('yellow_car'), self._exit_detection())
            return

        if self.mode == 'YOLO':
            detections, yellow, exit_box = self._detect_yolo(frame)
        else:
            detections, yellow, exit_box = self._detect_color_debug(frame)

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
            if encoding == 'mono8':
                frame = raw.reshape((height, width))
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except ValueError as exc:
            self.get_logger().warn('invalid image buffer: %s' % exc)
            return None

        self.get_logger().warn('unsupported image encoding: %s' % image_msg.encoding)
        return None

    def _detect_color_debug(self, frame):
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

        yellow = self._mask_to_detection(yellow_mask, frame.shape, 'yellow_car', 0.90)
        exit_box = self._mask_to_detection(green_mask, frame.shape, 'exit', 0.90)
        exit_box['bearing'] = self._bearing_from_center(exit_box['bbox_center_x'])
        detections = [item for item in (yellow, exit_box) if item['visible']]
        return detections, yellow, exit_box

    def _detect_yolo(self, frame):
        if self.model is None:
            return [], _empty_detection('yellow_car'), self._exit_detection()

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
            return [], _empty_detection('yellow_car'), self._exit_detection()

        if not results:
            return [], _empty_detection('yellow_car'), self._exit_detection()

        detections = self._parse_yolo_result(results[0])
        yellow = self._select_detection(detections, self.yellow_classes, fallback_to_best=True)
        if yellow['visible']:
            if yellow['class_name'] not in ('yellow_car', 'traffic_cone'):
                yellow['class_name'] = 'yellow_car'
        exit_box = self._select_detection(detections, self.exit_classes, fallback_to_best=False)
        if not exit_box['visible']:
            exit_box = self._exit_detection()
        else:
            exit_box['class_name'] = 'exit'
            exit_box['bearing'] = self._bearing_from_center(exit_box['bbox_center_x'])
        return detections, yellow, exit_box

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.get_logger().error('YOLO model not found: %s' % self.model_path)
            return None

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.get_logger().error('ultralytics is not installed: %s' % exc)
            return None

        try:
            model = YOLO(self.model_path)
        except Exception as exc:
            self.get_logger().error('failed to load YOLO model %s: %s' % (self.model_path, exc))
            return None

        self.get_logger().info('loaded YOLO model: %s' % self.model_path)
        return model

    def _mask_to_detection(self, mask, frame_shape, class_name, confidence):
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _empty_detection(class_name)

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < self.min_area:
            return _empty_detection(class_name)

        x, y, w, h = cv2.boundingRect(contour)
        image_height, image_width = frame_shape[:2]
        return {
            'visible': True,
            'class_name': class_name,
            'conf': confidence,
            'bbox_center_x': round((x + 0.5 * w) / image_width, 4),
            'bbox_center_y': round((y + 0.5 * h) / image_height, 4),
            'bbox_w': round(w / image_width, 4),
            'bbox_h': round(h / image_height, 4),
        }

    def _parse_yolo_result(self, result):
        detections = []
        names = getattr(result, 'names', {}) or {}
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return detections

        image_height, image_width = result.orig_shape[:2]
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = str(names.get(class_id, class_id))
            confidence = float(box.conf[0])
            x0, y0, x1, y1 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append({
                'visible': True,
                'class_name': class_name,
                'conf': round(confidence, 3),
                'bbox_center_x': round(((x0 + x1) * 0.5) / image_width, 4),
                'bbox_center_y': round(((y0 + y1) * 0.5) / image_height, 4),
                'bbox_w': round((x1 - x0) / image_width, 4),
                'bbox_h': round((y1 - y0) / image_height, 4),
                'x0': round(x0 / image_width, 4),
                'y0': round(y0 / image_height, 4),
                'x1': round(x1 / image_width, 4),
                'y1': round(y1 / image_height, 4),
            })
        return detections

    def _select_detection(self, detections, class_names, fallback_to_best):
        matches = [
            detection
            for detection in detections
            if detection['class_name'] in class_names
        ]
        if matches:
            return max(matches, key=lambda detection: detection['conf'])
        if fallback_to_best and detections:
            return max(detections, key=lambda detection: detection['conf'])
        return _empty_detection()

    def _exit_detection(self):
        detection = _empty_detection('exit')
        detection['bearing'] = 0.0
        return detection

    def _bearing_from_center(self, center_x):
        if center_x <= 0.0:
            return 0.0
        return round((float(center_x) - 0.5) * self.horizontal_fov_rad, 4)

    def _save_detection_visualization(self, frame, detections):
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        for detection in detections:
            if not detection.get('visible', False):
                continue
            x0 = int(float(detection.get('x0', 0.0)) * width)
            y0 = int(float(detection.get('y0', 0.0)) * height)
            x1 = int(float(detection.get('x1', 0.0)) * width)
            y1 = int(float(detection.get('y1', 0.0)) * height)
            if x1 <= x0 or y1 <= y0:
                cx = float(detection.get('bbox_center_x', 0.0)) * width
                cy = float(detection.get('bbox_center_y', 0.0)) * height
                bw = float(detection.get('bbox_w', 0.0)) * width
                bh = float(detection.get('bbox_h', 0.0)) * height
                x0 = int(cx - 0.5 * bw)
                y0 = int(cy - 0.5 * bh)
                x1 = int(cx + 0.5 * bw)
                y1 = int(cy + 0.5 * bh)
            color = (0, 220, 255) if detection.get('class_name') == 'traffic_cone' else (0, 180, 0)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            label = '%s %.2f' % (detection.get('class_name', ''), float(detection.get('conf', 0.0)))
            cv2.putText(annotated, label, (max(0, x0), max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        if not detections:
            cv2.putText(annotated, 'no YOLO detections', (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 220), 2, cv2.LINE_AA)

        path = os.path.join(self.artifacts_dir, 'first_yolo_detection.png')
        cv2.imwrite(path, annotated)
        self.get_logger().info('saved first YOLO detection visualization: %s' % path)

    def _publish_all(self, detections, yellow, exit_box):
        self._publish_json(self.detections_pub, {'detections': detections})
        self._publish_json(self.yellow_pub, yellow)
        self._publish_json(self.exit_pub, exit_box)

    @staticmethod
    def _publish_json(publisher, payload):
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except BaseException:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
