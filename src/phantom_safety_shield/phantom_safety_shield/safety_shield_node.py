import json
import math
import time

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # Allows local non-ROS chain tests to import safety helpers.
    rclpy = None
    Node = object

    class _Vector:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Twist:  # pragma: no cover - only used when ROS messages are unavailable.
        def __init__(self):
            self.linear = _Vector()
            self.angular = _Vector()

    class String:  # pragma: no cover
        def __init__(self):
            self.data = ''


def _clamp(value, lower, upper):
    return max(lower, min(float(value), upper))


def _as_float(payload, key, default):
    try:
        value = float(payload.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def apply_safety_shield(raw_cmd, front, rear=None, params=None):
    """Stateless local safety filter equivalent for debug chain injection tests."""
    values = {
        'front_hard_stop_m': 0.22,
        'front_soft_stop_m': 0.35,
        'front_slowdown_m': 0.50,
        'rear_hard_stop_m': 0.30,
        'rear_soft_stop_m': 0.55,
        'max_forward_vx': 0.32,
        'max_reverse_vx': -0.08,
        'max_wz': 0.75,
        'gap_escape_max_vx': 0.04,
    }
    if params:
        values.update(params)
    front = front or {}
    rear = rear or {}
    vx = float(raw_cmd.get('linear_x', 0.0))
    wz = float(raw_cmd.get('angular_z', 0.0))
    if not bool(front.get('valid', False)):
        return {'linear_x': 0.0, 'angular_z': 0.0}

    front_min = _as_float(front, 'front_min', _as_float(front, 'front_clearance', 0.0))
    front_unknown = bool(front.get('front_unknown', False))
    gap_escape_allowed = bool(front.get('gap_escape_allowed', False))
    if vx > 0.0 and front_unknown:
        vx = 0.0
        wz = _clamp(wz, -0.55, 0.55)
    elif vx > 0.0 and front_min > 0.0 and front_min < values['front_hard_stop_m']:
        if gap_escape_allowed:
            vx = min(vx, values['gap_escape_max_vx'])
            wz = _clamp(wz, -values['max_wz'], values['max_wz'])
        else:
            vx = 0.0
            wz = _clamp(wz, -0.55, 0.55)
    elif vx > 0.0 and values['front_hard_stop_m'] <= front_min < values['front_soft_stop_m']:
        vx = min(vx, 0.05)
        wz = _clamp(wz, -0.65, 0.65)
    elif vx > 0.0 and values['front_soft_stop_m'] <= front_min < values['front_slowdown_m']:
        vx = min(vx, 0.12)

    rear_valid = bool(rear.get('valid', False))
    if vx < 0.0:
        if not rear_valid:
            vx = 0.0
        else:
            rear_center = _as_float(rear, 'rear_center_min', 0.0)
            reverse_allowed = bool(rear.get('reverse_allowed', False))
            if rear_center <= 0.0 or rear_center < values['rear_hard_stop_m']:
                vx = 0.0
            elif values['rear_hard_stop_m'] <= rear_center < values['rear_soft_stop_m']:
                vx = max(vx, -0.03)
            elif reverse_allowed:
                vx = max(vx, -0.07)
            else:
                vx = 0.0

    return {
        'linear_x': round(_clamp(vx, values['max_reverse_vx'], values['max_forward_vx']), 4),
        'angular_z': round(_clamp(wz, -values['max_wz'], values['max_wz']), 4),
    }


class SafetyShieldNode(Node):
    def __init__(self):
        super().__init__('safety_shield_node')

        self.declare_parameter('raw_cmd_topic', '/cmd_vel_raw')
        self.declare_parameter('front_free_space_topic', '/nav/front_free_space')
        self.declare_parameter('rear_risk_topic', '/nav/rear_risk')
        self.declare_parameter('safe_cmd_topic', '/controller/cmd_vel')
        self.declare_parameter('safety_decision_topic', '/debug/safety_decision')
        self.declare_parameter('frequency_hz', 20.0)
        self.declare_parameter('cmd_timeout_s', 0.50)
        self.declare_parameter('front_timeout_s', 0.70)
        self.declare_parameter('rear_timeout_s', 0.80)
        self.declare_parameter('front_hard_stop_m', 0.22)
        self.declare_parameter('front_soft_stop_m', 0.35)
        self.declare_parameter('front_slowdown_m', 0.50)
        self.declare_parameter('rear_hard_stop_m', 0.30)
        self.declare_parameter('rear_soft_stop_m', 0.55)
        self.declare_parameter('max_forward_vx', 0.32)
        self.declare_parameter('max_reverse_vx', -0.08)
        self.declare_parameter('max_wz', 0.75)
        self.declare_parameter('gap_escape_max_vx', 0.04)
        self.declare_parameter('max_delta_vx_per_cycle', 0.03)
        self.declare_parameter('max_delta_wz_per_cycle', 0.10)

        self.raw_cmd_topic = str(self.get_parameter('raw_cmd_topic').value)
        self.front_topic = str(self.get_parameter('front_free_space_topic').value)
        self.rear_topic = str(self.get_parameter('rear_risk_topic').value)
        self.safe_cmd_topic = str(self.get_parameter('safe_cmd_topic').value)
        self.safety_decision_topic = str(self.get_parameter('safety_decision_topic').value)
        self.frequency_hz = float(self.get_parameter('frequency_hz').value)
        self.cmd_timeout_s = float(self.get_parameter('cmd_timeout_s').value)
        self.front_timeout_s = float(self.get_parameter('front_timeout_s').value)
        self.rear_timeout_s = float(self.get_parameter('rear_timeout_s').value)
        self.front_hard_stop_m = float(self.get_parameter('front_hard_stop_m').value)
        self.front_soft_stop_m = float(self.get_parameter('front_soft_stop_m').value)
        self.front_slowdown_m = float(self.get_parameter('front_slowdown_m').value)
        self.rear_hard_stop_m = float(self.get_parameter('rear_hard_stop_m').value)
        self.rear_soft_stop_m = float(self.get_parameter('rear_soft_stop_m').value)
        self.max_forward_vx = float(self.get_parameter('max_forward_vx').value)
        self.max_reverse_vx = float(self.get_parameter('max_reverse_vx').value)
        self.max_wz = float(self.get_parameter('max_wz').value)
        self.gap_escape_max_vx = float(self.get_parameter('gap_escape_max_vx').value)
        self.max_delta_vx = float(self.get_parameter('max_delta_vx_per_cycle').value)
        self.max_delta_wz = float(self.get_parameter('max_delta_wz_per_cycle').value)

        self.raw_cmd = None
        self.front = None
        self.rear = None
        self.last_raw_time = -999.0
        self.last_front_time = -999.0
        self.last_rear_time = -999.0
        self.last_safe_vx = 0.0
        self.last_safe_wz = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.safe_cmd_topic, 10)
        self.decision_pub = self.create_publisher(String, self.safety_decision_topic, 10)
        self.raw_cmd_sub = self.create_subscription(Twist, self.raw_cmd_topic, self.raw_cmd_callback, 10)
        self.front_sub = self.create_subscription(String, self.front_topic, self.front_callback, 10)
        self.rear_sub = self.create_subscription(String, self.rear_topic, self.rear_callback, 10)
        self.timer = self.create_timer(1.0 / max(self.frequency_hz, 1.0), self.publish_safe_cmd)

        self.get_logger().info(
            'safety_shield_node subscribed to %s, %s, %s; publishing %s'
            % (self.raw_cmd_topic, self.front_topic, self.rear_topic, self.safe_cmd_topic)
        )

    def raw_cmd_callback(self, raw_cmd):
        self.raw_cmd = raw_cmd
        self.last_raw_time = self._now()

    def front_callback(self, message):
        payload = self._parse_json(message.data, 'front_free_space')
        if payload is None:
            return
        self.front = payload
        self.last_front_time = self._now()

    def rear_callback(self, message):
        payload = self._parse_json(message.data, 'rear_risk')
        if payload is None:
            return
        self.rear = payload
        self.last_rear_time = self._now()

    def publish_safe_cmd(self):
        now = self._now()
        if self.raw_cmd is None or now - self.last_raw_time > self.cmd_timeout_s:
            self.publish_zero('cmd_timeout')
            return
        if self.front is None or now - self.last_front_time > self.front_timeout_s:
            self.publish_zero('front_timeout')
            return
        if not bool(self.front.get('valid', True)):
            self.publish_zero('front_invalid')
            return

        vx = float(self.raw_cmd.linear.x)
        wz = float(self.raw_cmd.angular.z)
        raw_vx = vx
        raw_wz = wz
        reason = 'pass'
        force_immediate_vx = False

        front_min = _as_float(self.front, 'front_min', _as_float(self.front, 'front_clearance', 0.0))
        front_unknown = bool(self.front.get('front_unknown', False))
        gap_escape_allowed = bool(self.front.get('gap_escape_allowed', False))
        if vx > 0.0 and front_unknown:
            vx = 0.0
            wz = _clamp(wz, -0.55, 0.55)
            force_immediate_vx = True
            reason = 'front_unknown_stop'
        elif vx > 0.0 and front_min > 0.0 and front_min < self.front_hard_stop_m:
            if gap_escape_allowed:
                vx = min(vx, self.gap_escape_max_vx)
                wz = _clamp(wz, -self.max_wz, self.max_wz)
                reason = 'front_hard_gap_limit'
            else:
                vx = 0.0
                wz = _clamp(wz, -0.55, 0.55)
                force_immediate_vx = True
                reason = 'front_hard_stop'
        elif vx > 0.0 and self.front_hard_stop_m <= front_min < self.front_soft_stop_m:
            vx = min(vx, 0.05)
            wz = _clamp(wz, -0.65, 0.65)
            reason = 'front_soft_limit'
        elif vx > 0.0 and self.front_soft_stop_m <= front_min < self.front_slowdown_m:
            vx = min(vx, 0.12)
            reason = 'front_slowdown_limit'

        rear_recent = self.rear is not None and now - self.last_rear_time <= self.rear_timeout_s
        rear_valid = rear_recent and bool(self.rear.get('valid', False))
        if vx < 0.0:
            if not rear_valid:
                vx = 0.0
                force_immediate_vx = True
                reason = 'rear_timeout_reverse_stop'
            else:
                rear_center = _as_float(self.rear, 'rear_center_min', 0.0)
                reverse_allowed = bool(self.rear.get('reverse_allowed', False))
                if rear_center <= 0.0 or rear_center < self.rear_hard_stop_m:
                    vx = 0.0
                    force_immediate_vx = True
                    reason = 'rear_hard_reverse_stop'
                elif self.rear_hard_stop_m <= rear_center < self.rear_soft_stop_m:
                    vx = max(vx, -0.03)
                    reason = 'rear_soft_reverse_limit'
                elif reverse_allowed:
                    vx = max(vx, -0.07)
                    reason = 'rear_reverse_allowed'
                else:
                    vx = 0.0
                    force_immediate_vx = True
                    reason = 'rear_reverse_not_allowed'

        vx = _clamp(vx, self.max_reverse_vx, self.max_forward_vx)
        wz = _clamp(wz, -self.max_wz, self.max_wz)
        if force_immediate_vx and vx == 0.0:
            self.last_safe_vx = 0.0
        vx, wz = self._smooth(vx, wz)
        self._publish(vx, wz)
        self._publish_decision(reason, raw_vx, raw_wz, vx, wz)

    def _smooth(self, target_vx, target_wz):
        vx = _clamp(target_vx, self.last_safe_vx - self.max_delta_vx, self.last_safe_vx + self.max_delta_vx)
        wz = _clamp(target_wz, self.last_safe_wz - self.max_delta_wz, self.last_safe_wz + self.max_delta_wz)
        self.last_safe_vx = vx
        self.last_safe_wz = wz
        return vx, wz

    def publish_zero(self, reason='zero'):
        self.last_safe_vx = 0.0
        self.last_safe_wz = 0.0
        self._publish(0.0, 0.0)
        raw_vx = float(self.raw_cmd.linear.x) if self.raw_cmd is not None else 0.0
        raw_wz = float(self.raw_cmd.angular.z) if self.raw_cmd is not None else 0.0
        self._publish_decision(reason, raw_vx, raw_wz, 0.0, 0.0)

    def _publish(self, vx, wz):
        safe_cmd = Twist()
        safe_cmd.linear.x = float(vx)
        safe_cmd.angular.z = float(wz)
        try:
            self.cmd_pub.publish(safe_cmd)
        except Exception as exc:
            if rclpy is not None and rclpy.ok():
                self.get_logger().warn('failed to publish safe cmd: %s' % exc)

    def _publish_decision(self, reason, raw_vx, raw_wz, safe_vx, safe_wz):
        front = self.front or {}
        rear = self.rear or {}
        payload = {
            'stamp': round(self._now(), 6),
            'reason': reason,
            'front_min': round(_as_float(front, 'front_min', _as_float(front, 'front_clearance', 0.0)), 3),
            'rear_center_min': round(_as_float(rear, 'rear_center_min', 0.0), 3),
            'reverse_allowed': bool(rear.get('reverse_allowed', False)),
            'raw_vx': round(float(raw_vx), 4),
            'raw_wz': round(float(raw_wz), 4),
            'safe_vx': round(float(safe_vx), 4),
            'safe_wz': round(float(safe_wz), 4),
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        try:
            self.decision_pub.publish(message)
        except Exception as exc:
            if rclpy is not None and rclpy.ok():
                self.get_logger().warn('failed to publish safety decision: %s' % exc)

    def _parse_json(self, data, label):
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn('invalid %s JSON: %s' % (label, exc))
            return None

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def destroy_node(self):
        try:
            for _ in range(10):
                self.publish_zero()
                time.sleep(0.05)
        finally:
            try:
                super().destroy_node()
            except BaseException:
                pass


def main(args=None):
    if rclpy is None:
        raise RuntimeError('rclpy is required to run safety_shield_node as a ROS2 node')
    rclpy.init(args=args)
    node = SafetyShieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error('safety_shield_node crashed: %s' % exc)
        raise
    finally:
        node.publish_zero('shutdown')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
