from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context):
    lidar_driver = LaunchConfiguration('lidar_driver').perform(context).strip().lower()
    lidar_port = LaunchConfiguration('lidar_port')
    scan_topic = LaunchConfiguration('scan_topic')
    camera_device = LaunchConfiguration('camera_device')
    image_topic = LaunchConfiguration('image_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    planner_cmd_topic = LaunchConfiguration('planner_cmd_topic')
    lidar_angle_offset_rad = LaunchConfiguration('lidar_angle_offset_rad')
    model_path = LaunchConfiguration('model_path')
    artifacts_dir = LaunchConfiguration('artifacts_dir')
    start_lidar = LaunchConfiguration('start_lidar')
    start_camera = LaunchConfiguration('start_camera')
    start_controller = LaunchConfiguration('start_controller')

    nodes = []

    if lidar_driver == 'ydlidar':
        nodes.append(Node(
            package='ydlidar_ros2_driver',
            executable='ydlidar_ros2_driver_node',
            name='ydlidar_ros2_driver_node',
            output='screen',
            condition=IfCondition(start_lidar),
            parameters=[{
                'port': lidar_port,
                'frame_id': 'lidar_frame',
                'ignore_array': '',
                'baudrate': 230400,
                'lidar_type': 1,
                'device_type': 0,
                'sample_rate': 9,
                'fixed_resolution': True,
                'auto_reconnect': True,
                'reversion': True,
                'inverted': True,
                'isSingleChannel': False,
                'intensity': False,
                'invalid_range_is_inf': False,
                'angle_min': -180.0,
                'angle_max': 180.0,
                'range_min': 0.12,
                'range_max': 16.0,
                'frequency': 12.0,
            }],
            remappings=[('scan', scan_topic)],
        ))
    elif lidar_driver == 'ldlidar':
        nodes.append(Node(
            package='ldlidar_stl_ros2',
            executable='ldlidar_stl_ros2_node',
            name='ldlidar_stl_ros2_node',
            output='screen',
            condition=IfCondition(start_lidar),
            parameters=[{
                'topic_name': 'scan',
                'product_name': 'LDLiDAR_LD19',
                'port_baudrate': 230400,
                'port_name': lidar_port,
                'frame_id': 'lidar_frame',
                'laser_scan_dir': True,
            }],
            remappings=[('scan', scan_topic)],
        ))
    else:
        nodes.append(Node(
            package='sllidar_ros2',
            executable='sllidar_node',
            name='sllidar_node',
            output='screen',
            condition=IfCondition(start_lidar),
            parameters=[{
                'channel_type': 'serial',
                'serial_baudrate': 115200,
                'serial_port': lidar_port,
                'frame_id': 'lidar_frame',
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': 'Sensitivity',
            }],
            remappings=[('scan', scan_topic)],
        ))

    nodes.extend([
        Node(
            package='ros_robot_controller',
            executable='ros_robot_controller',
            name='ros_robot_controller',
            output='screen',
            condition=IfCondition(start_controller),
            parameters=[{'imu_frame': 'imu_link'}],
        ),
        Node(
            package='controller',
            executable='odom_publisher',
            name='odom_publisher',
            output='screen',
            condition=IfCondition(start_controller),
            parameters=[{
                'base_frame_id': 'base_footprint',
                'odom_frame_id': 'odom',
                'pub_odom_topic': True,
            }],
        ),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            condition=IfCondition(start_camera),
            parameters=[{
                'video_device': camera_device,
                'framerate': 30.0,
                'io_method': 'mmap',
                'frame_id': 'rear_camera',
                'pixel_format': 'yuyv',
                'image_width': 640,
                'image_height': 480,
                'camera_name': 'rear_camera',
                'camera_info_url': '',
                'auto_white_balance': True,
                'autoexposure': True,
            }],
            remappings=[('image_raw', image_topic)],
        ),
        Node(
            package='phantom_detector',
            executable='detector_node',
            name='detector_node',
            output='screen',
            parameters=[{
                'mode': 'YOLO',
                'model_path': model_path,
                'image_topic': image_topic,
                'detections_topic': '/det/detections',
                'yellow_topic': '/det/yellow_boxes',
                'exit_topic': '/det/exit_boxes',
                'yellow_classes': ['traffic_cone', 'yellow_car'],
                'imgsz': 640,
                'conf': 0.25,
                'show': False,
                'artifacts_dir': artifacts_dir,
            }],
        ),
        Node(
            package='phantom_free_space',
            executable='free_space_node',
            name='free_space_node',
            output='screen',
            parameters=[{
                'scan_topic': scan_topic,
                'front_free_space_topic': '/nav/front_free_space',
                'features_topic': '/nav/local_obstacle_features',
                'lidar_angle_offset_rad': lidar_angle_offset_rad,
                'front_min_valid_ratio': 0.05,
                'front_hard_stop_m': 0.22,
                'front_soft_stop_m': 0.35,
                'front_slowdown_m': 0.50,
            }],
        ),
        Node(
            package='phantom_detector',
            executable='rear_perception_node',
            name='rear_perception_node',
            output='screen',
            parameters=[{
                'depth_topic': depth_topic,
                'rear_risk_topic': '/nav/rear_risk',
                'detections_topic': '/det/detections',
                'use_detections': True,
                'publish_rate_hz': 5.0,
                'depth_timeout_sec': 0.6,
                'rear_hard_stop_m': 0.30,
                'rear_soft_stop_m': 0.55,
                'z_bump_depth_jump_m': 0.06,
                'z_bump_trigger_score': 0.65,
                'z_bump_consecutive_frames': 3,
                'z_bump_cone_threshold_scale': 0.75,
            }],
        ),
        Node(
            package='phantom_planner_controller',
            executable='planner_controller_node',
            name='planner_controller_node',
            output='screen',
            parameters=[{
                'front_free_space_topic': '/nav/front_free_space',
                'rear_risk_topic': '/nav/rear_risk',
                'detections_topic': '/det/detections',
                'odom_topic': '/odom',
                'odom_fallback_topic': '/odom_raw',
                'cmd_vel_topic': planner_cmd_topic,
                'planner_state_topic': '/debug/planner_state',
                'stuck_status_topic': '/planner/stuck_status',
                'control_frequency_hz': 10.0,
                'front_timeout_sec': 0.5,
                'rear_timeout_sec': 0.8,
                'detection_timeout_sec': 0.8,
                'cruise_vx': 0.16,
                'escape_vx': 0.28,
                'avoid_front_vx': 0.06,
                'recover_reverse_vx': -0.06,
                'recover_wz': 0.55,
                'gap_escape_vx': 0.035,
                'gap_escape_min_wz': 0.35,
                'max_forward_vx': 0.32,
                'max_reverse_vx': -0.08,
                'max_wz': 0.75,
                'k_heading': 1.15,
                'front_hard_stop_m': 0.22,
                'front_soft_stop_m': 0.35,
                'front_slowdown_m': 0.50,
                'rear_pressure_escape_enter': 0.55,
                'rear_pressure_escape_exit': 0.35,
                'rear_pressure_cruise_max': 0.45,
                'min_escape_duration_s': 1.20,
                'escape_clear_duration_s': 1.00,
                'direction_lock_duration_s': 0.90,
                'switch_margin': 0.18,
                'keep_bonus': 0.12,
                'switch_penalty': 0.22,
                'oscillation_penalty': 0.15,
                'oscillation_window_s': 2.0,
                'stuck_enter_threshold': 0.75,
                'stuck_enter_duration_s': 0.80,
                'front_rear_soft_blocked_duration_s': 1.00,
                'cone_base_recover_enabled': True,
                'cone_base_recover_score_enter': 0.65,
                'cone_base_recover_score_exit': 0.35,
                'cone_base_recover_cooldown_s': 3.00,
                'cone_base_recover_stop_s': 0.20,
                'cone_base_recover_reverse_s': 0.60,
                'cone_base_recover_rotate_s': 0.80,
                'cone_base_recover_timeout_s': 2.20,
            }],
        ),
        Node(
            package='phantom_safety_shield',
            executable='safety_shield_node',
            name='safety_shield_node',
            output='screen',
            parameters=[{
                'raw_cmd_topic': planner_cmd_topic,
                'front_free_space_topic': '/nav/front_free_space',
                'rear_risk_topic': '/nav/rear_risk',
                'safe_cmd_topic': cmd_vel_topic,
                'safety_decision_topic': '/debug/safety_decision',
                'frequency_hz': 20.0,
                'cmd_timeout_s': 0.50,
                'front_timeout_s': 0.70,
                'rear_timeout_s': 0.80,
                'front_hard_stop_m': 0.22,
                'front_soft_stop_m': 0.35,
                'front_slowdown_m': 0.50,
                'rear_hard_stop_m': 0.30,
                'rear_soft_stop_m': 0.55,
                'max_forward_vx': 0.32,
                'max_reverse_vx': -0.08,
                'max_wz': 0.75,
                'gap_escape_max_vx': 0.04,
            }],
        ),
    ])

    return nodes


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('MACHINE_TYPE', 'JetAuto'),
        DeclareLaunchArgument('lidar_driver', default_value='sllidar',
                              description='sllidar, ydlidar, or ldlidar'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video2'),
        DeclareLaunchArgument('image_topic', default_value='/usb_cam/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/controller/cmd_vel'),
        DeclareLaunchArgument('planner_cmd_topic', default_value='/cmd_vel_raw'),
        DeclareLaunchArgument(
            'lidar_angle_offset_rad',
            default_value='3.14159',
            description='Rotate raw LaserScan angles into robot base frame; 3.14159 maps raw 180 deg to robot front.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/ubuntu/phantom_ws/models/yolo/phantom_cone_yellow_random200_best.pt',
        ),
        DeclareLaunchArgument('artifacts_dir', default_value='/home/ubuntu/phantom_ws/artifacts'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_controller', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
