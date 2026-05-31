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
                'features_topic': '/nav/local_obstacle_features',
                'obstacle_inflation_m': 0.22,
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
                'direction_lock_time_sec': 0.45,
                'direction_switch_margin': 0.18,
                'heading_change_penalty': 0.08,
            }],
        ),
        Node(
            package='phantom_planner_controller',
            executable='planner_controller_node',
            name='planner_controller_node',
            output='screen',
            parameters=[{
                'scan_topic': scan_topic,
                'odom_topic': '/odom',
                'detection_topics': ['/det/detections', '/det/yellow_boxes'],
                'depth_topic': depth_topic,
                'cmd_vel_topic': planner_cmd_topic,
                'stuck_status_topic': '/planner/stuck_status',
                'artifacts_dir': artifacts_dir,
                'safe_distance': 0.55,
                'emergency_stop_distance': 0.20,
                'obstacle_inflation_m': 0.22,
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
                'rear_stop_clearance_m': 0.22,
                'front_angle_offset_rad': 0.0,
                'cruise_max_linear': 0.18,
                'escape_max_linear': 0.35,
                'max_angular': 0.8,
                'direction_lock_time_sec': 0.55,
                'direction_switch_score_margin': 0.35,
                'heading_change_penalty': 0.18,
                'direction_switch_penalty': 0.35,
                'oscillation_penalty': 0.15,
                'stuck_score_threshold': 0.78,
                'stuck_score_hold_sec': 0.8,
                'recover_backup_linear': -0.06,
                'recover_forward_linear': 0.06,
                'probe_front_threshold_m': 0.10,
                'probe_tie_margin': 0.25,
                'min_probe_duration': 0.6,
                'max_probe_duration': 1.8,
                'probe_linear': 0.0,
                'probe_angular': 0.6,
                'rear_valid_ratio_min': 0.35,
                'rear_center_depth_min': 0.45,
                'rear_min_depth_min': 0.30,
                'recover_post_backup_stop_time_sec': 0.25,
                'recover_forward_clearance_m': 0.35,
                'max_linear_accel_mps2': 0.25,
                'max_angular_accel_rps2': 1.2,
            }],
        ),
        Node(
            package='phantom_safety_shield',
            executable='safety_shield_node',
            name='safety_shield_node',
            output='screen',
            parameters=[{
                'raw_cmd_topic': planner_cmd_topic,
                'features_topic': '/nav/local_obstacle_features',
                'safe_cmd_topic': cmd_vel_topic,
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
                'rear_stop_clearance_m': 0.22,
                'max_linear_accel_mps2': 0.25,
                'max_angular_accel_rps2': 1.2,
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
            'model_path',
            default_value='/home/ubuntu/phantom_ws/models/yolo/phantom_cone_yellow_random200_best.pt',
        ),
        DeclareLaunchArgument('artifacts_dir', default_value='/home/ubuntu/phantom_ws/artifacts'),
        DeclareLaunchArgument('start_lidar', default_value='true'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('start_controller', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
