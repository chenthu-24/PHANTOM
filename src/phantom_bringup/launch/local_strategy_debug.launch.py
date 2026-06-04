from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scan_topic = LaunchConfiguration('scan_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    image_topic = LaunchConfiguration('image_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    detector_mode = LaunchConfiguration('detector_mode')
    model_path = LaunchConfiguration('model_path')
    artifacts_dir = LaunchConfiguration('artifacts_dir')
    cmd_vel_raw_topic = LaunchConfiguration('cmd_vel_raw_topic')
    cmd_vel_safe_topic = LaunchConfiguration('cmd_vel_safe_topic')
    lidar_angle_offset_rad = LaunchConfiguration('lidar_angle_offset_rad')

    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_rect_raw'),
        DeclareLaunchArgument('image_topic', default_value='/usb_cam/image_raw'),
        DeclareLaunchArgument('odom_topic', default_value='/odom_raw'),
        DeclareLaunchArgument('detector_mode', default_value='COLOR_DEBUG'),
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/ubuntu/phantom_ws/models/yolo/phantom_cone_yellow_random200_best.pt',
        ),
        DeclareLaunchArgument('artifacts_dir', default_value='/tmp/phantom_strategy_debug'),
        DeclareLaunchArgument('cmd_vel_raw_topic', default_value='/cmd_vel_raw'),
        DeclareLaunchArgument('cmd_vel_safe_topic', default_value='/controller/cmd_vel'),
        DeclareLaunchArgument('lidar_angle_offset_rad', default_value='3.14159'),
        Node(
            package='phantom_free_space',
            executable='free_space_node',
            name='free_space_node',
            output='screen',
            parameters=[{
                'scan_topic': scan_topic,
                'front_free_space_topic': '/nav/front_free_space',
                'features_topic': '/nav/local_obstacle_features',
                'publish_legacy_features': True,
                'lidar_angle_sign': 1.0,
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
            }],
        ),
        Node(
            package='phantom_detector',
            executable='detector_node',
            name='detector_node',
            output='screen',
            parameters=[{
                'mode': detector_mode,
                'model_path': model_path,
                'image_topic': image_topic,
                'depth_topic': depth_topic,
                'subscribe_depth': True,
                'detections_topic': '/det/detections',
                'yellow_topic': '/det/yellow_boxes',
                'exit_topic': '/det/exit_boxes',
                'yellow_classes': ['traffic_cone', 'yellow_car'],
                'exit_classes': ['exit'],
                'conf': 0.25,
                'show': False,
                'artifacts_dir': artifacts_dir,
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
                'odom_topic': odom_topic,
                'odom_fallback_topic': '/odom',
                'cmd_vel_topic': cmd_vel_raw_topic,
            }],
        ),
        Node(
            package='phantom_safety_shield',
            executable='safety_shield_node',
            name='safety_shield_node',
            output='screen',
            parameters=[{
                'raw_cmd_topic': cmd_vel_raw_topic,
                'front_free_space_topic': '/nav/front_free_space',
                'rear_risk_topic': '/nav/rear_risk',
                'safe_cmd_topic': cmd_vel_safe_topic,
                'frequency_hz': 20.0,
            }],
        ),
    ])
