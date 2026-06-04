from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scan_topic = LaunchConfiguration('scan_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    image_topic = LaunchConfiguration('image_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    cmd_vel_raw_topic = LaunchConfiguration('cmd_vel_raw_topic')
    cmd_vel_safe_topic = LaunchConfiguration('cmd_vel_safe_topic')
    artifacts_dir = LaunchConfiguration('artifacts_dir')

    return LaunchDescription([
        SetEnvironmentVariable('ROS_DOMAIN_ID', '77'),
        DeclareLaunchArgument('scan_topic', default_value='/mock/scan'),
        DeclareLaunchArgument('depth_topic', default_value='/mock/depth/image_raw'),
        DeclareLaunchArgument('image_topic', default_value='/mock/usb_cam/image_raw'),
        DeclareLaunchArgument('odom_topic', default_value='/mock/odom'),
        DeclareLaunchArgument('cmd_vel_raw_topic', default_value='/mock/cmd_vel_raw'),
        DeclareLaunchArgument('cmd_vel_safe_topic', default_value='/mock/controller/cmd_vel'),
        DeclareLaunchArgument('artifacts_dir', default_value='/tmp/phantom_local_chain_debug'),
        Node(
            package='phantom_bringup',
            executable='local_mock_inputs_node',
            name='local_mock_inputs_node',
            output='screen',
            parameters=[{
                'scan_topic': scan_topic,
                'depth_topic': depth_topic,
                'image_topic': image_topic,
                'odom_topic': odom_topic,
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
                'publish_legacy_features': False,
            }],
        ),
        Node(
            package='phantom_detector',
            executable='detector_node',
            name='detector_node',
            output='screen',
            parameters=[{
                'mode': 'COLOR_DEBUG',
                'image_topic': image_topic,
                'depth_topic': depth_topic,
                'subscribe_depth': True,
                'detections_topic': '/det/detections',
                'yellow_topic': '/det/yellow_boxes',
                'exit_topic': '/det/exit_boxes',
                'artifacts_dir': artifacts_dir,
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
            package='phantom_planner_controller',
            executable='planner_controller_node',
            name='planner_controller_node',
            output='screen',
            parameters=[{
                'front_free_space_topic': '/nav/front_free_space',
                'rear_risk_topic': '/nav/rear_risk',
                'detections_topic': '/det/detections',
                'odom_topic': odom_topic,
                'odom_fallback_topic': '',
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
            }],
        ),
    ])
