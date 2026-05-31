from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='phantom_sensor_bridge',
            executable='sensor_bridge_node',
            name='sensor_bridge_node',
            output='screen',
            parameters=[{
                'scan_topic': '/perception/scan_or_depth',
                'ego_twist_topic': '/state/ego_twist',
            }],
        ),
        Node(
            package='phantom_free_space',
            executable='free_space_node',
            name='free_space_node',
            output='screen',
            parameters=[{
                'scan_topic': '/perception/scan_or_depth',
                'features_topic': '/nav/local_obstacle_features',
                'obstacle_inflation_m': 0.22,
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
            }],
        ),
        Node(
            package='phantom_planner_controller',
            executable='planner_controller_node',
            name='planner_controller_node',
            output='screen',
            parameters=[{
                'scan_topic': '/perception/scan_or_depth',
                'depth_topic': '/camera/depth/image_raw',
                'cmd_vel_topic': '/cmd_vel_raw',
                'obstacle_inflation_m': 0.22,
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
                'rear_stop_clearance_m': 0.22,
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
            }],
        ),
        Node(
            package='phantom_safety_shield',
            executable='safety_shield_node',
            name='safety_shield_node',
            output='screen',
            parameters=[{
                'raw_cmd_topic': '/cmd_vel_raw',
                'features_topic': '/nav/local_obstacle_features',
                'safe_cmd_topic': '/cmd_vel',
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
                'rear_stop_clearance_m': 0.22,
            }],
        ),
    ])
