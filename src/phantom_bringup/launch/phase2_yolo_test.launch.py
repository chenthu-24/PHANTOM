import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path = LaunchConfiguration('model_path')
    detector_mode = LaunchConfiguration('detector_mode')
    use_fake_image = LaunchConfiguration('use_fake_image')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.expanduser('~/phantom_ws/models/yolo/yolov8n.pt'),
            description='Local YOLOv8n model path.',
        ),
        DeclareLaunchArgument(
            'detector_mode',
            default_value='ros',
            description='detector_node mode: ros or test.',
        ),
        DeclareLaunchArgument(
            'use_fake_image',
            default_value='true',
            description='Publish PHASE2_FAKE_TEST_ONLY image frames from sensor_bridge_node.',
        ),
        Node(
            package='phantom_sensor_bridge',
            executable='sensor_bridge_node',
            name='sensor_bridge_node',
            output='screen',
            parameters=[{
                'scan_topic': '/perception/scan_or_depth',
                'image_topic': '/perception/image_raw',
                'ego_twist_topic': '/state/ego_twist',
                'use_fake_image': use_fake_image,
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
                'image_topic': '/perception/image_raw',
                'yellow_topic': '/det/yellow_boxes',
                'exit_topic': '/det/exit_boxes',
                'imgsz': 416,
                'conf': 0.4,
                'test_source': '0',
                'show': True,
            }],
        ),
        Node(
            package='phantom_tracker_predictor',
            executable='tracker_predictor_node',
            name='tracker_predictor_node',
            output='screen',
            parameters=[{
                'yellow_topic': '/det/yellow_boxes',
                'state_topic': '/track/yellow_state',
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
            package='phantom_tactical_fusion',
            executable='tactical_fusion_node',
            name='tactical_fusion_node',
            output='screen',
            parameters=[{
                'yellow_state_topic': '/track/yellow_state',
                'exit_topic': '/det/exit_boxes',
                'obstacle_features_topic': '/nav/local_obstacle_features',
                'tactics_features_topic': '/tactics/features',
            }],
        ),
        Node(
            package='phantom_mode_manager',
            executable='mode_manager_node',
            name='mode_manager_node',
            output='screen',
            parameters=[{
                'features_topic': '/tactics/features',
                'mode_topic': '/tactics/mode',
                'recover_stuck_threshold': 0.8,
                'recover_stuck_hold_sec': 0.8,
            }],
        ),
        Node(
            package='phantom_subgoal_generator',
            executable='subgoal_generator_node',
            name='subgoal_generator_node',
            output='screen',
            parameters=[{
                'features_topic': '/tactics/features',
                'mode_topic': '/tactics/mode',
                'subgoal_topic': '/tactics/subgoal_pose',
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
                'stuck_status_topic': '/planner/stuck_status',
                'obstacle_inflation_m': 0.22,
                'front_stop_clearance_m': 0.30,
                'side_stop_clearance_m': 0.24,
                'rear_stop_clearance_m': 0.22,
                'direction_lock_time_sec': 0.55,
                'direction_switch_score_margin': 0.35,
                'stuck_score_threshold': 0.78,
                'stuck_score_hold_sec': 0.8,
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
                'max_linear_accel_mps2': 0.25,
                'max_angular_accel_rps2': 1.2,
            }],
        ),
    ])
