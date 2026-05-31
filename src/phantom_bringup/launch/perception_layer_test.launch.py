import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sensor_mode = LaunchConfiguration('mode')
    detector_mode = LaunchConfiguration('detector_mode')
    model_path = LaunchConfiguration('model_path')
    real_image_topic = LaunchConfiguration('real_image_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='FAKE',
            description='sensor_bridge_node mode: FAKE or REAL.',
        ),
        DeclareLaunchArgument(
            'detector_mode',
            default_value='COLOR_DEBUG',
            description='detector_node mode: COLOR_DEBUG or YOLO.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.expanduser('~/phantom_ws/models/yolo/yolov8n.pt'),
            description='YOLOv8n model path used when detector_mode:=yolo.',
        ),
        DeclareLaunchArgument(
            'real_image_topic',
            default_value='/camera/color/image_raw',
            description='RealSense color Image topic forwarded in sensor REAL mode.',
        ),
        Node(
            package='phantom_sensor_bridge',
            executable='sensor_bridge_node',
            name='sensor_bridge_node',
            output='screen',
            parameters=[{
                'mode': sensor_mode,
                'image_topic': '/perception/image_raw',
                'scan_topic': '/perception/scan_or_depth',
                'real_image_topic': real_image_topic,
                'real_scan_topic': '/scan',
                'camera_frame_id': 'camera_link',
                'scan_frame_id': 'base_link',
                'publish_rate_hz': 10.0,
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
                'horizontal_fov_rad': 1.0472,
                'imgsz': 640,
                'conf': 0.25,
                'show': False,
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
                'horizontal_fov_rad': 1.0472,
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
    ])
