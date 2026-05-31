# Integrated Escape Test

This launch runs the simplified YOLO + LiDAR + chassis controller on JetAuto.

## Workspace

Use the ROS2 container workspace:

```bash
docker exec -it -u ubuntu jetauto bash
cd ~/phantom_ws
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Hardware Checks

```bash
ls -l /dev/ttyUSB0
ls -l /dev/rrc
ls -l /dev/ttyACM1
ros2 topic list
ros2 topic echo /scan --once
```

On this car `/dev/ttyUSB0` is the LiDAR serial port. `/dev/rrc` points to
`/dev/ttyACM0` and is used by the chassis controller. The rear RealSense color
stream appears as `/dev/video2`; `/dev/video0` is a depth stream and `/dev/rrc`
is not a V4L2 camera device on this unit.

## Run

```bash
ros2 launch phantom_bringup integrated_escape_test.launch.py
```

Useful overrides:

```bash
ros2 launch phantom_bringup integrated_escape_test.launch.py lidar_driver:=ydlidar
ros2 launch phantom_bringup integrated_escape_test.launch.py camera_device:=/dev/video1
ros2 launch phantom_bringup integrated_escape_test.launch.py cmd_vel_topic:=/cmd_vel
ros2 launch phantom_bringup integrated_escape_test.launch.py planner_cmd_topic:=/cmd_vel_raw
```

## Topics

- Free-space publishes `/nav/local_obstacle_features`.
- Planner subscribes `/scan`, `/odom`, `/det/detections`, `/det/yellow_boxes`.
- Planner publishes raw `geometry_msgs/Twist` to `/cmd_vel_raw` by default.
- Safety shield subscribes `/cmd_vel_raw` and `/nav/local_obstacle_features`, then publishes `/controller/cmd_vel`.
- Planner stuck/recover diagnostics are published on `/planner/stuck_status`.
- YOLO model: `~/phantom_ws/models/yolo/phantom_cone_yellow_random200_best.pt`.

## Artifacts

The first LiDAR free-space image is written to:

```text
~/phantom_ws/artifacts/first_lidar_freespace.png
```

The first YOLO image is written to:

```text
~/phantom_ws/artifacts/first_yolo_detection.png
```

Copy them back to the local machine with:

```bash
scp -r jetauto@166.111.55.186:~/docker/tmp/phantom_artifacts ./artifacts_from_jetauto
```

The command above is used after copying from the container to the host.

## Emergency Stop

Press `Ctrl+C` in the launch terminal. The planner publishes zero velocity in
its shutdown path. From another terminal, a direct stop to the base interface is:

```bash
ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist "{}"
```

## Current Limits

This is a reactive sector controller, not AC-MPC. It has no map, no dynamic
obstacle prediction, no acceleration constraints beyond capped Twist output,
and no optimal trajectory rollout. It now adds safety inflation, front-corner
turn limiting, command smoothing, and a short recover sequence, but final
parameters must be validated only with real LiDAR/odom/chassis logs. AC-MPC can later replace only the
`_make_cmd()` decision layer while keeping the same `/scan`, `/det/detections`,
and `/controller/cmd_vel` interfaces.
