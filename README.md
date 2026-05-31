# PHANTOM / JetAuto ROS2 Workspace

PHANTOM is a ROS2 workspace for JetAuto autonomous navigation experiments. The current strategy uses a front LiDAR, rear depth camera, rear RGB detector, a planner/controller state machine, and a final safety shield before any chassis command reaches the low-level controller.

This repository is intended to support local development first, then deployment to the JetAuto robot for controlled testing.

## Control Strategy

Final V2 control chain:

```text
Front LiDAR /scan
  -> free_space_node
  -> /nav/front_free_space

Rear Depth Camera
  -> rear_perception_node
  -> /nav/rear_risk

Rear RGB Camera + YOLO / color debug
  -> detector_node
  -> /det/detections
  -> /det/yellow_boxes

/nav/front_free_space + /nav/rear_risk + /det/detections + /odom_raw or /odom
  -> planner_controller_node
  -> /cmd_vel_raw
  -> safety_shield_node
  -> /controller/cmd_vel
```

The planner owns behavior decisions. The safety shield is deliberately narrow: it limits speed, handles stale sensors, blocks unsafe reverse commands, and publishes zero velocity on shutdown.

## Workspace Layout

```text
src/
  phantom_bringup/              Launch files for local debug and robot integration
  phantom_free_space/           Front LiDAR free-space extraction
  phantom_detector/             Rear RGB detection and rear depth risk perception
  phantom_planner_controller/   Main CRUISE / ESCAPE / AVOID_FRONT / RECOVER / STOP planner
  phantom_safety_shield/        Final velocity gate before /controller/cmd_vel
  phantom_sensor_bridge/        Local fake/real sensor bridge utilities
  phantom_tracker_predictor/    Detection tracking and prediction helpers
  phantom_tactical_fusion/      Older tactical feature fusion layer
  phantom_mode_manager/         Older tactical mode manager
  phantom_subgoal_generator/    Older subgoal generation layer

tools/
  mock_strategy_contract_check.py  Pure-Python contract checks for strategy JSON and safety rules

models/, dataset*/, data*/, runs/
  YOLO models, training data, and experiment outputs

artifacts*/
  Captured logs and debug outputs from local or robot tests
```

## Key ROS2 Packages

| Package | Main executable | Role |
| --- | --- | --- |
| `phantom_free_space` | `free_space_node` | Subscribes to `/scan`, extracts front/side sector statistics, publishes `/nav/front_free_space`. |
| `phantom_detector` | `detector_node` | Subscribes to rear RGB image, runs YOLO or color-debug detection, publishes `/det/detections` and `/det/yellow_boxes`. |
| `phantom_detector` | `rear_perception_node` | Subscribes to rear depth image, computes rear clearance/risk, publishes `/nav/rear_risk`. |
| `phantom_planner_controller` | `planner_controller_node` | Fuses front free space, rear risk, detections, and odom; publishes `/cmd_vel_raw`. |
| `phantom_safety_shield` | `safety_shield_node` | Applies final timeout, front/rear danger, speed limit, smoothing, and shutdown-stop logic; publishes `/controller/cmd_vel`. |
| `phantom_bringup` | launch files | Starts strategy, perception, and integration test graphs. |

## Topic Contract

### Inputs

| Topic | Type | Source |
| --- | --- | --- |
| `/scan` | `sensor_msgs/msg/LaserScan` | Front LiDAR |
| `/usb_cam/image_raw` | `sensor_msgs/msg/Image` | Rear RGB camera |
| `/camera/depth/image_rect_raw` or `/camera/depth/image_raw` or `/depth/image_raw` | `sensor_msgs/msg/Image` | Rear depth camera |
| `/odom_raw` or `/odom` | `nav_msgs/msg/Odometry` | Odometry |

### Intermediate Topics

| Topic | Type | Producer |
| --- | --- | --- |
| `/nav/front_free_space` | `std_msgs/msg/String` JSON | `free_space_node` |
| `/nav/rear_risk` | `std_msgs/msg/String` JSON | `rear_perception_node` |
| `/det/detections` | `std_msgs/msg/String` JSON list | `detector_node` |
| `/det/yellow_boxes` | `std_msgs/msg/String` JSON | `detector_node` |

### Velocity Topics

| Topic | Type | Producer | Notes |
| --- | --- | --- | --- |
| `/cmd_vel_raw` | `geometry_msgs/msg/Twist` | `planner_controller_node` | Planner output only. |
| `/controller/cmd_vel` | `geometry_msgs/msg/Twist` | `safety_shield_node` | Final command topic for JetAuto controller. |

## Main Behaviors

`planner_controller_node` implements:

- `CRUISE`: low-speed forward motion toward the best front free-space heading.
- `ESCAPE`: faster forward motion when rear pressure or YOLO threat is detected.
- `AVOID_FRONT`: slow forward motion and stronger turning when front is softly blocked.
- `RECOVER`: timed stop, reverse-if-allowed, and rotate recovery sequence.
- `STOP`: zero command when front free-space is invalid/stale or no safe action exists.

It also includes direction locking and oscillation suppression:

- direction lock duration: `0.9 s`
- switch margin: `0.18`
- keep bonus: `0.12`
- switch penalty: `0.22`
- oscillation penalty: `0.30`

## Safety Shield

`safety_shield_node` subscribes to `/cmd_vel_raw`, `/nav/front_free_space`, and `/nav/rear_risk`, then publishes `/controller/cmd_vel`.

Core rules:

- stop if `/cmd_vel_raw` is stale for more than `0.5 s`
- stop if `/nav/front_free_space` is stale for more than `0.7 s`
- do not force-stop forward cruise when rear risk is stale
- forbid reverse if rear risk is stale or invalid
- stop forward motion on front hard danger: `front_min < 0.22 m`
- gradually limit forward speed for soft/slowdown zones
- stop reverse on rear hard danger: `rear_center_min < 0.30 m`
- limit reverse speed in rear soft danger
- clamp velocity to `vx [-0.08, 0.32]`, `wz [-0.75, 0.75]`
- publish zero velocity repeatedly during shutdown

## Launch Files

Safe local strategy launch:

```bash
ros2 launch phantom_bringup local_strategy_debug.launch.py
```

This launch starts only strategy nodes:

- `free_space_node`
- `rear_perception_node`
- `detector_node`
- `planner_controller_node`
- `safety_shield_node`

It does not start the real chassis driver, LiDAR driver, camera driver, or any SSH connection.

Other launch files:

- `perception_layer_test.launch.py`: older perception-layer test graph.
- `phase1_test.launch.py`: older phase 1 local test graph.
- `phase2_yolo_test.launch.py`: older YOLO/tactical pipeline test graph.
- `integrated_escape_test.launch.py`: robot integration launch; use only in a safe test environment.

## Local Checks

On a development machine without ROS2, run pure Python checks:

```bash
python -m py_compile $(find . -name "*.py")
python tools/mock_strategy_contract_check.py
```

On Windows PowerShell:

```powershell
$files = Get-ChildItem -Path . -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\(build|install|log)\\' } | Select-Object -ExpandProperty FullName
python -m py_compile @files
python tools\mock_strategy_contract_check.py
```

On a ROS2 machine:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch phantom_bringup local_strategy_debug.launch.py
```

## Robot Deployment Notes

Before running on JetAuto:

1. Confirm ROS2 build succeeds on the robot.
2. Confirm real topics exist:
   - `/scan`
   - rear RGB image topic, usually `/usb_cam/image_raw`
   - rear depth topic, one of `/camera/depth/image_rect_raw`, `/camera/depth/image_raw`, `/depth/image_raw`
   - `/odom_raw` or `/odom`
3. Echo perception outputs before any motion test:
   - `/nav/front_free_space`
   - `/nav/rear_risk`
   - `/det/detections`
4. Confirm `/cmd_vel_raw` is produced by planner.
5. Confirm `/controller/cmd_vel` is produced only by `safety_shield_node`.
6. Keep the robot physically supervised during low-speed tests.

Do not start robot integration launch files on a desk or unsafe floor area.

## Git Hygiene

Large datasets, generated artifacts, training runs, and robot logs should be kept out of normal commits unless they are intentionally being archived. The source code and launch files under `src/`, plus small tools under `tools/`, are the main files expected to be reviewed on GitHub.
