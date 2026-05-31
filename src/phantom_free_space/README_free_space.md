# PHANTOM Free Space Node

`free_space_node.py` is a ROS1 `rospy` node for JetAuto. It consumes a 2D LiDAR `sensor_msgs/LaserScan`, splits the scan into angular sectors, estimates which directions are free or risky, publishes a recommended heading, and writes a top-down debug image.

## Input

- `/scan` (`sensor_msgs/LaserScan`) by default.
- On this JetAuto, the real LiDAR serial interface was identified as `/dev/ttyUSB0`.
- A working driver test used:

```bash
rosrun rplidar_ros rplidarNode __name:=rplidar_a1 \
  _serial_port:=/dev/ttyUSB0 _serial_baudrate:=115200 \
  _frame_id:=lidar_frame _inverted:=false \
  _angle_compensate:=true _scan_mode:=Boost
```

## Outputs

- `/free_space/sectors` (`std_msgs/Float32MultiArray`)
  - Fixed stride: 8 floats per sector.
  - Layout: `[sector_id, angle_min, angle_max, angle_center, min_range, mean_range, score, is_free]`
  - Angles are radians; `is_free` is `1.0` or `0.0`.
- `/free_space/best_heading` (`std_msgs/Float32`)
  - Recommended heading in radians, relative to the robot forward direction.
- `/free_space/status` (`std_msgs/String`)
  - JSON status with stamp, scan topic, sector count, best heading, best score, free sector count, and state.
- `/free_space/debug_image_path` (`std_msgs/String`)
  - Absolute path of the latest saved debug image.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `~scan_topic` | `/scan` | Input LaserScan topic |
| `~num_sectors` | `36` | Number of angular sectors |
| `~safety_distance` | `0.45` | Minimum safe clearance in meters |
| `~max_range` | `6.0` | Max range used by free-space scoring |
| `~min_valid_points_per_sector` | `3` | Unknown if fewer valid points are present |
| `~forward_weight` | `0.35` | Preference for headings near straight ahead |
| `~clearance_weight` | `0.45` | Preference for larger clearance |
| `~width_weight` | `0.20` | Preference for wider continuous free groups |
| `~image_size` | `800` | Debug image size in pixels |
| `~meters_per_pixel` | `0.01` | Rendering scale |
| `~debug_dir` | `~/free_space_debug` | Output directory for PNG images |
| `~save_every_n_frames` | `5` | Save a debug image every N processed frames |
| `~publish_rate_limit_hz` | `10.0` | Output rate limit |

## Algorithm

1. Build scan angles from `angle_min + i * angle_increment`.
2. Filter ranges that are NaN, inf, below `range_min`, above `range_max`, or above configured `max_range`.
3. Divide the actual scan angle span into `num_sectors`.
4. For each sector compute valid point count, min range, mean range, clearance, obstacle risk, free flag, and score.
5. Mark sectors with too few valid points as unknown, not free.
6. Merge adjacent free sectors into groups, including wraparound groups.
7. Pick the best free group using clearance, forward preference, and width; publish the group's center heading.
8. If no free group exists, publish `state: no_free_space` and heading `0.0`.

## Start

```bash
source /opt/ros/melodic/setup.bash
source ~/jetauto_ws/devel/setup.bash
export ROS_PACKAGE_PATH=~/phantom_ws/src:$ROS_PACKAGE_PATH

roslaunch phantom_free_space free_space_node.launch scan_topic:=/scan
```

If launching the LiDAR manually is needed:

```bash
rosrun rplidar_ros rplidarNode __name:=rplidar_a1 \
  _serial_port:=/dev/ttyUSB0 _serial_baudrate:=115200 \
  _frame_id:=lidar_frame _inverted:=false \
  _angle_compensate:=true _scan_mode:=Boost
```

## Check Outputs

```bash
rostopic info /scan
rostopic echo -n 1 /free_space/status
rostopic echo -n 1 /free_space/best_heading
rostopic echo -n 1 /free_space/sectors
rostopic echo -n 1 /free_space/debug_image_path
ls -lh ~/free_space_debug/
file ~/free_space_debug/free_space_latest.png
```

Copy the latest debug image to the local machine:

```bash
scp jetauto@166.111.55.186:~/free_space_debug/free_space_latest.png .
```

## Troubleshooting

- Cannot find `/scan`: start the real LiDAR driver and verify `/dev/ttyUSB0` exists.
- Cannot find any real LiDAR interface: compare serial scans before and after unplugging the LiDAR; on this car `/dev/ttyUSB0` disappeared when the LiDAR was disconnected.
- Only PointCloud2 exists: add a PointCloud2-to-2D LaserScan conversion or a sector distance extractor before this node.
- Serial device exists but no ROS topic: verify the driver/model/baudrate; for this car RPLidar A1 settings produced `/scan`.
- `roscore` is not running: start `roscore` or the JetAuto bringup before launching this node.
- Image not generated: check `~/free_space_debug`, `cv2`, and `numpy`.
- All sectors blocked: reduce nearby obstacles, verify LiDAR orientation, or tune `safety_distance`.
- `cv2` or `numpy` missing: install the corresponding Python packages in the active ROS Python environment.
