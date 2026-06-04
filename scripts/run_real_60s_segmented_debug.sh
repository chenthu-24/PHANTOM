#!/usr/bin/env bash
set -eo pipefail

ART="${1:-artifacts/real_60s_debug}"
LIDAR_DRIVER="${2:-sllidar}"
CAMERA_DEVICE="${3:-/dev/video2}"
SEGMENTS="${4:-6}"

cd /home/ubuntu/phantom_ws
source /opt/ros/humble/setup.bash
source /home/ubuntu/ros2_ws/install/setup.bash 2>/dev/null || true
source /home/ubuntu/phantom_ws/install/setup.bash 2>/dev/null || true
set -u

rm -rf "$ART"
mkdir -p "$ART"

stop_robot() {
  timeout 3 ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true
  sleep 0.2
  timeout 3 ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" >/dev/null 2>&1 || true
}

cleanup_nodes() {
  local pids
  pids=$(ps aux | awk '/ros_robot_controller|odom_publisher|usb_cam_node_exe|detector_node|free_space_node|rear_perception_node|planner_controller_node|safety_shield_node|sllidar_node|ydlidar_ros2_driver_node|ldlidar_stl_ros2_node|relay_cmd_vel_10s|record_debug_topics/ && !/awk/ {print $2}')
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
    sleep 1
    pids=$(ps aux | awk '/ros_robot_controller|odom_publisher|usb_cam_node_exe|detector_node|free_space_node|rear_perception_node|planner_controller_node|safety_shield_node|sllidar_node|ydlidar_ros2_driver_node|ldlidar_stl_ros2_node|relay_cmd_vel_10s|record_debug_topics/ && !/awk/ {print $2}')
    if [ -n "$pids" ]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

trap 'stop_robot; cleanup_nodes' EXIT INT TERM

ros2 launch phantom_bringup integrated_escape_test.launch.py \
  lidar_driver:="$LIDAR_DRIVER" \
  camera_device:="$CAMERA_DEVICE" \
  cmd_vel_topic:=/phantom/disabled_cmd_vel \
  artifacts_dir:="/home/ubuntu/phantom_ws/$ART" \
  > "$ART/launch.log" 2>&1 &
launch_pid=$!

ready=0
for _ in $(seq 1 120); do
  topics="$(timeout 3 ros2 topic list 2>/dev/null || true)"
  printf '%s\n' "$topics" > "$ART/topic_list_latest.txt"
  if printf '%s\n' "$topics" | grep -qx /scan \
    && printf '%s\n' "$topics" | grep -qx /nav/front_free_space \
    && printf '%s\n' "$topics" | grep -qx /nav/rear_risk \
    && printf '%s\n' "$topics" | grep -qx /det/detections \
    && printf '%s\n' "$topics" | grep -qx /debug/planner_state \
    && printf '%s\n' "$topics" | grep -qx /debug/safety_decision \
    && printf '%s\n' "$topics" | grep -qx /cmd_vel_raw \
    && printf '%s\n' "$topics" | grep -qx /phantom/disabled_cmd_vel; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "READY_TIMEOUT" > "$ART/launch_exit.txt"
  cp "$ART/topic_list_latest.txt" "$ART/topic_list_during_launch.txt" 2>/dev/null || true
  stop_robot
  kill "$launch_pid" 2>/dev/null || true
  cleanup_nodes
  wait "$launch_pid" || true
  exit 0
fi

timeout 3 ros2 topic list > "$ART/topic_list_during_launch.txt" 2>&1 || true
for topic in /scan /nav/front_free_space /nav/rear_risk /det/detections /cmd_vel_raw /phantom/disabled_cmd_vel /controller/cmd_vel /odom /odom_raw /debug/planner_state /debug/safety_decision; do
  echo "=== $topic ===" >> "$ART/topic_info_during_launch.txt"
  timeout 3 ros2 topic info "$topic" -v >> "$ART/topic_info_during_launch.txt" 2>&1 || true
done

record_duration=$((SEGMENTS * 11 + 8))
python3 scripts/record_debug_topics.py "$ART" --duration "$record_duration" > "$ART/recorder_stdout.txt" 2>&1 &
recorder_pid=$!
sleep 1

for segment in $(seq 1 "$SEGMENTS"); do
  echo "$(date -Is) segment_${segment}_start" >> "$ART/segment_markers.txt"
  python3 scripts/relay_cmd_vel_10s.py --source /phantom/disabled_cmd_vel --dest /controller/cmd_vel --duration 10.0 > "$ART/relay_segment_${segment}.txt" 2>&1 || true
  echo "$(date -Is) segment_${segment}_stop" >> "$ART/segment_markers.txt"
  stop_robot
  sleep 0.5
done

stop_robot
kill "$launch_pid" 2>/dev/null || true
wait "$launch_pid" || echo "LAUNCH_EXIT=$?" > "$ART/launch_exit.txt"
cleanup_nodes
wait "$recorder_pid" || true
sleep 0.5

python3 scripts/analyze_real_10s_debug.py "$ART" > "$ART/analyze_stdout.txt" 2>&1 || true
cp "$ART/first_yolo_detection.png" "$ART/yolo_debug_raw.png" 2>/dev/null || true
python3 scripts/export_debug_images.py "$ART" > "$ART/export_images_stdout.txt" 2>&1 || true

ls -lh "$ART"
cat "$ART/analysis_summary.txt" 2>/dev/null || true
