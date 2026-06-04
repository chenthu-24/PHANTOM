#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


class RunResult:
    def __init__(self, returncode, stdout=''):
        self.returncode = returncode
        self.stdout = stdout


SETUP = (
    'set -e; '
    'cd /home/ubuntu/phantom_ws; '
    'source /opt/ros/humble/setup.bash; '
    'source /home/ubuntu/ros2_ws/install/setup.bash 2>/dev/null || true; '
    'source /home/ubuntu/phantom_ws/install/setup.bash 2>/dev/null || true'
)

ZERO_TWIST = (
    '{linear: {x: 0.0, y: 0.0, z: 0.0}, '
    'angular: {x: 0.0, y: 0.0, z: 0.0}}'
)


def _bash_cmd(command):
    return ['/bin/bash', '-lc', '%s; %s' % (SETUP, command)]


def _ros_args(args):
    return _bash_cmd('exec ' + shlex.join(args))


def _popen(args, stdout_path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    handle = stdout_path.open('w', encoding='utf-8')
    proc = subprocess.Popen(
        args,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd='/home/ubuntu/phantom_ws',
        preexec_fn=os.setsid,
    )
    return proc, handle


def _run(command, timeout=10, output_path=None):
    kwargs = {
        'cwd': '/home/ubuntu/phantom_ws',
        'text': True,
        'stdout': subprocess.PIPE,
        'stderr': subprocess.STDOUT,
        'timeout': timeout,
    }
    try:
        result = subprocess.run(_bash_cmd(command), **kwargs)
    except subprocess.TimeoutExpired as exc:
        result = RunResult(124, (exc.stdout or '') if isinstance(exc.stdout, str) else '')
    if output_path is not None:
        output_path.write_text(result.stdout or '', encoding='utf-8', errors='replace')
    return result


def _terminate_group(proc, name, log, sigint_timeout=3.0, sigterm_timeout=2.0):
    if proc.poll() is not None:
        log.append({'name': name, 'action': 'already_exited', 'returncode': proc.returncode})
        return
    pgid = os.getpgid(proc.pid)
    for sig, label, timeout_s in (
        (signal.SIGINT, 'SIGINT', sigint_timeout),
        (signal.SIGTERM, 'SIGTERM', sigterm_timeout),
        (signal.SIGKILL, 'SIGKILL', 1.0),
    ):
        try:
            os.killpg(pgid, sig)
            log.append({'name': name, 'action': label, 'pgid': pgid})
        except ProcessLookupError:
            log.append({'name': name, 'action': 'missing_after_%s' % label})
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.append({'name': name, 'action': 'exited_after_%s' % label, 'returncode': proc.returncode})
                return
            time.sleep(0.05)


def _publish_zero(count, interval_s, log):
    ok_count = 0
    for index in range(count):
        result = _run(
            'timeout 4 ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist %s >/dev/null 2>&1 || true'
            % shlex.quote(ZERO_TWIST),
            timeout=12,
        )
        ok_count += 1 if result.returncode == 0 else 0
        log.append({'zero_index': index + 1, 'returncode': result.returncode})
        time.sleep(interval_s)
    return ok_count


def _topic_list():
    result = _run('timeout 12 ros2 topic list 2>/dev/null || true', timeout=20)
    return set((result.stdout or '').splitlines()), result.stdout or ''


def _wait_ready(root, timeout_s):
    required = {
        '/scan',
        '/nav/front_free_space',
        '/nav/rear_risk',
        '/det/detections',
        '/debug/planner_state',
        '/debug/safety_decision',
        '/cmd_vel_raw',
        '/phantom/disabled_cmd_vel',
    }
    deadline = time.monotonic() + timeout_s
    latest_text = ''
    while time.monotonic() < deadline:
        topics, latest_text = _topic_list()
        (root / 'topic_list_latest.txt').write_text(latest_text, encoding='utf-8', errors='replace')
        if required.issubset(topics):
            (root / 'topic_list_during_launch.txt').write_text(latest_text, encoding='utf-8', errors='replace')
            return True
        time.sleep(1.0)
    (root / 'topic_list_during_launch.txt').write_text(latest_text, encoding='utf-8', errors='replace')
    return False


def _record_topic_info(root):
    lines = []
    topics = [
        '/scan',
        '/nav/front_free_space',
        '/nav/rear_risk',
        '/det/detections',
        '/debug/planner_state',
        '/debug/safety_decision',
        '/cmd_vel_raw',
        '/phantom/disabled_cmd_vel',
        '/controller/cmd_vel',
        '/odom',
        '/odom_raw',
    ]
    for topic in topics:
        lines.append('=== %s ===' % topic)
        result = _run('timeout 12 ros2 topic info %s -v 2>&1 || true' % shlex.quote(topic), timeout=20)
        lines.append(result.stdout or '')
    (root / 'topic_info_during_launch.txt').write_text('\n'.join(lines), encoding='utf-8', errors='replace')


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-dir', default='artifacts/real_60s_debug')
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--lidar-driver', default='sllidar')
    parser.add_argument('--camera-device', default='/dev/video2')
    parser.add_argument('--ready-timeout', type=float, default=120.0)
    args = parser.parse_args(argv)

    root = Path(args.artifact_dir)
    if root.exists():
        subprocess.run(['rm', '-rf', str(root)], cwd='/home/ubuntu/phantom_ws', check=False)
    root.mkdir(parents=True, exist_ok=True)

    events = []
    procs = []
    handles = []
    zero_log = []
    success = False
    try:
        _publish_zero(3, 0.15, zero_log)
        launch_cmd = _ros_args([
            'ros2',
            'launch',
            'phantom_bringup',
            'integrated_escape_test.launch.py',
            'lidar_driver:=%s' % args.lidar_driver,
            'camera_device:=%s' % args.camera_device,
            'cmd_vel_topic:=/phantom/disabled_cmd_vel',
            'artifacts_dir:=/home/ubuntu/phantom_ws/%s' % args.artifact_dir,
        ])
        launch_proc, launch_handle = _popen(launch_cmd, root / 'launch.log')
        procs.append(('launch', launch_proc))
        handles.append(launch_handle)
        events.append({'event': 'launch_started', 'pid': launch_proc.pid, 'time': time.time()})

        if not _wait_ready(root, args.ready_timeout):
            events.append({'event': 'ready_timeout', 'time': time.time()})
            return 2
        _record_topic_info(root)

        recorder_proc, recorder_handle = _popen(
            _ros_args(['python3', 'scripts/record_debug_topics.py', args.artifact_dir, '--duration', str(args.duration + 60.0)]),
            root / 'recorder_stdout.txt',
        )
        procs.append(('recorder', recorder_proc))
        handles.append(recorder_handle)
        events.append({'event': 'recorder_started', 'pid': recorder_proc.pid, 'time': time.time()})
        time.sleep(1.0)

        relay_proc, relay_handle = _popen(
            _ros_args([
                'python3',
                'scripts/relay_cmd_vel_10s.py',
                '--source',
                '/phantom/disabled_cmd_vel',
                '--dest',
                '/controller/cmd_vel',
                '--duration',
                str(args.duration),
            ]),
            root / 'relay_60s_stdout.txt',
        )
        procs.append(('relay', relay_proc))
        handles.append(relay_handle)
        motion_start = time.time()
        events.append({'event': 'continuous_motion_started', 'pid': relay_proc.pid, 'time': motion_start})
        try:
            relay_rc = relay_proc.wait(timeout=args.duration + 35.0)
        except subprocess.TimeoutExpired:
            motion_end = time.time()
            events.append({
                'event': 'relay_wait_timeout',
                'duration_s': round(motion_end - motion_start, 3),
                'time': motion_end,
            })
            _terminate_group(relay_proc, 'relay', events)
            relay_rc = 0 if motion_end - motion_start >= args.duration else 1
        else:
            motion_end = time.time()
        events.append({
            'event': 'continuous_motion_finished',
            'returncode': relay_rc,
            'duration_s': round(motion_end - motion_start, 3),
            'time': motion_end,
        })

        _publish_zero(5, 0.2, zero_log)
        time.sleep(2.0)
        _terminate_group(launch_proc, 'launch', events)
        time.sleep(1.0)
        _publish_zero(5, 0.2, zero_log)
        time.sleep(1.0)
        _terminate_group(recorder_proc, 'recorder', events)
        success = relay_rc == 0
        return 0 if success else 3
    except KeyboardInterrupt:
        events.append({'event': 'keyboard_interrupt', 'time': time.time()})
        return 130
    except Exception as exc:
        events.append({'event': 'exception', 'detail': repr(exc), 'time': time.time()})
        return 1
    finally:
        for name, proc in reversed(procs):
            if proc.poll() is None:
                _terminate_group(proc, name, events)
        _publish_zero(5, 0.2, zero_log)
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        (root / 'run_events.json').write_text(json.dumps(events, indent=2, sort_keys=True), encoding='utf-8')
        (root / 'stop_confirmed.json').write_text(
            json.dumps({'zero_commands': zero_log, 'success': len(zero_log) >= 3}, indent=2, sort_keys=True),
            encoding='utf-8',
        )
        _run('python3 scripts/analyze_real_10s_debug.py %s' % shlex.quote(args.artifact_dir), timeout=30, output_path=root / 'analyze_stdout.txt')
        _run('cp %s %s 2>/dev/null || true' % (
            shlex.quote(str(root / 'first_yolo_detection.png')),
            shlex.quote(str(root / 'yolo_debug_raw.png')),
        ), timeout=5)
        _run('python3 scripts/export_debug_images.py %s' % shlex.quote(args.artifact_dir), timeout=30, output_path=root / 'export_images_stdout.txt')
        print(root / 'analysis_summary.txt')
        print(root / 'lidar_debug.png')
        print(root / 'yolo_debug.png')
        print(root / 'z_bump_debug.png')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
