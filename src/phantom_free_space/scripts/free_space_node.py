#!/usr/bin/env python3
import json
import math
import os
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray, String


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class FreeSpaceNode:
    def __init__(self):
        rospy.init_node("free_space_node")

        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.num_sectors = int(rospy.get_param("~num_sectors", 36))
        self.safety_distance = float(rospy.get_param("~safety_distance", 0.45))
        # Vehicle envelope inflation: body half-width, wheel overhang, control error, and margin.
        self.obstacle_inflation_m = float(rospy.get_param("~obstacle_inflation_m", 0.22))
        self.front_stop_clearance_m = float(rospy.get_param("~front_stop_clearance_m", 0.30))
        self.side_stop_clearance_m = float(rospy.get_param("~side_stop_clearance_m", 0.24))
        self.direction_lock_time_sec = float(rospy.get_param("~direction_lock_time_sec", 0.45))
        self.direction_switch_margin = float(rospy.get_param("~direction_switch_margin", 0.18))
        self.heading_change_penalty = float(rospy.get_param("~heading_change_penalty", 0.08))
        self.configured_max_range = float(rospy.get_param("~max_range", 6.0))
        self.min_valid_points = int(rospy.get_param("~min_valid_points_per_sector", 3))
        self.forward_weight = float(rospy.get_param("~forward_weight", 0.35))
        self.clearance_weight = float(rospy.get_param("~clearance_weight", 0.45))
        self.width_weight = float(rospy.get_param("~width_weight", 0.20))
        self.image_size = int(rospy.get_param("~image_size", 800))
        self.meters_per_pixel = float(rospy.get_param("~meters_per_pixel", 0.01))
        self.debug_dir = os.path.expanduser(rospy.get_param("~debug_dir", "~/free_space_debug"))
        self.save_every_n_frames = max(1, int(rospy.get_param("~save_every_n_frames", 5)))
        self.publish_rate_limit_hz = float(rospy.get_param("~publish_rate_limit_hz", 10.0))

        self.latest_image_path = os.path.join(self.debug_dir, "free_space_latest.png")
        self.frames_dir = os.path.join(self.debug_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

        self.sectors_pub = rospy.Publisher("/free_space/sectors", Float32MultiArray, queue_size=1)
        self.best_heading_pub = rospy.Publisher("/free_space/best_heading", Float32, queue_size=1)
        self.status_pub = rospy.Publisher("/free_space/status", String, queue_size=1)
        self.image_path_pub = rospy.Publisher("/free_space/debug_image_path", String, queue_size=1)

        self.frame_count = 0
        self.last_publish_time = 0.0
        self.last_scan_stamp = 0.0
        self.held_heading = 0.0
        self.last_heading_switch_time = -999.0
        self.heading_switch_count = 0

        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.loginfo("free_space_node subscribed to %s", self.scan_topic)

    def scan_callback(self, msg):
        now = time.time()
        if self.publish_rate_limit_hz > 0.0:
            min_period = 1.0 / self.publish_rate_limit_hz
            if now - self.last_publish_time < min_period:
                return
        self.last_publish_time = now
        self.last_scan_stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()

        angles, ranges, valid_mask, effective_max_range = self.preprocess_scan(msg)
        sectors = self.compute_sectors(angles, ranges, valid_mask, effective_max_range)
        best_sector = self.select_best_sector(sectors)

        image_path = ""
        self.frame_count += 1
        if self.frame_count == 1 or self.frame_count % self.save_every_n_frames == 0:
            image_path = self.render_debug_image(angles, ranges, valid_mask, sectors, best_sector)

        self.publish_outputs(sectors, best_sector, image_path)

    def preprocess_scan(self, msg):
        count = len(msg.ranges)
        if count == 0:
            return np.array([]), np.array([]), np.array([], dtype=bool), self.configured_max_range

        angles = msg.angle_min + np.arange(count, dtype=np.float32) * msg.angle_increment
        raw_ranges = np.asarray(msg.ranges, dtype=np.float32)

        range_min = msg.range_min if msg.range_min > 0.0 else 0.05
        msg_max = msg.range_max if msg.range_max > range_min else self.configured_max_range
        if self.configured_max_range > 0.0:
            effective_max_range = min(msg_max, self.configured_max_range)
        else:
            effective_max_range = msg_max

        valid_mask = (
            np.isfinite(raw_ranges)
            & (raw_ranges >= range_min)
            & (raw_ranges <= msg_max)
            & (raw_ranges <= effective_max_range)
        )
        return angles, raw_ranges, valid_mask, effective_max_range

    def compute_sectors(self, angles, ranges, valid_mask, effective_max_range):
        if len(angles) == 0 or self.num_sectors <= 0:
            return []

        angle_min = float(np.min(angles))
        angle_max = float(np.max(angles))
        span = max(angle_max - angle_min, 1e-6)
        sector_width = span / float(self.num_sectors)
        sectors = []

        for sector_id in range(self.num_sectors):
            sec_min = angle_min + sector_id * sector_width
            sec_max = angle_min + (sector_id + 1) * sector_width
            if sector_id == self.num_sectors - 1:
                in_sector = (angles >= sec_min) & (angles <= sec_max)
            else:
                in_sector = (angles >= sec_min) & (angles < sec_max)

            valid = in_sector & valid_mask
            valid_ranges = ranges[valid]
            valid_count = int(valid_ranges.size)
            angle_center = normalize_angle((sec_min + sec_max) * 0.5)

            if valid_count >= self.min_valid_points:
                raw_min_range = float(np.min(valid_ranges))
                raw_mean_range = float(np.mean(valid_ranges))
                min_range = self.inflate_clearance(raw_min_range)
                mean_range = self.inflate_clearance(raw_mean_range)
                clearance = min_range
                obstacle_risk = 1.0 / max(clearance, 1e-3)
                is_free = min_range > self.safety_distance
                unknown = False
            else:
                raw_min_range = 0.0
                raw_mean_range = 0.0
                min_range = 0.0
                mean_range = 0.0
                clearance = 0.0
                obstacle_risk = 0.0
                is_free = False
                unknown = True

            clearance_score = clamp(clearance / max(effective_max_range, 1e-3), 0.0, 1.0)
            forward_score = max(0.0, math.cos(angle_center))
            width_score = 0.0
            score = 0.0
            if is_free:
                score = (
                    self.clearance_weight * clearance_score
                    + self.forward_weight * forward_score
                    + self.width_weight * width_score
                )

            sectors.append(
                {
                    "sector_id": sector_id,
                    "angle_min": normalize_angle(sec_min),
                    "angle_max": normalize_angle(sec_max),
                    "angle_center": angle_center,
                    "valid_count": valid_count,
                    "min_range": min_range,
                    "mean_range": mean_range,
                    "raw_min_range": raw_min_range,
                    "raw_mean_range": raw_mean_range,
                    "clearance": clearance,
                    "obstacle_risk": obstacle_risk,
                    "is_free": bool(is_free),
                    "unknown": bool(unknown),
                    "score": float(score),
                    "width_score": width_score,
                }
            )

        self._apply_width_scores(sectors)
        return sectors

    def select_best_sector(self, sectors):
        free_groups = self._free_groups(sectors)
        if not free_groups:
            return {
                "state": "no_free_space",
                "best_heading": 0.0,
                "best_score": 0.0,
                "sector_id": -1,
                "group_size": 0,
            }

        best = None
        for group in free_groups:
            group_size = len(group)
            width_score = clamp(group_size / float(max(self.num_sectors, 1)), 0.0, 1.0)
            mean_score = float(np.mean([s["score"] for s in group]))
            mean_clearance = float(np.mean([s["clearance"] for s in group]))
            heading = self._group_heading(group)
            forward_score = max(0.0, math.cos(heading))
            smooth_penalty = self.heading_change_penalty * abs(normalize_angle(heading - self.held_heading))
            group_score = mean_score + self.width_weight * width_score + 0.1 * forward_score - smooth_penalty
            candidate = {
                "state": "ok",
                "best_heading": heading,
                "best_score": group_score,
                "sector_id": int(group[len(group) // 2]["sector_id"]),
                "group_size": group_size,
                "mean_clearance": mean_clearance,
            }
            if best is None or candidate["best_score"] > best["best_score"]:
                best = candidate
        held_group = min(
            (self._candidate_from_group(group) for group in free_groups),
            key=lambda item: abs(normalize_angle(item["best_heading"] - self.held_heading)),
        )
        now = time.time()
        if now - self.last_heading_switch_time < self.direction_lock_time_sec:
            if held_group["best_score"] + self.direction_switch_margin >= best["best_score"]:
                best = held_group
        if abs(normalize_angle(best["best_heading"] - self.held_heading)) > 0.35:
            self.heading_switch_count += 1
            self.last_heading_switch_time = now
        self.held_heading = best["best_heading"]
        return best

    def _candidate_from_group(self, group):
        group_size = len(group)
        width_score = clamp(group_size / float(max(self.num_sectors, 1)), 0.0, 1.0)
        mean_score = float(np.mean([s["score"] for s in group]))
        mean_clearance = float(np.mean([s["clearance"] for s in group]))
        heading = self._group_heading(group)
        forward_score = max(0.0, math.cos(heading))
        group_score = mean_score + self.width_weight * width_score + 0.1 * forward_score
        return {
            "state": "ok",
            "best_heading": heading,
            "best_score": group_score,
            "sector_id": int(group[len(group) // 2]["sector_id"]),
            "group_size": group_size,
            "mean_clearance": mean_clearance,
        }

    def render_debug_image(self, angles, ranges, valid_mask, sectors, best_sector):
        image_size = self.image_size
        origin_x = image_size // 2
        origin_y = int(image_size * 0.62)
        scale = 1.0 / max(self.meters_per_pixel, 1e-4)

        img = np.full((image_size, image_size, 3), 245, dtype=np.uint8)
        cv2.circle(img, (origin_x, origin_y), 8, (30, 30, 30), -1)
        cv2.arrowedLine(img, (origin_x, origin_y), (origin_x, origin_y - 45), (30, 30, 30), 2, tipLength=0.25)

        max_draw_range = max(self.configured_max_range, self.safety_distance * 3.0)
        radius_px = int(max_draw_range * scale)

        overlay = img.copy()
        for sector in sectors:
            color = (80, 180, 80) if sector["is_free"] else (50, 120, 230)
            if sector["unknown"]:
                color = (180, 180, 180)
            p1 = self._point_on_image(origin_x, origin_y, sector["angle_min"], radius_px)
            p2 = self._point_on_image(origin_x, origin_y, sector["angle_max"], radius_px)
            pts = np.array([(origin_x, origin_y), p1, p2], dtype=np.int32)
            cv2.fillConvexPoly(overlay, pts, color)
            cv2.line(img, (origin_x, origin_y), p1, color, 1)
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0.0, img)

        safe_radius = int(self.safety_distance * scale)
        cv2.circle(img, (origin_x, origin_y), safe_radius, (0, 165, 255), 1)

        for angle, distance, valid in zip(angles, ranges, valid_mask):
            if not valid:
                continue
            px, py = self._metric_to_image(origin_x, origin_y, float(angle), float(distance), scale)
            if 0 <= px < image_size and 0 <= py < image_size:
                cv2.circle(img, (px, py), 2, (40, 40, 40), -1)

        if best_sector and best_sector.get("state") == "ok":
            heading = best_sector["best_heading"]
            end = self._point_on_image(origin_x, origin_y, heading, int(max(0.8, self.safety_distance * 2.0) * scale))
            cv2.arrowedLine(img, (origin_x, origin_y), end, (255, 80, 0), 4, tipLength=0.25)

        free_count = sum(1 for sector in sectors if sector["is_free"])
        heading_deg = math.degrees(best_sector.get("best_heading", 0.0)) if best_sector else 0.0
        lines = [
            "scan: %s" % self.scan_topic,
            "best: %.1f deg  state: %s" % (heading_deg, best_sector.get("state", "unknown") if best_sector else "unknown"),
            "free sectors: %d/%d  safety: %.2fm" % (free_count, self.num_sectors, self.safety_distance),
            "stamp: %.3f" % self.last_scan_stamp,
        ]
        for index, text in enumerate(lines):
            cv2.putText(img, text, (12, 24 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        frame_path = os.path.join(self.frames_dir, "free_space_%s_%03d.png" % (timestamp, self.frame_count))
        cv2.imwrite(self.latest_image_path, img)
        cv2.imwrite(frame_path, img)
        return self.latest_image_path

    def publish_outputs(self, sectors, best_sector, image_path):
        array_msg = Float32MultiArray()
        data = []
        for sector in sectors:
            data.extend(
                [
                    float(sector["sector_id"]),
                    float(sector["angle_min"]),
                    float(sector["angle_max"]),
                    float(sector["angle_center"]),
                    float(sector["min_range"]),
                    float(sector["mean_range"]),
                    float(sector["score"]),
                    1.0 if sector["is_free"] else 0.0,
                ]
            )
        array_msg.data = data
        self.sectors_pub.publish(array_msg)

        heading = float(best_sector.get("best_heading", 0.0)) if best_sector else 0.0
        self.best_heading_pub.publish(Float32(data=heading))

        free_count = sum(1 for sector in sectors if sector["is_free"])
        status = {
            "stamp": self.last_scan_stamp,
            "scan_topic": self.scan_topic,
            "num_sectors": self.num_sectors,
            "best_heading": heading,
            "best_score": float(best_sector.get("best_score", 0.0)) if best_sector else 0.0,
            "free_sector_count": free_count,
            "state": best_sector.get("state", "unknown") if best_sector else "unknown",
            "obstacle_inflation_m": self.obstacle_inflation_m,
            "front_blocked": self._sector_window_blocked(sectors, -0.28, 0.28, self.front_stop_clearance_m),
            "left_front_blocked": self._sector_window_blocked(sectors, 0.28, 0.95, self.side_stop_clearance_m),
            "right_front_blocked": self._sector_window_blocked(sectors, -0.95, -0.28, self.side_stop_clearance_m),
            "heading_switch_count": self.heading_switch_count,
        }
        self.status_pub.publish(String(data=json.dumps(status, sort_keys=True)))

        if image_path:
            self.image_path_pub.publish(String(data=image_path))

    def _apply_width_scores(self, sectors):
        for group in self._free_groups(sectors):
            width_score = clamp(len(group) / float(max(self.num_sectors, 1)), 0.0, 1.0)
            for sector in group:
                sector["width_score"] = width_score
                sector["score"] = float(sector["score"] + self.width_weight * width_score)

    def _free_groups(self, sectors):
        groups = []
        current = []
        for sector in sectors:
            if sector["is_free"]:
                current.append(sector)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        if len(groups) > 1 and sectors[0]["is_free"] and sectors[-1]["is_free"]:
            merged = groups[-1] + groups[0]
            groups = [merged] + groups[1:-1]
        return groups

    def inflate_clearance(self, distance):
        return max(0.0, float(distance) - max(0.0, self.obstacle_inflation_m))

    @staticmethod
    def _sector_window_blocked(sectors, angle_min, angle_max, threshold):
        clearances = [
            sector["clearance"]
            for sector in sectors
            if angle_min <= sector["angle_center"] <= angle_max
        ]
        return bool(clearances and min(clearances) < threshold)

    @staticmethod
    def _group_heading(group):
        vectors = np.array([[math.cos(s["angle_center"]), math.sin(s["angle_center"])] for s in group])
        mean_vec = np.mean(vectors, axis=0)
        return normalize_angle(math.atan2(float(mean_vec[1]), float(mean_vec[0])))

    @staticmethod
    def _metric_to_image(origin_x, origin_y, angle, distance, scale):
        x_m = distance * math.sin(angle)
        y_m = distance * math.cos(angle)
        return int(origin_x + x_m * scale), int(origin_y - y_m * scale)

    @staticmethod
    def _point_on_image(origin_x, origin_y, angle, radius_px):
        return int(origin_x + math.sin(angle) * radius_px), int(origin_y - math.cos(angle) * radius_px)


def main():
    node = FreeSpaceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
