import argparse
import atexit
import importlib.util
import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np


def load_base_module():
    candidates = [
        Path(__file__).resolve().with_name("line_following_v1_7_obstacle_detour.py"),
        (
            Path(__file__).resolve().parents[2]
            / "07_obstacle_detour"
            / "code"
            / "line_following_v1_7_obstacle_detour.py"
        ),
    ]
    base_path = None
    for candidate in candidates:
        if candidate.exists():
            base_path = candidate
            break
    if base_path is None:
        raise RuntimeError(
            "unable to find line_following_v1_7_obstacle_detour.py in the current folder "
            "or the original repo layout"
        )
    spec = importlib.util.spec_from_file_location("obstacle_v17_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load base module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


WAIT_FOR_MAZE_TEMPLATE = "WAIT_FOR_MAZE_TEMPLATE"
MAZE_CONFIRMED_COUNTDOWN = "MAZE_CONFIRMED_COUNTDOWN"
MAZE_NAVIGATION_ACTIVE = "MAZE_NAVIGATION_ACTIVE"
MAZE_FINISHED = "MAZE_FINISHED"

MAZE_TEMPLATE_FILENAME = "maze_template.png"
MAZE_TEMPLATE_MATCH_THRESHOLD = 0.60
MAZE_TEMPLATE_MARGIN = 0.03
MAZE_CONFIRMATION_HITS = 3
MAZE_COUNTDOWN_SECONDS = 5.0
MAZE_TEMPLATE_SIZE = (202, 202)
MAZE_WHITE_GRAY_MIN = 150
MAZE_WHITE_VALUE_MIN = 150
MAZE_WHITE_SAT_MAX = 72
MAZE_WHITE_OPEN_KERNEL = 3
MAZE_WHITE_CLOSE_KERNEL = 7
MAZE_COMPONENT_MIN_AREA_RATIO = 0.035
MAZE_SCANLINE_Y_RATIOS = (0.86, 0.76, 0.66, 0.56, 0.46, 0.36)
MAZE_SCANLINE_HALF_HEIGHT = 4
MAZE_SCANLINE_MIN_WIDTH = 18
MAZE_CENTERLINE_TARGET_WIDTH_RATIO = 0.33
MAZE_CONFIDENCE_TRACK_MIN = 0.25
MAZE_CONFIDENCE_RECOVER_MIN = 0.18
MAZE_LOW_CONFIDENCE_HITS = 4
MAZE_TRACK_CRUISE_SPEED = 28
MAZE_TRACK_MIN_SPEED = 16
MAZE_APPROACH_SPEED = 18
MAZE_COMMAND_SMOOTHING = 0.42
MAZE_HEADING_ERROR_GAIN = 34.0
MAZE_LOOKAHEAD_GAIN = 18.0
MAZE_APPROACH_TRIGGER = 0.15
MAZE_BRANCH_SCORE_STRONG = 0.22
MAZE_FRONT_BLOCK_APPROACH = 0.38
MAZE_FRONT_BLOCK_STRONG = 0.62
MAZE_FORWARD_DEPTH_KEEP_MIN = 0.46
MAZE_WIDTH_GAIN_BRANCH_MIN = 0.18
MAZE_APPROACH_HITS = 2
MAZE_DECISION_HITS = 2
MAZE_COMMIT_RIGHT_MIN_SECONDS = 0.42
MAZE_COMMIT_LEFT_MIN_SECONDS = 0.50
MAZE_COMMIT_UTURN_MIN_SECONDS = 0.95
MAZE_COMMIT_RIGHT_MAX_SECONDS = 1.20
MAZE_COMMIT_LEFT_MAX_SECONDS = 1.30
MAZE_COMMIT_UTURN_MAX_SECONDS = 1.65
MAZE_COMMIT_OUTER_SPEED = 28
MAZE_COMMIT_INNER_SPEED = 6
MAZE_COMMIT_BLOCKED_INNER_SPEED = -4
MAZE_COMMIT_UTURN_OUTER_SPEED = 30
MAZE_COMMIT_UTURN_INNER_SPEED = -18
MAZE_COMMIT_RECENTER_ERROR = 26
MAZE_COMMIT_RECENTER_CONFIDENCE = 0.40
MAZE_RECOVER_PAUSE_SECONDS = 0.10
MAZE_RECOVER_REVERSE_SECONDS = 0.18
MAZE_RECOVER_SEARCH_SECONDS = 0.70
MAZE_RECOVER_REVERSE_SPEED = -10
MAZE_RECOVER_SEARCH_OUTER_SPEED = 22
MAZE_RECOVER_SEARCH_INNER_SPEED = -10
MAZE_EXIT_FORWARD_DEPTH_MIN = 0.95
MAZE_EXIT_WIDTH_RATIO_MIN = 0.55
MAZE_EXIT_CONFIRMATION_HITS = 7
MAZE_EXIT_RUNOUT_SECONDS = 1.0

TEMPLATE_CACHE = {}
MISSING_TEMPLATE_WARNED = set()


def clamp(value, low, high):
    return max(low, min(high, value))


def spans_from_columns(active_columns):
    spans = []
    start = None
    for idx, is_active in enumerate(active_columns):
        if is_active and start is None:
            start = idx
        elif (not is_active) and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(active_columns) - 1))
    return spans


def choose_guided_span(spans, target_x, min_width):
    if not spans:
        return None

    best_span = None
    best_score = -1e9
    for span in spans:
        width = span[1] - span[0] + 1
        if width < min_width:
            continue
        center = (span[0] + span[1]) / 2.0
        score = width - abs(center - target_x) * 0.75
        if score > best_score:
            best_score = score
            best_span = span
    return best_span


def keep_bottom_seed_component(mask):
    if mask is None or mask.size == 0:
        return mask

    frame_h, frame_w = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    min_area = int(frame_h * frame_w * MAZE_COMPONENT_MIN_AREA_RATIO)
    seed_y0 = int(frame_h * 0.78)
    seed_x0 = int(frame_w * 0.30)
    seed_x1 = int(frame_w * 0.70)
    seed_region = labels[seed_y0:frame_h, seed_x0:seed_x1]

    candidate_labels = []
    if seed_region.size > 0:
        candidate_labels = [int(label) for label in np.unique(seed_region) if label > 0]

    best_label = 0
    best_score = -1.0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_seed = label in candidate_labels
        bottom_reach = (y + h) / float(frame_h)
        width_bias = min(1.0, w / float(frame_w * 0.85))
        score = area + bottom_reach * frame_w * frame_h * 0.25 + width_bias * frame_w * frame_h * 0.10
        if touches_seed:
            score += frame_w * frame_h
        if score > best_score:
            best_score = score
            best_label = label

    if best_label <= 0:
        return mask
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def build_drivable_mask(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, gray_mask = cv2.threshold(gray_blur, MAZE_WHITE_GRAY_MIN, 255, cv2.THRESH_BINARY)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 0, MAZE_WHITE_VALUE_MIN], dtype=np.uint8),
        np.array([180, MAZE_WHITE_SAT_MAX, 255], dtype=np.uint8),
    )
    drivable_mask = cv2.bitwise_or(gray_mask, hsv_mask)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MAZE_WHITE_OPEN_KERNEL, MAZE_WHITE_OPEN_KERNEL),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MAZE_WHITE_CLOSE_KERNEL, MAZE_WHITE_CLOSE_KERNEL),
    )
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    drivable_mask = keep_bottom_seed_component(drivable_mask)
    return drivable_mask


def sample_corridor_span(mask, y, target_x):
    frame_h, frame_w = mask.shape[:2]
    y0 = clamp(int(y) - MAZE_SCANLINE_HALF_HEIGHT, 0, frame_h - 1)
    y1 = clamp(int(y) + MAZE_SCANLINE_HALF_HEIGHT + 1, y0 + 1, frame_h)
    band = mask[y0:y1, :]
    if band.size == 0:
        return None
    active_columns = (np.mean(band > 0, axis=0) >= 0.45).astype(np.uint8)
    spans = spans_from_columns(active_columns)
    best_span = choose_guided_span(spans, target_x, MAZE_SCANLINE_MIN_WIDTH)
    if best_span is None:
        return None
    left, right = best_span
    center = (left + right) / 2.0
    return {
        "y": int((y0 + y1 - 1) / 2),
        "left": int(left),
        "right": int(right),
        "center": float(center),
        "width": int(right - left + 1),
        "all_spans": spans,
    }


def compute_perception_confidence(frame_shape, drivable_mask, scanlines):
    frame_h, frame_w = frame_shape[:2]
    area_ratio = cv2.countNonZero(drivable_mask) / float(frame_h * frame_w)
    valid_scanlines = [sample for sample in scanlines if sample is not None]
    coverage = len(valid_scanlines) / float(len(scanlines)) if scanlines else 0.0
    if valid_scanlines:
        centers = np.array([sample["center"] for sample in valid_scanlines], dtype=np.float32)
        widths = np.array([sample["width"] for sample in valid_scanlines], dtype=np.float32)
        center_std = float(np.std(centers))
        width_mean = float(np.mean(widths))
    else:
        center_std = frame_w
        width_mean = 0.0
    stability = 1.0 - clamp(center_std / float(frame_w * 0.16), 0.0, 1.0)
    width_quality = clamp(width_mean / float(frame_w * 0.42), 0.0, 1.0)
    area_quality = clamp(area_ratio / 0.36, 0.0, 1.0)
    return clamp(
        0.32 * coverage + 0.28 * stability + 0.22 * width_quality + 0.18 * area_quality,
        0.0,
        1.0,
    )


def analyze_maze_frame(frame_bgr, previous_center_x=None):
    frame_h, frame_w = frame_bgr.shape[:2]
    drivable_mask = build_drivable_mask(frame_bgr)
    frame_center_x = frame_w / 2.0
    target_x = frame_center_x if previous_center_x is None else float(previous_center_x)

    scanlines = []
    guided_target_x = target_x
    for ratio in MAZE_SCANLINE_Y_RATIOS:
        sample = sample_corridor_span(drivable_mask, frame_h * ratio, guided_target_x)
        scanlines.append(sample)
        if sample is not None:
            guided_target_x = sample["center"]

    valid_scanlines = [sample for sample in scanlines if sample is not None]
    corridor_center_x = None
    corridor_width = 0
    heading_error = 0.0
    lookahead_error = 0.0
    forward_depth = 0.0
    centerline_points = []
    width_gain = 0.0
    left_branch_score = 0.0
    right_branch_score = 0.0
    front_block_score = 1.0

    if valid_scanlines:
        bottom_sample = valid_scanlines[0]
        corridor_center_x = int(round(bottom_sample["center"]))
        corridor_width = int(bottom_sample["width"])
        heading_error = float(bottom_sample["center"] - frame_center_x)
        centerline_points = [(int(round(sample["center"])), sample["y"]) for sample in valid_scanlines]

        top_sample = valid_scanlines[-1]
        lookahead_error = float(top_sample["center"] - bottom_sample["center"])
        forward_depth = clamp(1.0 - (top_sample["y"] / float(frame_h)), 0.0, 1.0)

        widths = [float(sample["width"]) for sample in valid_scanlines]
        bottom_width = widths[0]
        upper_width = float(np.mean(widths[-2:])) if len(widths) >= 2 else widths[0]
        width_gain = clamp((upper_width - bottom_width) / float(frame_w * 0.35), 0.0, 1.0)

        low_left = float(bottom_sample["left"])
        low_right = float(bottom_sample["right"])
        top_left = float(top_sample["left"])
        top_right = float(top_sample["right"])
        left_expansion = clamp((low_left - top_left) / float(frame_w * 0.30), 0.0, 1.0)
        right_expansion = clamp((top_right - low_right) / float(frame_w * 0.30), 0.0, 1.0)
        left_branch_score = clamp(left_expansion * 0.75 + width_gain * 0.25, 0.0, 1.0)
        right_branch_score = clamp(right_expansion * 0.75 + width_gain * 0.25, 0.0, 1.0)

        front_roi = drivable_mask[
            int(frame_h * 0.20) : int(frame_h * 0.56),
            int(frame_w * 0.34) : int(frame_w * 0.66),
        ]
        front_white_ratio = (
            cv2.countNonZero(front_roi) / float(front_roi.size) if front_roi.size > 0 else 0.0
        )
        front_signal = clamp(max(forward_depth, front_white_ratio / 0.35), 0.0, 1.0)
        front_block_score = clamp(1.0 - front_signal, 0.0, 1.0)

    confidence = compute_perception_confidence(frame_bgr.shape, drivable_mask, scanlines)
    corridor_width_ratio = corridor_width / float(frame_w) if frame_w > 0 else 0.0
    exit_candidate = (
        forward_depth >= MAZE_EXIT_FORWARD_DEPTH_MIN
        and corridor_width_ratio >= MAZE_EXIT_WIDTH_RATIO_MIN
        and left_branch_score < 0.10
        and right_branch_score < 0.10
        and confidence >= 0.55
    )

    return {
        "drivable_mask": drivable_mask,
        "wall_mask": cv2.bitwise_not(drivable_mask),
        "scanlines": scanlines,
        "centerline_points": centerline_points,
        "centerline_x": corridor_center_x,
        "corridor_center_x": corridor_center_x,
        "corridor_width": corridor_width,
        "corridor_width_ratio": corridor_width_ratio,
        "heading_error": heading_error,
        "lookahead_error": lookahead_error,
        "corridor_error": heading_error,
        "forward_depth": forward_depth,
        "left_branch_score": left_branch_score,
        "right_branch_score": right_branch_score,
        "front_block_score": front_block_score,
        "width_gain": width_gain,
        "confidence": confidence,
        "corridor_confidence": confidence,
        "left_open": left_branch_score >= MAZE_BRANCH_SCORE_STRONG,
        "right_open": right_branch_score >= MAZE_BRANCH_SCORE_STRONG,
        "front_blocked": front_block_score >= MAZE_FRONT_BLOCK_STRONG,
        "front_danger": front_block_score >= 0.82,
        "front_clear": front_block_score <= MAZE_FRONT_BLOCK_APPROACH * 0.55,
        "exit_candidate": exit_candidate,
    }


class MazeNavigator:
    def __init__(self, detect_exit=True):
        self.mode = "TRACK_CORRIDOR"
        self.detect_exit = detect_exit
        self.previous_center_x = None
        self.last_heading_direction = 1
        self.last_left_speed = 0.0
        self.last_right_speed = 0.0
        self.low_confidence_hits = 0
        self.approach_hits = 0
        self.right_hits = 0
        self.left_hits = 0
        self.block_hits = 0
        self.commit_direction = ""
        self.commit_min_until = -999.0
        self.commit_max_until = -999.0
        self.recover_phase = ""
        self.recover_direction = 1
        self.recover_until = -999.0
        self.exit_hits = 0
        self.exit_until = -999.0
        self.last_front_block_score = 0.0
        self.finished = False

    def _emit_command(self, left_speed, right_speed, mode):
        alpha = MAZE_COMMAND_SMOOTHING
        left = (self.last_left_speed * alpha) + (float(left_speed) * (1.0 - alpha))
        right = (self.last_right_speed * alpha) + (float(right_speed) * (1.0 - alpha))
        left = clamp(int(round(left)), -base.MAX_SPEED, base.MAX_SPEED)
        right = clamp(int(round(right)), -base.MAX_SPEED, base.MAX_SPEED)
        self.last_left_speed = float(left)
        self.last_right_speed = float(right)
        return {"left_speed": left, "right_speed": right, "mode": mode, "done": False}

    def _track_command(self, perception, approach=False):
        frame_half = base.PROCESS_SIZE[0] / 2.0
        heading_norm = 0.0 if frame_half <= 0 else perception["heading_error"] / frame_half
        lookahead_norm = 0.0 if frame_half <= 0 else perception["lookahead_error"] / frame_half
        steer = (heading_norm * MAZE_HEADING_ERROR_GAIN) + (lookahead_norm * MAZE_LOOKAHEAD_GAIN)
        width_ratio = perception.get("corridor_width_ratio", 0.0)
        confidence = perception.get("confidence", 0.0)
        speed_factor = clamp(0.45 + confidence * 0.45 + width_ratio * 0.25, 0.0, 1.0)
        base_speed = MAZE_TRACK_MIN_SPEED + (MAZE_TRACK_CRUISE_SPEED - MAZE_TRACK_MIN_SPEED) * speed_factor
        if approach:
            base_speed = min(base_speed, MAZE_APPROACH_SPEED)
        left = base_speed + steer
        right = base_speed - steer
        return self._emit_command(left, right, "APPROACH_JUNCTION" if approach else "TRACK_CORRIDOR")

    def _start_commit(self, direction, now):
        self.commit_direction = direction
        if direction == "RIGHT":
            self.mode = "COMMIT_RIGHT"
            self.commit_min_until = now + MAZE_COMMIT_RIGHT_MIN_SECONDS
            self.commit_max_until = now + MAZE_COMMIT_RIGHT_MAX_SECONDS
        elif direction == "LEFT":
            self.mode = "COMMIT_LEFT"
            self.commit_min_until = now + MAZE_COMMIT_LEFT_MIN_SECONDS
            self.commit_max_until = now + MAZE_COMMIT_LEFT_MAX_SECONDS
        else:
            self.mode = "COMMIT_UTURN"
            self.commit_min_until = now + MAZE_COMMIT_UTURN_MIN_SECONDS
            self.commit_max_until = now + MAZE_COMMIT_UTURN_MAX_SECONDS
        return self._commit_command()

    def _commit_command(self):
        front_blocked = self.last_front_block_score >= MAZE_FRONT_BLOCK_STRONG
        if self.commit_direction == "RIGHT":
            inner = MAZE_COMMIT_BLOCKED_INNER_SPEED if front_blocked else MAZE_COMMIT_INNER_SPEED
            return self._emit_command(MAZE_COMMIT_OUTER_SPEED, inner, "COMMIT_RIGHT")
        if self.commit_direction == "LEFT":
            inner = MAZE_COMMIT_BLOCKED_INNER_SPEED if front_blocked else MAZE_COMMIT_INNER_SPEED
            return self._emit_command(inner, MAZE_COMMIT_OUTER_SPEED, "COMMIT_LEFT")
        return self._emit_command(MAZE_COMMIT_UTURN_OUTER_SPEED, MAZE_COMMIT_UTURN_INNER_SPEED, "COMMIT_UTURN")

    def _start_recover(self, now):
        self.mode = "RECOVER"
        self.recover_phase = "pause"
        self.recover_until = now + MAZE_RECOVER_PAUSE_SECONDS
        return self._emit_command(0, 0, "RECOVER")

    def _update_recover(self, perception, now):
        if (
            perception["confidence"] >= MAZE_CONFIDENCE_TRACK_MIN
            and perception["forward_depth"] >= 0.20
            and perception["centerline_x"] is not None
        ):
            self.mode = "TRACK_CORRIDOR"
            return self._track_command(perception)

        if now >= self.recover_until:
            if self.recover_phase == "pause":
                self.recover_phase = "reverse"
                self.recover_until = now + MAZE_RECOVER_REVERSE_SECONDS
            elif self.recover_phase == "reverse":
                self.recover_phase = "search"
                self.recover_until = now + MAZE_RECOVER_SEARCH_SECONDS
            else:
                self.recover_phase = "reverse"
                self.recover_until = now + MAZE_RECOVER_REVERSE_SECONDS

        if self.recover_phase == "pause":
            return self._emit_command(0, 0, "RECOVER")
        if self.recover_phase == "reverse":
            return self._emit_command(
                MAZE_RECOVER_REVERSE_SPEED,
                MAZE_RECOVER_REVERSE_SPEED,
                "RECOVER_REVERSE",
            )
        if self.recover_direction > 0:
            return self._emit_command(
                MAZE_RECOVER_SEARCH_OUTER_SPEED,
                MAZE_RECOVER_SEARCH_INNER_SPEED,
                "RECOVER_SEARCH_R",
            )
        return self._emit_command(
            MAZE_RECOVER_SEARCH_INNER_SPEED,
            MAZE_RECOVER_SEARCH_OUTER_SPEED,
            "RECOVER_SEARCH_L",
        )

    def update(self, perception, now):
        if self.finished:
            return {"left_speed": 0, "right_speed": 0, "mode": "DONE", "done": True}

        self.last_front_block_score = perception["front_block_score"]
        if perception["centerline_x"] is not None:
            self.previous_center_x = perception["centerline_x"]
        if abs(perception["heading_error"]) >= 8:
            self.last_heading_direction = 1 if perception["heading_error"] > 0 else -1

        if self.mode == "EXIT_RUNOUT":
            if now < self.exit_until:
                return self._emit_command(MAZE_TRACK_MIN_SPEED + 6, MAZE_TRACK_MIN_SPEED + 6, "EXIT_RUNOUT")
            self.mode = "DONE"
            self.finished = True
            return {"left_speed": 0, "right_speed": 0, "mode": "DONE", "done": True}

        if self.detect_exit:
            if perception["exit_candidate"]:
                self.exit_hits += 1
            else:
                self.exit_hits = 0
            if self.exit_hits >= MAZE_EXIT_CONFIRMATION_HITS:
                self.mode = "EXIT_RUNOUT"
                self.exit_until = now + MAZE_EXIT_RUNOUT_SECONDS
                return self._emit_command(MAZE_TRACK_MIN_SPEED + 6, MAZE_TRACK_MIN_SPEED + 6, "EXIT_RUNOUT")
        else:
            self.exit_hits = 0

        if perception["confidence"] < MAZE_CONFIDENCE_RECOVER_MIN:
            self.low_confidence_hits += 1
        else:
            self.low_confidence_hits = 0

        if self.mode == "RECOVER":
            return self._update_recover(perception, now)

        if self.low_confidence_hits >= MAZE_LOW_CONFIDENCE_HITS:
            if perception["heading_error"] != 0:
                self.recover_direction = 1 if perception["heading_error"] > 0 else -1
            else:
                self.recover_direction = self.last_heading_direction
            return self._start_recover(now)

        if self.mode.startswith("COMMIT_"):
            recentered = (
                perception["confidence"] >= MAZE_COMMIT_RECENTER_CONFIDENCE
                and abs(perception["heading_error"]) <= MAZE_COMMIT_RECENTER_ERROR
                and perception["forward_depth"] >= MAZE_FORWARD_DEPTH_KEEP_MIN
            )
            if now >= self.commit_min_until and (recentered or now >= self.commit_max_until):
                self.mode = "TRACK_CORRIDOR"
                self.commit_direction = ""
                return self._track_command(perception)
            return self._commit_command()

        junction_candidate = max(
            perception["left_branch_score"],
            perception["right_branch_score"],
            perception["front_block_score"],
            perception["width_gain"],
        ) >= MAZE_APPROACH_TRIGGER

        self.right_hits = self.right_hits + 1 if perception["right_branch_score"] >= MAZE_BRANCH_SCORE_STRONG else 0
        self.left_hits = self.left_hits + 1 if perception["left_branch_score"] >= MAZE_BRANCH_SCORE_STRONG else 0
        self.block_hits = self.block_hits + 1 if perception["front_block_score"] >= MAZE_FRONT_BLOCK_STRONG else 0

        if self.mode == "APPROACH_JUNCTION":
            if self.right_hits >= MAZE_DECISION_HITS:
                self.last_heading_direction = 1
                return self._start_commit("RIGHT", now)
            if (
                self.block_hits >= MAZE_DECISION_HITS
                and perception["forward_depth"] < MAZE_FORWARD_DEPTH_KEEP_MIN
            ):
                if self.left_hits >= MAZE_DECISION_HITS:
                    self.last_heading_direction = -1
                    return self._start_commit("LEFT", now)
                return self._start_commit("UTURN", now)
            if (
                self.left_hits >= MAZE_DECISION_HITS
                and perception["forward_depth"] < MAZE_FORWARD_DEPTH_KEEP_MIN
            ):
                self.last_heading_direction = -1
                return self._start_commit("LEFT", now)
            if not junction_candidate and perception["front_block_score"] < MAZE_FRONT_BLOCK_APPROACH * 0.75:
                self.mode = "TRACK_CORRIDOR"
                self.approach_hits = 0
                return self._track_command(perception)
            return self._track_command(perception, approach=True)

        if junction_candidate:
            self.approach_hits += 1
        else:
            self.approach_hits = 0

        if self.approach_hits >= MAZE_APPROACH_HITS:
            self.mode = "APPROACH_JUNCTION"
            return self._track_command(perception, approach=True)

        self.mode = "TRACK_CORRIDOR"
        return self._track_command(perception)


def ensure_default_maze_template():
    template_path = Path(__file__).with_name(MAZE_TEMPLATE_FILENAME)
    if template_path.exists():
        return template_path

    canvas = np.zeros(MAZE_TEMPLATE_SIZE, dtype=np.uint8)

    cv2.rectangle(canvas, (18, 18), (184, 184), 255, 10)

    cv2.line(canvas, (54, 63), (154, 63), 255, 8)
    cv2.line(canvas, (54, 63), (54, 77), 255, 8)
    cv2.line(canvas, (112, 63), (112, 144), 255, 8)
    cv2.line(canvas, (154, 63), (154, 126), 255, 8)
    cv2.line(canvas, (54, 92), (54, 144), 255, 8)
    cv2.line(canvas, (54, 144), (93, 144), 255, 8)
    cv2.line(canvas, (112, 144), (154, 144), 255, 8)

    cv2.rectangle(canvas, (76, 85), (103, 123), 255, -1)
    cv2.rectangle(canvas, (121, 83), (148, 123), 255, -1)
    cv2.rectangle(canvas, (91, 120), (103, 144), 255, -1)

    if not cv2.imwrite(str(template_path), canvas):
        raise RuntimeError(f"unable to write default maze template to {template_path}")
    return template_path


def load_local_template(template_filename):
    if template_filename in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[template_filename]

    template_path = Path(__file__).with_name(template_filename)
    if not template_path.exists():
        if template_filename not in MISSING_TEMPLATE_WARNED:
            print(f"[warn] template missing: {template_path}")
            MISSING_TEMPLATE_WARNED.add(template_filename)
        TEMPLATE_CACHE[template_filename] = None
        return None

    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
        TEMPLATE_CACHE[template_filename] = None
        return None

    TEMPLATE_CACHE[template_filename] = template
    return template


def compute_template_scores(binary):
    templates = {
        "MAZE": load_local_template(MAZE_TEMPLATE_FILENAME),
        "ALARM": base.load_sign_template(base.ALARM_TEMPLATE_FILENAME),
        "TRAFFIC": base.load_sign_template(base.TRAFFIC_TEMPLATE_FILENAME),
        "MUSIC": base.load_sign_template(base.MUSIC_TEMPLATE_FILENAME),
        "OBSTACLE": base.load_sign_template(base.OBSTACLE_TEMPLATE_FILENAME),
    }

    scores = {}
    for label, template in templates.items():
        if template is None:
            continue
        sample = binary
        if sample.shape != template.shape:
            sample = cv2.resize(
                sample,
                (template.shape[1], template.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        xor_frame = cv2.bitwise_xor(sample, template)
        difference_pixels = cv2.countNonZero(xor_frame)
        scores[label] = 1.0 - difference_pixels / float(xor_frame.shape[0] * xor_frame.shape[1])
    return scores


def classify_maze_sign(sign_roi):
    if sign_roi is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "no_sign",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "other_label": "UNKNOWN",
            "other_confidence": 0.0,
            "sign_found": False,
        }

    roi = sign_roi["warped"]
    if roi is None or roi.size == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "empty_roi",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "other_label": "UNKNOWN",
            "other_confidence": 0.0,
            "sign_found": True,
        }

    binary = base.preprocess_sign_symbol(roi)
    if binary is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "empty_inner",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "other_label": "UNKNOWN",
            "other_confidence": 0.0,
            "sign_found": True,
        }

    scores = compute_template_scores(binary)
    other_result = base.classify_sign_symbol(None, sign_roi)
    maze_score = scores.get("MAZE", -1.0)
    best_other_label = "NONE"
    best_other_score = -1.0
    for label, score in scores.items():
        if label == "MAZE":
            continue
        if score > best_other_score:
            best_other_score = score
            best_other_label = label

    maze_confident = (
        maze_score >= MAZE_TEMPLATE_MATCH_THRESHOLD
        and maze_score >= best_other_score + MAZE_TEMPLATE_MARGIN
    )

    if maze_confident:
        return {
            "label": "MAZE",
            "confidence": maze_score,
            "reason": "xor_template_match_maze",
            "scores": scores,
            "binary": binary,
            "maze_confident": True,
            "maze_score": maze_score,
            "best_other_label": best_other_label,
            "best_other_score": best_other_score,
            "other_label": other_result.get("label", "UNKNOWN"),
            "other_confidence": other_result.get("confidence", 0.0),
            "sign_found": True,
        }

    return {
        "label": other_result.get("label", "UNKNOWN"),
        "confidence": other_result.get("confidence", max(0.0, maze_score)),
        "reason": other_result.get("reason", "maze_not_confident"),
        "scores": scores,
        "binary": binary,
        "maze_confident": False,
        "maze_score": maze_score,
        "best_other_label": best_other_label,
        "best_other_score": best_other_score,
        "other_label": other_result.get("label", "UNKNOWN"),
        "other_confidence": other_result.get("confidence", 0.0),
        "sign_found": True,
    }


def draw_overlay(frame, state, sign_roi, sign_result, fps, maze_hits, countdown_deadline, now, perception=None, drive_command=None):
    view = frame.copy()
    if sign_roi is not None and sign_roi.get("quad") is not None:
        cv2.polylines(view, [sign_roi["quad"]], True, (255, 0, 255), 2)
    if sign_roi is not None and sign_roi.get("outer_rect") is not None:
        x, y, w, h = sign_roi["outer_rect"]
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 255), 2)

    top_lines = [
        f"State: {state}",
        f"Maze hits: {maze_hits}/{MAZE_CONFIRMATION_HITS}",
        f"FPS: {fps:4.1f}",
    ]

    if sign_result is not None:
        maze_score = sign_result.get("maze_score", sign_result.get("scores", {}).get("MAZE", -1.0))
        detected_label = sign_result.get("label", "UNKNOWN")
        top_lines.append(f"Maze score: {maze_score:0.3f}")
        top_lines.append(f"Detected: {detected_label}")
        top_lines.append(
            f"Other: {sign_result.get('best_other_label', 'NONE')} {sign_result.get('best_other_score', -1.0):0.3f}"
        )

    if state == MAZE_CONFIRMED_COUNTDOWN:
        remaining = max(0, int(np.ceil(countdown_deadline - now)))
        top_lines.append(f"Countdown: {remaining}s")
    elif state == MAZE_NAVIGATION_ACTIVE and drive_command is not None:
        top_lines.append(
            f"Drive: {drive_command['mode']} L{drive_command['left_speed']:>3} R{drive_command['right_speed']:>3}"
        )

    for index, line in enumerate(top_lines):
        cv2.putText(
            view,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if sign_result is not None and sign_result.get("binary") is not None:
        preview = sign_result["binary"]
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        preview = cv2.resize(preview, (120, 120), interpolation=cv2.INTER_NEAREST)
        h, w = view.shape[:2]
        y0 = clamp(12, 0, h - preview.shape[0])
        x0 = clamp(w - preview.shape[1] - 12, 0, w - preview.shape[1])
        view[y0 : y0 + preview.shape[0], x0 : x0 + preview.shape[1]] = preview
        cv2.rectangle(
            view,
            (x0, y0),
            (x0 + preview.shape[1], y0 + preview.shape[0]),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            view,
            sign_result["label"],
            (x0, y0 + preview.shape[0] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if sign_result.get("maze_confident") else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if perception is not None:
        frame_h, frame_w = view.shape[:2]
        cv2.line(view, (frame_w // 2, 0), (frame_w // 2, frame_h - 1), (255, 80, 80), 1)

        scanlines = perception.get("scanlines", [])
        for sample in scanlines:
            if sample is None:
                continue
            y = int(sample["y"])
            left = int(sample["left"])
            right = int(sample["right"])
            center = int(round(sample["center"]))
            cv2.line(view, (left, y), (right, y), (0, 255, 255), 2)
            cv2.circle(view, (left, y), 3, (255, 180, 0), -1)
            cv2.circle(view, (right, y), 3, (255, 180, 0), -1)
            cv2.circle(view, (center, y), 4, (0, 255, 0), -1)

        points = perception.get("centerline_points", [])
        if len(points) >= 2:
            cv2.polylines(
                view,
                [np.array(points, dtype=np.int32)],
                False,
                (255, 255, 0),
                2,
            )

        info_lines = [
            f"heading: {perception.get('heading_error', 0.0):6.1f}",
            f"lookahead: {perception.get('lookahead_error', 0.0):6.1f}",
            f"depth: {perception.get('forward_depth', 0.0):0.2f}",
            f"branch L/R: {perception.get('left_branch_score', 0.0):0.2f}/{perception.get('right_branch_score', 0.0):0.2f}",
            f"front block: {perception.get('front_block_score', 0.0):0.2f}",
            f"conf: {perception.get('confidence', 0.0):0.2f} width: {perception.get('corridor_width', 0)}",
        ]
        for index, line in enumerate(info_lines):
            cv2.putText(
                view,
                line,
                (12, frame_h - 122 + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        mask_small = cv2.resize(perception["drivable_mask"], (160, 90), interpolation=cv2.INTER_NEAREST)
        mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        y0 = frame_h - mask_small.shape[0] - 12
        x0 = 12
        view[y0 : y0 + mask_small.shape[0], x0 : x0 + mask_small.shape[1]] = mask_small
        cv2.rectangle(view, (x0, y0), (x0 + mask_small.shape[1], y0 + mask_small.shape[0]), (255, 255, 255), 2)
        cv2.putText(
            view,
            "Drivable mask",
            (x0, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return view


def update_lcd(lcd_display, state, sign_result, countdown_deadline, now, drive_command=None):
    if state == WAIT_FOR_MAZE_TEMPLATE:
        if sign_result is not None and sign_result.get("scores"):
            detected = sign_result.get("label", "UNKNOWN")
            lcd_display.update("Waiting Maze", detected)
        else:
            lcd_display.update("Waiting Maze", "Show template")
        return

    if state == MAZE_CONFIRMED_COUNTDOWN:
        remaining = max(0, int(np.ceil(countdown_deadline - now)))
        lcd_display.update("Maze", f"Start in {remaining}")
        return

    if state == MAZE_NAVIGATION_ACTIVE:
        mode_text = "Running"
        if drive_command is not None:
            mode_text = drive_command["mode"][:16]
        lcd_display.update("Maze Run", mode_text)
        return

    if state == MAZE_FINISHED:
        lcd_display.update("Maze Done", "Stopped")
        return


def install_stop_handlers(controller):
    def safe_stop():
        try:
            controller.stop()
        except Exception:
            pass

    def stop_and_exit(signum=None, frame=None):
        safe_stop()
        raise SystemExit(0)

    atexit.register(safe_stop)
    signal.signal(signal.SIGINT, stop_and_exit)
    signal.signal(signal.SIGTERM, stop_and_exit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument(
        "--record-output",
        default="",
        help="Optional output h264 path from the Pi camera.",
    )
    parser.add_argument(
        "--debug-video-output",
        default="",
        help="Optional mp4 path for the processed overlay video.",
    )
    args = parser.parse_args()
    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

    ensure_default_maze_template()

    base.GPIO.setwarnings(False)
    base.GPIO.setmode(base.GPIO.BCM)

    controller = base.CarMotorController()
    lcd_display = base.LCD1602Display()
    install_stop_handlers(controller)

    picam2 = base.Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": base.CAPTURE_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)

    h264_recording = False
    if args.record_output:
        encoder = base.H264Encoder(bitrate=8_000_000)
        picam2.start_recording(encoder, base.FileOutput(args.record_output))
        h264_recording = True
    else:
        picam2.start()

    debug_writer = None
    state = WAIT_FOR_MAZE_TEMPLATE
    maze_hits = 0
    countdown_deadline = -999.0
    last_frame_time = time.perf_counter()
    fps = 0.0
    navigator = MazeNavigator()
    last_reported_label = None
    last_reported_reason = None

    try:
        if args.debug_video_output:
            debug_path = Path(args.debug_video_output)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(
                str(debug_path),
                fourcc,
                20.0,
                base.PROCESS_SIZE,
            )
            if not debug_writer.isOpened():
                raise RuntimeError(f"unable to open debug video writer: {debug_path}")

        while True:
            now = time.perf_counter()
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            full_frame = picam2.capture_array()
            full_frame = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(full_frame, base.PROCESS_SIZE, interpolation=cv2.INTER_AREA)

            sign_roi = None
            sign_result = None
            perception = None
            drive_command = None

            if state == WAIT_FOR_MAZE_TEMPLATE:
                sign_roi = base.detect_sign_roi(frame)
                sign_result = classify_maze_sign(sign_roi)
                current_label = sign_result.get("label", "UNKNOWN")
                current_conf = sign_result.get("confidence", 0.0)
                current_reason = sign_result.get("reason", "")
                if current_label != last_reported_label or current_reason != last_reported_reason:
                    print(
                        "[sign] "
                        f"label={current_label} confidence={current_conf:.3f} "
                        f"reason={current_reason} sign_found={sign_result.get('sign_found', False)} "
                        f"maze_score={sign_result.get('maze_score', -1.0):.3f}"
                    )
                    last_reported_label = current_label
                    last_reported_reason = current_reason
                if sign_result["maze_confident"]:
                    maze_hits += 1
                else:
                    maze_hits = 0

                controller.hold_stop()
                if maze_hits >= MAZE_CONFIRMATION_HITS:
                    state = MAZE_CONFIRMED_COUNTDOWN
                    countdown_deadline = now + MAZE_COUNTDOWN_SECONDS
                    controller.stop()

            elif state == MAZE_CONFIRMED_COUNTDOWN:
                controller.hold_stop()
                if now >= countdown_deadline:
                    state = MAZE_NAVIGATION_ACTIVE
                    navigator = MazeNavigator()
                    controller.stop()

            elif state == MAZE_NAVIGATION_ACTIVE:
                perception = analyze_maze_frame(frame, navigator.previous_center_x)
                drive_command = navigator.update(perception, now)
                if drive_command["done"]:
                    state = MAZE_FINISHED
                    controller.stop()
                else:
                    controller.set_tank_drive(drive_command["left_speed"], drive_command["right_speed"])

            elif state == MAZE_FINISHED:
                controller.hold_stop()

            update_lcd(lcd_display, state, sign_result, countdown_deadline, now, drive_command)
            overlay = draw_overlay(
                frame,
                state,
                sign_roi,
                sign_result,
                fps,
                maze_hits,
                countdown_deadline,
                now,
                perception,
                drive_command,
            )

            if debug_writer is not None:
                debug_writer.write(overlay)

            if not args.headless:
                cv2.imshow("Maze Navigation", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("f"), ord("F")):
                    state = MAZE_FINISHED
                    controller.stop()
            else:
                time.sleep(0.001)

    finally:
        controller.close()
        lcd_display.close()
        if debug_writer is not None:
            debug_writer.release()
        base.GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()


if __name__ == "__main__":
    main()
