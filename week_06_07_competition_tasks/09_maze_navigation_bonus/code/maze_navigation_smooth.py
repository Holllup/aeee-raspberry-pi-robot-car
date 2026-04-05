import argparse
import atexit
import csv
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
    spec = importlib.util.spec_from_file_location("maze_smooth_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load base module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


COUNTDOWN = "COUNTDOWN"
RUNNING = "RUNNING"
DONE = "DONE"

TRACK = "TRACK"
SCAN = "SCAN"
TURN = "TURN"
UTURN_RECOVERY = "UTURN_RECOVERY"

MAZE_WALL_DARK_THRESHOLD = 118
MAZE_WALL_CANNY_LOW = 55
MAZE_WALL_CANNY_HIGH = 140
MAZE_ADAPTIVE_BLOCK_SIZE = 31
MAZE_ADAPTIVE_C = 7
MAZE_FREE_BRIGHT_THRESHOLD = 112

MAZE_HORIZON_CUTOFF_RATIO = 0.35
MAZE_DECISION_TOP_WIDTH_RATIO = 0.62
MAZE_DECISION_BOTTOM_MARGIN_RATIO = 0.02
MAZE_GROUND_SEED_WIDTH_RATIO = 0.20
MAZE_GROUND_SEED_HEIGHT_RATIO = 0.08

MAZE_LOWER_BAND_Y_RATIO = 0.73
MAZE_LOWER_BAND_H_RATIO = 0.13
MAZE_UPPER_BAND_Y_RATIO = 0.55
MAZE_UPPER_BAND_H_RATIO = 0.12
MAZE_FREE_COLUMN_MIN_RATIO = 0.36
MAZE_CORRIDOR_CONFIDENCE_MIN = 0.23
MAZE_CORRIDOR_REACQUIRE_CONFIDENCE = 0.28
MAZE_CORRIDOR_CENTER_ALPHA = 0.72
MAZE_CENTERED_ERROR_PIXELS = 34

MAZE_FRONT_GATE_MIN = 0.25
MAZE_FRONT_SPACE_MIN = 0.23
MAZE_FRONT_BLOCKED_SPACE_MAX = 0.13
MAZE_SIDE_GATE_MIN = 0.24
MAZE_SIDE_SPACE_MIN = 0.20
MAZE_BRANCH_SCORE_MIN = 0.28
MAZE_SCAN_DECIDE_SCORE_MIN = 0.24

MAZE_NAV_BASE_SPEED = 24
MAZE_NAV_MIN_SPEED = 14
MAZE_NAV_MAX_STEER = 16
MAZE_NAV_STEER_GAIN = 28.0
MAZE_WALL_BALANCE_DEADBAND = 0.04
MAZE_WALL_REPULSION_GAIN = 15.0
MAZE_WALL_REPULSION_MAX_STEER = 8
MAZE_TRACK_MEMORY_SECONDS = 0.35
MAZE_LOST_SCAN_TIMEOUT_SECONDS = 0.55

MAZE_RIGHT_OPEN_CONFIRM_FRAMES = 3
MAZE_FRONT_BLOCKED_CONFIRM_FRAMES = 2
MAZE_LOW_CONF_CONFIRM_FRAMES = 4
MAZE_DECISION_COOLDOWN_SECONDS = 0.45

MAZE_SCAN_TURN_SPEED = 18
MAZE_SCAN_RIGHT_SECONDS = 0.34
MAZE_SCAN_LEFT_SECONDS = 0.72

MAZE_TURN_SPEED = 28
MAZE_UTURN_SPEED = 30
MAZE_RIGHT_TURN_MAX_SECONDS = 1.25
MAZE_LEFT_TURN_MAX_SECONDS = 1.50
MAZE_UTURN_MAX_SECONDS = 2.30
MAZE_TURN_ALIGN_HITS_REQUIRED = 3

MAZE_TRACK_DECISION_HOLD_SECONDS = 0.18


def clamp(value, low, high):
    return max(low, min(high, value))


def roi_rect(frame_shape, x_ratio, y_ratio, w_ratio, h_ratio):
    frame_h, frame_w = frame_shape[:2]
    x0 = clamp(int(frame_w * x_ratio), 0, frame_w - 1)
    y0 = clamp(int(frame_h * y_ratio), 0, frame_h - 1)
    x1 = clamp(int(frame_w * (x_ratio + w_ratio)), x0 + 1, frame_w)
    y1 = clamp(int(frame_h * (y_ratio + h_ratio)), y0 + 1, frame_h)
    return (x0, y0, x1 - x0, y1 - y0)


def occupancy_ratio(mask, rect):
    x, y, w, h = rect
    roi = mask[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    return cv2.countNonZero(roi) / float(roi.size)


def build_decision_mask(frame_shape):
    frame_h, frame_w = frame_shape[:2]
    horizon_y = clamp(int(frame_h * MAZE_HORIZON_CUTOFF_RATIO), 0, frame_h - 1)
    top_half_width = int(frame_w * MAZE_DECISION_TOP_WIDTH_RATIO * 0.5)
    center_x = frame_w // 2
    left_top_x = clamp(center_x - top_half_width, 0, frame_w - 1)
    right_top_x = clamp(center_x + top_half_width, left_top_x + 1, frame_w - 1)
    bottom_margin = int(frame_w * MAZE_DECISION_BOTTOM_MARGIN_RATIO)
    polygon = np.array(
        [
            [left_top_x, horizon_y],
            [right_top_x, horizon_y],
            [frame_w - 1 - bottom_margin, frame_h - 1],
            [bottom_margin, frame_h - 1],
        ],
        dtype=np.int32,
    )
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask, polygon, horizon_y


def build_wall_mask(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, dark_mask = cv2.threshold(
        blurred,
        MAZE_WALL_DARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )
    _, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        MAZE_ADAPTIVE_BLOCK_SIZE,
        MAZE_ADAPTIVE_C,
    )
    edges = cv2.Canny(blurred, MAZE_WALL_CANNY_LOW, MAZE_WALL_CANNY_HIGH)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    wall_mask = cv2.bitwise_or(dark_mask, otsu_mask)
    wall_mask = cv2.bitwise_or(wall_mask, adaptive_mask)
    wall_mask = cv2.bitwise_or(wall_mask, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return wall_mask


def _build_free_masks(frame_bgr, wall_mask):
    frame_h, frame_w = frame_bgr.shape[:2]
    decision_mask, decision_polygon, horizon_y = build_decision_mask(frame_bgr.shape)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright_mask = cv2.threshold(
        blurred,
        MAZE_FREE_BRIGHT_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    free_mask_raw = cv2.bitwise_and(bright_mask, cv2.bitwise_not(wall_mask))
    free_mask_raw = cv2.bitwise_and(free_mask_raw, decision_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    free_mask_raw = cv2.morphologyEx(free_mask_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    free_mask_raw = cv2.morphologyEx(free_mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)

    seed_width = max(8, int(frame_w * MAZE_GROUND_SEED_WIDTH_RATIO))
    seed_height = max(6, int(frame_h * MAZE_GROUND_SEED_HEIGHT_RATIO))
    seed_x = clamp((frame_w - seed_width) // 2, 0, frame_w - seed_width)
    seed_y = clamp(frame_h - seed_height, 0, frame_h - 1)
    seed_rect = (seed_x, seed_y, seed_width, seed_height)

    label_count, labels, _, _ = cv2.connectedComponentsWithStats(free_mask_raw, connectivity=8)
    connected_free = np.zeros_like(free_mask_raw)

    if label_count > 1:
        seed_roi = labels[seed_y : seed_y + seed_height, seed_x : seed_x + seed_width]
        seed_labels = np.unique(seed_roi)
        seed_labels = seed_labels[seed_labels != 0]
        if seed_labels.size > 0:
            connected_free[np.isin(labels, seed_labels)] = 255

    ignored_free = cv2.bitwise_and(free_mask_raw, cv2.bitwise_not(connected_free))
    return {
        "decision_mask": decision_mask,
        "decision_polygon": decision_polygon,
        "horizon_y": horizon_y,
        "free_mask_raw": free_mask_raw,
        "ground_connected_free_mask": connected_free,
        "ignored_free_mask": ignored_free,
        "seed_rect": seed_rect,
    }


def build_ground_connected_free_mask(frame_bgr, wall_mask):
    return _build_free_masks(frame_bgr, wall_mask)["ground_connected_free_mask"]


def pick_best_free_span(free_columns, target_x):
    spans = []
    start = None
    for idx, is_free in enumerate(free_columns):
        if is_free and start is None:
            start = idx
        elif not is_free and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(free_columns) - 1))

    if not spans:
        return None

    best_span = None
    best_score = -1e9
    for span in spans:
        center = (span[0] + span[1]) / 2.0
        width = span[1] - span[0] + 1
        score = width - abs(center - target_x) * 0.38
        if score > best_score:
            best_score = score
            best_span = span
    return best_span


def _build_direction_rois(frame_shape):
    rois = {
        "front_gate": roi_rect(frame_shape, 0.38, 0.73, 0.24, 0.13),
        "front_space": roi_rect(frame_shape, 0.31, 0.53, 0.38, 0.18),
        "right_gate": roi_rect(frame_shape, 0.68, 0.69, 0.22, 0.16),
        "right_space": roi_rect(frame_shape, 0.64, 0.49, 0.26, 0.20),
        "left_gate": roi_rect(frame_shape, 0.10, 0.69, 0.22, 0.16),
        "left_space": roi_rect(frame_shape, 0.10, 0.49, 0.26, 0.20),
    }
    return rois


def _build_wall_pressure_rois(frame_shape):
    return {
        "right_wall_near": roi_rect(frame_shape, 0.80, 0.67, 0.14, 0.21),
        "right_wall_mid": roi_rect(frame_shape, 0.72, 0.51, 0.16, 0.18),
        "left_wall_near": roi_rect(frame_shape, 0.06, 0.67, 0.14, 0.21),
        "left_wall_mid": roi_rect(frame_shape, 0.12, 0.51, 0.16, 0.18),
    }


def analyze_maze_perception(frame_bgr, previous_center_x=None):
    frame_h, frame_w = frame_bgr.shape[:2]
    wall_mask = build_wall_mask(frame_bgr)
    mask_bundle = _build_free_masks(frame_bgr, wall_mask)
    ground_free = mask_bundle["ground_connected_free_mask"]

    lower_band_rect = roi_rect(frame_bgr.shape, 0.0, MAZE_LOWER_BAND_Y_RATIO, 1.0, MAZE_LOWER_BAND_H_RATIO)
    upper_band_rect = roi_rect(frame_bgr.shape, 0.0, MAZE_UPPER_BAND_Y_RATIO, 1.0, MAZE_UPPER_BAND_H_RATIO)

    lx, ly, lw, lh = lower_band_rect
    ux, uy, uw, uh = upper_band_rect
    lower_band = ground_free[ly : ly + lh, lx : lx + lw]
    upper_band = ground_free[uy : uy + uh, ux : ux + uw]

    corridor_center_x = None
    corridor_width = 0
    corridor_error = 0.0
    corridor_confidence = 0.0
    free_span = None

    if lower_band.size > 0 and upper_band.size > 0:
        lower_occ = np.count_nonzero(lower_band > 0, axis=0) / float(max(1, lower_band.shape[0]))
        upper_occ = np.count_nonzero(upper_band > 0, axis=0) / float(max(1, upper_band.shape[0]))
        column_free_score = lower_occ * 0.68 + upper_occ * 0.32
        free_columns = column_free_score >= MAZE_FREE_COLUMN_MIN_RATIO
        target_x = (frame_w / 2.0) if previous_center_x is None else previous_center_x
        free_span = pick_best_free_span(free_columns, target_x)
        if free_span is not None:
            span_start, span_end = free_span
            measured_center_x = (span_start + span_end) / 2.0
            if previous_center_x is None:
                corridor_center_x = int(round(measured_center_x))
            else:
                smoothed_center = (
                    MAZE_CORRIDOR_CENTER_ALPHA * measured_center_x
                    + (1.0 - MAZE_CORRIDOR_CENTER_ALPHA) * previous_center_x
                )
                corridor_center_x = int(round(smoothed_center))
            corridor_width = int(span_end - span_start + 1)
            width_conf = clamp(corridor_width / float(frame_w * 0.36), 0.0, 1.0)
            density_conf = clamp(np.mean(column_free_score[span_start : span_end + 1]) / 0.65, 0.0, 1.0)
            corridor_confidence = 0.55 * width_conf + 0.45 * density_conf
            corridor_error = corridor_center_x - (frame_w / 2.0)

    direction_rois = _build_direction_rois(frame_bgr.shape)
    scores = {}
    for name, rect in direction_rois.items():
        scores[name] = occupancy_ratio(ground_free, rect)

    front_open_score = 0.60 * scores["front_gate"] + 0.40 * scores["front_space"]
    right_open_score = 0.60 * scores["right_gate"] + 0.40 * scores["right_space"]
    left_open_score = 0.60 * scores["left_gate"] + 0.40 * scores["left_space"]

    front_entry_gate = scores["front_gate"] >= MAZE_FRONT_GATE_MIN
    right_entry_gate = scores["right_gate"] >= MAZE_SIDE_GATE_MIN
    left_entry_gate = scores["left_gate"] >= MAZE_SIDE_GATE_MIN

    front_inside_space = scores["front_space"] >= MAZE_FRONT_SPACE_MIN
    right_inside_space = scores["right_space"] >= MAZE_SIDE_SPACE_MIN
    left_inside_space = scores["left_space"] >= MAZE_SIDE_SPACE_MIN

    front_open = front_entry_gate and front_inside_space
    right_open = right_entry_gate and right_inside_space
    left_open = left_entry_gate and left_inside_space
    front_blocked = (not front_entry_gate) or (scores["front_space"] <= MAZE_FRONT_BLOCKED_SPACE_MAX)

    wall_rois = _build_wall_pressure_rois(frame_bgr.shape)
    wall_scores = {name: occupancy_ratio(wall_mask, rect) for name, rect in wall_rois.items()}
    right_wall_pressure = 0.62 * wall_scores["right_wall_near"] + 0.38 * wall_scores["right_wall_mid"]
    left_wall_pressure = 0.62 * wall_scores["left_wall_near"] + 0.38 * wall_scores["left_wall_mid"]
    wall_balance = right_wall_pressure - left_wall_pressure

    return {
        "wall_mask": wall_mask,
        "free_mask_raw": mask_bundle["free_mask_raw"],
        "ground_connected_free_mask": ground_free,
        "ignored_free_mask": mask_bundle["ignored_free_mask"],
        "decision_mask": mask_bundle["decision_mask"],
        "decision_polygon": mask_bundle["decision_polygon"],
        "horizon_y": mask_bundle["horizon_y"],
        "seed_rect": mask_bundle["seed_rect"],
        "corridor_center_x": corridor_center_x,
        "corridor_width": corridor_width,
        "corridor_error": corridor_error,
        "corridor_confidence": corridor_confidence,
        "free_span": free_span,
        "front_entry_gate": front_entry_gate,
        "right_entry_gate": right_entry_gate,
        "left_entry_gate": left_entry_gate,
        "front_inside_space": front_inside_space,
        "right_inside_space": right_inside_space,
        "left_inside_space": left_inside_space,
        "front_open_score": front_open_score,
        "right_open_score": right_open_score,
        "left_open_score": left_open_score,
        "front_blocked": front_blocked,
        "front_open": front_open,
        "right_open": right_open,
        "left_open": left_open,
        "direction_scores": scores,
        "direction_rois": direction_rois,
        "wall_rois": wall_rois,
        "wall_scores": wall_scores,
        "right_wall_pressure": right_wall_pressure,
        "left_wall_pressure": left_wall_pressure,
        "wall_balance": wall_balance,
    }


def tint_mask(image, mask, color, alpha):
    if not np.any(mask):
        return image
    overlay = image.copy()
    overlay[mask] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    image[mask] = blended[mask]
    return image


class MazeSmoothNavigator:
    def __init__(self):
        self.mode = TRACK
        self.previous_center_x = None
        self.last_confident_time = time.perf_counter()
        self.right_open_hits = 0
        self.front_blocked_hits = 0
        self.low_conf_hits = 0
        self.decision_cooldown_until = -999.0
        self.scan_phase = ""
        self.scan_phase_until = -999.0
        self.scan_scores = {"right": 0.0, "front": 0.0, "left": 0.0}
        self.scan_reason = ""
        self.turn_action = ""
        self.turn_until = -999.0
        self.turn_align_hits = 0
        self.finished = False

    def _follow_command(self, perception, mode_name=TRACK):
        frame_half = base.PROCESS_SIZE[0] / 2.0
        target_x = perception["corridor_center_x"]
        if target_x is None:
            target_x = self.previous_center_x
        if target_x is None:
            target_x = frame_half

        error = target_x - frame_half
        normalized_error = 0.0 if frame_half <= 0 else error / frame_half
        steer = normalized_error * MAZE_NAV_STEER_GAIN

        wall_balance = perception["wall_balance"]
        if abs(wall_balance) <= MAZE_WALL_BALANCE_DEADBAND:
            wall_balance = 0.0
        wall_steer = clamp(
            -wall_balance * MAZE_WALL_REPULSION_GAIN,
            -MAZE_WALL_REPULSION_MAX_STEER,
            MAZE_WALL_REPULSION_MAX_STEER,
        )
        steer = clamp(steer + wall_steer, -MAZE_NAV_MAX_STEER, MAZE_NAV_MAX_STEER)

        confidence = perception["corridor_confidence"]
        speed_penalty = min(6.0, abs(error) * 0.035) + (1.0 - clamp(confidence, 0.0, 1.0)) * 4.0
        base_speed = clamp(MAZE_NAV_BASE_SPEED - speed_penalty, MAZE_NAV_MIN_SPEED, MAZE_NAV_BASE_SPEED)

        left_speed = int(clamp(base_speed + steer, -base.MAX_SPEED, base.MAX_SPEED))
        right_speed = int(clamp(base_speed - steer, -base.MAX_SPEED, base.MAX_SPEED))
        return {"left_speed": left_speed, "right_speed": right_speed, "mode": mode_name, "done": False}

    def _start_scan(self, now, reason, perception):
        self.mode = SCAN
        self.scan_phase = "RIGHT_CHECK"
        self.scan_phase_until = now + MAZE_SCAN_RIGHT_SECONDS
        self.scan_reason = reason
        self.scan_scores = {
            "right": perception["right_open_score"],
            "front": perception["front_open_score"],
            "left": perception["left_open_score"],
        }

    def _scan_command(self):
        if self.scan_phase == "RIGHT_CHECK":
            return {
                "left_speed": MAZE_SCAN_TURN_SPEED,
                "right_speed": -MAZE_SCAN_TURN_SPEED,
                "mode": "SCAN_RIGHT",
                "done": False,
            }
        return {
            "left_speed": -MAZE_SCAN_TURN_SPEED,
            "right_speed": MAZE_SCAN_TURN_SPEED,
            "mode": "SCAN_LEFT",
            "done": False,
        }

    def _begin_turn(self, action, now):
        self.mode = TURN if action in ("RIGHT", "LEFT") else UTURN_RECOVERY
        self.turn_action = action
        if action == "RIGHT":
            self.turn_until = now + MAZE_RIGHT_TURN_MAX_SECONDS
        elif action == "LEFT":
            self.turn_until = now + MAZE_LEFT_TURN_MAX_SECONDS
        else:
            self.turn_until = now + MAZE_UTURN_MAX_SECONDS
        self.turn_align_hits = 0
        self.decision_cooldown_until = now + MAZE_DECISION_COOLDOWN_SECONDS

    def _turn_command(self):
        if self.turn_action == "RIGHT":
            return {"left_speed": MAZE_TURN_SPEED, "right_speed": -MAZE_TURN_SPEED, "mode": "TURN_RIGHT", "done": False}
        if self.turn_action == "LEFT":
            return {"left_speed": -MAZE_TURN_SPEED, "right_speed": MAZE_TURN_SPEED, "mode": "TURN_LEFT", "done": False}
        return {"left_speed": MAZE_UTURN_SPEED, "right_speed": -MAZE_UTURN_SPEED, "mode": "UTURN", "done": False}

    def _direction_decision(self):
        if self.scan_scores["right"] >= MAZE_SCAN_DECIDE_SCORE_MIN:
            return "RIGHT"
        if self.scan_scores["front"] >= MAZE_SCAN_DECIDE_SCORE_MIN:
            return "FRONT"
        if self.scan_scores["left"] >= MAZE_SCAN_DECIDE_SCORE_MIN:
            return "LEFT"
        return "UTURN"

    def _update_track_counters(self, perception):
        self.right_open_hits = (
            min(MAZE_RIGHT_OPEN_CONFIRM_FRAMES + 2, self.right_open_hits + 1)
            if perception["right_open"] and perception["right_open_score"] >= MAZE_BRANCH_SCORE_MIN
            else max(0, self.right_open_hits - 1)
        )
        self.front_blocked_hits = (
            min(MAZE_FRONT_BLOCKED_CONFIRM_FRAMES + 2, self.front_blocked_hits + 1)
            if perception["front_blocked"]
            else max(0, self.front_blocked_hits - 1)
        )
        self.low_conf_hits = (
            min(MAZE_LOW_CONF_CONFIRM_FRAMES + 2, self.low_conf_hits + 1)
            if perception["corridor_confidence"] < MAZE_CORRIDOR_CONFIDENCE_MIN
            else max(0, self.low_conf_hits - 1)
        )

    def update(self, perception, now):
        if self.finished:
            return {"left_speed": 0, "right_speed": 0, "mode": "DONE", "done": True}

        if perception["corridor_center_x"] is not None:
            self.previous_center_x = perception["corridor_center_x"]
        if perception["corridor_confidence"] >= MAZE_CORRIDOR_CONFIDENCE_MIN:
            self.last_confident_time = now

        if self.mode == SCAN:
            self.scan_scores["right"] = max(self.scan_scores["right"], perception["right_open_score"])
            self.scan_scores["front"] = max(self.scan_scores["front"], perception["front_open_score"])
            self.scan_scores["left"] = max(self.scan_scores["left"], perception["left_open_score"])

            if self.scan_phase == "RIGHT_CHECK":
                if perception["right_open"] and perception["right_open_score"] >= MAZE_SCAN_DECIDE_SCORE_MIN:
                    self._begin_turn("RIGHT", now)
                    return self._turn_command()
                if now >= self.scan_phase_until:
                    self.scan_phase = "LEFT_SWEEP"
                    self.scan_phase_until = now + MAZE_SCAN_LEFT_SECONDS
            else:
                if now >= self.scan_phase_until:
                    decision = self._direction_decision()
                    if decision == "FRONT":
                        self.mode = TRACK
                        self.decision_cooldown_until = now + MAZE_DECISION_COOLDOWN_SECONDS
                        return self._follow_command(perception)
                    self._begin_turn(decision, now)
                    return self._turn_command()
            return self._scan_command()

        if self.mode in (TURN, UTURN_RECOVERY):
            aligned = (
                perception["corridor_confidence"] >= MAZE_CORRIDOR_REACQUIRE_CONFIDENCE
                and abs(perception["corridor_error"]) <= MAZE_CENTERED_ERROR_PIXELS
            )
            if aligned:
                self.turn_align_hits += 1
            else:
                self.turn_align_hits = 0

            if self.turn_align_hits >= MAZE_TURN_ALIGN_HITS_REQUIRED:
                self.mode = TRACK
                self.turn_action = ""
                return self._follow_command(perception)
            if now >= self.turn_until and self.mode == TURN:
                self.mode = TRACK
                self.turn_action = ""
                return self._follow_command(perception)
            if now >= self.turn_until and self.mode == UTURN_RECOVERY:
                self.mode = TRACK
                self.turn_action = ""
                return self._follow_command(perception, mode_name="TRACK_RECOVER")
            return self._turn_command()

        self._update_track_counters(perception)

        if self.front_blocked_hits >= MAZE_FRONT_BLOCKED_CONFIRM_FRAMES:
            self._start_scan(now, "front_blocked", perception)
            return self._scan_command()

        if now >= self.decision_cooldown_until and self.right_open_hits >= MAZE_RIGHT_OPEN_CONFIRM_FRAMES:
            self._start_scan(now, "right_branch", perception)
            return self._scan_command()

        if (
            self.low_conf_hits >= MAZE_LOW_CONF_CONFIRM_FRAMES
            and max(
                perception["right_open_score"],
                perception["front_open_score"],
                perception["left_open_score"],
            )
            >= MAZE_SCAN_DECIDE_SCORE_MIN
        ):
            self._start_scan(now, "low_conf_branch", perception)
            return self._scan_command()

        time_since_confident = now - self.last_confident_time
        if (
            perception["corridor_confidence"] < MAZE_CORRIDOR_CONFIDENCE_MIN
            and time_since_confident >= MAZE_LOST_SCAN_TIMEOUT_SECONDS
        ):
            self._start_scan(now, "lost_track", perception)
            return self._scan_command()

        return self._follow_command(perception)


def draw_maze_overlay(frame, perception, command, navigator, fps):
    view = frame.copy()

    trusted = perception["ground_connected_free_mask"] > 0
    ignored = perception["ignored_free_mask"] > 0
    walls = perception["wall_mask"] > 0

    view = tint_mask(view, trusted, (0, 200, 0), 0.65)
    view = tint_mask(view, ignored, (0, 180, 255), 0.65)
    view = tint_mask(view, walls, (40, 40, 40), 0.50)

    cv2.polylines(view, [perception["decision_polygon"]], True, (255, 255, 0), 2)
    cv2.line(
        view,
        (0, perception["horizon_y"]),
        (view.shape[1] - 1, perception["horizon_y"]),
        (255, 180, 0),
        1,
    )

    seed_x, seed_y, seed_w, seed_h = perception["seed_rect"]
    cv2.rectangle(view, (seed_x, seed_y), (seed_x + seed_w, seed_y + seed_h), (255, 255, 255), 2)

    for name, rect in perception["direction_rois"].items():
        x, y, w, h = rect
        if "front" in name:
            color = (0, 255, 0) if perception["front_open"] else (0, 0, 255)
        elif "right" in name:
            color = (0, 255, 0) if perception["right_open"] else (0, 165, 255)
        else:
            color = (0, 255, 0) if perception["left_open"] else (0, 165, 255)
        thickness = 2 if "gate" in name else 1
        cv2.rectangle(view, (x, y), (x + w, y + h), color, thickness)

    for name, rect in perception["wall_rois"].items():
        x, y, w, h = rect
        if "right" in name:
            color = (0, 200, 255)
        else:
            color = (255, 200, 0)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 1)

    frame_center_x = view.shape[1] // 2
    cv2.line(view, (frame_center_x, 0), (frame_center_x, view.shape[0] - 1), (255, 0, 0), 1)
    if perception["corridor_center_x"] is not None:
        corridor_x = int(perception["corridor_center_x"])
        cv2.line(view, (corridor_x, 0), (corridor_x, view.shape[0] - 1), (0, 255, 255), 2)

    free_span = perception["free_span"]
    if free_span is not None:
        y = int(view.shape[0] * 0.90)
        cv2.line(view, (int(free_span[0]), y), (int(free_span[1]), y), (0, 255, 255), 4)

    mode_text = command["mode"] if command is not None else navigator.mode
    top_lines = [
        f"Mode: {navigator.mode} / {mode_text}",
        f"FPS: {fps:4.1f}",
        (
            f"Corridor conf {perception['corridor_confidence']:.2f} "
            f"err {perception['corridor_error']:+5.1f}"
        ),
        (
            f"R {perception['right_open_score']:.2f}  "
            f"F {perception['front_open_score']:.2f}  "
            f"L {perception['left_open_score']:.2f}"
        ),
        (
            f"Gate R{int(perception['right_entry_gate'])} "
            f"F{int(perception['front_entry_gate'])} "
            f"L{int(perception['left_entry_gate'])}  "
            f"Blocked {int(perception['front_blocked'])}"
        ),
        (
            f"Wall R {perception['right_wall_pressure']:.2f}  "
            f"L {perception['left_wall_pressure']:.2f}  "
            f"bal {perception['wall_balance']:+.2f}"
        ),
        "Green=trusted free  Orange=ignored free  Dark=wall",
    ]

    if navigator.mode == SCAN:
        top_lines.append(
            f"Scan {navigator.scan_phase}  best R{navigator.scan_scores['right']:.2f} "
            f"F{navigator.scan_scores['front']:.2f} L{navigator.scan_scores['left']:.2f}"
        )

    if command is not None:
        top_lines.append(f"Drive L{command['left_speed']:>3} R{command['right_speed']:>3}")

    for index, line in enumerate(top_lines):
        cv2.putText(
            view,
            line,
            (12, 26 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    mini_masks = [
        ("wall", perception["wall_mask"]),
        ("raw", perception["free_mask_raw"]),
        ("trusted", perception["ground_connected_free_mask"]),
    ]
    for index, (label, mask) in enumerate(mini_masks):
        preview = cv2.resize(mask, (120, 68), interpolation=cv2.INTER_NEAREST)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        x0 = 12 + index * 132
        y0 = view.shape[0] - preview.shape[0] - 12
        view[y0 : y0 + preview.shape[0], x0 : x0 + preview.shape[1]] = preview
        cv2.rectangle(view, (x0, y0), (x0 + preview.shape[1], y0 + preview.shape[0]), (255, 255, 255), 2)
        cv2.putText(
            view,
            label,
            (x0 + 4, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return view


def update_lcd(lcd_display, state, command, countdown_deadline, now):
    if state == COUNTDOWN:
        remaining = max(0, int(np.ceil(countdown_deadline - now)))
        lcd_display.update("Maze Smooth", f"Start in {remaining}")
        return
    if state == RUNNING and command is not None:
        lcd_display.update("Maze Smooth", command["mode"][:16])
        return
    lcd_display.update("Maze Smooth", "Stopped")


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
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before starting maze navigation.",
    )
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
    parser.add_argument(
        "--metrics-output",
        default="",
        help="Optional CSV path for frame-by-frame maze perception metrics.",
    )
    args = parser.parse_args()
    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

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
    metrics_file = None
    metrics_writer = None
    navigator = MazeSmoothNavigator()
    state = COUNTDOWN
    countdown_deadline = time.perf_counter() + max(0.0, args.start_delay)
    last_frame_time = time.perf_counter()
    fps = 0.0

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

        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
            metrics_writer = csv.DictWriter(
                metrics_file,
                fieldnames=[
                    "t",
                    "state",
                    "mode",
                    "left_speed",
                    "right_speed",
                    "corridor_confidence",
                    "corridor_error",
                    "front_open_score",
                    "right_open_score",
                    "left_open_score",
                    "front_entry_gate",
                    "right_entry_gate",
                    "left_entry_gate",
                    "front_blocked",
                    "front_open",
                    "right_open",
                    "left_open",
                    "right_wall_pressure",
                    "left_wall_pressure",
                    "wall_balance",
                ],
            )
            metrics_writer.writeheader()

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

            perception = analyze_maze_perception(frame, navigator.previous_center_x)
            command = None

            if state == COUNTDOWN:
                controller.hold_stop()
                if now >= countdown_deadline:
                    state = RUNNING
                    navigator = MazeSmoothNavigator()
                    controller.stop()
            elif state == RUNNING:
                command = navigator.update(perception, now)
                if command["done"]:
                    state = DONE
                    controller.stop()
                else:
                    controller.set_tank_drive(command["left_speed"], command["right_speed"])
            else:
                controller.hold_stop()

            update_lcd(lcd_display, state, command, countdown_deadline, now)

            if metrics_writer is not None:
                row = {
                    "t": f"{now:.3f}",
                    "state": state,
                    "mode": "" if command is None else command["mode"],
                    "left_speed": 0 if command is None else command["left_speed"],
                    "right_speed": 0 if command is None else command["right_speed"],
                    "corridor_confidence": f"{perception['corridor_confidence']:.4f}",
                    "corridor_error": f"{perception['corridor_error']:.2f}",
                    "front_open_score": f"{perception['front_open_score']:.4f}",
                    "right_open_score": f"{perception['right_open_score']:.4f}",
                    "left_open_score": f"{perception['left_open_score']:.4f}",
                    "front_entry_gate": int(perception["front_entry_gate"]),
                    "right_entry_gate": int(perception["right_entry_gate"]),
                    "left_entry_gate": int(perception["left_entry_gate"]),
                    "front_blocked": int(perception["front_blocked"]),
                    "front_open": int(perception["front_open"]),
                    "right_open": int(perception["right_open"]),
                    "left_open": int(perception["left_open"]),
                    "right_wall_pressure": f"{perception['right_wall_pressure']:.4f}",
                    "left_wall_pressure": f"{perception['left_wall_pressure']:.4f}",
                    "wall_balance": f"{perception['wall_balance']:.4f}",
                }
                metrics_writer.writerow(row)
                metrics_file.flush()

            overlay = draw_maze_overlay(frame, perception, command, navigator, fps)
            if state == COUNTDOWN:
                cv2.putText(
                    overlay,
                    f"Start in {max(0, int(np.ceil(countdown_deadline - now)))}",
                    (12, overlay.shape[0] - 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if debug_writer is not None:
                debug_writer.write(overlay)

            if not args.headless:
                cv2.imshow("Maze Navigation Smooth", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("f"), ord("F")):
                    state = DONE
                    controller.stop()
            else:
                time.sleep(0.001)

    finally:
        controller.close()
        lcd_display.close()
        if debug_writer is not None:
            debug_writer.release()
        if metrics_file is not None:
            metrics_file.close()
        base.GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()


if __name__ == "__main__":
    main()
