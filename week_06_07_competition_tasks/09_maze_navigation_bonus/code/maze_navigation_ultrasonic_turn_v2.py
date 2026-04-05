import argparse
import atexit
import importlib.util
import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_local_module(module_name, filename):
    module_path = Path(__file__).resolve().with_name(filename)
    if not module_path.exists():
        raise RuntimeError(f"unable to find local module: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smooth = load_local_module("maze_navigation_smooth_local", "maze_navigation_smooth.py")
base = smooth.base


MODE_RECORD_VALIDATE = "record-validate"
MODE_RUN_MAZE = "run-maze"

COUNTDOWN = "COUNTDOWN"
RUNNING = "RUNNING"
DONE = "DONE"

FORWARD = "FORWARD"
STOP_SETTLE = "STOP_SETTLE"
CAPTURE_FRONT = "CAPTURE_FRONT"
ROTATE_SCAN_RIGHT = "ROTATE_SCAN_RIGHT"
SETTLE_SCAN_RIGHT = "SETTLE_SCAN_RIGHT"
CAPTURE_RIGHT = "CAPTURE_RIGHT"
RETURN_CENTER_FROM_RIGHT = "RETURN_CENTER_FROM_RIGHT"
SETTLE_CENTER_AFTER_RIGHT = "SETTLE_CENTER_AFTER_RIGHT"
ROTATE_SCAN_LEFT = "ROTATE_SCAN_LEFT"
SETTLE_SCAN_LEFT = "SETTLE_SCAN_LEFT"
CAPTURE_LEFT = "CAPTURE_LEFT"
RETURN_CENTER_FROM_LEFT = "RETURN_CENTER_FROM_LEFT"
SETTLE_CENTER_AFTER_LEFT = "SETTLE_CENTER_AFTER_LEFT"
TURN_RIGHT_COMMIT = "TURN_RIGHT_COMMIT"
TURN_LEFT_COMMIT = "TURN_LEFT_COMMIT"
TURN_UTURN_COMMIT = "TURN_UTURN_COMMIT"
FORWARD_RECOVER = "FORWARD_RECOVER"
SEARCH_FORWARD = "SEARCH_FORWARD"
ALIGN_GATE = "ALIGN_GATE"
PASS_GATE = "PASS_GATE"
SEARCH_NEXT = "SEARCH_NEXT"
ADVANCE_OPEN = "ADVANCE_OPEN"
BACKOFF = "BACKOFF"

TURN_RIGHT = "RIGHT"
TURN_LEFT = "LEFT"
TURN_FRONT = "FRONT"
TURN_UTURN = "UTURN"

ULTRASONIC_SAMPLE_HZ = 12.0
ULTRASONIC_MAX_WAIT_SECONDS = 0.03
ULTRASONIC_VALID_MIN_CM = 2.0
ULTRASONIC_VALID_MAX_CM = 400.0
SEARCH_CLEARANCE_CM = 22.0
RIGHT_REAR_BOOST = 6
TURN_90_SECONDS = 0.80
SCAN_SETTLE_SECONDS = 0.12
FORWARD_RECOVER_SECONDS = 0.25

VISUAL_CENTER_BLOCK_THRESHOLD = 0.50
VISUAL_WALL_RATIO_BLOCK_THRESHOLD = 0.50
VISUAL_HARD_BLOCK_SCORE = 0.28
VISUAL_OPEN_MIN = 0.46
VISUAL_COMBINED_OPEN_MIN = 0.54
MARKER_RATIO_THRESHOLD = 0.002
MARKER_SCAN_DISTANCE_CM = 30.0
MARKER_MEMORY_SECONDS = 1.15
MARKER_STEER_GAIN = 6.0
GATE_CONFIDENCE_THRESHOLD = 0.28
ALIGN_GATE_OFFSET_THRESHOLD = 0.08
ALIGN_SPIN_OFFSET_THRESHOLD = 0.10
GATE_TARGET_Y_RATIO = 0.78
SEARCH_SMALL_SECONDS = 0.18
SEARCH_LARGE_SECONDS = 0.34
SEARCH_FORWARD_BURST_SECONDS = 0.22
GATE_LOSS_GRACE_SECONDS = 0.18
GATE_COMMIT_SECONDS = 3.0
GATE_BLIND_DRIVE_SECONDS = 2.0

GATE_TURN_THRESHOLD = 0.12
GATE_TARGET_SHIFT_RATIO = 0.12
GATE_SAFE_MARGIN_RATIO = 0.18
SINGLE_POST_GAP_RATIO = 0.36
ALIGN_NEAR_WIDTH_RATIO = 0.14
PASS_BLIND_STEER_SCALE = 0.75
PASS_BLIND_SPEED_SCALE = 0.56

TURN_LABEL_LEFT = "LEFT"
TURN_LABEL_RIGHT = "RIGHT"
TURN_LABEL_STRAIGHT = "STRAIGHT"

POST_MODE_BOTH = "BOTH"
POST_MODE_RED_ONLY = "RED_ONLY"
POST_MODE_GREEN_ONLY = "GREEN_ONLY"
POST_MODE_NONE = "NONE"


def clamp(value, low, high):
    return max(low, min(high, value))


def format_distance(distance_cm):
    if distance_cm is None:
        return "---"
    return f"{distance_cm:4.1f}"


def tint_mask(image, mask, color=(0, 0, 255), alpha=0.32):
    if mask is None or not np.any(mask):
        return image
    tinted = image.copy()
    tinted[mask > 0] = color
    return cv2.addWeighted(tinted, alpha, image, 1.0 - alpha, 0)


def normalized_progress(value, threshold):
    threshold = max(1e-6, float(threshold))
    return float(value) / threshold


def dominant_blob_center_x(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1.0:
        return -1.0
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        return -1.0
    return float(moments["m10"] / moments["m00"])


def midpoint(point_a, point_b):
    return (
        0.5 * (float(point_a[0]) + float(point_b[0])),
        0.5 * (float(point_a[1]) + float(point_b[1])),
    )


def line_x_at_y(point_a, point_b, target_y):
    x1, y1 = float(point_a[0]), float(point_a[1])
    x2, y2 = float(point_b[0]), float(point_b[1])
    if abs(y2 - y1) < 1e-6:
        return 0.5 * (x1 + x2)
    t = clamp((float(target_y) - y1) / (y2 - y1), 0.0, 1.0)
    return x1 + ((x2 - x1) * t)


def line_intersection(point_a1, point_a2, point_b1, point_b2):
    x1, y1 = float(point_a1[0]), float(point_a1[1])
    x2, y2 = float(point_a2[0]), float(point_a2[1])
    x3, y3 = float(point_b1[0]), float(point_b1[1])
    x4, y4 = float(point_b2[0]), float(point_b2[1])
    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denominator) < 1e-6:
        return None
    det_a = (x1 * y2) - (y1 * x2)
    det_b = (x3 * y4) - (y3 * x4)
    x = ((det_a * (x3 - x4)) - ((x1 - x2) * det_b)) / denominator
    y = ((det_a * (y3 - y4)) - ((y1 - y2) * det_b)) / denominator
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (float(x), float(y))


def classify_gate_turn_label(turn_bias, threshold=GATE_TURN_THRESHOLD):
    if turn_bias <= -float(threshold):
        return TURN_LABEL_LEFT
    if turn_bias >= float(threshold):
        return TURN_LABEL_RIGHT
    return TURN_LABEL_STRAIGHT


def filter_marker_mask(
    mask,
    frame_shape,
    min_area_ratio=0.0035,
    min_aspect=1.30,
    min_height_ratio=0.16,
):
    frame_h, frame_w = frame_shape[:2]
    min_area = float(frame_h * frame_w) * float(min_area_ratio)
    min_height = float(frame_h) * float(min_height_ratio)
    filtered = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        if h < min_height:
            continue
        aspect = float(h) / float(w)
        bbox_fill = area / float(w * h)
        if aspect < min_aspect and bbox_fill < 0.58:
            continue
        cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)
        kept += 1
        if kept >= 3:
            break
    return filtered


def extract_color_strip_candidates(mask, frame_shape, max_candidates=6):
    frame_h, frame_w = frame_shape[:2]
    frame_area = float(frame_h * frame_w)
    candidates = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = float(h) / float(w)
        fill = area / float(w * h)
        area_ratio = area / max(1.0, frame_area)
        score = (
            clamp(area_ratio / 0.020, 0.0, 1.0) * 0.32
            + clamp((aspect - 1.0) / 5.0, 0.0, 1.0) * 0.42
            + clamp((fill - 0.20) / 0.70, 0.0, 1.0) * 0.26
        )
        candidates.append(
            ColorStripCandidate(
                bbox=(int(x), int(y), int(w), int(h)),
                center_x=float(x + (w * 0.5)),
                center_y=float(y + (h * 0.5)),
                top_mid=(float(x + (w * 0.5)), float(y)),
                bottom_mid=(float(x + (w * 0.5)), float(y + h)),
                height=float(h),
                width=float(w),
                area=float(area),
                score=float(score),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def select_best_gate(red_candidates, green_candidates, frame_shape, config):
    frame_h, frame_w = frame_shape[:2]
    best_gate = None
    best_score = -1.0
    min_gate_width = frame_w * 0.05
    max_gate_width = frame_w * 0.82

    for red in red_candidates:
        red_bottom = red.bbox[1] + red.bbox[3]
        for green in green_candidates:
            green_bottom = green.bbox[1] + green.bbox[3]
            if red.center_x >= green.center_x:
                continue
            width_px = green.center_x - red.center_x
            if not (min_gate_width <= width_px <= max_gate_width):
                continue

            height_similarity = 1.0 - min(1.0, abs(red.height - green.height) / max(red.height, green.height, 1.0))
            bottom_similarity = 1.0 - min(1.0, abs(red_bottom - green_bottom) / max(red.height, green.height, 1.0))
            top_pair_mid = midpoint(red.top_mid, green.top_mid)
            bottom_pair_mid = midpoint(red.bottom_mid, green.bottom_mid)
            cross_intersection = line_intersection(
                red.top_mid,
                green.bottom_mid,
                red.bottom_mid,
                green.top_mid,
            )
            if cross_intersection is None:
                target_y = frame_h * GATE_TARGET_Y_RATIO
                center_x = line_x_at_y(top_pair_mid, bottom_pair_mid, target_y)
                cross_intersection = (float(center_x), float(target_y))
            else:
                center_x = cross_intersection[0]
            offset = clamp((center_x - (frame_w * 0.5)) / max(1.0, frame_w * 0.5), -1.0, 1.0)
            turn_bias = float(offset)
            turn_label = classify_gate_turn_label(turn_bias)
            target_x = float(center_x)
            target_shift = width_px * float(config.gate_target_shift_ratio)
            if turn_label == TURN_LABEL_LEFT:
                target_x += target_shift
            elif turn_label == TURN_LABEL_RIGHT:
                target_x -= target_shift
            safe_left_x = red.center_x + (width_px * float(config.gate_safe_margin_ratio))
            safe_right_x = green.center_x - (width_px * float(config.gate_safe_margin_ratio))
            if safe_left_x < safe_right_x:
                target_x = clamp(target_x, safe_left_x, safe_right_x)
            target_offset = clamp((target_x - (frame_w * 0.5)) / max(1.0, frame_w * 0.5), -1.0, 1.0)
            center_bias = 1.0 - min(1.0, abs(offset))
            width_ratio = width_px / max(1.0, frame_w)
            width_bias = clamp((width_ratio - 0.08) / 0.42, 0.0, 1.0)
            red_outer_bias = clamp(((frame_w * 0.5) - red.center_x) / max(1.0, frame_w * 0.5), 0.0, 1.0)
            green_outer_bias = clamp((green.center_x - (frame_w * 0.5)) / max(1.0, frame_w * 0.5), 0.0, 1.0)
            outer_bias = 0.5 * (red_outer_bias + green_outer_bias)
            confidence = (
                red.score * 0.16
                + green.score * 0.16
                + height_similarity * 0.17
                + bottom_similarity * 0.17
                + width_bias * 0.22
                + outer_bias * 0.08
                + center_bias * 0.04
            )
            if confidence > best_score:
                best_score = confidence
                best_gate = GateCandidate(
                    red=red,
                    green=green,
                    center_x=float(center_x),
                    width_px=float(width_px),
                    offset=float(offset),
                    confidence=float(confidence),
                    top_pair_mid=top_pair_mid,
                    bottom_pair_mid=bottom_pair_mid,
                    cross_intersection=cross_intersection,
                    target_x=float(target_x),
                    target_offset=float(target_offset),
                    turn_bias=float(turn_bias),
                    turn_label=turn_label,
                )
    return best_gate


@dataclass
class UltrasonicReading:
    distance_cm: Optional[float]
    valid: bool
    timestamp: float


@dataclass
class DriveCommand:
    left_speed: int
    right_speed: int
    mode: str
    done: bool = False


@dataclass
class VisualWallObservation:
    wall_mask: np.ndarray = field(repr=False)
    raw_red_mask: np.ndarray = field(repr=False)
    raw_green_mask: np.ndarray = field(repr=False)
    red_mask: np.ndarray = field(repr=False)
    green_mask: np.ndarray = field(repr=False)
    wall_ratio: float
    center_block_ratio: float
    left_block_ratio: float
    right_block_ratio: float
    lower_open_ratio: float
    left_red_ratio: float
    left_green_ratio: float
    right_red_ratio: float
    right_green_ratio: float
    left_marker_score: float
    right_marker_score: float
    red_center_x: float
    green_center_x: float
    gate_center_x: float
    gate_target_x: float
    gate_width_px: float
    gate_width_ratio: float
    gate_offset: float
    gate_target_offset: float
    gate_turn_bias: float
    gate_turn_label: str
    gate_visible: bool
    gate_confidence: float
    red_visible: bool
    green_visible: bool
    single_post_mode: str
    opening_bias: float
    visual_open_score: float
    front_blocked_visual: bool
    confidence: float
    rois: Dict[str, tuple] = field(default_factory=dict, repr=False)
    red_candidates: List["ColorStripCandidate"] = field(default_factory=list, repr=False)
    green_candidates: List["ColorStripCandidate"] = field(default_factory=list, repr=False)
    best_gate: Optional["GateCandidate"] = field(default=None, repr=False)


@dataclass
class ColorStripCandidate:
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    top_mid: Tuple[float, float]
    bottom_mid: Tuple[float, float]
    height: float
    width: float
    area: float
    score: float


@dataclass
class GateCandidate:
    red: ColorStripCandidate
    green: ColorStripCandidate
    center_x: float
    width_px: float
    offset: float
    confidence: float
    top_pair_mid: Tuple[float, float]
    bottom_pair_mid: Tuple[float, float]
    cross_intersection: Tuple[float, float]
    target_x: float
    target_offset: float
    turn_bias: float
    turn_label: str


@dataclass
class DirectionalScan:
    direction: str
    distance_cm: Optional[float]
    distance_score: float
    visual_open_score: float
    combined_score: float
    center_block_ratio: float
    wall_ratio: float
    is_open: bool
    blocked_reason: str


@dataclass
class MazeConfig:
    mode: str = MODE_RUN_MAZE
    trig_pin: int = 21
    echo_pin: int = 20
    trigger_cm: float = 20.0
    clear_cm: float = SEARCH_CLEARANCE_CM
    follow_speed: int = 22
    spin_speed: int = 26
    right_rear_boost: int = RIGHT_REAR_BOOST
    right_rear_spin_boost: int = 0
    start_delay: float = 0.0
    scan_right_seconds: float = 0.55
    scan_left_seconds: float = 0.55
    turn_right_90_seconds: float = TURN_90_SECONDS
    turn_left_90_seconds: float = TURN_90_SECONDS
    uturn_seconds: float = TURN_90_SECONDS * 2.0
    scan_settle_seconds: float = SCAN_SETTLE_SECONDS
    forward_recover_seconds: float = FORWARD_RECOVER_SECONDS
    visual_open_min: float = VISUAL_OPEN_MIN
    visual_combined_open_min: float = VISUAL_COMBINED_OPEN_MIN
    visual_center_block_threshold: float = VISUAL_CENTER_BLOCK_THRESHOLD
    visual_wall_ratio_block_threshold: float = VISUAL_WALL_RATIO_BLOCK_THRESHOLD
    forward_steer_gain: float = 22.0
    max_forward_steer: float = 8.0
    turn_finish_center_max: float = 0.24
    turn_finish_open_min: float = 0.46
    turn_finish_bias_max: float = 0.20
    turn_finish_extra_seconds: float = 0.35
    scan_trigger_hits: int = 2
    visual_trigger_hits: int = 3
    scan_cooldown_seconds: float = 0.55
    emergency_stop_cm: float = 11.0
    hard_turn_center_threshold: float = 0.50
    hard_turn_wall_threshold: float = 0.38
    hard_turn_hits: int = 2
    marker_ratio_threshold: float = MARKER_RATIO_THRESHOLD
    marker_hits: int = 2
    marker_scan_distance_cm: float = MARKER_SCAN_DISTANCE_CM
    marker_memory_seconds: float = MARKER_MEMORY_SECONDS
    marker_steer_gain: float = MARKER_STEER_GAIN
    search_spin_seconds: float = 0.42
    gate_pass_seconds: float = 0.42
    reverse_seconds: float = 0.30
    reverse_speed: int = 20
    gate_confidence_threshold: float = GATE_CONFIDENCE_THRESHOLD
    align_gate_offset_threshold: float = ALIGN_GATE_OFFSET_THRESHOLD
    align_spin_offset_threshold: float = ALIGN_SPIN_OFFSET_THRESHOLD
    search_small_seconds: float = SEARCH_SMALL_SECONDS
    search_large_seconds: float = SEARCH_LARGE_SECONDS
    search_forward_burst_seconds: float = SEARCH_FORWARD_BURST_SECONDS
    gate_loss_grace_seconds: float = GATE_LOSS_GRACE_SECONDS
    gate_commit_seconds: float = GATE_COMMIT_SECONDS
    gate_blind_drive_seconds: float = GATE_BLIND_DRIVE_SECONDS
    advance_open_seconds: float = 0.55
    align_spin_speed_min: int = 18
    align_spin_speed_gain: float = 18.0
    gate_target_shift_ratio: float = GATE_TARGET_SHIFT_RATIO
    gate_safe_margin_ratio: float = GATE_SAFE_MARGIN_RATIO
    single_post_gap_ratio: float = SINGLE_POST_GAP_RATIO
    align_near_width_ratio: float = ALIGN_NEAR_WIDTH_RATIO
    pass_blind_steer_scale: float = PASS_BLIND_STEER_SCALE
    pass_blind_speed_scale: float = PASS_BLIND_SPEED_SCALE


class TrimmedCarMotorController(base.CarMotorController):
    RIGHT_REAR_INDEX = 1

    def __init__(self, right_rear_boost=RIGHT_REAR_BOOST, right_rear_spin_boost=0):
        self.right_rear_boost = int(max(0, right_rear_boost))
        self.right_rear_spin_boost = int(max(0, right_rear_spin_boost))
        super().__init__()

    def set_tank_drive(self, left_speed, right_speed, straight_mode=False):
        per_motor_speeds = [0, 0, 0, 0]
        for index in base.LEFT_MOTOR_INDEXES:
            per_motor_speeds[index] = int(left_speed)
        for index in base.RIGHT_MOTOR_INDEXES:
            per_motor_speeds[index] = int(right_speed)

        if (
            straight_mode
            and self.right_rear_boost > 0
            and left_speed > 0
            and right_speed > 0
        ):
            per_motor_speeds[self.RIGHT_REAR_INDEX] += self.right_rear_boost
        elif (
            self.right_rear_spin_boost > 0
            and left_speed != 0
            and right_speed != 0
            and ((left_speed > 0 and right_speed < 0) or (left_speed < 0 and right_speed > 0))
        ):
            if right_speed > 0:
                per_motor_speeds[self.RIGHT_REAR_INDEX] += self.right_rear_spin_boost
            else:
                per_motor_speeds[self.RIGHT_REAR_INDEX] -= self.right_rear_spin_boost

        payload = bytearray(b"#ba")
        for index, speed in enumerate(per_motor_speeds):
            direction = base.MOTOR_FORWARD_DIRS[index]
            if speed < 0:
                direction = base.reverse_direction(direction)
            payload.extend(direction.encode("ascii"))

        for speed in per_motor_speeds:
            magnitude = int(clamp(abs(speed), 0, 65535))
            payload.append(magnitude & 0xFF)
            payload.append((magnitude >> 8) & 0xFF)
        self._write_payload(payload)


class UltrasonicSensor:
    def __init__(self, gpio, trig_pin, echo_pin, sample_hz=ULTRASONIC_SAMPLE_HZ):
        self.gpio = gpio
        self.trig_pin = int(trig_pin)
        self.echo_pin = int(echo_pin)
        self.sample_interval = 1.0 / max(1.0, float(sample_hz))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._samples = deque(maxlen=5)
        self._last_reading = UltrasonicReading(distance_cm=None, valid=False, timestamp=time.perf_counter())

        self.gpio.setup(self.trig_pin, self.gpio.OUT)
        self.gpio.setup(self.echo_pin, self.gpio.IN)
        self.gpio.output(self.trig_pin, False)
        time.sleep(0.05)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _measure_once(self):
        self.gpio.output(self.trig_pin, False)
        time.sleep(0.0002)
        self.gpio.output(self.trig_pin, True)
        time.sleep(0.00001)
        self.gpio.output(self.trig_pin, False)

        deadline = time.perf_counter() + ULTRASONIC_MAX_WAIT_SECONDS
        while self.gpio.input(self.echo_pin) == 0:
            if time.perf_counter() >= deadline:
                return None

        pulse_start = time.perf_counter()
        deadline = pulse_start + ULTRASONIC_MAX_WAIT_SECONDS
        while self.gpio.input(self.echo_pin) == 1:
            if time.perf_counter() >= deadline:
                return None

        pulse_end = time.perf_counter()
        distance_cm = (pulse_end - pulse_start) * 17150.0
        if not (ULTRASONIC_VALID_MIN_CM <= distance_cm <= ULTRASONIC_VALID_MAX_CM):
            return None
        return float(distance_cm)

    def _run(self):
        while not self._stop_event.is_set():
            started = time.perf_counter()
            distance_cm = self._measure_once()
            timestamp = time.perf_counter()
            with self._lock:
                if distance_cm is not None:
                    self._samples.append(distance_cm)
                    filtered = float(median(self._samples))
                    self._last_reading = UltrasonicReading(
                        distance_cm=filtered,
                        valid=True,
                        timestamp=timestamp,
                    )
                else:
                    self._last_reading = UltrasonicReading(
                        distance_cm=self._last_reading.distance_cm if self._samples else None,
                        valid=False,
                        timestamp=timestamp,
                    )
            remaining = self.sample_interval - (time.perf_counter() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def get_reading(self):
        with self._lock:
            reading = self._last_reading
            return UltrasonicReading(reading.distance_cm, reading.valid, reading.timestamp)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.gpio.output(self.trig_pin, False)


class NullLCDDisplay:
    def update(self, line_1, line_2, force=False):
        return None

    def close(self):
        return None


class NullMotorController:
    def stop(self):
        return None

    def hold_stop(self):
        return None

    def set_tank_drive(self, left_speed, right_speed, straight_mode=False):
        return None

    def close(self):
        return None


def build_black_wall_observation(frame_bgr, config):
    wall_mask_raw = smooth.build_wall_mask(frame_bgr)
    frame_h, frame_w = frame_bgr.shape[:2]
    wall_mask = wall_mask_raw.copy()
    color_y0 = int(frame_h * 0.20)
    color_roi_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    color_roi_mask[color_y0:, :] = 255
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    b_channel, g_channel, r_channel = cv2.split(frame_bgr)
    _, a_channel, _ = cv2.split(lab)
    _, cr_channel, cb_channel = cv2.split(ycrcb)
    sat_mask = cv2.inRange(hsv, (0, 45, 0), (180, 255, 255))
    value_mask = cv2.inRange(hsv, (0, 0, 20), (180, 255, 255))
    white_suppress = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
    color_gate = cv2.bitwise_and(cv2.bitwise_and(sat_mask, value_mask), cv2.bitwise_not(white_suppress))
    red_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 45), (18, 255, 255)),
        cv2.inRange(hsv, (162, 70, 45), (180, 255, 255)),
    )
    red_lowlight_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 36, 8), (18, 255, 150)),
        cv2.inRange(hsv, (162, 36, 8), (180, 255, 150)),
    )
    red_rgb = np.where(
        (r_channel.astype(np.int16) >= g_channel.astype(np.int16) + 10)
        & (r_channel.astype(np.int16) >= b_channel.astype(np.int16) + 10)
        & (r_channel >= 34),
        255,
        0,
    ).astype(np.uint8)
    red_lab = np.where(
        (a_channel >= 130)
        & (r_channel >= 30),
        255,
        0,
    ).astype(np.uint8)
    red_ycrcb = np.where(
        (cr_channel >= 132)
        & (cr_channel.astype(np.int16) >= cb_channel.astype(np.int16) + 4)
        & (r_channel >= 30),
        255,
        0,
    ).astype(np.uint8)
    green_hsv = cv2.inRange(hsv, (46, 96, 45), (86, 255, 255))
    green_rgb = np.where(
        (g_channel.astype(np.int16) >= r_channel.astype(np.int16) + 24)
        & (g_channel.astype(np.int16) >= b_channel.astype(np.int16) + 8)
        & (g_channel >= 78),
        255,
        0,
    ).astype(np.uint8)
    green_lab = np.where(
        (a_channel <= 118)
        & (g_channel >= 62),
        255,
        0,
    ).astype(np.uint8)
    red_seed = cv2.bitwise_or(
        cv2.bitwise_or(red_hsv, red_lowlight_hsv),
        cv2.bitwise_or(red_rgb, cv2.bitwise_or(red_lab, red_ycrcb)),
    )
    red_mask = cv2.bitwise_and(red_seed, color_gate)
    green_mask = cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_or(green_hsv, green_rgb), green_lab), color_gate)
    red_mask = cv2.bitwise_and(red_mask, color_roi_mask)
    green_mask = cv2.bitwise_and(green_mask, color_roi_mask)
    raw_red_mask = red_mask.copy()
    raw_green_mask = green_mask.copy()
    color_kernel = np.ones((5, 5), dtype=np.uint8)
    red_close_kernel = np.ones((7, 7), dtype=np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_close_kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, color_kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, color_kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, color_kernel, iterations=1)
    red_mask = filter_marker_mask(
        red_mask,
        frame_bgr.shape,
        min_area_ratio=0.00035,
        min_aspect=0.95,
        min_height_ratio=0.06,
    )
    green_mask = filter_marker_mask(
        green_mask,
        frame_bgr.shape,
        min_area_ratio=0.00045,
        min_aspect=1.05,
        min_height_ratio=0.07,
    )
    rois = {
        "full": smooth.roi_rect(frame_bgr.shape, 0.10, 0.22, 0.80, 0.64),
        "center": smooth.roi_rect(frame_bgr.shape, 0.36, 0.60, 0.28, 0.18),
        "left": smooth.roi_rect(frame_bgr.shape, 0.14, 0.58, 0.18, 0.20),
        "right": smooth.roi_rect(frame_bgr.shape, 0.68, 0.58, 0.18, 0.20),
        "open_strip": smooth.roi_rect(frame_bgr.shape, 0.22, 0.66, 0.56, 0.12),
        "marker_left": smooth.roi_rect(frame_bgr.shape, 0.00, 0.20, 0.50, 0.80),
        "marker_right": smooth.roi_rect(frame_bgr.shape, 0.50, 0.20, 0.50, 0.80),
    }

    wall_ratio = smooth.occupancy_ratio(wall_mask, rois["full"])
    center_block_ratio = smooth.occupancy_ratio(wall_mask, rois["center"])
    left_block_ratio = smooth.occupancy_ratio(wall_mask, rois["left"])
    right_block_ratio = smooth.occupancy_ratio(wall_mask, rois["right"])
    lower_wall_ratio = smooth.occupancy_ratio(wall_mask, rois["open_strip"])
    red_candidates = extract_color_strip_candidates(red_mask, frame_bgr.shape)
    green_candidates = extract_color_strip_candidates(green_mask, frame_bgr.shape)
    best_gate = select_best_gate(red_candidates, green_candidates, frame_bgr.shape, config)

    left_red_ratio = smooth.occupancy_ratio(red_mask, rois["marker_left"])
    left_green_ratio = smooth.occupancy_ratio(green_mask, rois["marker_left"])
    right_red_ratio = smooth.occupancy_ratio(red_mask, rois["marker_right"])
    right_green_ratio = smooth.occupancy_ratio(green_mask, rois["marker_right"])
    red_center_x = red_candidates[0].center_x if red_candidates else -1.0
    green_center_x = green_candidates[0].center_x if green_candidates else -1.0
    lower_open_ratio = clamp(1.0 - lower_wall_ratio, 0.0, 1.0)
    opening_bias = left_block_ratio - right_block_ratio
    left_marker_score = clamp(left_red_ratio * 1.45 - left_green_ratio * 0.55, 0.0, 1.0)
    right_marker_score = clamp(right_green_ratio * 1.45 - right_red_ratio * 0.55, 0.0, 1.0)
    red_visible = bool(red_candidates)
    green_visible = bool(green_candidates)
    if red_visible and green_visible:
        single_post_mode = POST_MODE_BOTH
    elif red_visible:
        single_post_mode = POST_MODE_RED_ONLY
    elif green_visible:
        single_post_mode = POST_MODE_GREEN_ONLY
    else:
        single_post_mode = POST_MODE_NONE
    gate_visible = bool(best_gate is not None and best_gate.confidence >= config.gate_confidence_threshold)
    if gate_visible:
        gate_center_x = best_gate.center_x
        gate_target_x = best_gate.target_x
        gate_width_px = best_gate.width_px
        gate_width_ratio = best_gate.width_px / max(1.0, frame_w)
        gate_offset = best_gate.offset
        gate_target_offset = best_gate.target_offset
        gate_turn_bias = best_gate.turn_bias
        gate_turn_label = best_gate.turn_label
        gate_confidence = best_gate.confidence
    else:
        gate_center_x = frame_w * 0.5
        gate_target_x = gate_center_x
        gate_width_px = 0.0
        gate_width_ratio = 0.0
        gate_offset = 0.0
        gate_target_offset = 0.0
        gate_turn_bias = 0.0
        gate_turn_label = TURN_LABEL_STRAIGHT
        gate_confidence = 0.0

    wall_mask_display = cv2.bitwise_and(
        wall_mask,
        cv2.bitwise_not(cv2.bitwise_or(red_mask, green_mask)),
    )

    weighted_block = (
        center_block_ratio * 0.64
        + left_block_ratio * 0.12
        + right_block_ratio * 0.12
        + lower_wall_ratio * 0.12
    )
    visual_open_score = clamp(0.74 * (1.0 - weighted_block) + 0.26 * lower_open_ratio, 0.0, 1.0)
    front_blocked_visual = (
        center_block_ratio >= config.visual_center_block_threshold
        or (wall_ratio >= config.visual_wall_ratio_block_threshold and lower_open_ratio <= 0.50)
        or visual_open_score <= VISUAL_HARD_BLOCK_SCORE
    )

    edge_crop = wall_mask[int(frame_h * 0.48) : int(frame_h * 0.82), int(frame_w * 0.18) : int(frame_w * 0.82)]
    if edge_crop.size > 0:
        edge_ratio = cv2.countNonZero(edge_crop) / float(edge_crop.size)
    else:
        edge_ratio = 0.0
    confidence = clamp(max(wall_ratio, edge_ratio) * 2.2 + 0.12, 0.0, 1.0)

    return VisualWallObservation(
        wall_mask=wall_mask_display,
        raw_red_mask=raw_red_mask,
        raw_green_mask=raw_green_mask,
        red_mask=red_mask,
        green_mask=green_mask,
        wall_ratio=float(wall_ratio),
        center_block_ratio=float(center_block_ratio),
        left_block_ratio=float(left_block_ratio),
        right_block_ratio=float(right_block_ratio),
        lower_open_ratio=float(lower_open_ratio),
        left_red_ratio=float(left_red_ratio),
        left_green_ratio=float(left_green_ratio),
        right_red_ratio=float(right_red_ratio),
        right_green_ratio=float(right_green_ratio),
        left_marker_score=float(left_marker_score),
        right_marker_score=float(right_marker_score),
        red_center_x=float(red_center_x),
        green_center_x=float(green_center_x),
        gate_center_x=float(gate_center_x),
        gate_target_x=float(gate_target_x),
        gate_width_px=float(gate_width_px),
        gate_width_ratio=float(gate_width_ratio),
        gate_offset=float(gate_offset),
        gate_target_offset=float(gate_target_offset),
        gate_turn_bias=float(gate_turn_bias),
        gate_turn_label=gate_turn_label,
        gate_visible=bool(gate_visible),
        gate_confidence=float(gate_confidence),
        red_visible=bool(red_visible),
        green_visible=bool(green_visible),
        single_post_mode=single_post_mode,
        red_candidates=red_candidates,
        green_candidates=green_candidates,
        best_gate=best_gate,
        opening_bias=float(opening_bias),
        visual_open_score=float(visual_open_score),
        front_blocked_visual=bool(front_blocked_visual),
        confidence=float(confidence),
        rois=rois,
    )


def classify_direction(direction, reading, visual, config):
    if reading.valid and reading.distance_cm is not None:
        if config.clear_cm <= config.trigger_cm:
            distance_score = 1.0 if reading.distance_cm > config.clear_cm else 0.0
        else:
            distance_score = clamp(
                (reading.distance_cm - config.trigger_cm) / (config.clear_cm - config.trigger_cm),
                0.0,
                1.0,
            )
    else:
        distance_score = 0.40

    combined_score = clamp(distance_score * 0.62 + visual.visual_open_score * 0.38, 0.0, 1.0)
    blocked_reason = ""
    is_open = False

    if reading.valid and reading.distance_cm is not None and reading.distance_cm <= config.trigger_cm:
        blocked_reason = "dist<=trigger"
    elif visual.front_blocked_visual and visual.visual_open_score < config.visual_open_min:
        blocked_reason = "visual_blocked"
    elif (
        reading.valid
        and reading.distance_cm is not None
        and reading.distance_cm >= config.clear_cm
        and visual.visual_open_score >= config.visual_open_min
    ):
        is_open = True
        blocked_reason = "clear"
    elif combined_score >= config.visual_combined_open_min and not visual.front_blocked_visual:
        is_open = True
        blocked_reason = "combined"
    else:
        blocked_reason = "ambiguous"

    return DirectionalScan(
        direction=direction,
        distance_cm=reading.distance_cm,
        distance_score=float(distance_score),
        visual_open_score=float(visual.visual_open_score),
        combined_score=float(combined_score),
        center_block_ratio=float(visual.center_block_ratio),
        wall_ratio=float(visual.wall_ratio),
        is_open=bool(is_open),
        blocked_reason=blocked_reason,
    )


class MazeNavigator:
    def __init__(self, config):
        self.config = config
        self.mode = SEARCH_FORWARD
        self.state_started_at = time.perf_counter()
        self.last_decision = ""
        self.last_trigger_reason = ""
        self.search_pattern = [("left", self.config.search_small_seconds + (self.config.search_large_seconds * 1.35))]
        self.search_step_index = 0
        self.search_step_started_at = self.state_started_at
        self.search_cycle_count = 0
        self.last_gate_seen_at = -999.0
        self.last_gate_offset = 0.0
        self.backoff_return_mode = SEARCH_FORWARD
        self.pass_gate_started_at = self.state_started_at
        self.committed_gate_offset = 0.0
        self.align_ready_hits = 0
        self.pass_last_seen_at = -999.0
        self.pass_last_offset = 0.0
        self.pass_last_width_ratio = 0.0
        self.pass_peak_width_ratio = 0.0
        self.turn_bias_history = deque(maxlen=3)
        self.turn_candidate_label = TURN_LABEL_STRAIGHT
        self.turn_candidate_hits = 0
        self.current_turn_bias = 0.0
        self.locked_turn_label = TURN_LABEL_STRAIGHT
        self.pass_turn_label = TURN_LABEL_STRAIGHT
        self.single_post_candidate_mode = POST_MODE_NONE
        self.single_post_candidate_hits = 0
        self.current_post_mode = POST_MODE_NONE
        self.pass_last_target_offset = 0.0
        self.pass_last_combined_steer = 0.0
        self.pass_last_gate_width_px = 0.0
        self.pass_last_single_post_mode = POST_MODE_NONE
        self.pass_blind_active = False

    def _reset_turn_tracking(self):
        self.turn_bias_history.clear()
        self.turn_candidate_label = TURN_LABEL_STRAIGHT
        self.turn_candidate_hits = 0
        self.current_turn_bias = 0.0
        self.locked_turn_label = TURN_LABEL_STRAIGHT
        self.pass_turn_label = TURN_LABEL_STRAIGHT

    def _reset_pass_tracking(self):
        self.single_post_candidate_mode = POST_MODE_NONE
        self.single_post_candidate_hits = 0
        self.current_post_mode = POST_MODE_NONE
        self.pass_last_target_offset = 0.0
        self.pass_last_combined_steer = 0.0
        self.pass_last_gate_width_px = 0.0
        self.pass_last_single_post_mode = POST_MODE_NONE
        self.pass_blind_active = False

    def _set_mode(self, mode, now):
        self.mode = mode
        self.state_started_at = now
        if mode != ALIGN_GATE:
            self.align_ready_hits = 0
        if mode != PASS_GATE:
            self.pass_blind_active = False
            self.current_post_mode = POST_MODE_NONE
        if mode == PASS_GATE:
            self.pass_gate_started_at = now
            self.pass_last_seen_at = now
            self.pass_last_width_ratio = 0.0
            self.pass_peak_width_ratio = 0.0

    def _emit(self, left_speed, right_speed, mode, done=False):
        return DriveCommand(
            left_speed=int(clamp(left_speed, -base.MAX_SPEED, base.MAX_SPEED)),
            right_speed=int(clamp(right_speed, -base.MAX_SPEED, base.MAX_SPEED)),
            mode=mode,
            done=done,
        )

    def _stop_command(self, mode_name):
        return self._emit(0, 0, mode_name)

    def _forward_command(self, mode_name):
        return self._emit(self.config.follow_speed, self.config.follow_speed, mode_name)

    def _reverse_command(self, mode_name):
        return self._emit(-self.config.reverse_speed, -self.config.reverse_speed, mode_name)

    def _open_area_forward_command(self, visual, mode_name):
        steer = clamp(
            -visual.opening_bias * self.config.marker_steer_gain * 8.0,
            -self.config.max_forward_steer,
            self.config.max_forward_steer,
        )
        speed = max(12, int(self.config.follow_speed * 0.80))
        return self._emit(speed + steer, speed - steer, mode_name)

    def _emit_with_steer(self, base_speed, steer, mode_name):
        return self._emit(base_speed + steer, base_speed - steer, mode_name)

    def _compute_gate_steer(self, offset, gain_scale):
        return clamp(
            offset * self.config.marker_steer_gain * gain_scale,
            -self.config.max_forward_steer,
            self.config.max_forward_steer,
        )

    def _update_turn_tracking(self, visual):
        if visual.gate_visible:
            self.turn_bias_history.append(float(visual.gate_turn_bias))
            self.current_turn_bias = float(median(self.turn_bias_history))
            candidate = classify_gate_turn_label(self.current_turn_bias)
            if candidate == self.turn_candidate_label:
                self.turn_candidate_hits += 1
            else:
                self.turn_candidate_label = candidate
                self.turn_candidate_hits = 1
            if candidate == TURN_LABEL_STRAIGHT:
                if abs(self.current_turn_bias) < (GATE_TURN_THRESHOLD * 0.60):
                    self.locked_turn_label = TURN_LABEL_STRAIGHT
            elif self.turn_candidate_hits >= 3:
                self.locked_turn_label = candidate
        elif self.mode not in (ALIGN_GATE, PASS_GATE):
            self._reset_turn_tracking()

    def _resolve_turn_label(self, visual):
        if self.mode == PASS_GATE and self.pass_turn_label != TURN_LABEL_STRAIGHT:
            return self.pass_turn_label
        if self.locked_turn_label != TURN_LABEL_STRAIGHT:
            return self.locked_turn_label
        return visual.gate_turn_label

    def _gate_forward_command(self, visual, mode_name, speed=None):
        speed = self.config.follow_speed if speed is None else speed
        steer = self._compute_gate_steer(visual.gate_target_offset, 12.0)
        return self._emit_with_steer(speed, steer, mode_name)

    def _gate_forward_from_offset(self, offset, mode_name, speed=None):
        speed = self.config.follow_speed if speed is None else speed
        steer = self._compute_gate_steer(offset, 10.0)
        return self._emit_with_steer(speed, steer, mode_name)

    def _gate_blind_drive_command(self, offset, mode_name):
        base_speed = max(15, int(self.config.follow_speed * 0.92))
        steer = self._compute_gate_steer(offset, 8.0)
        return self._emit_with_steer(base_speed, steer, mode_name)

    def _gate_blind_drive_from_steer(self, steer, mode_name):
        base_speed = max(12, int(self.config.follow_speed * self.config.pass_blind_speed_scale))
        blind_steer = clamp(
            steer * self.config.pass_blind_steer_scale,
            -self.config.max_forward_steer,
            self.config.max_forward_steer,
        )
        return self._emit_with_steer(base_speed, blind_steer, mode_name)

    def _pass_gate_search_command(self):
        if self.pass_last_single_post_mode == POST_MODE_RED_ONLY:
            return self._spin_command(TURN_RIGHT, PASS_GATE)
        if self.pass_last_single_post_mode == POST_MODE_GREEN_ONLY:
            return self._spin_command(TURN_LEFT, PASS_GATE)
        if self.pass_turn_label == TURN_LABEL_RIGHT:
            return self._spin_command(TURN_RIGHT, PASS_GATE)
        if self.pass_turn_label == TURN_LABEL_LEFT:
            return self._spin_command(TURN_LEFT, PASS_GATE)
        if self.pass_last_combined_steer >= 0.0:
            return self._spin_command(TURN_RIGHT, PASS_GATE)
        return self._spin_command(TURN_LEFT, PASS_GATE)

    def _approach_gate_command(self, offset, width_ratio, mode_name, pass_mode=False, single_post=False):
        abs_offset = abs(offset)
        if pass_mode:
            base_speed = int(
                clamp(
                    self.config.follow_speed * (0.64 + width_ratio * 0.76 - abs_offset * 0.18),
                    max(12, int(self.config.follow_speed * (0.58 if single_post else 0.56))),
                    max(13, int(self.config.follow_speed * (0.76 if single_post else 0.82))),
                )
            )
            steer = self._compute_gate_steer(offset, 11.8 if single_post else 10.0)
        else:
            base_speed = int(
                clamp(
                    self.config.follow_speed * (0.52 + width_ratio * 0.80 - abs_offset * 0.35),
                    max(11, int(self.config.follow_speed * 0.48)),
                    max(12, int(self.config.follow_speed * 0.82)),
                )
            )
            if width_ratio >= self.config.align_near_width_ratio:
                base_speed = min(base_speed, max(11, int(self.config.follow_speed * 0.62)))
                steer = self._compute_gate_steer(offset, 11.0)
            else:
                steer = self._compute_gate_steer(offset, 13.5)
        return self._emit_with_steer(base_speed, steer, mode_name), float(steer)

    def _spin_command(self, direction, mode_name):
        if direction == TURN_RIGHT:
            return self._emit(self.config.spin_speed, -self.config.spin_speed, mode_name)
        return self._emit(-self.config.spin_speed, self.config.spin_speed, mode_name)

    def _align_spin_command(self, offset, mode_name):
        align_speed = int(
            clamp(
                self.config.align_spin_speed_min + abs(offset) * self.config.align_spin_speed_gain * self.config.spin_speed,
                self.config.align_spin_speed_min,
                self.config.spin_speed,
            )
        )
        if offset >= 0.0:
            return self._emit(align_speed, -align_speed, mode_name)
        return self._emit(-align_speed, align_speed, mode_name)

    def _commit_pass_gate(self, now, visual, offset):
        self.last_decision = "PASS_GATE"
        self.pass_gate_started_at = now
        self.committed_gate_offset = offset
        self.pass_last_offset = offset
        self.pass_last_target_offset = offset
        self.pass_last_gate_width_px = max(self.pass_last_gate_width_px, visual.gate_width_px)
        self.pass_last_single_post_mode = POST_MODE_BOTH
        self.current_post_mode = POST_MODE_BOTH
        resolved_turn_label = self._resolve_turn_label(visual)
        self.pass_turn_label = resolved_turn_label
        self.pass_last_combined_steer = self._compute_gate_steer(offset, 10.0)
        self._set_mode(PASS_GATE, now)
        return self._gate_forward_from_offset(
            offset,
            PASS_GATE,
            speed=max(12, int(self.config.follow_speed * 0.72)),
        )

    def _reset_search_sequence(self, now):
        self.search_step_index = 0
        self.search_step_started_at = now
        self.search_cycle_count = 0

    def _enter_search_mode(self, mode, now):
        self._reset_turn_tracking()
        self._reset_pass_tracking()
        self._set_mode(mode, now)
        self._reset_search_sequence(now)

    def _current_search_step(self):
        return self.search_pattern[self.search_step_index]

    def _advance_search_step(self, now):
        self.search_step_index += 1
        if self.search_step_index >= len(self.search_pattern):
            self.search_step_index = 0
            self.search_cycle_count += 1
        self.search_step_started_at = now

    def _search_command(self, mode_name, now):
        step_direction, step_duration = self._current_search_step()
        if step_duration > 0.0 and (now - self.search_step_started_at) >= step_duration:
            self._advance_search_step(now)
            if self.search_step_index == 0 and self.search_cycle_count > 0:
                return None
        return self._spin_command(TURN_LEFT, mode_name)

    def _resolve_pass_post_mode(self, visual):
        if visual.gate_visible:
            self.single_post_candidate_mode = POST_MODE_BOTH
            self.single_post_candidate_hits = 0
            return POST_MODE_BOTH
        observed_mode = visual.single_post_mode
        if observed_mode in (POST_MODE_RED_ONLY, POST_MODE_GREEN_ONLY):
            if observed_mode == self.single_post_candidate_mode:
                self.single_post_candidate_hits += 1
            else:
                self.single_post_candidate_mode = observed_mode
                self.single_post_candidate_hits = 1
            if self.single_post_candidate_hits >= 1:
                return observed_mode
            if self.pass_last_single_post_mode in (POST_MODE_RED_ONLY, POST_MODE_GREEN_ONLY):
                return self.pass_last_single_post_mode
        else:
            self.single_post_candidate_mode = POST_MODE_NONE
            self.single_post_candidate_hits = 0
        return POST_MODE_NONE

    def _single_post_target(self, visual, post_mode):
        frame_w = visual.red_mask.shape[1]
        half_frame_w = max(1.0, frame_w * 0.5)
        remembered_gate_width_px = max(self.pass_last_gate_width_px, visual.gate_width_px, frame_w * 0.18)
        target_x = None
        if post_mode == POST_MODE_RED_ONLY and visual.red_candidates:
            target_x = visual.red_candidates[0].center_x + (
                remembered_gate_width_px * self.config.single_post_gap_ratio
            )
        elif post_mode == POST_MODE_GREEN_ONLY and visual.green_candidates:
            target_x = visual.green_candidates[0].center_x - (
                remembered_gate_width_px * self.config.single_post_gap_ratio
            )
        if target_x is None:
            return None, 0.0, 0.0
        target_x = clamp(target_x, 0.0, frame_w - 1.0)
        target_offset = clamp((target_x - half_frame_w) / half_frame_w, -1.0, 1.0)
        width_ratio = clamp(remembered_gate_width_px / max(1.0, frame_w), 0.0, 1.0)
        return float(target_x), float(target_offset), float(width_ratio)

    def _remember_pass_guidance(self, now, visual, offset, steer, post_mode):
        self.pass_last_seen_at = now
        self.pass_last_offset = offset
        self.pass_last_target_offset = offset
        self.pass_last_combined_steer = float(steer)
        self.pass_last_single_post_mode = post_mode
        self.current_post_mode = post_mode
        self.pass_blind_active = False
        if visual.gate_width_px > 0.0:
            self.pass_last_gate_width_px = max(self.pass_last_gate_width_px, visual.gate_width_px)
        if visual.gate_width_ratio > 0.0:
            self.pass_last_width_ratio = visual.gate_width_ratio
            self.pass_peak_width_ratio = max(self.pass_peak_width_ratio, visual.gate_width_ratio)

    def update(self, reading, visual, now):
        self._update_turn_tracking(visual)
        if self.mode != PASS_GATE:
            self.current_post_mode = visual.single_post_mode
            self.pass_blind_active = False
        if visual.gate_visible:
            self.last_gate_seen_at = now
            self.last_gate_offset = visual.gate_target_offset

        too_close = bool(
            reading.valid
            and reading.distance_cm is not None
            and reading.distance_cm <= self.config.emergency_stop_cm
        )

        if self.mode == BACKOFF:
            if (now - self.state_started_at) >= self.config.reverse_seconds:
                self.last_trigger_reason = "resume_search"
                self._enter_search_mode(self.backoff_return_mode, now)
                return self._search_command(self.mode, now)
            return self._reverse_command(BACKOFF)

        if too_close and not visual.gate_visible:
            self.last_trigger_reason = "ultra_backoff"
            self.last_decision = "BACKOFF"
            self.backoff_return_mode = SEARCH_NEXT if self.mode in (ALIGN_GATE, PASS_GATE, SEARCH_NEXT) else SEARCH_FORWARD
            self._set_mode(BACKOFF, now)
            return self._reverse_command(BACKOFF)

        if self.mode in (SEARCH_FORWARD, SEARCH_NEXT):
            if visual.gate_visible:
                self.last_trigger_reason = "gate_found"
                self.last_decision = "ALIGN_GATE"
                self._set_mode(ALIGN_GATE, now)
                if abs(visual.gate_target_offset) >= 0.55:
                    return self._align_spin_command(visual.gate_target_offset, ALIGN_GATE)
                command, _ = self._approach_gate_command(
                    visual.gate_target_offset,
                    visual.gate_width_ratio,
                    ALIGN_GATE,
                    pass_mode=False,
                )
                return command
            search_command = self._search_command(self.mode, now)
            if search_command is None:
                if too_close:
                    self.last_trigger_reason = "search_blocked"
                    self.backoff_return_mode = SEARCH_NEXT
                    self.last_decision = "BACKOFF"
                    self._set_mode(BACKOFF, now)
                    return self._reverse_command(BACKOFF)
                self.last_trigger_reason = "advance_open"
                self.last_decision = "ADVANCE_OPEN"
                self._set_mode(ADVANCE_OPEN, now)
                return self._open_area_forward_command(visual, ADVANCE_OPEN)
            return search_command

        if self.mode == ALIGN_GATE:
            if visual.gate_visible:
                if visual.gate_confidence >= self.config.gate_confidence_threshold and abs(
                    visual.gate_target_offset
                ) <= self.config.align_gate_offset_threshold:
                    self.align_ready_hits += 1
                else:
                    self.align_ready_hits = 0
                if self.align_ready_hits >= 2:
                    return self._commit_pass_gate(now, visual, visual.gate_target_offset)
                if abs(visual.gate_target_offset) >= max(self.config.align_spin_offset_threshold, 0.38):
                    return self._align_spin_command(visual.gate_target_offset, ALIGN_GATE)
                command, _ = self._approach_gate_command(
                    visual.gate_target_offset,
                    visual.gate_width_ratio,
                    ALIGN_GATE,
                    pass_mode=False,
                )
                return command

            if (now - self.last_gate_seen_at) <= self.config.gate_blind_drive_seconds:
                if abs(self.last_gate_offset) >= max(self.config.align_spin_offset_threshold, 0.38):
                    return self._align_spin_command(self.last_gate_offset, ALIGN_GATE)
                return self._gate_blind_drive_command(self.last_gate_offset, ALIGN_GATE)

            self.last_trigger_reason = "gate_lost_align"
            self._enter_search_mode(SEARCH_FORWARD, now)
            return self._search_command(SEARCH_FORWARD, now)

        if self.mode == PASS_GATE:
            if visual.gate_visible:
                command, steer = self._approach_gate_command(
                    visual.gate_target_offset,
                    visual.gate_width_ratio,
                    PASS_GATE,
                    pass_mode=True,
                )
                self._remember_pass_guidance(now, visual, visual.gate_target_offset, steer, POST_MODE_BOTH)
                return command

            post_mode = self._resolve_pass_post_mode(visual)
            if post_mode in (POST_MODE_RED_ONLY, POST_MODE_GREEN_ONLY):
                _, target_offset, width_ratio = self._single_post_target(visual, post_mode)
                if width_ratio > 0.0:
                    command, steer = self._approach_gate_command(
                        target_offset,
                        width_ratio,
                        PASS_GATE,
                        pass_mode=True,
                        single_post=True,
                    )
                    self._remember_pass_guidance(now, visual, target_offset, steer, post_mode)
                    return command

            blind_drive_active = (now - self.pass_last_seen_at) <= self.config.gate_blind_drive_seconds
            if blind_drive_active:
                self.current_post_mode = POST_MODE_NONE
                self.pass_blind_active = True
                return self._gate_blind_drive_from_steer(self.pass_last_combined_steer, PASS_GATE)

            passed_gate = (
                self.pass_peak_width_ratio >= 0.22
                and self.pass_last_width_ratio >= 0.16
                and visual.lower_open_ratio >= 0.48
            )
            timed_out = (now - self.pass_gate_started_at) >= self.config.gate_commit_seconds
            if passed_gate or timed_out:
                self.last_trigger_reason = "passed_gate"
                self._enter_search_mode(SEARCH_FORWARD, now)
                return self._search_command(SEARCH_FORWARD, now)
            self.last_trigger_reason = "pass_gate_search"
            self.pass_blind_active = False
            return self._pass_gate_search_command()

        if self.mode == ADVANCE_OPEN:
            if visual.gate_visible:
                self.last_trigger_reason = "gate_found_after_advance"
                self.last_decision = "ALIGN_GATE"
                self._set_mode(ALIGN_GATE, now)
                return self._align_spin_command(visual.gate_target_offset, ALIGN_GATE)
            if too_close:
                self.last_trigger_reason = "advance_blocked"
                self.backoff_return_mode = SEARCH_FORWARD
                self.last_decision = "BACKOFF"
                self._set_mode(BACKOFF, now)
                return self._reverse_command(BACKOFF)
            if (now - self.state_started_at) >= self.config.advance_open_seconds:
                self.last_trigger_reason = "resume_search"
                self._enter_search_mode(SEARCH_FORWARD, now)
                return self._search_command(SEARCH_FORWARD, now)
            return self._open_area_forward_command(visual, ADVANCE_OPEN)

        return self._emit(0, 0, DONE, done=True)


def draw_overlay(frame, reading, visual, command, navigator, fps, run_state, config):
    view = frame.copy()
    frame_h, frame_w = view.shape[:2]
    cv2.line(view, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 255, 255), 1)
    if visual.gate_visible:
        gate_x = int(clamp(visual.gate_center_x, 0, frame_w - 1))
        target_x = int(clamp(visual.gate_target_x, 0, frame_w - 1))
        cv2.line(view, (gate_x, 0), (gate_x, frame_h), (0, 200, 255), 1)
        cv2.line(view, (target_x, 0), (target_x, frame_h), (255, 255, 0), 2)
        if visual.best_gate is not None:
            rx, ry, rw, rh = visual.best_gate.red.bbox
            gx, gy, gw, gh = visual.best_gate.green.bbox
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
            cv2.rectangle(view, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            red_top = tuple(int(v) for v in visual.best_gate.red.top_mid)
            red_bottom = tuple(int(v) for v in visual.best_gate.red.bottom_mid)
            green_top = tuple(int(v) for v in visual.best_gate.green.top_mid)
            green_bottom = tuple(int(v) for v in visual.best_gate.green.bottom_mid)
            gate_cross = (
                int(clamp(visual.best_gate.cross_intersection[0], 0, frame_w - 1)),
                int(clamp(visual.best_gate.cross_intersection[1], 0, frame_h - 1)),
            )
            cv2.line(view, red_top, green_bottom, (255, 180, 0), 2)
            cv2.line(view, red_bottom, green_top, (255, 180, 0), 2)
            cv2.circle(view, red_top, 4, (0, 0, 255), -1)
            cv2.circle(view, red_bottom, 4, (0, 0, 255), -1)
            cv2.circle(view, green_top, 4, (0, 255, 0), -1)
            cv2.circle(view, green_bottom, 4, (0, 255, 0), -1)
            cv2.circle(view, gate_cross, 5, (0, 255, 255), -1)
            cv2.circle(view, (target_x, gate_cross[1]), 5, (255, 255, 0), -1)

    gate_text = "Gate YES" if visual.gate_visible else "Gate NO"
    turn_label = visual.gate_turn_label
    turn_bias = visual.gate_turn_bias
    post_mode = visual.single_post_mode
    blind_text = "NO"
    if navigator is not None:
        turn_label = navigator.pass_turn_label if navigator.mode == PASS_GATE and navigator.pass_turn_label != TURN_LABEL_STRAIGHT else navigator.locked_turn_label
        if turn_label == TURN_LABEL_STRAIGHT:
            turn_label = visual.gate_turn_label
        turn_bias = navigator.current_turn_bias
        post_mode = navigator.current_post_mode
        blind_text = "YES" if navigator.pass_blind_active else "NO"
    lines = [
        f"{command.mode}  Dist {format_distance(reading.distance_cm)} cm  FPS {fps:4.1f}",
        f"{gate_text}  offset {visual.gate_offset:+.2f}  target {visual.gate_target_offset:+.2f}  conf {visual.gate_confidence:.2f}",
        f"Turn {turn_label[:1]}  bias {turn_bias:+.2f}  target_x {visual.gate_target_x:5.1f}",
        f"Post {post_mode}  blind {blind_text}",
    ]

    if navigator is not None:
        lines.append(
            f"Trig {navigator.last_trigger_reason or '-'}  Decision {navigator.last_decision or '-'}"
        )

    y = 24
    for text in lines:
        cv2.putText(
            view,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 20

    panel_w, panel_h = 220, 124
    x0 = 12
    y0 = frame_h - panel_h - 18
    wall_small = cv2.resize(visual.wall_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    red_small = cv2.resize(visual.red_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    green_small = cv2.resize(visual.green_mask, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    mask_panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    mask_panel[wall_small > 0] = (0, 0, 0)
    mask_panel[red_small > 0] = (0, 0, 255)
    mask_panel[green_small > 0] = (0, 255, 0)
    view[y0 : y0 + panel_h, x0 : x0 + panel_w] = mask_panel
    cv2.rectangle(view, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 2)
    cv2.putText(
        view,
        "mask: white floor / black wall / red / green",
        (x0, y0 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return view


def update_lcd(lcd_display, config, reading, visual, command):
    if config.mode == MODE_RECORD_VALIDATE:
        line_1 = "Recording Mask"
        line_2 = f"L{visual.left_red_ratio:.2f} R{visual.right_green_ratio:.2f}"
    else:
        label_map = {
            SEARCH_FORWARD: "Search",
            ALIGN_GATE: "Align Gate",
            PASS_GATE: "Pass Gate",
            SEARCH_NEXT: "Next Gate",
            ADVANCE_OPEN: "Open Area",
            BACKOFF: "Backoff",
            DONE: "Stopped",
            COUNTDOWN: "Countdown",
            "START": "Starting",
        }
        line_1 = label_map.get(command.mode, command.mode[:16])
        turn_char = visual.gate_turn_label[:1]
        if reading.valid and reading.distance_cm is not None:
            line_2 = f"D{reading.distance_cm:4.1f} {turn_char} {visual.single_post_mode[:1]}"
        else:
            line_2 = f"NoD {turn_char} {visual.single_post_mode[:1]}"
    lcd_display.update(line_1, line_2)


def install_stop_handlers(controller, ultrasonic_sensor):
    def safe_stop():
        try:
            if controller is not None:
                controller.stop()
        except Exception:
            pass
        try:
            if ultrasonic_sensor is not None:
                ultrasonic_sensor.close()
        except Exception:
            pass

    def stop_and_exit(signum=None, frame=None):
        safe_stop()
        raise SystemExit(0)

    atexit.register(safe_stop)
    signal.signal(signal.SIGINT, stop_and_exit)
    signal.signal(signal.SIGTERM, stop_and_exit)


def create_lcd_display():
    try:
        return base.LCD1602Display()
    except Exception as exc:
        print(f"[warn] LCD init failed, continuing without LCD: {exc}")
        return NullLCDDisplay()


def initialize_gpio_runtime():
    try:
        base.GPIO.cleanup()
    except Exception:
        pass
    base.GPIO.setwarnings(False)
    base.GPIO.setmode(base.GPIO.BCM)


def try_initialize_lcd_runtime():
    try:
        initialize_gpio_runtime()
        return create_lcd_display(), True
    except Exception as exc:
        print(f"[warn] LCD GPIO init failed, continuing without LCD: {exc}")
        return NullLCDDisplay(), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(MODE_RECORD_VALIDATE, MODE_RUN_MAZE), default=MODE_RUN_MAZE)
    parser.add_argument("--trig-pin", type=int, default=21, help="GPIO pin for ultrasonic trigger.")
    parser.add_argument("--echo-pin", type=int, default=20, help="GPIO pin for ultrasonic echo.")
    parser.add_argument("--trigger-cm", type=float, default=20.0, help="Distance that triggers a scan or wall response.")
    parser.add_argument("--clear-cm", type=float, default=SEARCH_CLEARANCE_CM, help="Distance considered clear during decision making.")
    parser.add_argument("--follow-speed", type=int, default=22, help="Forward speed.")
    parser.add_argument("--spin-speed", type=int, default=26, help="In-place scanning speed.")
    parser.add_argument("--right-rear-boost", type=int, default=RIGHT_REAR_BOOST, help="Extra speed added to the right rear wheel during straight forward motion.")
    parser.add_argument("--right-rear-spin-boost", type=int, default=0, help="Extra magnitude added to the right rear wheel during in-place rotation, useful if that wheel struggles to spin.")
    parser.add_argument("--turn-90-seconds", type=float, default=TURN_90_SECONDS, help="Legacy default duration for a 90 degree spin.")
    parser.add_argument("--scan-right-seconds", type=float, default=None, help="Rotation time from center to the right scan pose.")
    parser.add_argument("--scan-left-seconds", type=float, default=None, help="Rotation time from center to the left scan pose.")
    parser.add_argument("--turn-right-90-seconds", type=float, default=None, help="Rotation time for a right 90 degree turn.")
    parser.add_argument("--turn-left-90-seconds", type=float, default=None, help="Rotation time for a left 90 degree turn.")
    parser.add_argument("--uturn-seconds", type=float, default=None, help="Rotation time for a U-turn.")
    parser.add_argument("--scan-settle-seconds", type=float, default=SCAN_SETTLE_SECONDS, help="Pause after each scan rotation before sampling.")
    parser.add_argument("--forward-recover-seconds", type=float, default=FORWARD_RECOVER_SECONDS, help="Short forward burst after a turn.")
    parser.add_argument("--forward-steer-gain", type=float, default=22.0, help="How strongly visual left/right imbalance steers forward motion.")
    parser.add_argument("--max-forward-steer", type=float, default=8.0, help="Maximum visual steering correction while moving forward.")
    parser.add_argument("--turn-finish-center-max", type=float, default=0.24, help="Maximum center block ratio allowed before a turn can finish.")
    parser.add_argument("--turn-finish-open-min", type=float, default=0.46, help="Minimum visual open score required before a turn can finish.")
    parser.add_argument("--turn-finish-bias-max", type=float, default=0.20, help="Maximum opening bias magnitude allowed before a turn can finish.")
    parser.add_argument("--turn-finish-extra-seconds", type=float, default=0.35, help="Extra timeout allowed while visually settling after a turn.")
    parser.add_argument("--scan-trigger-hits", type=int, default=2, help="Consecutive near-wall hits required before entering scan.")
    parser.add_argument("--visual-trigger-hits", type=int, default=3, help="Consecutive visual blocked hits required before entering scan.")
    parser.add_argument("--scan-cooldown-seconds", type=float, default=0.55, help="Cooldown after a decision before scanning can trigger again.")
    parser.add_argument("--emergency-stop-cm", type=float, default=11.0, help="Immediate stop-and-scan distance, bypassing cooldown.")
    parser.add_argument("--hard-turn-center-threshold", type=float, default=0.50, help="Center block ratio that can directly force an in-place turn.")
    parser.add_argument("--hard-turn-wall-threshold", type=float, default=0.38, help="Overall wall ratio that can directly force an in-place turn.")
    parser.add_argument("--hard-turn-hits", type=int, default=2, help="Consecutive strong visual hits required before forcing an in-place turn.")
    parser.add_argument("--marker-ratio-threshold", type=float, default=MARKER_RATIO_THRESHOLD, help="Minimum red/green marker ratio required before treating a doorway marker as valid.")
    parser.add_argument("--marker-hits", type=int, default=2, help="Consecutive marker hits required before remembering a left/right branch hint.")
    parser.add_argument("--marker-scan-distance-cm", type=float, default=MARKER_SCAN_DISTANCE_CM, help="If a doorway marker is visible inside this distance, stop and scan early.")
    parser.add_argument("--marker-memory-seconds", type=float, default=MARKER_MEMORY_SECONDS, help="How long a red/green doorway hint should influence the next decision.")
    parser.add_argument("--marker-steer-gain", type=float, default=MARKER_STEER_GAIN, help="How strongly visible doorway markers bias the forward steering direction.")
    parser.add_argument("--search-spin-seconds", type=float, default=0.42, help="How long each left/right search spin segment lasts while looking for the next gate.")
    parser.add_argument("--search-small-seconds", type=float, default=SEARCH_SMALL_SECONDS, help="Search dwell for the small left/right sweep steps.")
    parser.add_argument("--search-large-seconds", type=float, default=SEARCH_LARGE_SECONDS, help="Search dwell for the large left/right sweep steps.")
    parser.add_argument("--search-forward-burst-seconds", type=float, default=SEARCH_FORWARD_BURST_SECONDS, help="Short forward burst between front-half search cycles.")
    parser.add_argument("--gate-pass-seconds", type=float, default=0.42, help="How long to keep driving forward after the gate markers leave view.")
    parser.add_argument("--gate-commit-seconds", type=float, default=GATE_COMMIT_SECONDS, help="How long to drive straight after committing to a gate, ignoring further gate updates.")
    parser.add_argument("--advance-open-seconds", type=float, default=0.55, help="How long to drive toward the whiter open floor region after a full search cycle misses the next gate.")
    parser.add_argument("--gate-confidence-threshold", type=float, default=GATE_CONFIDENCE_THRESHOLD, help="Minimum paired red/green gate confidence required before following a gate.")
    parser.add_argument("--align-gate-offset-threshold", type=float, default=ALIGN_GATE_OFFSET_THRESHOLD, help="Gate offset threshold below which the car switches from align to pass.")
    parser.add_argument("--align-spin-offset-threshold", type=float, default=ALIGN_SPIN_OFFSET_THRESHOLD, help="Gate offset threshold above which the car uses in-place alignment spins.")
    parser.add_argument("--align-spin-speed-min", type=int, default=18, help="Minimum in-place rotation speed while aligning the gate center.")
    parser.add_argument("--align-spin-speed-gain", type=float, default=18.0, help="How aggressively alignment spin speed grows with gate offset.")
    parser.add_argument("--gate-loss-grace-seconds", type=float, default=GATE_LOSS_GRACE_SECONDS, help="How long to trust the last seen gate before falling back to search.")
    parser.add_argument("--gate-blind-drive-seconds", type=float, default=GATE_BLIND_DRIVE_SECONDS, help="How long to keep driving toward the last gate center after one side of the gate disappears from view.")
    parser.add_argument("--gate-target-shift-ratio", type=float, default=GATE_TARGET_SHIFT_RATIO, help="How far the safe target line shifts toward the outside of the turn while approaching a gate.")
    parser.add_argument("--gate-safe-margin-ratio", type=float, default=GATE_SAFE_MARGIN_RATIO, help="How much of the gate width to reserve on each side as a no-drive safety margin.")
    parser.add_argument("--single-post-gap-ratio", type=float, default=SINGLE_POST_GAP_RATIO, help="How far the car center should stay from a single remaining gate post after the other post disappears.")
    parser.add_argument("--align-near-width-ratio", type=float, default=ALIGN_NEAR_WIDTH_RATIO, help="Gate width ratio at which alignment slows down to track the safe target line more carefully.")
    parser.add_argument("--pass-blind-steer-scale", type=float, default=PASS_BLIND_STEER_SCALE, help="How much of the last remembered turn command to keep while both gate posts are lost.")
    parser.add_argument("--pass-blind-speed-scale", type=float, default=PASS_BLIND_SPEED_SCALE, help="Forward speed scale used during the short remembered blind pass through a gate.")
    parser.add_argument("--reverse-seconds", type=float, default=0.30, help="How long to reverse when the ultrasonic sensor says the car is too close to a wall.")
    parser.add_argument("--reverse-speed", type=int, default=20, help="Reverse speed used during ultrasonic backoff.")
    parser.add_argument("--start-delay", type=float, default=0.0, help="Optional delay before motion starts.")
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument("--record-output", default="", help="Optional output h264 path from the Pi camera.")
    parser.add_argument("--debug-video-output", default="", help="Optional mp4 path for the processed overlay video.")
    args = parser.parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

    legacy_turn_90 = max(0.2, float(args.turn_90_seconds))
    scan_right_seconds = legacy_turn_90 * 0.70 if args.scan_right_seconds is None else float(args.scan_right_seconds)
    scan_left_seconds = legacy_turn_90 * 0.70 if args.scan_left_seconds is None else float(args.scan_left_seconds)
    turn_right_90_seconds = legacy_turn_90 if args.turn_right_90_seconds is None else float(args.turn_right_90_seconds)
    turn_left_90_seconds = legacy_turn_90 if args.turn_left_90_seconds is None else float(args.turn_left_90_seconds)
    uturn_seconds = max(turn_right_90_seconds, turn_left_90_seconds) * 2.0 if args.uturn_seconds is None else float(args.uturn_seconds)

    config = MazeConfig(
        mode=args.mode,
        trig_pin=args.trig_pin,
        echo_pin=args.echo_pin,
        trigger_cm=max(2.0, float(args.trigger_cm)),
        clear_cm=max(float(args.trigger_cm) + 1.0, float(args.clear_cm)),
        follow_speed=max(0, int(args.follow_speed)),
        spin_speed=max(0, int(args.spin_speed)),
        right_rear_boost=max(0, int(args.right_rear_boost)),
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
        start_delay=max(0.0, float(args.start_delay)),
        scan_right_seconds=max(0.15, scan_right_seconds),
        scan_left_seconds=max(0.15, scan_left_seconds),
        turn_right_90_seconds=max(0.2, turn_right_90_seconds),
        turn_left_90_seconds=max(0.2, turn_left_90_seconds),
        uturn_seconds=max(0.4, uturn_seconds),
        scan_settle_seconds=max(0.02, float(args.scan_settle_seconds)),
        forward_recover_seconds=max(0.0, float(args.forward_recover_seconds)),
        forward_steer_gain=max(0.0, float(args.forward_steer_gain)),
        max_forward_steer=max(0.0, float(args.max_forward_steer)),
        turn_finish_center_max=clamp(float(args.turn_finish_center_max), 0.02, 0.95),
        turn_finish_open_min=clamp(float(args.turn_finish_open_min), 0.02, 0.98),
        turn_finish_bias_max=clamp(float(args.turn_finish_bias_max), 0.02, 0.95),
        turn_finish_extra_seconds=max(0.0, float(args.turn_finish_extra_seconds)),
        scan_trigger_hits=max(1, int(args.scan_trigger_hits)),
        visual_trigger_hits=max(1, int(args.visual_trigger_hits)),
        scan_cooldown_seconds=max(0.0, float(args.scan_cooldown_seconds)),
        emergency_stop_cm=max(2.0, float(args.emergency_stop_cm)),
        hard_turn_center_threshold=clamp(float(args.hard_turn_center_threshold), 0.05, 0.98),
        hard_turn_wall_threshold=clamp(float(args.hard_turn_wall_threshold), 0.05, 0.98),
        hard_turn_hits=max(1, int(args.hard_turn_hits)),
        marker_ratio_threshold=clamp(float(args.marker_ratio_threshold), 0.001, 0.20),
        marker_hits=max(1, int(args.marker_hits)),
        marker_scan_distance_cm=max(4.0, float(args.marker_scan_distance_cm)),
        marker_memory_seconds=max(0.0, float(args.marker_memory_seconds)),
        marker_steer_gain=max(0.0, float(args.marker_steer_gain)),
        search_spin_seconds=max(0.05, float(args.search_spin_seconds)),
        search_small_seconds=max(0.04, float(args.search_small_seconds)),
        search_large_seconds=max(0.08, float(args.search_large_seconds)),
        search_forward_burst_seconds=max(0.0, float(args.search_forward_burst_seconds)),
        gate_pass_seconds=max(0.0, float(args.gate_pass_seconds)),
        gate_commit_seconds=max(0.2, float(args.gate_commit_seconds)),
        advance_open_seconds=max(0.0, float(args.advance_open_seconds)),
        gate_confidence_threshold=clamp(float(args.gate_confidence_threshold), 0.02, 0.99),
        align_gate_offset_threshold=clamp(float(args.align_gate_offset_threshold), 0.01, 0.50),
        align_spin_offset_threshold=clamp(float(args.align_spin_offset_threshold), 0.05, 0.95),
        align_spin_speed_min=max(8, int(args.align_spin_speed_min)),
        align_spin_speed_gain=max(1.0, float(args.align_spin_speed_gain)),
        gate_loss_grace_seconds=max(0.0, float(args.gate_loss_grace_seconds)),
        gate_blind_drive_seconds=max(0.0, float(args.gate_blind_drive_seconds)),
        gate_target_shift_ratio=clamp(float(args.gate_target_shift_ratio), 0.0, 0.35),
        gate_safe_margin_ratio=clamp(float(args.gate_safe_margin_ratio), 0.0, 0.30),
        single_post_gap_ratio=clamp(float(args.single_post_gap_ratio), 0.12, 0.65),
        align_near_width_ratio=clamp(float(args.align_near_width_ratio), 0.05, 0.50),
        pass_blind_steer_scale=clamp(float(args.pass_blind_steer_scale), 0.10, 1.00),
        pass_blind_speed_scale=clamp(float(args.pass_blind_speed_scale), 0.20, 0.90),
        reverse_seconds=max(0.0, float(args.reverse_seconds)),
        reverse_speed=max(0, int(args.reverse_speed)),
    )

    controller = NullMotorController()
    lcd_display = NullLCDDisplay()
    ultrasonic_sensor = None
    gpio_initialized = False
    if config.mode == MODE_RUN_MAZE:
        initialize_gpio_runtime()
        gpio_initialized = True
        controller = TrimmedCarMotorController(
            right_rear_boost=config.right_rear_boost,
            right_rear_spin_boost=config.right_rear_spin_boost,
        )
        lcd_display = create_lcd_display()
        ultrasonic_sensor = UltrasonicSensor(base.GPIO, config.trig_pin, config.echo_pin)
    else:
        lcd_display, gpio_initialized = try_initialize_lcd_runtime()
    install_stop_handlers(controller, ultrasonic_sensor)

    picam2 = base.Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": base.CAPTURE_SIZE, "format": "BGR888"}
    )
    picam2.configure(camera_config)

    h264_recording = False
    if args.record_output:
        record_path = Path(args.record_output)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        encoder = base.H264Encoder(bitrate=8_000_000)
        picam2.start_recording(encoder, base.FileOutput(str(record_path)))
        h264_recording = True
    else:
        picam2.start()

    debug_writer = None
    navigator = MazeNavigator(config) if config.mode == MODE_RUN_MAZE else None
    run_state = COUNTDOWN if config.start_delay > 0.0 else RUNNING
    countdown_deadline = time.perf_counter() + config.start_delay
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

        while True:
            now = time.perf_counter()
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            full_frame = picam2.capture_array()
            frame = cv2.resize(full_frame, base.PROCESS_SIZE, interpolation=cv2.INTER_AREA)

            if ultrasonic_sensor is not None:
                reading = ultrasonic_sensor.get_reading()
            else:
                reading = UltrasonicReading(distance_cm=None, valid=False, timestamp=now)
            visual = build_black_wall_observation(frame, config)
            command = DriveCommand(0, 0, COUNTDOWN if run_state == COUNTDOWN else DONE)

            if run_state == COUNTDOWN:
                controller.hold_stop()
                if now >= countdown_deadline:
                    run_state = RUNNING
                    if config.mode == MODE_RUN_MAZE:
                        navigator = MazeNavigator(config)
                    controller.stop()
                    command = DriveCommand(0, 0, "START")
            elif run_state == RUNNING:
                if config.mode == MODE_RECORD_VALIDATE:
                    controller.hold_stop()
                    command = DriveCommand(0, 0, "VALIDATE")
                else:
                    command = navigator.update(reading, visual, now)
                    controller.set_tank_drive(
                        command.left_speed,
                        command.right_speed,
                        straight_mode=command.mode in (FORWARD, FORWARD_RECOVER),
                    )
            else:
                controller.hold_stop()

            update_lcd(lcd_display, config, reading, visual, command)
            overlay = draw_overlay(frame, reading, visual, command, navigator, fps, run_state, config)

            if run_state == COUNTDOWN:
                remaining = max(0, int(np.ceil(countdown_deadline - now)))
                cv2.putText(
                    overlay,
                    f"Start in {remaining}",
                    (12, overlay.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if debug_writer is not None:
                debug_writer.write(overlay)

            if not args.headless:
                cv2.imshow("Maze Validation / Navigation", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("f"), ord("F")):
                    run_state = DONE
                    controller.stop()
            else:
                time.sleep(0.001)

    finally:
        try:
            if ultrasonic_sensor is not None:
                ultrasonic_sensor.close()
        except Exception:
            pass
        controller.close()
        lcd_display.close()
        if debug_writer is not None:
            debug_writer.release()
        if gpio_initialized:
            base.GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()


if __name__ == "__main__":
    main()
