import argparse
import atexit
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import RPi.GPIO as GPIO
import serial


CAPTURE_SIZE = (1280, 720)
PROCESS_SIZE = (640, 360)

BLACK_THRESHOLD = 80
MIN_LINE_AREA = 180
PRIMARY_ROI_CENTER_RATIO = 0.58
PRIMARY_ROI_HEIGHT_RATIO = 0.24
RECOVERY_ROI_CENTER_RATIO = 0.74
RECOVERY_ROI_HEIGHT_RATIO = 0.40
CURVE_SAMPLE_ROI_CENTER_RATIOS = (0.52, 0.62, 0.72, 0.82)
CURVE_SAMPLE_ROI_HEIGHT_RATIO = 0.12
ENTRY_MIN_VALID_POINTS = 2
ENTRY_MIN_SPREAD_PX = 14
ENTRY_MIN_SLOPE_PX_PER_ROW = 0.10
ENTRY_SHARP_SPREAD_PX = 46
ENTRY_SHARP_CURVATURE_PX = 24
ENTRY_MEMORY_SECONDS = 0.50
ENTRY_ASSIST_SCORE_THRESHOLD = 0.42

PURPLE_CONFIRMATION_HITS = 3
PURPLE_TRIGGER_COOLDOWN_SECONDS = 2.0
PURPLE_SCANLINE_Y_RATIO = 0.62
PURPLE_SCANLINE_X_START_RATIO = 0.45
PURPLE_TARGET_RGB = np.array([225, 115, 165], dtype=np.int16)
PURPLE_RGB_DISTANCE_THRESHOLD = 65
PURPLE_MIN_MATCH_PIXELS = 70
PURPLE_MIN_RUN_PIXELS = 40
CENTER_PURPLE_SCANLINE_Y_RATIO = 0.62
CENTER_PURPLE_BAND_HALF_HEIGHT = 8
CENTER_PURPLE_MIN_MATCH_PIXELS = 60
CENTER_PURPLE_MIN_RUN_PIXELS = 18
CENTER_PURPLE_HUE_LOW = 118
CENTER_PURPLE_HUE_HIGH = 179
CENTER_PURPLE_SAT_LOW = 20
CENTER_PURPLE_VAL_LOW = 100

SERIAL_PORT_CANDIDATES = (
    "/dev/ttyAMA0",
    "/dev/serial0",
    "/dev/ttyAMA10",
    "/dev/ttyS0",
)
SERIAL_BAUD_RATE = 57600
SERIAL_TIMEOUT_SECONDS = 0.1
LEFT_MOTOR_INDEXES = (2, 3)
RIGHT_MOTOR_INDEXES = (0, 1)
MOTOR_FORWARD_DIRS = ("f", "f", "f", "f")

BASE_SPEED = 30
MAX_SPEED = 70
STEERING_GAIN = 42.0
GENTLE_ENTRY_STEERING_GAIN = 50.0
SHARP_ENTRY_STEERING_GAIN = 60.0
DEADBAND_PIXELS = 12
CONTROL_HZ = 120.0
LINE_TIMEOUT_SECONDS = 0.15
RECOVERY_HOLD_SECONDS = 0.05
CORNER_ERROR_THRESHOLD = 34
CORNER_PIVOT_TIMEOUT_SECONDS = 0.72
CORNER_PIVOT_SPEED = 30
SHARP_CORNER_PIVOT_SPEED = 40
SHARP_CORNER_PIVOT_TIMEOUT_SECONDS = 0.95
RECOVERY_TURN_TIMEOUT_SECONDS = 1.15
RECOVERY_TURN_SPEED = 28
RECOVERY_FORWARD_SPEED = 4
SHARP_RECOVERY_TURN_SPEED = 34
SHARP_RECOVERY_FORWARD_SPEED = 0
TURN_DIRECTION_LOCK_THRESHOLD = 22
TURN_DIRECTION_MEMORY_SECONDS = 0.80
EDGE_DIRECTION_MARGIN_RATIO = 0.18

LCD_RS = 22
LCD_E = 17
LCD_D4 = 25
LCD_D5 = 18
LCD_D6 = 24
LCD_D7 = 23
LCD_WIDTH = 16
LCD_CHR = True
LCD_CMD = False
LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0
LCD_E_PULSE = 0.0005
LCD_E_DELAY = 0.0005
LCD_REFRESH_SECONDS = 0.2

SERVO_PIN = 27
SERVO_PWM_FREQUENCY = 50
SERVO_PURPLE_ANGLE = 90
SERVO_HOME_ANGLE = 180
SERVO_PURPLE_HOLD_SECONDS = 0.25
CENTER_PURPLE_STOP_ANGLE = 120
CENTER_PURPLE_STOP_WAIT_SECONDS = 5.0
CENTER_PURPLE_TRIGGER_COOLDOWN_SECONDS = 6.0
MUSIC_DISPLAY_SECONDS = 5.0
SORTING_DISPLAY_SECONDS = 5.0
SORTING_POST_COOLDOWN_SECONDS = 2.0
MUSIC_REMOTE_COMMAND = "ssh nenene@10.176.97.136 '~/play_mario.sh'"
MUSIC_REMOTE_TIMEOUT_SECONDS = 20.0
MUSIC_RELEASE_IGNORE_SECONDS = 5.0
ALARM_RESUME_BASE_SPEED = 30
ALARM_RESUME_SECONDS = 5.0
ALARM_RESUME_TURN_GAIN_SCALE = 1.25
TRAFFIC_RELEASE_IGNORE_SECONDS = 2.0
LOST_LINE_CONTINUOUS_TURN_SPEED = 32
LOST_LINE_CONTINUOUS_FORWARD_SPEED = -2
ALARM_RED_PIN = 20
ALARM_BLUE_PIN = 21
ALARM_LED_BLINK_SECONDS = 0.5
SIGN_PINK_HUE_LOW = 118
SIGN_PINK_HUE_HIGH = 179
SIGN_PINK_SAT_LOW = 20
SIGN_PINK_VAL_LOW = 90
SIGN_MIN_BORDER_AREA = 8000
SIGN_MIN_ASPECT_RATIO = 0.65
SIGN_MAX_ASPECT_RATIO = 1.8
SIGN_INNER_MARGIN_RATIO = 0.08
SIGN_WARP_SIZE = (240, 240)
SIGN_SYMBOL_SAT_THRESHOLD = 55
SIGN_SYMBOL_VAL_MAX = 235
SORT_BLUE_HUE_LOW = 78
SORT_BLUE_HUE_HIGH = 150
SORT_BLUE_SAT_LOW = 35
SORT_BLUE_VAL_LOW = 35
SORT_BLUE_RGB_CENTER = np.array([104, 114, 153], dtype=np.int16)
SORT_BLUE_RGB_DISTANCE_THRESHOLD = 140
# Red-cast blue sample (from your video/screenshot) is treated as BLUE class.
SORT_BLUE_RGB_CENTER_REDCAST = np.array([166, 108, 86], dtype=np.int16)
SORT_BLUE_RGB_DISTANCE_THRESHOLD_REDCAST = 75
SORT_GREEN_HUE_LOW = 35
SORT_GREEN_HUE_HIGH = 95
SORT_GREEN_SAT_LOW = 50
SORT_GREEN_VAL_LOW = 40
SORT_GREEN_RGB_CENTER = np.array([84, 153, 70], dtype=np.int16)
SORT_GREEN_RGB_DISTANCE_THRESHOLD = 86
SORT_MIN_CONTOUR_AREA = 220
SORT_MIN_BOX_SIZE = 18
SORT_INNER_MARGIN_RATIO = 0.04
SORT_POLY_APPROX_RATIO = 0.04
SORT_SQUARE_MIN_ASPECT = 0.72
SORT_SQUARE_MAX_ASPECT = 1.30
SORT_CIRCLE_MIN_VERTICES = 5
SORT_MAX_VERTICES_FOR_SHAPE = 18
SORT_MIN_SOLIDITY = 0.80
SORT_MIN_FILL_RATIO = 0.44
SORT_CIRCLE_MIN_CIRCULARITY = 0.60
SORT_NMS_IOU_THRESHOLD = 0.45
SIGN_TEMPLATE_MATCH_THRESHOLD = 0.78
TRAFFIC_TEMPLATE_MATCH_THRESHOLD = 0.54
TRAFFIC_TEMPLATE_MARGIN = 0.03
ALARM_TEMPLATE_FILENAME = "alarm_template.png"
TRAFFIC_TEMPLATE_FILENAME = "traffic_light_template.png"
MUSIC_TEMPLATE_FILENAME = "music_template.png"
TRAFFIC_LIGHT_HOUSING_X_RATIO = 1.184
TRAFFIC_LIGHT_HOUSING_Y_RATIO = 0.302
TRAFFIC_LIGHT_HOUSING_W_RATIO = 0.165
TRAFFIC_LIGHT_HOUSING_H_RATIO = 0.616
TRAFFIC_LIGHT_LAMP_X_RATIO = 0.40
TRAFFIC_LIGHT_LAMP_W_RATIO = 0.43
TRAFFIC_LIGHT_LAMP_H_RATIO = 0.19
TRAFFIC_LIGHT_RED_Y_RATIO = 0.08
TRAFFIC_LIGHT_GREEN_Y_RATIO = 0.28
TRAFFIC_LIGHT_RED_HUE_LOW_1 = 0
TRAFFIC_LIGHT_RED_HUE_HIGH_1 = 15
TRAFFIC_LIGHT_RED_HUE_LOW_2 = 160
TRAFFIC_LIGHT_RED_HUE_HIGH_2 = 179
TRAFFIC_LIGHT_GREEN_HUE_LOW = 40
TRAFFIC_LIGHT_GREEN_HUE_HIGH = 95
TRAFFIC_LIGHT_COLOR_SAT_LOW = 90
TRAFFIC_LIGHT_BRIGHT_MIN = 170
TRAFFIC_LIGHT_BRIGHT_OFFSET = 35
TRAFFIC_LIGHT_BRIGHT_CONTOUR_MIN_AREA = 20
TRAFFIC_LIGHT_COLOR_DOMINANCE = 20
TRAFFIC_LIGHT_GREEN_CONFIRM_HITS = 3
TRAFFIC_LIGHT_GREEN_MIN_WAIT_SECONDS = 0.8
SIGN_TEMPLATE_CACHE = {}


def clamp(value, low, high):
    return max(low, min(high, value))


def rect_iou(rect_a, rect_b):
    if rect_a is None or rect_b is None:
        return 0.0
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ax1 = ax + aw
    ay1 = ay + ah
    bx1 = bx + bw
    by1 = by + bh
    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
    union_area = float(aw * ah + bw * bh) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def reverse_direction(direction):
    return "r" if direction == "f" else "f"


class CarMotorController:
    def __init__(self):
        self.serial = self._open_serial_port()
        self.port_name = self.serial.port
        self.last_command = None
        self.stop()

    def _open_serial_port(self):
        attempted_ports = []
        for port in SERIAL_PORT_CANDIDATES:
            if not Path(port).exists():
                continue
            attempted_ports.append(port)
            try:
                conn = serial.Serial(
                    port=port,
                    baudrate=SERIAL_BAUD_RATE,
                    timeout=SERIAL_TIMEOUT_SECONDS,
                    write_timeout=SERIAL_TIMEOUT_SECONDS,
                )
                conn.reset_input_buffer()
                conn.reset_output_buffer()
                return conn
            except serial.SerialException:
                continue
        if attempted_ports:
            raise RuntimeError(f"unable to open serial port from {', '.join(attempted_ports)}")
        raise RuntimeError(f"no serial port found in candidates: {', '.join(SERIAL_PORT_CANDIDATES)}")

    def _write_payload(self, payload):
        if payload == self.last_command:
            return
        self.serial.write(payload)
        self.serial.flush()
        self.last_command = bytes(payload)
        time.sleep(0.002)

    def _force_write_payload(self, payload, repeats=3, pause_seconds=0.01):
        for _ in range(repeats):
            self.serial.write(payload)
            self.serial.flush()
            time.sleep(pause_seconds)
        self.last_command = bytes(payload)

    def stop(self):
        self._force_write_payload(b"#ha")

    def hold_stop(self):
        self._write_payload(b"#ha")

    def set_tank_drive(self, left_speed, right_speed):
        per_motor_speeds = [0, 0, 0, 0]
        for index in LEFT_MOTOR_INDEXES:
            per_motor_speeds[index] = int(left_speed)
        for index in RIGHT_MOTOR_INDEXES:
            per_motor_speeds[index] = int(right_speed)

        payload = bytearray(b"#ba")
        for index, speed in enumerate(per_motor_speeds):
            direction = MOTOR_FORWARD_DIRS[index]
            if speed < 0:
                direction = reverse_direction(direction)
            payload.extend(direction.encode("ascii"))

        for speed in per_motor_speeds:
            magnitude = int(clamp(abs(speed), 0, 65535))
            payload.append(magnitude & 0xFF)
            payload.append((magnitude >> 8) & 0xFF)
        self._write_payload(payload)

    def close(self):
        try:
            self.stop()
        finally:
            self.serial.close()


class LCD1602Display:
    def __init__(self):
        self.last_lines = ("", "")
        self.last_update = 0.0

        for pin in (LCD_E, LCD_RS, LCD_D4, LCD_D5, LCD_D6, LCD_D7):
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, False)

        self._send_byte(0x33, LCD_CMD)
        self._send_byte(0x32, LCD_CMD)
        self._send_byte(0x28, LCD_CMD)
        self._send_byte(0x0C, LCD_CMD)
        self._send_byte(0x06, LCD_CMD)
        self._send_byte(0x01, LCD_CMD)
        time.sleep(LCD_E_DELAY)

    def _toggle_enable(self):
        time.sleep(LCD_E_DELAY)
        GPIO.output(LCD_E, True)
        time.sleep(LCD_E_PULSE)
        GPIO.output(LCD_E, False)
        time.sleep(LCD_E_DELAY)

    def _send_byte(self, bits, mode):
        GPIO.output(LCD_RS, mode)

        GPIO.output(LCD_D4, bool(bits & 0x10))
        GPIO.output(LCD_D5, bool(bits & 0x20))
        GPIO.output(LCD_D6, bool(bits & 0x40))
        GPIO.output(LCD_D7, bool(bits & 0x80))
        self._toggle_enable()

        GPIO.output(LCD_D4, bool(bits & 0x01))
        GPIO.output(LCD_D5, bool(bits & 0x02))
        GPIO.output(LCD_D6, bool(bits & 0x04))
        GPIO.output(LCD_D7, bool(bits & 0x08))
        self._toggle_enable()

    def _write_line(self, message, line_address):
        self._send_byte(line_address, LCD_CMD)
        text = message.ljust(LCD_WIDTH)[:LCD_WIDTH]
        for ch in text:
            self._send_byte(ord(ch), LCD_CHR)

    def update(self, line_1, line_2, force=False):
        now = time.perf_counter()
        if not force and (now - self.last_update) < LCD_REFRESH_SECONDS:
            return
        lines = (line_1[:LCD_WIDTH], line_2[:LCD_WIDTH])
        if force or lines != self.last_lines:
            self._write_line(lines[0], LCD_LINE_1)
            self._write_line(lines[1], LCD_LINE_2)
            self.last_lines = lines
        self.last_update = now

    def close(self):
        self.update("Line Following", "Stopped", force=True)
        time.sleep(0.2)


class ServoController:
    def __init__(self):
        GPIO.setup(SERVO_PIN, GPIO.OUT)
        self.pwm = GPIO.PWM(SERVO_PIN, SERVO_PWM_FREQUENCY)
        self.pwm.start(0)
        self._lock = threading.Lock()
        self._busy = False
        self.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.3)

    @staticmethod
    def _angle_to_duty_cycle(angle):
        angle = max(0, min(180, angle))
        return 2.5 + (angle / 180.0) * 10.0

    def set_angle(self, angle, settle_seconds=0.5):
        self.pwm.ChangeDutyCycle(self._angle_to_duty_cycle(angle))
        time.sleep(settle_seconds)
        self.pwm.ChangeDutyCycle(0)

    def pulse_once(self, target_angle=SERVO_PURPLE_ANGLE, hold_seconds=SERVO_PURPLE_HOLD_SECONDS):
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _worker():
            try:
                self.set_angle(target_angle, settle_seconds=0.12)
                time.sleep(max(0.0, hold_seconds))
                self.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.12)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_worker, daemon=True).start()
        return True

    def close(self):
        self.pwm.ChangeDutyCycle(0)
        time.sleep(0.1)
        self.pwm.stop()


class AlarmLedController:
    def __init__(self):
        self.pins = (ALARM_RED_PIN, ALARM_BLUE_PIN)
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, False)
        self._state = 0
        self._last_toggle_time = -999.0

    def all_off(self):
        GPIO.output(ALARM_RED_PIN, False)
        GPIO.output(ALARM_BLUE_PIN, False)

    def update_alarm_pattern(self, now):
        if (now - self._last_toggle_time) >= ALARM_LED_BLINK_SECONDS:
            self._state = 1 - self._state
            self._last_toggle_time = now
        GPIO.output(ALARM_RED_PIN, self._state == 0)
        GPIO.output(ALARM_BLUE_PIN, self._state == 1)

    def close(self):
        self.all_off()


class ControlState:
    def __init__(self, motor_armed):
        self.lock = threading.Lock()
        self.running = True
        self.motor_armed = motor_armed
        self.pause_active = False
        self.center_x = None
        self.frame_width = PROCESS_SIZE[0]
        self.last_line_time = 0.0
        self.error_pixels = 0
        self.left_speed = 0
        self.right_speed = 0
        self.last_nonzero_error = 0
        self.recovery_turn_direction = 0
        self.recovery_turn_time = 0.0
        self.entry_assist_direction = 0
        self.entry_assist_until = -999.0
        self.entry_kind = "lost"
        self.entry_score = 0.0
        self.base_speed_override = None
        self.base_speed_override_until = -999.0

    def update_detection(self, center_x, frame_width, now):
        with self.lock:
            self.center_x = center_x
            self.frame_width = frame_width
            self.last_line_time = now
            base_speed_override = None
            if self.base_speed_override is not None and now <= self.base_speed_override_until:
                base_speed_override = self.base_speed_override
            elif self.base_speed_override is not None and now > self.base_speed_override_until:
                self.base_speed_override = None
                self.base_speed_override_until = -999.0
            self.left_speed, self.right_speed, self.error_pixels = compute_drive_command(
                center_x,
                frame_width,
                self.entry_kind,
                self.entry_score,
                base_speed_override,
            )
            if self.error_pixels != 0:
                self.last_nonzero_error = self.error_pixels

            direction_candidate = 0
            edge_margin = frame_width * EDGE_DIRECTION_MARGIN_RATIO
            if center_x <= edge_margin:
                direction_candidate = -1
            elif center_x >= (frame_width - edge_margin):
                direction_candidate = 1
            elif self.error_pixels <= -TURN_DIRECTION_LOCK_THRESHOLD:
                direction_candidate = -1
            elif self.error_pixels >= TURN_DIRECTION_LOCK_THRESHOLD:
                direction_candidate = 1
            if direction_candidate != 0:
                self.recovery_turn_direction = direction_candidate
                self.recovery_turn_time = now

    def mark_line_lost(self):
        with self.lock:
            self.center_x = None
            self.error_pixels = 0

    def set_entry_assist(self, direction, now):
        with self.lock:
            self.entry_assist_direction = direction
            self.entry_assist_until = now + ENTRY_MEMORY_SECONDS

    def set_entry_profile(self, kind, score):
        with self.lock:
            self.entry_kind = kind
            self.entry_score = score

    def set_base_speed_override(self, base_speed, until_time):
        with self.lock:
            self.base_speed_override = base_speed
            self.base_speed_override_until = until_time

    def set_pause(self, active):
        with self.lock:
            self.pause_active = active

    def set_motor_armed(self, armed):
        with self.lock:
            self.motor_armed = armed

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "motor_armed": self.motor_armed,
                "pause_active": self.pause_active,
                "center_x": self.center_x,
                "frame_width": self.frame_width,
                "last_line_time": self.last_line_time,
                "error_pixels": self.error_pixels,
                "left_speed": self.left_speed,
                "right_speed": self.right_speed,
                "last_nonzero_error": self.last_nonzero_error,
                "recovery_turn_direction": self.recovery_turn_direction,
                "recovery_turn_time": self.recovery_turn_time,
                "entry_assist_direction": self.entry_assist_direction,
                "entry_assist_until": self.entry_assist_until,
                "entry_kind": self.entry_kind,
                "entry_score": self.entry_score,
                "base_speed_override": self.base_speed_override,
                "base_speed_override_until": self.base_speed_override_until,
            }

    def stop(self):
        with self.lock:
            self.running = False


def compute_drive_command(
    center_x,
    frame_width,
    entry_kind="lost",
    entry_score=0.0,
    base_speed_override=None,
):
    frame_center_x = frame_width // 2
    error_pixels = center_x - frame_center_x
    if abs(error_pixels) <= DEADBAND_PIXELS:
        error_pixels = 0
    normalized_error = error_pixels / (frame_width / 2.0)
    effective_base_speed = BASE_SPEED if base_speed_override is None else base_speed_override
    base_speed = effective_base_speed
    steering_gain = STEERING_GAIN
    turn_gain_scale = effective_base_speed / float(BASE_SPEED)
    if entry_kind == "sharp" and entry_score >= ENTRY_ASSIST_SCORE_THRESHOLD:
        steering_gain = SHARP_ENTRY_STEERING_GAIN
    elif entry_kind == "gentle" and entry_score >= ENTRY_ASSIST_SCORE_THRESHOLD:
        steering_gain = GENTLE_ENTRY_STEERING_GAIN
    if base_speed_override is not None:
        turn_gain_scale = ALARM_RESUME_TURN_GAIN_SCALE
    turn_delta = steering_gain * turn_gain_scale * normalized_error
    left_speed = clamp(base_speed + turn_delta, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(base_speed - turn_delta, -MAX_SPEED, MAX_SPEED)
    return int(left_speed), int(right_speed), int(error_pixels)


def find_line_center_in_roi(frame_bgr, roi_center_ratio, roi_height_ratio):
    frame_height = frame_bgr.shape[0]
    roi_height = max(1, int(frame_height * roi_height_ratio))
    roi_center = int(frame_height * roi_center_ratio)
    roi_top = max(0, roi_center - roi_height // 2)
    roi_bottom = min(frame_height, roi_top + roi_height)

    roi_bgr = frame_bgr[roi_top:roi_bottom, :]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_LINE_AREA]
    if not contours:
        return None, mask, roi_top, roi_bottom, None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, mask, roi_top, roi_bottom, contour

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"]) + roi_top
    return (center_x, center_y), mask, roi_top, roi_bottom, contour


def find_line_center(frame_bgr):
    result = find_line_center_in_roi(frame_bgr, PRIMARY_ROI_CENTER_RATIO, PRIMARY_ROI_HEIGHT_RATIO)
    if result[0] is not None:
        return result, "primary"
    return find_line_center_in_roi(frame_bgr, RECOVERY_ROI_CENTER_RATIO, RECOVERY_ROI_HEIGHT_RATIO), "recovery"


def sample_curve_points(frame_bgr):
    points = []
    for roi_center_ratio in CURVE_SAMPLE_ROI_CENTER_RATIOS:
        center, _, _, _, _ = find_line_center_in_roi(
            frame_bgr, roi_center_ratio, CURVE_SAMPLE_ROI_HEIGHT_RATIO
        )
        if center is not None:
            points.append(center)
    return points


def infer_curve_entry(points):
    if len(points) < ENTRY_MIN_VALID_POINTS:
        return {
            "direction": 0,
            "kind": "lost",
            "score": 0.0,
            "spread_px": None,
            "slope": None,
            "curvature_px": None,
        }

    ordered = sorted(points, key=lambda point: point[1])
    xs = [point[0] for point in ordered]
    ys = [point[1] for point in ordered]
    x_top = float(xs[0])
    x_bottom = float(xs[-1])
    y_top = float(ys[0])
    y_bottom = float(ys[-1])
    dy = max(1.0, y_bottom - y_top)
    slope = (x_bottom - x_top) / dy
    spread = float(max(xs) - min(xs))

    curvature = 0.0
    if len(xs) >= 3:
        mid_idx = len(xs) // 2
        curvature = x_bottom - 2.0 * float(xs[mid_idx]) + x_top

    direction = 0
    if abs(slope) >= ENTRY_MIN_SLOPE_PX_PER_ROW or spread >= ENTRY_MIN_SPREAD_PX:
        direction = 1 if slope >= 0 else -1

    if abs(slope) < ENTRY_MIN_SLOPE_PX_PER_ROW and spread < ENTRY_MIN_SPREAD_PX:
        curve_kind = "straight"
    elif spread >= ENTRY_SHARP_SPREAD_PX or abs(curvature) >= ENTRY_SHARP_CURVATURE_PX:
        curve_kind = "sharp"
    else:
        curve_kind = "gentle"

    score = min(
        1.0,
        0.60 * (spread / ENTRY_SHARP_SPREAD_PX)
        + 0.25 * (abs(slope) / ENTRY_MIN_SLOPE_PX_PER_ROW)
        + 0.15 * (abs(curvature) / ENTRY_SHARP_CURVATURE_PX),
    )
    return {
        "direction": direction,
        "kind": curve_kind,
        "score": round(score, 4),
        "spread_px": round(spread, 3),
        "slope": round(slope, 4),
        "curvature_px": round(curvature, 3),
    }


def detect_right_purple_square(frame_bgr):
    frame_height, frame_width = frame_bgr.shape[:2]
    y = int(frame_height * PURPLE_SCANLINE_Y_RATIO)
    y = max(0, min(frame_height - 1, y))
    x_start = int(frame_width * PURPLE_SCANLINE_X_START_RATIO)
    x_start = max(0, min(frame_width - 1, x_start))

    scanline_rgb = frame_bgr[y, x_start:, ::-1].astype(np.int16)
    if scanline_rgb.size == 0:
        return None

    diff = scanline_rgb - PURPLE_TARGET_RGB
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    matched = np.where(dist <= PURPLE_RGB_DISTANCE_THRESHOLD)[0]
    match_count = int(matched.size)
    if match_count < PURPLE_MIN_MATCH_PIXELS:
        return None

    max_run = 1
    run = 1
    for idx in range(1, match_count):
        if int(matched[idx]) == int(matched[idx - 1]) + 1:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    if max_run < PURPLE_MIN_RUN_PIXELS:
        return None

    x0 = x_start + int(matched[0])
    x1 = x_start + int(matched[-1])
    w = max(1, x1 - x0 + 1)
    return {"rect": (x0, max(0, y - 3), w, 7), "area": float(max_run)}


def detect_centerline_purple(frame_bgr):
    frame_height, frame_width = frame_bgr.shape[:2]
    y = int(frame_height * CENTER_PURPLE_SCANLINE_Y_RATIO)
    y = max(0, min(frame_height - 1, y))

    band_top = max(0, y - CENTER_PURPLE_BAND_HALF_HEIGHT)
    band_bottom = min(frame_height, y + CENTER_PURPLE_BAND_HALF_HEIGHT + 1)
    band_bgr = frame_bgr[band_top:band_bottom, :]
    if band_bgr.size == 0:
        return None

    # Use HSV for the center trigger because the real marker looks more pink/magenta
    # than the earlier hard-coded purple RGB sample.
    band_hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        band_hsv,
        np.array([CENTER_PURPLE_HUE_LOW, CENTER_PURPLE_SAT_LOW, CENTER_PURPLE_VAL_LOW], dtype=np.uint8),
        np.array([CENTER_PURPLE_HUE_HIGH, 255, 255], dtype=np.uint8),
    )
    matched_columns = np.where(mask.max(axis=0) > 0)[0]
    match_count = int(matched_columns.size)
    if match_count < CENTER_PURPLE_MIN_MATCH_PIXELS:
        return {
            "detected": False,
            "rect": None,
            "area": 0.0,
            "match_count": match_count,
            "max_run": 0,
        }

    max_run = 1
    run = 1
    for idx in range(1, match_count):
        if int(matched_columns[idx]) == int(matched_columns[idx - 1]) + 1:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    if max_run < CENTER_PURPLE_MIN_RUN_PIXELS:
        return {
            "detected": False,
            "rect": None,
            "area": 0.0,
            "match_count": match_count,
            "max_run": int(max_run),
        }

    x0 = int(matched_columns[0])
    x1 = int(matched_columns[-1])
    w = max(1, x1 - x0 + 1)
    return {
        "detected": True,
        "rect": (x0, max(0, y - 3), w, 7),
        "area": float(max_run),
        "match_count": match_count,
        "max_run": int(max_run),
    }


def order_quad_points(points):
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmin(diff)],
            pts[np.argmax(s)],
            pts[np.argmax(diff)],
        ],
        dtype=np.float32,
    )


def detect_sign_roi(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([SIGN_PINK_HUE_LOW, SIGN_PINK_SAT_LOW, SIGN_PINK_VAL_LOW], dtype=np.uint8),
        np.array([SIGN_PINK_HUE_HIGH, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_height, frame_width = frame_bgr.shape[:2]

    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < SIGN_MIN_BORDER_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < SIGN_MIN_ASPECT_RATIO or aspect_ratio > SIGN_MAX_ASPECT_RATIO:
            continue
        if w < frame_width * 0.18 or h < frame_height * 0.18:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        score = area
        if best is None or score > best["score"]:
            best = {"rect": (x, y, w, h), "score": score, "box": box}

    if best is None:
        return None

    x, y, w, h = best["rect"]
    margin_x = max(4, int(w * SIGN_INNER_MARGIN_RATIO))
    margin_y = max(4, int(h * SIGN_INNER_MARGIN_RATIO))
    inner_x0 = clamp(x + margin_x, 0, frame_width - 1)
    inner_y0 = clamp(y + margin_y, 0, frame_height - 1)
    inner_x1 = clamp(x + w - margin_x, inner_x0 + 1, frame_width)
    inner_y1 = clamp(y + h - margin_y, inner_y0 + 1, frame_height)
    inner_rect = (inner_x0, inner_y0, inner_x1 - inner_x0, inner_y1 - inner_y0)
    ordered_box = order_quad_points(best["box"])
    dst = np.array(
        [
            [0, 0],
            [SIGN_WARP_SIZE[0] - 1, 0],
            [SIGN_WARP_SIZE[0] - 1, SIGN_WARP_SIZE[1] - 1],
            [0, SIGN_WARP_SIZE[1] - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered_box, dst)
    warped = cv2.warpPerspective(frame_bgr, matrix, SIGN_WARP_SIZE)
    return {
        "outer_rect": (x, y, w, h),
        "inner_rect": inner_rect,
        "quad": ordered_box.astype(np.int32),
        "warped": warped,
    }


def preprocess_sign_symbol(warped_sign_bgr):
    inner_margin_x = int(warped_sign_bgr.shape[1] * SIGN_INNER_MARGIN_RATIO)
    inner_margin_y = int(warped_sign_bgr.shape[0] * SIGN_INNER_MARGIN_RATIO)
    roi = warped_sign_bgr[
        inner_margin_y : warped_sign_bgr.shape[0] - inner_margin_y,
        inner_margin_x : warped_sign_bgr.shape[1] - inner_margin_x,
    ]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sat_mask = hsv[:, :, 1] >= SIGN_SYMBOL_SAT_THRESHOLD
    dark_mask = gray <= SIGN_SYMBOL_VAL_MAX
    binary = np.where(sat_mask & dark_mask, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def clean_binary_mask(mask):
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def extract_inner_board_mask(roi_bgr):
    """Keep only the inner white paper region; ignore anything outside the frame."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 95], dtype=np.uint8),
        np.array([179, 80, 255], dtype=np.uint8),
    )
    white_mask = clean_binary_mask(white_mask)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(white_mask.shape, 255, dtype=np.uint8)
    contour = max(contours, key=cv2.contourArea)
    board_mask = np.zeros_like(white_mask)
    cv2.drawContours(board_mask, [contour], -1, 255, thickness=-1)
    # Keep the full inner sheet so edge-near shapes are not truncated.
    return board_mask


def refine_mask_by_rgb_distance(roi_bgr, hsv_mask, target_rgb, distance_threshold):
    # Use float math to avoid overflow in squared distance.
    roi_rgb = roi_bgr[:, :, ::-1].astype(np.float32)
    target = target_rgb.astype(np.float32)
    diff = roi_rgb - target
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    rgb_mask = (dist <= distance_threshold).astype(np.uint8) * 255
    return cv2.bitwise_and(hsv_mask, rgb_mask)


def refine_mask_by_multi_rgb_distance(roi_bgr, base_mask, targets_with_thresholds):
    merged = np.zeros(base_mask.shape, dtype=np.uint8)
    for target_rgb, distance_threshold in targets_with_thresholds:
        candidate = refine_mask_by_rgb_distance(
            roi_bgr, base_mask, target_rgb, distance_threshold
        )
        merged = cv2.bitwise_or(merged, candidate)
    return merged


def resolve_color_overlap(roi_bgr, blue_mask, green_mask):
    """Ensure one pixel belongs to only one color class (blue or green)."""
    overlap = cv2.bitwise_and(blue_mask, green_mask)
    if cv2.countNonZero(overlap) == 0:
        return blue_mask, green_mask

    rgb = roi_bgr[:, :, ::-1].astype(np.float32)
    blue_center_1 = SORT_BLUE_RGB_CENTER.astype(np.float32)
    blue_center_2 = SORT_BLUE_RGB_CENTER_REDCAST.astype(np.float32)
    green_center = SORT_GREEN_RGB_CENTER.astype(np.float32)

    blue_dist_1 = np.sqrt(np.sum((rgb - blue_center_1) ** 2, axis=2))
    blue_dist_2 = np.sqrt(np.sum((rgb - blue_center_2) ** 2, axis=2))
    blue_dist = np.minimum(blue_dist_1, blue_dist_2)
    green_dist = np.sqrt(np.sum((rgb - green_center) ** 2, axis=2))

    overlap_bool = overlap > 0
    assign_blue = np.zeros_like(overlap, dtype=np.uint8)
    assign_green = np.zeros_like(overlap, dtype=np.uint8)
    assign_blue[overlap_bool & (blue_dist <= green_dist)] = 255
    assign_green[overlap_bool & (green_dist < blue_dist)] = 255

    keep_blue = cv2.bitwise_and(blue_mask, cv2.bitwise_not(overlap))
    keep_green = cv2.bitwise_and(green_mask, cv2.bitwise_not(overlap))
    blue_mask = cv2.bitwise_or(keep_blue, assign_blue)
    green_mask = cv2.bitwise_or(keep_green, assign_green)
    return blue_mask, green_mask


def contour_is_square_by_corners(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        return False
    approx = cv2.approxPolyDP(contour, SORT_POLY_APPROX_RATIO * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return False
    x, y, w, h = cv2.boundingRect(approx)
    if h <= 0:
        return False
    aspect_ratio = w / float(h)
    if not (SORT_SQUARE_MIN_ASPECT <= aspect_ratio <= SORT_SQUARE_MAX_ASPECT):
        return False
    area = cv2.contourArea(contour)
    if area <= 0.0:
        return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0.0:
        return False
    solidity = area / hull_area
    fill_ratio = area / float(max(1, w * h))
    return solidity >= SORT_MIN_SOLIDITY and fill_ratio >= SORT_MIN_FILL_RATIO


def contour_is_circle_by_corners(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0.0:
        return False
    approx = cv2.approxPolyDP(contour, SORT_POLY_APPROX_RATIO * perimeter, True)
    vertices = len(approx)
    if vertices < SORT_CIRCLE_MIN_VERTICES or vertices > SORT_MAX_VERTICES_FOR_SHAPE:
        return False
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0.0:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    if h <= 0:
        return False
    ratio = w / float(h)
    if not (0.70 <= ratio <= 1.35):
        return False
    circularity = 4.0 * np.pi * area / max(1e-6, perimeter * perimeter)
    if circularity < SORT_CIRCLE_MIN_CIRCULARITY:
        return False
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0.0:
        return False
    solidity = area / hull_area
    return solidity >= SORT_MIN_SOLIDITY


def count_sorting_objects(warped_sign_bgr):
    if warped_sign_bgr is None or warped_sign_bgr.size == 0:
        return None
    h, w = warped_sign_bgr.shape[:2]
    margin_x = max(2, int(w * SORT_INNER_MARGIN_RATIO))
    margin_y = max(2, int(h * SORT_INNER_MARGIN_RATIO))
    roi = warped_sign_bgr[margin_y : h - margin_y, margin_x : w - margin_x]
    if roi.size == 0:
        return None

    board_mask = extract_inner_board_mask(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Blue class is defined by RGB references (normal blue + red-cast blue),
    # because camera white-balance can shift blue objects toward brown/red.
    blue_base_mask = cv2.inRange(
        hsv,
        np.array([0, SORT_BLUE_SAT_LOW, SORT_BLUE_VAL_LOW], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    blue_mask = refine_mask_by_multi_rgb_distance(
        roi,
        blue_base_mask,
        (
            (SORT_BLUE_RGB_CENTER, SORT_BLUE_RGB_DISTANCE_THRESHOLD),
            (SORT_BLUE_RGB_CENTER_REDCAST, SORT_BLUE_RGB_DISTANCE_THRESHOLD_REDCAST),
        ),
    )
    green_mask = cv2.inRange(
        hsv,
        np.array([SORT_GREEN_HUE_LOW, SORT_GREEN_SAT_LOW, SORT_GREEN_VAL_LOW], dtype=np.uint8),
        np.array([SORT_GREEN_HUE_HIGH, 255, 255], dtype=np.uint8),
    )
    green_mask = refine_mask_by_rgb_distance(
        roi, green_mask, SORT_GREEN_RGB_CENTER, SORT_GREEN_RGB_DISTANCE_THRESHOLD
    )
    blue_mask, green_mask = resolve_color_overlap(roi, blue_mask, green_mask)
    blue_mask = cv2.bitwise_and(blue_mask, board_mask)
    green_mask = cv2.bitwise_and(green_mask, board_mask)
    blue_mask = clean_binary_mask(blue_mask)
    green_mask = clean_binary_mask(green_mask)

    counts = {"BC": 0, "BS": 0, "GC": 0, "GS": 0}

    def process_mask(mask, prefix):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < SORT_MIN_CONTOUR_AREA:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw < SORT_MIN_BOX_SIZE or ch < SORT_MIN_BOX_SIZE:
                continue
            # Use the lecture logic: classify by polygon corner count.
            if contour_is_square_by_corners(contour):
                candidates.append(("S", (x, y, cw, ch), area))
            elif contour_is_circle_by_corners(contour):
                candidates.append(("C", (x, y, cw, ch), area))
        # NMS-style dedup: suppress overlapping duplicate detections.
        candidates.sort(key=lambda item: item[2], reverse=True)
        kept = []
        for label, rect, area in candidates:
            if any(rect_iou(rect, other_rect) >= SORT_NMS_IOU_THRESHOLD for _, other_rect, _ in kept):
                continue
            kept.append((label, rect, area))
        circles = sum(1 for label, _, _ in kept if label == "C")
        squares = sum(1 for label, _, _ in kept if label == "S")
        counts[f"{prefix}C"] += circles
        counts[f"{prefix}S"] += squares
        return circles + squares
    process_mask(blue_mask, "B")
    process_mask(green_mask, "G")
    counts["TOTAL"] = counts["BC"] + counts["BS"] + counts["GC"] + counts["GS"]
    counts["BLUE"] = counts["BC"] + counts["BS"]
    counts["GREEN"] = counts["GC"] + counts["GS"]
    counts["CIRCLE"] = counts["BC"] + counts["GC"]
    counts["SQUARE"] = counts["BS"] + counts["GS"]
    return counts


def load_sign_template(template_filename):
    if template_filename in SIGN_TEMPLATE_CACHE:
        return SIGN_TEMPLATE_CACHE[template_filename]
    template_path = Path(__file__).with_name(template_filename)
    if not template_path.exists():
        SIGN_TEMPLATE_CACHE[template_filename] = None
        return None
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
        SIGN_TEMPLATE_CACHE[template_filename] = None
        return None
    SIGN_TEMPLATE_CACHE[template_filename] = template
    return template


def run_music_remote_command():
    try:
        completed = subprocess.run(
            MUSIC_REMOTE_COMMAND,
            shell=True,
            capture_output=True,
            text=True,
            timeout=MUSIC_REMOTE_TIMEOUT_SECONDS,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        output_text = stdout or stderr or f"exit {completed.returncode}"
        success = completed.returncode == 0
        return success, output_text
    except Exception as exc:
        return False, str(exc)


def detect_traffic_light_state(frame_bgr, sign_roi):
    frame_height, frame_width = frame_bgr.shape[:2]
    if sign_roi is None:
        return {"state": "NONE", "rect": None, "center": None}
    x, y, w, h = sign_roi["outer_rect"]
    hx = clamp(int(round(x + w * TRAFFIC_LIGHT_HOUSING_X_RATIO)), 0, frame_width - 1)
    hy = clamp(int(round(y + h * TRAFFIC_LIGHT_HOUSING_Y_RATIO)), 0, frame_height - 1)
    hw = max(8, int(round(w * TRAFFIC_LIGHT_HOUSING_W_RATIO)))
    hh = max(12, int(round(h * TRAFFIC_LIGHT_HOUSING_H_RATIO)))
    hx1 = clamp(hx + hw, hx + 1, frame_width)
    hy1 = clamp(hy + hh, hy + 1, frame_height)
    housing = frame_bgr[hy:hy1, hx:hx1]
    if housing.size == 0:
        return {"state": "NONE", "rect": (hx, hy, hx1 - hx, hy1 - hy), "center": None}

    lamp_x = int(round((hx1 - hx) * TRAFFIC_LIGHT_LAMP_X_RATIO))
    lamp_w = max(6, int(round((hx1 - hx) * TRAFFIC_LIGHT_LAMP_W_RATIO)))
    lamp_h = max(6, int(round((hy1 - hy) * TRAFFIC_LIGHT_LAMP_H_RATIO)))
    red_y = int(round((hy1 - hy) * TRAFFIC_LIGHT_RED_Y_RATIO))
    green_y = int(round((hy1 - hy) * TRAFFIC_LIGHT_GREEN_Y_RATIO))

    def clip_roi(local_x, local_y, local_w, local_h):
        x0 = clamp(local_x, 0, (hx1 - hx) - 1)
        y0 = clamp(local_y, 0, (hy1 - hy) - 1)
        x1 = clamp(local_x + local_w, x0 + 1, (hx1 - hx))
        y1 = clamp(local_y + local_h, y0 + 1, (hy1 - hy))
        return x0, y0, x1, y1

    rx0, ry0, rx1, ry1 = clip_roi(lamp_x, red_y, lamp_w, lamp_h)
    gx0, gy0, gx1, gy1 = clip_roi(lamp_x, green_y, lamp_w, lamp_h)
    red_roi = housing[ry0:ry1, rx0:rx1]
    green_roi = housing[gy0:gy1, gx0:gx1]
    def classify_bright_spot(roi_bgr, abs_rect):
        if roi_bgr.size == 0:
            return None
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        max_value = int(gray.max())
        threshold_value = max(TRAFFIC_LIGHT_BRIGHT_MIN, max_value - TRAFFIC_LIGHT_BRIGHT_OFFSET)
        _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < TRAFFIC_LIGHT_BRIGHT_CONTOUR_MIN_AREA:
            return None

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        mean_b, mean_g, mean_r = [float(v) for v in roi_bgr.mean(axis=(0, 1))]
        x0, y0, w0, h0 = abs_rect
        result = {
            "center": (x0 + cx, y0 + cy),
            "rect": abs_rect,
            "rgb": (int(round(mean_r)), int(round(mean_g)), int(round(mean_b))),
            "area": area,
        }
        if mean_g >= mean_r + TRAFFIC_LIGHT_COLOR_DOMINANCE and mean_g >= mean_b + TRAFFIC_LIGHT_COLOR_DOMINANCE:
            result["state"] = "GREEN"
            return result
        # Red lamp can shift toward magenta/white under the Pi camera, so allow
        # red-dominant or red-blue combined bright spots to count as RED.
        if (
            mean_r >= mean_g + TRAFFIC_LIGHT_COLOR_DOMINANCE
            and mean_r >= mean_b - 5
        ) or (
            mean_r >= mean_g + 8
            and mean_b >= mean_g + 8
        ):
            result["state"] = "RED"
            return result
        return None

    red_result = classify_bright_spot(red_roi, (hx + rx0, hy + ry0, rx1 - rx0, ry1 - ry0))
    green_result = classify_bright_spot(green_roi, (hx + gx0, hy + gy0, gx1 - gx0, gy1 - gy0))

    if red_result is not None and red_result["state"] == "RED":
        return {
            "state": "RED",
            "rect": red_result["rect"],
            "center": red_result["center"],
        }
    if green_result is not None and green_result["state"] == "GREEN":
        return {
            "state": "GREEN",
            "rect": green_result["rect"],
            "center": green_result["center"],
        }
    return {
        "state": "NONE",
        "rect": (hx, hy, hx1 - hx, hy1 - hy),
        "center": (hx + (hx1 - hx) // 2, hy + (hy1 - hy) // 2),
    }


def classify_sign_symbol(frame_bgr, sign_roi):
    if sign_roi is None:
        return {"label": "UNKNOWN", "confidence": 0.0, "reason": "no_sign"}

    roi = sign_roi["warped"]
    if roi.size == 0:
        return {"label": "UNKNOWN", "confidence": 0.0, "reason": "empty_roi"}

    binary = preprocess_sign_symbol(roi)
    if binary is None:
        return {"label": "UNKNOWN", "confidence": 0.0, "reason": "empty_inner"}

    templates = {
        "ALARM": load_sign_template(ALARM_TEMPLATE_FILENAME),
        "TRAFFIC": load_sign_template(TRAFFIC_TEMPLATE_FILENAME),
        "MUSIC": load_sign_template(MUSIC_TEMPLATE_FILENAME),
    }
    if all(template is None for template in templates.values()):
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "missing_templates",
            "binary": binary,
        }

    scores = {}
    best_label = "UNKNOWN"
    best_score = -1.0
    for label, template in templates.items():
        if template is None:
            continue
        sample = binary
        if sample.shape != template.shape:
            sample = cv2.resize(sample, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_NEAREST)
        xor_frame = cv2.bitwise_xor(sample, template)
        difference_pixels = cv2.countNonZero(xor_frame)
        match_score = 1.0 - difference_pixels / float(xor_frame.shape[0] * xor_frame.shape[1])
        scores[label] = match_score
        if match_score > best_score:
            best_score = match_score
            best_label = label

    traffic_score = scores.get("TRAFFIC", -1.0)
    alarm_score = scores.get("ALARM", -1.0)
    if (
        traffic_score >= TRAFFIC_TEMPLATE_MATCH_THRESHOLD
        and traffic_score >= alarm_score + TRAFFIC_TEMPLATE_MARGIN
    ):
        return {
            "label": "TRAFFIC",
            "confidence": traffic_score,
            "reason": "xor_template_match_traffic",
            "binary": binary,
            "match_score": traffic_score,
            "scores": scores,
        }

    if best_score >= SIGN_TEMPLATE_MATCH_THRESHOLD:
        return {
            "label": best_label,
            "confidence": best_score,
            "reason": "xor_template_match",
            "binary": binary,
            "match_score": best_score,
            "scores": scores,
        }

    return {
        "label": "UNKNOWN",
        "confidence": max(0.0, best_score),
        "reason": "xor_template_match_below_threshold",
        "binary": binary,
        "match_score": best_score,
        "scores": scores,
    }


def install_stop_handlers(controller, control_state):
    def safe_stop():
        try:
            control_state.stop()
        except Exception:
            pass
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


def resolve_recovery_turn_direction(state, now):
    if state["entry_assist_direction"] != 0 and now <= state["entry_assist_until"]:
        return state["entry_assist_direction"]
    if state["recovery_turn_direction"] != 0 and (now - state["recovery_turn_time"]) <= TURN_DIRECTION_MEMORY_SECONDS:
        return state["recovery_turn_direction"]
    if state["last_nonzero_error"] > 0:
        return 1
    if state["last_nonzero_error"] < 0:
        return -1
    return 1


def motor_control_loop(controller, control_state):
    interval = 1.0 / CONTROL_HZ
    while True:
        loop_started = time.perf_counter()
        state = control_state.snapshot()
        if not state["running"]:
            break

        line_is_recent = (
            state["center_x"] is not None
            and (loop_started - state["last_line_time"]) <= LINE_TIMEOUT_SECONDS
        )
        time_since_line = loop_started - state["last_line_time"]
        sharp_entry_active = (
            state["entry_kind"] == "sharp"
            and state["entry_assist_direction"] != 0
            and loop_started <= state["entry_assist_until"]
            and state["entry_score"] >= ENTRY_ASSIST_SCORE_THRESHOLD
        )

        if state["pause_active"]:
            controller.hold_stop()
        elif state["motor_armed"] and line_is_recent:
            controller.set_tank_drive(state["left_speed"], state["right_speed"])
        elif state["motor_armed"] and time_since_line <= RECOVERY_HOLD_SECONDS:
            controller.hold_stop()
        elif state["motor_armed"] and (
            abs(state["last_nonzero_error"]) >= CORNER_ERROR_THRESHOLD
            or (state["entry_assist_direction"] != 0 and loop_started <= state["entry_assist_until"])
        ) and time_since_line <= (
            SHARP_CORNER_PIVOT_TIMEOUT_SECONDS if sharp_entry_active else CORNER_PIVOT_TIMEOUT_SECONDS
        ):
            turn_direction = resolve_recovery_turn_direction(state, loop_started)
            pivot_speed = SHARP_CORNER_PIVOT_SPEED if sharp_entry_active else CORNER_PIVOT_SPEED
            controller.set_tank_drive(
                int(clamp(pivot_speed * turn_direction, -MAX_SPEED, MAX_SPEED)),
                int(clamp(-pivot_speed * turn_direction, -MAX_SPEED, MAX_SPEED)),
            )
        elif state["motor_armed"] and time_since_line <= RECOVERY_TURN_TIMEOUT_SECONDS:
            turn_direction = resolve_recovery_turn_direction(state, loop_started)
            recovery_forward_speed = (
                SHARP_RECOVERY_FORWARD_SPEED if sharp_entry_active else RECOVERY_FORWARD_SPEED
            )
            recovery_turn_speed = SHARP_RECOVERY_TURN_SPEED if sharp_entry_active else RECOVERY_TURN_SPEED
            left_speed = recovery_forward_speed + recovery_turn_speed * turn_direction
            right_speed = recovery_forward_speed - recovery_turn_speed * turn_direction
            controller.set_tank_drive(
                int(clamp(left_speed, -MAX_SPEED, MAX_SPEED)),
                int(clamp(right_speed, -MAX_SPEED, MAX_SPEED)),
            )
        elif state["motor_armed"]:
            turn_direction = resolve_recovery_turn_direction(state, loop_started)
            left_speed = LOST_LINE_CONTINUOUS_FORWARD_SPEED + LOST_LINE_CONTINUOUS_TURN_SPEED * turn_direction
            right_speed = LOST_LINE_CONTINUOUS_FORWARD_SPEED - LOST_LINE_CONTINUOUS_TURN_SPEED * turn_direction
            controller.set_tank_drive(
                int(clamp(left_speed, -MAX_SPEED, MAX_SPEED)),
                int(clamp(right_speed, -MAX_SPEED, MAX_SPEED)),
            )
        else:
            controller.hold_stop()

        remaining = interval - (time.perf_counter() - loop_started)
        if remaining > 0:
            time.sleep(remaining)


def draw_overlay(
    frame,
    center,
    contour,
    roi_top,
    roi_bottom,
    error_pixels,
    left_speed,
    right_speed,
    fps,
    mode,
    curve_points,
    entry,
    purple_detection,
    center_purple_detection,
    sign_roi,
    sign_result,
    traffic_light_result,
    paused,
):
    height, width = frame.shape[:2]
    frame_center_x = width // 2
    cv2.rectangle(frame, (0, roi_top), (width - 1, roi_bottom - 1), (255, 200, 0), 2)
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, height - 1), (255, 0, 0), 2)
    center_scanline_y = int(height * CENTER_PURPLE_SCANLINE_Y_RATIO)
    band_top = max(0, center_scanline_y - CENTER_PURPLE_BAND_HALF_HEIGHT)
    band_bottom = min(height - 1, center_scanline_y + CENTER_PURPLE_BAND_HALF_HEIGHT)
    cv2.rectangle(frame, (0, band_top), (width - 1, band_bottom), (0, 0, 255), 2)
    if contour is not None:
        contour_on_frame = contour.copy()
        contour_on_frame[:, 0, 1] += roi_top
        cv2.drawContours(frame, [contour_on_frame], -1, (0, 255, 255), 2)
    if center is not None:
        cv2.circle(frame, center, 7, (0, 0, 255), -1)
    for point in curve_points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)
    if purple_detection is not None:
        x, y, w, h = purple_detection["rect"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(
            frame,
            "RedDot",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if center_purple_detection is not None and center_purple_detection.get("detected"):
        x, y, w, h = center_purple_detection["rect"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 0, 255), 2)
        cv2.putText(
            frame,
            "Center Purple",
            (max(12, x - 24), max(48, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if sign_roi is not None:
        x, y, w, h = sign_roi["outer_rect"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)
        ix, iy, iw, ih = sign_roi["inner_rect"]
        cv2.rectangle(frame, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)
        if "quad" in sign_roi:
            cv2.polylines(frame, [sign_roi["quad"].reshape(-1, 1, 2)], True, (255, 255, 0), 2)
    if sign_result is not None:
        cv2.putText(
            frame,
            f"SIGN {sign_result['label']} {sign_result.get('match_score', 0.0):.2f}",
            (18, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if traffic_light_result is not None and traffic_light_result.get("rect") is not None:
        x, y, w, h = traffic_light_result["rect"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"TL {traffic_light_result['state']}",
            (max(12, x - 10), max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    center_status = "YES" if center_purple_detection is not None and center_purple_detection.get("detected") else "NO"
    center_match_count = 0 if center_purple_detection is None else int(center_purple_detection.get("match_count", 0))
    center_run = 0 if center_purple_detection is None else int(center_purple_detection.get("max_run", 0))
    cv2.putText(
        frame,
        f"CENTER PURPLE {center_status} M{center_match_count} R{center_run}",
        (18, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    status = "PAUSE" if paused else mode
    cv2.putText(frame, f"FPS {fps:4.1f} {status}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"ERR {error_pixels:+4d} L{left_speed:>3} R{right_speed:>3}", (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"ENTRY {entry['kind']} D{entry['direction']:+d} S{entry['score']:.2f}",
        (18, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="store_true", help="Start driving immediately.")
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument(
        "--record-output",
        default="",
        help="Optional output mp4 path. When set, save processed frames while running.",
    )
    parser.add_argument(
        "--debug-video-output",
        default="",
        help="Optional debug mp4 path. When set, save processed overlay frames for review.",
    )
    args = parser.parse_args()

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    controller = CarMotorController()
    control_state = ControlState(motor_armed=args.arm)
    lcd_display = LCD1602Display()
    servo = ServoController()
    alarm_led = AlarmLedController()
    lcd_display.update("Line Following", "Starting", force=True)
    install_stop_handlers(controller, control_state)

    control_thread = threading.Thread(
        target=motor_control_loop,
        args=(controller, control_state),
        daemon=True,
    )
    control_thread.start()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"})
    picam2.configure(config)
    h264_recording = False
    debug_writer = None
    if args.record_output:
        # Use Picamera2 native H264 encoder for robust on-device recording.
        encoder = H264Encoder(bitrate=8_000_000)
        picam2.start_recording(encoder, FileOutput(args.record_output))
        h264_recording = True
    else:
        picam2.start()

    last_line_time = 0.0
    last_nonzero_error = 0
    recovery_turn_direction = 0
    recovery_turn_time = 0.0
    last_entry_direction = 0
    last_entry_kind = "lost"
    last_entry_score = 0.0
    last_entry_time = -999.0
    purple_hits = 0
    last_purple_trigger_time = -999.0
    last_center_purple_trigger_time = -999.0
    center_purple_sequence_active = False
    center_purple_sequence_until = -999.0
    center_purple_sign_result = None
    center_purple_sign_roi = None
    center_sort_counts = None
    sorting_display_until = -999.0
    sorting_post_cooldown_until = -999.0
    traffic_wait_active = False
    traffic_wait_sign_roi = None
    traffic_light_result = None
    traffic_wait_started_at = -999.0
    traffic_green_hits = 0
    traffic_release_ignore_until = -999.0
    music_display_until = -999.0
    music_command_pending = False
    music_command_status = ""
    music_command_status_until = -999.0
    alarm_resume_until = -999.0
    fps = 0.0
    last_frame_time = time.perf_counter()

    try:
        if args.debug_video_output:
            debug_path = Path(args.debug_video_output)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(
                str(debug_path),
                fourcc,
                20.0,
                PROCESS_SIZE,
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
            frame = cv2.resize(full_frame, PROCESS_SIZE, interpolation=cv2.INTER_AREA)

            purple_detection = detect_right_purple_square(frame)
            center_purple_detection = detect_centerline_purple(frame)
            center_purple_detected = (
                center_purple_detection is not None and center_purple_detection.get("detected")
            )
            sign_roi = None
            sign_result = None
            traffic_light_result = None
            allow_purple_trigger = (
                now >= traffic_release_ignore_until
                and now >= sorting_post_cooldown_until
            )
            if purple_detection is not None and allow_purple_trigger:
                purple_hits += 1
            else:
                purple_hits = 0
            if (
                purple_hits >= PURPLE_CONFIRMATION_HITS
                and (now - last_purple_trigger_time) >= PURPLE_TRIGGER_COOLDOWN_SECONDS
            ):
                last_purple_trigger_time = now
                purple_hits = 0
                servo.pulse_once(target_angle=SERVO_PURPLE_ANGLE, hold_seconds=SERVO_PURPLE_HOLD_SECONDS)
            if (
                center_purple_detected
                and allow_purple_trigger
                and not center_purple_sequence_active
                and (now - last_center_purple_trigger_time) >= CENTER_PURPLE_TRIGGER_COOLDOWN_SECONDS
            ):
                center_purple_sequence_active = True
                center_purple_sequence_until = now + CENTER_PURPLE_STOP_WAIT_SECONDS
                last_center_purple_trigger_time = now
                center_purple_sign_result = None
                center_purple_sign_roi = None
                center_sort_counts = None
                sorting_display_until = -999.0
                sorting_post_cooldown_until = -999.0
                control_state.set_pause(True)
                controller.stop()
                servo.set_angle(CENTER_PURPLE_STOP_ANGLE, settle_seconds=0.2)
            if center_purple_sequence_active and now >= center_purple_sequence_until:
                servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
                center_purple_sequence_active = False
                center_purple_sequence_until = -999.0
            elif center_purple_sequence_active:
                sign_roi = detect_sign_roi(frame)
                sign_result = classify_sign_symbol(frame, sign_roi)
                center_purple_sign_roi = sign_roi
                center_purple_sign_result = sign_result
                if sign_roi is not None:
                    frame_sort_counts = count_sorting_objects(sign_roi["warped"])
                    if frame_sort_counts is not None and frame_sort_counts.get("TOTAL", 0) > 0:
                        center_sort_counts = frame_sort_counts
                        if sorting_display_until <= now:
                            sorting_display_until = now + SORTING_DISPLAY_SECONDS
                            sorting_post_cooldown_until = (
                                sorting_display_until + SORTING_POST_COOLDOWN_SECONDS
                            )
                if sign_result is not None and sign_result.get("label") == "ALARM":
                    servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
                    center_purple_sequence_active = False
                    center_purple_sequence_until = -999.0
                    alarm_resume_until = now + ALARM_RESUME_SECONDS
                    control_state.set_pause(False)
                    control_state.set_base_speed_override(ALARM_RESUME_BASE_SPEED, alarm_resume_until)
                elif sign_result is not None and sign_result.get("label") == "TRAFFIC":
                    center_purple_sequence_active = False
                    center_purple_sequence_until = -999.0
                    traffic_wait_active = True
                    traffic_wait_sign_roi = sign_roi
                    traffic_wait_started_at = now
                    traffic_green_hits = 0
                    control_state.set_pause(True)
                elif sign_result is not None and sign_result.get("label") == "MUSIC":
                    servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
                    center_purple_sequence_active = False
                    center_purple_sequence_until = -999.0
                    music_display_until = now + MUSIC_DISPLAY_SECONDS
                    music_command_pending = True
                    music_command_status = ""
                    music_command_status_until = -999.0
                    control_state.set_pause(True)

            if traffic_wait_active:
                current_sign_roi = detect_sign_roi(frame)
                if current_sign_roi is not None:
                    current_sign_result = classify_sign_symbol(frame, current_sign_roi)
                    current_rect = current_sign_roi["outer_rect"]
                    locked_rect = None if traffic_wait_sign_roi is None else traffic_wait_sign_roi["outer_rect"]
                    if traffic_wait_sign_roi is None:
                        if current_sign_result.get("label") == "TRAFFIC":
                            traffic_wait_sign_roi = current_sign_roi
                    else:
                        if (
                            current_sign_result.get("label") == "TRAFFIC"
                            or rect_iou(current_rect, locked_rect) >= 0.35
                        ):
                            traffic_wait_sign_roi = current_sign_roi
                traffic_light_result = detect_traffic_light_state(frame, traffic_wait_sign_roi)
                if traffic_light_result["state"] == "GREEN":
                    traffic_green_hits += 1
                else:
                    traffic_green_hits = 0
                if (
                    traffic_light_result["state"] == "GREEN"
                    and traffic_green_hits >= TRAFFIC_LIGHT_GREEN_CONFIRM_HITS
                    and now - traffic_wait_started_at >= TRAFFIC_LIGHT_GREEN_MIN_WAIT_SECONDS
                ):
                    traffic_wait_active = False
                    traffic_wait_sign_roi = None
                    traffic_wait_started_at = -999.0
                    traffic_green_hits = 0
                    traffic_release_ignore_until = now + TRAFFIC_RELEASE_IGNORE_SECONDS
                    servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.2)
                    alarm_resume_until = now + ALARM_RESUME_SECONDS
                    control_state.set_pause(False)
                    control_state.set_base_speed_override(ALARM_RESUME_BASE_SPEED, alarm_resume_until)
                else:
                    control_state.set_pause(True)

            if now >= music_display_until and music_display_until > 0:
                music_display_until = -999.0

            if music_command_pending and music_display_until <= 0:
                success, output_text = run_music_remote_command()
                music_command_pending = False
                music_command_status = output_text[:16]
                music_command_status_until = now + 2.0
                music_cooldown_start = time.perf_counter()
                traffic_release_ignore_until = max(
                    traffic_release_ignore_until,
                    music_cooldown_start + MUSIC_RELEASE_IGNORE_SECONDS,
                )
                if not success:
                    last_center_purple_trigger_time = now

            curve_points = sample_curve_points(frame)
            entry = infer_curve_entry(curve_points)
            control_state.set_entry_profile(entry["kind"], entry["score"])
            if entry["direction"] != 0 and entry["kind"] != "straight":
                last_entry_direction = entry["direction"]
                last_entry_kind = entry["kind"]
                last_entry_score = entry["score"]
                last_entry_time = now
                control_state.set_entry_assist(entry["direction"], now)

            (center, mask, roi_top, roi_bottom, contour), roi_mode = find_line_center(frame)
            left_speed = 0
            right_speed = 0
            error_pixels = 0
            pause_active = (
                center_purple_sequence_active
                or traffic_wait_active
                or (now < music_display_until)
                or music_command_pending
                or (center_sort_counts is not None and now < sorting_display_until)
            )
            control_state.set_pause(pause_active)
            state_snapshot = control_state.snapshot()

            if center is not None:
                last_line_time = now
                base_speed_override = None
                if state_snapshot.get("base_speed_override") is not None and now <= state_snapshot.get(
                    "base_speed_override_until", -999.0
                ):
                    base_speed_override = state_snapshot["base_speed_override"]
                left_speed, right_speed, error_pixels = compute_drive_command(
                    center[0], frame.shape[1], entry["kind"], entry["score"], base_speed_override
                )
                control_state.update_detection(center[0], frame.shape[1], now)
                if error_pixels != 0:
                    last_nonzero_error = error_pixels
                edge_margin = frame.shape[1] * EDGE_DIRECTION_MARGIN_RATIO
                if center[0] <= edge_margin:
                    recovery_turn_direction = -1
                    recovery_turn_time = now
                elif center[0] >= (frame.shape[1] - edge_margin):
                    recovery_turn_direction = 1
                    recovery_turn_time = now
                elif error_pixels <= -TURN_DIRECTION_LOCK_THRESHOLD:
                    recovery_turn_direction = -1
                    recovery_turn_time = now
                elif error_pixels >= TURN_DIRECTION_LOCK_THRESHOLD:
                    recovery_turn_direction = 1
                    recovery_turn_time = now

                mode = roi_mode
            else:
                control_state.mark_line_lost()
                time_since_line = now - last_line_time
                entry_assist_active = (
                    (now - last_entry_time) <= ENTRY_MEMORY_SECONDS
                    and last_entry_kind in ("gentle", "sharp")
                    and last_entry_score >= ENTRY_ASSIST_SCORE_THRESHOLD
                )
                turn_direction = (
                    recovery_turn_direction
                    if (now - recovery_turn_time) <= TURN_DIRECTION_MEMORY_SECONDS
                    else (-1 if last_nonzero_error < 0 else 1)
                )
                if entry_assist_active and last_entry_direction != 0:
                    turn_direction = last_entry_direction
                if state_snapshot["motor_armed"] and time_since_line <= RECOVERY_HOLD_SECONDS:
                    mode = "hold"
                elif state_snapshot["motor_armed"] and (
                    abs(last_nonzero_error) >= CORNER_ERROR_THRESHOLD or entry_assist_active
                ) and time_since_line <= CORNER_PIVOT_TIMEOUT_SECONDS:
                    mode = "corner+entry" if entry_assist_active else "corner"
                elif state_snapshot["motor_armed"] and time_since_line <= RECOVERY_TURN_TIMEOUT_SECONDS:
                    left_speed = RECOVERY_FORWARD_SPEED + RECOVERY_TURN_SPEED * turn_direction
                    right_speed = RECOVERY_FORWARD_SPEED - RECOVERY_TURN_SPEED * turn_direction
                    mode = "search"
                else:
                    mode = "stop"

            state_snapshot = control_state.snapshot()
            alarm_resume_active = now <= alarm_resume_until
            if center_sort_counts is not None and now < sorting_display_until:
                alarm_led.all_off()
                lcd_display.update(
                    f"BC:{center_sort_counts['BC']} BS:{center_sort_counts['BS']}",
                    f"GC:{center_sort_counts['GC']} GS:{center_sort_counts['GS']}",
                )
            elif center_purple_sequence_active:
                alarm_led.all_off()
                remaining = max(0.0, center_purple_sequence_until - now)
                sign_label = "UNKNOWN"
                if center_purple_sign_result is not None:
                    sign_label = center_purple_sign_result["label"]
                lcd_display.update(sign_label, f"Wait {remaining:>4.1f}s")
            elif traffic_wait_active:
                alarm_led.all_off()
                traffic_state = "NONE"
                if traffic_light_result is not None:
                    traffic_state = traffic_light_result["state"]
                lcd_display.update("TRAFFIC SIGN", f"TL {traffic_state}")
            elif now < music_display_until:
                alarm_led.all_off()
                remaining = max(0.0, music_display_until - now)
                lcd_display.update("MUSIC", f"Wait {remaining:>4.1f}s")
            elif music_command_pending:
                alarm_led.all_off()
                lcd_display.update("MUSIC", "Playing...")
            elif now < music_command_status_until:
                alarm_led.all_off()
                lcd_display.update("MUSIC DONE", music_command_status)
            elif alarm_resume_active:
                alarm_led.update_alarm_pattern(now)
                remaining = max(0.0, alarm_resume_until - now)
                lcd_display.update("ALARM RUN", f"V40 {remaining:>4.1f}s")
            elif center_purple_detected:
                alarm_led.all_off()
                lcd_display.update("Center Line", "Purple")
            elif state_snapshot["motor_armed"]:
                alarm_led.all_off()
                lcd_display.update(
                    f"E{int(state_snapshot['error_pixels']):+4d}",
                    f"L{int(state_snapshot['left_speed']):>3} R{int(state_snapshot['right_speed']):>3}",
                )
            else:
                alarm_led.all_off()
                lcd_display.update("Line Following", "SAFE")

            state_snapshot = control_state.snapshot()
            draw_overlay(
                frame,
                center,
                contour,
                roi_top,
                roi_bottom,
                int(state_snapshot["error_pixels"]),
                int(state_snapshot["left_speed"]),
                int(state_snapshot["right_speed"]),
                fps,
                mode,
                curve_points,
                entry,
                purple_detection,
                center_purple_detection,
                center_purple_sign_roi if center_purple_sequence_active else (traffic_wait_sign_roi if traffic_wait_active else sign_roi),
                center_purple_sign_result if center_purple_sequence_active else sign_result,
                traffic_light_result,
                pause_active,
            )
            if debug_writer is not None:
                debug_writer.write(frame)

            if not args.headless:
                cv2.imshow("Line Following Minimal", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("m"), ord("M")):
                    new_armed = not state_snapshot["motor_armed"]
                    control_state.set_motor_armed(new_armed)
                    if not new_armed:
                        controller.stop()
            else:
                time.sleep(0.001)

    finally:
        control_state.stop()
        control_thread.join(timeout=0.5)
        controller.close()
        servo.close()
        alarm_led.close()
        lcd_display.close()
        if debug_writer is not None:
            debug_writer.release()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()


if __name__ == "__main__":
    main()
