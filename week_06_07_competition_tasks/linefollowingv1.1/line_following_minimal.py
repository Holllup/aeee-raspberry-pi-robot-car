import argparse
import atexit
import csv
import json
import signal
import threading
import time
from datetime import datetime
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

PURPLE_RIGHT_REGION_RATIO = 0.55
PURPLE_MIN_AREA = 90
PURPLE_MIN_SOLIDITY = 0.35
PURPLE_MIN_FILL_RATIO = 0.20
PURPLE_MAX_ASPECT_ERROR = 1.20
PURPLE_MIN_SATURATION = 15
PURPLE_MIN_VALUE = 40
PURPLE_CONFIRMATION_HITS = 8
PURPLE_ACTION_HOLD_SECONDS = 5.0
PURPLE_TRIGGER_COOLDOWN_SECONDS = 8.0
PURPLE_EMERGENCY_STOP_BURST = 2
PURPLE_SCANLINE_Y_RATIO = 0.62
PURPLE_SCANLINE_X_START_RATIO = 0.40
PURPLE_SCANLINE_TARGET_RGB = np.array([225, 115, 165], dtype=np.int16)
PURPLE_SCANLINE_DISTANCE_THRESHOLD = 65
PURPLE_SCANLINE_MIN_MATCH_PIXELS = 80
PURPLE_SCANLINE_MIN_RUN_PIXELS = 48

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

BASE_SPEED = 24
MAX_SPEED = 70
STEERING_GAIN = 42.0
DEADBAND_PIXELS = 12
CONTROL_HZ = 120.0
LINE_TIMEOUT_SECONDS = 0.15
RECOVERY_HOLD_SECONDS = 0.05
CORNER_ERROR_THRESHOLD = 34
CORNER_PIVOT_TIMEOUT_SECONDS = 0.72
CORNER_PIVOT_SPEED = 30
RECOVERY_TURN_TIMEOUT_SECONDS = 1.15
RECOVERY_TURN_SPEED = 28
RECOVERY_FORWARD_SPEED = 4
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


class RunSessionLogger:
    def __init__(self, session_root, duration_seconds):
        self.duration_seconds = max(1.0, float(duration_seconds))
        self.started_at = time.perf_counter()
        self.start_wall_time = datetime.now()
        timestamp = self.start_wall_time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(session_root).expanduser() / f"run_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.session_dir / "run_video.h264"
        self.events_csv_path = self.session_dir / "events.csv"
        self.summary_path = self.session_dir / "summary.json"
        self.total_frames = 0
        self.line_hits = 0
        self.purple_hits = 0
        self.purple_triggers = 0
        self._csv_file = self.events_csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(
            [
                "t_s",
                "frame",
                "line_found",
                "line_center_x",
                "error_px",
                "left_speed",
                "right_speed",
                "mode",
                "purple_found",
                "purple_trigger",
                "purple_pause_active",
                "purple_best_area",
                "purple_roi_mode",
                "purple_rect_x",
                "purple_rect_y",
                "purple_rect_w",
                "purple_rect_h",
            ]
        )

    def elapsed_seconds(self):
        return time.perf_counter() - self.started_at

    def should_stop(self):
        return self.elapsed_seconds() >= self.duration_seconds

    def log_frame(
        self,
        line_found,
        line_center_x,
        error_px,
        left_speed,
        right_speed,
        mode,
        purple_found,
        purple_trigger,
        purple_pause_active,
        purple_best_area,
        purple_roi_mode,
        purple_rect,
    ):
        self.total_frames += 1
        if line_found:
            self.line_hits += 1
        if purple_found:
            self.purple_hits += 1
        if purple_trigger:
            self.purple_triggers += 1
        self._writer.writerow(
            [
                round(self.elapsed_seconds(), 4),
                self.total_frames,
                int(bool(line_found)),
                "" if line_center_x is None else int(line_center_x),
                int(error_px),
                int(left_speed),
                int(right_speed),
                str(mode),
                int(bool(purple_found)),
                int(bool(purple_trigger)),
                int(bool(purple_pause_active)),
                round(float(purple_best_area), 2),
                purple_roi_mode,
                "" if purple_rect is None else int(purple_rect[0]),
                "" if purple_rect is None else int(purple_rect[1]),
                "" if purple_rect is None else int(purple_rect[2]),
                "" if purple_rect is None else int(purple_rect[3]),
            ]
        )

    def close(self):
        elapsed = max(1e-6, self.elapsed_seconds())
        fps = self.total_frames / elapsed
        summary = {
            "session_dir": str(self.session_dir),
            "start_time": self.start_wall_time.isoformat(timespec="seconds"),
            "duration_seconds": round(elapsed, 3),
            "target_duration_seconds": self.duration_seconds,
            "video_file": str(self.video_path),
            "events_csv": str(self.events_csv_path),
            "frames": self.total_frames,
            "fps_avg": round(fps, 3),
            "line_detect_rate": round(self.line_hits / self.total_frames, 4) if self.total_frames else 0.0,
            "purple_detect_rate": round(self.purple_hits / self.total_frames, 4) if self.total_frames else 0.0,
            "purple_trigger_count": self.purple_triggers,
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._csv_file.close()


def clamp(value, low, high):
    return max(low, min(high, value))


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
        self.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.3)

    @staticmethod
    def _angle_to_duty_cycle(angle):
        angle = max(0, min(180, angle))
        return 2.5 + (angle / 180.0) * 10.0

    def set_angle(self, angle, settle_seconds=0.5):
        self.pwm.ChangeDutyCycle(self._angle_to_duty_cycle(angle))
        time.sleep(settle_seconds)
        self.pwm.ChangeDutyCycle(0)

    def close(self):
        self.pwm.ChangeDutyCycle(0)
        time.sleep(0.1)
        self.pwm.stop()


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

    def update_detection(self, center_x, frame_width, now):
        with self.lock:
            self.center_x = center_x
            self.frame_width = frame_width
            self.last_line_time = now
            self.left_speed, self.right_speed, self.error_pixels = compute_drive_command(
                center_x, frame_width
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
            }

    def stop(self):
        with self.lock:
            self.running = False


def compute_drive_command(center_x, frame_width):
    frame_center_x = frame_width // 2
    error_pixels = center_x - frame_center_x
    if abs(error_pixels) <= DEADBAND_PIXELS:
        error_pixels = 0
    normalized_error = error_pixels / (frame_width / 2.0)
    turn_delta = STEERING_GAIN * normalized_error
    left_speed = clamp(BASE_SPEED + turn_delta, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(BASE_SPEED - turn_delta, -MAX_SPEED, MAX_SPEED)
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

    # Convert scanline BGR -> RGB for distance check.
    scanline_rgb = frame_bgr[y, x_start:, ::-1].astype(np.int16)
    if scanline_rgb.size == 0:
        return None
    diff = scanline_rgb - PURPLE_SCANLINE_TARGET_RGB
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    matched = np.where(dist <= PURPLE_SCANLINE_DISTANCE_THRESHOLD)[0]
    match_count = int(matched.size)
    if match_count < PURPLE_SCANLINE_MIN_MATCH_PIXELS:
        return None

    # Require a long contiguous match run to suppress scattered-pixel false positives.
    max_run = 1
    run = 1
    for i in range(1, match_count):
        if int(matched[i]) == int(matched[i - 1]) + 1:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    if max_run < PURPLE_SCANLINE_MIN_RUN_PIXELS:
        return None

    x0 = x_start + int(matched[0])
    x1 = x_start + int(matched[-1])
    w = max(1, x1 - x0 + 1)
    rect = (x0, max(0, y - 3), w, 7)
    area_proxy = float(max_run)
    return {"rect": rect, "area": area_proxy, "roi_mode": "scanline"}
    return None


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

        if state["pause_active"]:
            controller.hold_stop()
        elif state["motor_armed"] and line_is_recent:
            controller.set_tank_drive(state["left_speed"], state["right_speed"])
        elif state["motor_armed"] and time_since_line <= RECOVERY_HOLD_SECONDS:
            controller.hold_stop()
        elif state["motor_armed"] and (
            abs(state["last_nonzero_error"]) >= CORNER_ERROR_THRESHOLD
            or (state["entry_assist_direction"] != 0 and loop_started <= state["entry_assist_until"])
        ) and time_since_line <= CORNER_PIVOT_TIMEOUT_SECONDS:
            turn_direction = resolve_recovery_turn_direction(state, loop_started)
            controller.set_tank_drive(
                int(clamp(CORNER_PIVOT_SPEED * turn_direction, -MAX_SPEED, MAX_SPEED)),
                int(clamp(-CORNER_PIVOT_SPEED * turn_direction, -MAX_SPEED, MAX_SPEED)),
            )
        elif state["motor_armed"] and time_since_line <= RECOVERY_TURN_TIMEOUT_SECONDS:
            turn_direction = resolve_recovery_turn_direction(state, loop_started)
            left_speed = RECOVERY_FORWARD_SPEED + RECOVERY_TURN_SPEED * turn_direction
            right_speed = RECOVERY_FORWARD_SPEED - RECOVERY_TURN_SPEED * turn_direction
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
    paused,
):
    height, width = frame.shape[:2]
    frame_center_x = width // 2
    cv2.rectangle(frame, (0, roi_top), (width - 1, roi_bottom - 1), (255, 200, 0), 2)
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, height - 1), (255, 0, 0), 2)
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
        "--duration-seconds",
        type=float,
        default=20.0,
        help="Auto-stop after this many seconds (default: 20).",
    )
    parser.add_argument(
        "--session-root",
        default="/home/jacob/line_follow_runs",
        help="Directory for timestamped run logs/videos.",
    )
    parser.add_argument(
        "--record-output",
        default="",
        help="Optional manual h264 output path. If omitted, auto uses timestamped session path.",
    )
    args = parser.parse_args()

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)

    controller = CarMotorController()
    control_state = ControlState(motor_armed=args.arm)
    lcd_display = LCD1602Display()
    servo = ServoController()
    lcd_display.update("Line Following", "Starting", force=True)
    install_stop_handlers(controller, control_state)

    control_thread = threading.Thread(
        target=motor_control_loop,
        args=(controller, control_state),
        daemon=True,
    )
    control_thread.start()

    session_logger = RunSessionLogger(args.session_root, args.duration_seconds)
    record_path = args.record_output if args.record_output else str(session_logger.video_path)
    print(f"[RUN] session={session_logger.session_dir}")
    print(f"[RUN] video={record_path}")
    print(f"[RUN] events={session_logger.events_csv_path}")
    print(f"[RUN] summary={session_logger.summary_path}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"})
    picam2.configure(config)
    h264_recording = False
    # Use Picamera2 native H264 encoder for robust on-device recording.
    encoder = H264Encoder(bitrate=8_000_000)
    picam2.start_recording(encoder, FileOutput(record_path))
    h264_recording = True

    last_line_time = 0.0
    last_nonzero_error = 0
    recovery_turn_direction = 0
    recovery_turn_time = 0.0
    last_entry_direction = 0
    last_entry_kind = "lost"
    last_entry_score = 0.0
    last_entry_time = -999.0
    purple_hits = 0
    purple_pause_until = 0.0
    last_purple_trigger_time = -999.0
    servo_is_purple_pose = False
    fps = 0.0
    last_frame_time = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            purple_triggered_this_frame = False
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            full_frame = picam2.capture_array()
            full_frame = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(full_frame, PROCESS_SIZE, interpolation=cv2.INTER_AREA)

            purple_detection = detect_right_purple_square(frame)
            if purple_detection is not None:
                purple_hits += 1
            else:
                purple_hits = 0
            if (
                purple_hits >= PURPLE_CONFIRMATION_HITS
                and (now - last_purple_trigger_time) >= PURPLE_TRIGGER_COOLDOWN_SECONDS
            ):
                purple_pause_until = now + PURPLE_ACTION_HOLD_SECONDS
                last_purple_trigger_time = now
                purple_hits = 0
                purple_triggered_this_frame = True
                # Immediate hard stop to avoid any control-loop lag.
                for _ in range(PURPLE_EMERGENCY_STOP_BURST):
                    controller.stop()
                control_state.set_pause(True)
                print(
                    f"[EVENT] purple_trigger t={session_logger.elapsed_seconds():.3f}s "
                    f"hold={PURPLE_ACTION_HOLD_SECONDS:.1f}s "
                    f"mode={purple_detection.get('roi_mode', 'unknown')} "
                    f"area={purple_detection.get('area', 0):.1f}"
                )
                if not servo_is_purple_pose:
                    servo.set_angle(SERVO_PURPLE_ANGLE, settle_seconds=0.35)
                    servo_is_purple_pose = True

            curve_points = sample_curve_points(frame)
            entry = infer_curve_entry(curve_points)
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
            pause_active = now < purple_pause_until
            control_state.set_pause(pause_active)
            state_snapshot = control_state.snapshot()

            if pause_active:
                mode = "purple"
                lcd_display.update("State", "Red Dot", force=True)
            elif center is not None:
                if servo_is_purple_pose:
                    servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.35)
                    servo_is_purple_pose = False
                last_line_time = now
                left_speed, right_speed, error_pixels = compute_drive_command(center[0], frame.shape[1])
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
                if servo_is_purple_pose:
                    servo.set_angle(SERVO_HOME_ANGLE, settle_seconds=0.35)
                    servo_is_purple_pose = False

            state_snapshot = control_state.snapshot()
            if pause_active:
                lcd_display.update("State", "Red Dot")
            elif state_snapshot["motor_armed"]:
                lcd_display.update(
                    f"E{int(state_snapshot['error_pixels']):+4d}",
                    f"L{int(state_snapshot['left_speed']):>3} R{int(state_snapshot['right_speed']):>3}",
                )
            else:
                lcd_display.update("Line Following", "SAFE")

            session_logger.log_frame(
                line_found=(center is not None),
                line_center_x=(None if center is None else center[0]),
                error_px=int(state_snapshot["error_pixels"]),
                left_speed=int(state_snapshot["left_speed"]),
                right_speed=int(state_snapshot["right_speed"]),
                mode=mode,
                purple_found=(purple_detection is not None),
                purple_trigger=purple_triggered_this_frame,
                purple_pause_active=pause_active,
                purple_best_area=(0.0 if purple_detection is None else purple_detection.get("area", 0.0)),
                purple_roi_mode=("" if purple_detection is None else purple_detection.get("roi_mode", "")),
                purple_rect=(None if purple_detection is None else purple_detection.get("rect")),
            )

            if session_logger.should_stop():
                print(f"[RUN] duration reached: {args.duration_seconds:.1f}s, stopping.")
                break

            if not args.headless:
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
                    pause_active,
                )
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
        lcd_display.close()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()
        session_logger.close()
        print(f"[RUN] saved: {session_logger.summary_path}")


if __name__ == "__main__":
    main()
