import argparse
import atexit
import csv
import json
import signal
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2
import serial


WINDOW_NAME = "Line Following Car"
MASK_WINDOW_NAME = "Line Mask"
FRAME_SIZE = (320, 240)
SHOW_MASK_WINDOW = False

ROI_CENTER_RATIO = 0.5
ROI_HEIGHT_RATIO = 0.2
RECOVERY_ROI_CENTER_RATIO = 0.76
RECOVERY_ROI_HEIGHT_RATIO = 0.44
BLACK_THRESHOLD = 80
MIN_CONTOUR_AREA = 250

SERIAL_PORT_CANDIDATES = (
    "/dev/serial0",
    "/dev/ttyAMA0",
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
STEERING_SIGN = 1.0
DEADBAND_PIXELS = 12
ARM_ON_START = False
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
MAX_CENTER_JUMP_PIXELS = 110
LOG_SAMPLE_HZ = 20.0
LOG_DIR = Path.home() / "line_following_logs"
DEBUG_CAPTURE_DIR = Path.home() / "line_following_debug"
DEBUG_PRE_EVENT_SECONDS = 3.0
DEBUG_BUFFER_MAX_FRAMES = 120
DEBUG_EVENT_COOLDOWN_SECONDS = 1.0


def clamp(value, low, high):
    return max(low, min(high, value))


def reverse_direction(direction):
    return "r" if direction == "f" else "f"


def find_line_center_in_roi(frame_bgr, black_threshold, roi_center_ratio, roi_height_ratio):
    frame_height = frame_bgr.shape[0]
    roi_height = max(1, int(frame_height * roi_height_ratio))
    roi_center = int(frame_height * roi_center_ratio)
    roi_top = max(0, roi_center - roi_height // 2)
    roi_bottom = min(frame_height, roi_top + roi_height)

    roi_bgr = frame_bgr[roi_top:roi_bottom, :]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, black_threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

    if not contours:
        return None, mask, roi_top, roi_bottom, None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, mask, roi_top, roi_bottom, contour

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"]) + roi_top
    return (center_x, center_y), mask, roi_top, roi_bottom, contour


def find_line_center(frame_bgr, black_threshold):
    primary = find_line_center_in_roi(
        frame_bgr,
        black_threshold,
        ROI_CENTER_RATIO,
        ROI_HEIGHT_RATIO,
    )
    if primary[0] is not None:
        return primary

    return find_line_center_in_roi(
        frame_bgr,
        black_threshold,
        RECOVERY_ROI_CENTER_RATIO,
        RECOVERY_ROI_HEIGHT_RATIO,
    )


class CarMotorController:
    # UART transport matching the robot command format like "#ha".
    def __init__(self, baud_rate, port_candidates):
        self.serial = self._open_serial_port(port_candidates, baud_rate)
        self.port_name = self.serial.port
        self.last_command = None
        self.stop()

    def _open_serial_port(self, port_candidates, baud_rate):
        attempted_ports = []
        for port in port_candidates:
            if not Path(port).exists():
                continue
            attempted_ports.append(port)
            try:
                connection = serial.Serial(
                    port=port,
                    baudrate=baud_rate,
                    timeout=SERIAL_TIMEOUT_SECONDS,
                    write_timeout=SERIAL_TIMEOUT_SECONDS,
                )
                connection.reset_input_buffer()
                connection.reset_output_buffer()
                return connection
            except serial.SerialException:
                continue

        if attempted_ports:
            raise RuntimeError(
                f"unable to open serial port from {', '.join(attempted_ports)}"
            )
        raise RuntimeError(
            f"no serial port found in candidates: {', '.join(port_candidates)}"
        )

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

    def hold_stop(self):
        self._write_payload(b"#ha")

    def stop(self):
        self._force_write_payload(b"#ha")

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


def install_stop_handlers(controller):
    def stop_and_exit(signum=None, frame=None):
        try:
            if controller is not None:
                controller.stop()
        finally:
            raise SystemExit(0)

    atexit.register(lambda: controller is not None and controller.stop())
    signal.signal(signal.SIGINT, stop_and_exit)
    signal.signal(signal.SIGTERM, stop_and_exit)


class ControlState:
    def __init__(self, motor_armed, base_speed, steering_gain):
        self.lock = threading.Lock()
        self.running = True
        self.motor_armed = motor_armed
        self.base_speed = base_speed
        self.steering_gain = steering_gain
        self.center_x = None
        self.frame_width = FRAME_SIZE[0]
        self.last_line_time = 0.0
        self.error_pixels = 0
        self.left_speed = 0
        self.right_speed = 0
        self.last_nonzero_error = 0
        self.recovery_turn_direction = 0
        self.recovery_turn_time = 0.0
        self.last_confirmed_center_x = None

    def update_detection(self, center_x, frame_width):
        with self.lock:
            frame_center_x = frame_width / 2.0
            if self.last_confirmed_center_x is not None:
                opposite_sides = (
                    (center_x - frame_center_x) * (self.last_confirmed_center_x - frame_center_x) < 0
                )
                jump_too_large = abs(center_x - self.last_confirmed_center_x) >= MAX_CENTER_JUMP_PIXELS
                if opposite_sides and jump_too_large:
                    return False

            self.center_x = center_x
            self.frame_width = frame_width
            self.last_line_time = time.perf_counter()
            self.last_confirmed_center_x = center_x
            self.left_speed, self.right_speed, self.error_pixels = compute_drive_command(
                center_x,
                frame_width,
                self.base_speed,
                self.steering_gain,
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
                self.recovery_turn_time = self.last_line_time
            return True

    def mark_line_lost(self):
        with self.lock:
            self.center_x = None
            self.error_pixels = 0

    def set_motor_armed(self, motor_armed):
        with self.lock:
            self.motor_armed = motor_armed

    def update_base_speed(self, base_speed):
        with self.lock:
            self.base_speed = base_speed
            if self.center_x is not None:
                self.left_speed, self.right_speed, self.error_pixels = compute_drive_command(
                    self.center_x,
                    self.frame_width,
                    self.base_speed,
                    self.steering_gain,
                )

    def update_steering_gain(self, steering_gain):
        with self.lock:
            self.steering_gain = steering_gain
            if self.center_x is not None:
                self.left_speed, self.right_speed, self.error_pixels = compute_drive_command(
                    self.center_x,
                    self.frame_width,
                    self.base_speed,
                    self.steering_gain,
                )

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "motor_armed": self.motor_armed,
                "base_speed": self.base_speed,
                "steering_gain": self.steering_gain,
                "center_x": self.center_x,
                "last_line_time": self.last_line_time,
                "error_pixels": self.error_pixels,
                "left_speed": self.left_speed,
                "right_speed": self.right_speed,
                "last_nonzero_error": self.last_nonzero_error,
                "recovery_turn_direction": self.recovery_turn_direction,
                "recovery_turn_time": self.recovery_turn_time,
            }

    def stop(self):
        with self.lock:
            self.running = False


class TelemetryLogger:
    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"line_following_{timestamp}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "wall_time",
                "mono_time",
                "fps",
                "motor_armed",
                "line_found",
                "center_x",
                "error_pixels",
                "left_speed",
                "right_speed",
                "recovery_mode",
                "recovery_turn_direction",
                "black_threshold",
                "base_speed",
                "steering_gain",
            ]
        )
        self.last_write_time = 0.0
        self.min_interval = 1.0 / LOG_SAMPLE_HZ

    def log(
        self,
        fps,
        motor_armed,
        center,
        error_pixels,
        left_speed,
        right_speed,
        recovery_mode,
        recovery_turn_direction,
        black_threshold,
        base_speed,
        steering_gain,
    ):
        now = time.perf_counter()
        if (now - self.last_write_time) < self.min_interval:
            return

        self.writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{now:.6f}",
                f"{fps:.2f}",
                int(motor_armed),
                int(center is not None),
                "" if center is None else center[0],
                error_pixels,
                left_speed,
                right_speed,
                recovery_mode,
                recovery_turn_direction,
                black_threshold,
                base_speed,
                f"{steering_gain:.2f}",
            ]
        )
        self.file.flush()
        self.last_write_time = now

    def close(self):
        self.file.close()


class FrameDebugRecorder:
    def __init__(self):
        DEBUG_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=DEBUG_BUFFER_MAX_FRAMES)
        self.last_event_time = 0.0
        self.event_index = 0

    def push(self, frame, metadata):
        now = time.perf_counter()
        self.buffer.append(
            {
                "time": now,
                "frame": frame.copy(),
                "metadata": dict(metadata),
            }
        )
        while self.buffer and (now - self.buffer[0]["time"]) > DEBUG_PRE_EVENT_SECONDS:
            self.buffer.popleft()

    def capture_event(self, reason, frame, mask, metadata):
        now = time.perf_counter()
        if (now - self.last_event_time) < DEBUG_EVENT_COOLDOWN_SECONDS:
            return None

        self.event_index += 1
        self.last_event_time = now
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        event_dir = DEBUG_CAPTURE_DIR / f"{timestamp}_{self.event_index:03d}_{reason}"
        event_dir.mkdir(parents=True, exist_ok=True)

        for index, buffered in enumerate(self.buffer):
            cv2.imwrite(str(event_dir / f"pre_{index:02d}.jpg"), buffered["frame"])
            (event_dir / f"pre_{index:02d}.json").write_text(
                json.dumps(buffered["metadata"], ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

        cv2.imwrite(str(event_dir / "event.jpg"), frame)
        cv2.imwrite(str(event_dir / "mask.jpg"), mask)
        (event_dir / "event.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return event_dir


def resolve_recovery_turn_direction(state, now):
    if (
        state["recovery_turn_direction"] != 0
        and (now - state["recovery_turn_time"]) <= TURN_DIRECTION_MEMORY_SECONDS
    ):
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

        if controller is not None:
            if state["motor_armed"] and line_is_recent:
                controller.set_tank_drive(state["left_speed"], state["right_speed"])
            elif state["motor_armed"] and time_since_line <= RECOVERY_HOLD_SECONDS:
                controller.set_tank_drive(state["left_speed"], state["right_speed"])
            elif (
                state["motor_armed"]
                and abs(state["last_nonzero_error"]) >= CORNER_ERROR_THRESHOLD
                and time_since_line <= CORNER_PIVOT_TIMEOUT_SECONDS
            ):
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


def compute_drive_command(center_x, frame_width, base_speed, steering_gain):
    frame_center_x = frame_width // 2
    error_pixels = center_x - frame_center_x
    if abs(error_pixels) <= DEADBAND_PIXELS:
        error_pixels = 0

    normalized_error = error_pixels / (frame_width / 2.0)
    turn_delta = STEERING_SIGN * steering_gain * normalized_error

    left_speed = clamp(base_speed + turn_delta, -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(base_speed - turn_delta, -MAX_SPEED, MAX_SPEED)
    return int(left_speed), int(right_speed), error_pixels


def draw_overlay(
    frame,
    center,
    contour,
    roi_top,
    roi_bottom,
    error_pixels,
    left_speed,
    right_speed,
    motor_armed,
    controller_status,
    black_threshold,
    base_speed,
    steering_gain,
    fps,
):
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2

    cv2.rectangle(
        frame,
        (0, roi_top),
        (frame_width - 1, roi_bottom - 1),
        (255, 200, 0),
        2,
    )
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height - 1), (255, 0, 0), 2)

    if contour is not None:
        contour_on_frame = contour.copy()
        contour_on_frame[:, 0, 1] += roi_top
        cv2.drawContours(frame, [contour_on_frame], -1, (0, 255, 255), 2)

    if center is not None:
        cv2.circle(frame, center, 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Line center: ({center[0]}, {center[1]})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "Line not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"Error(px): {error_pixels:4d}  Left: {left_speed:3d}  Right: {right_speed:3d}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Mode: {'ARMED' if motor_armed else 'SAFE'}  UART: {controller_status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if motor_armed else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:4.1f}  Ctrl: {CONTROL_HZ:.0f}Hz  Threshold: {black_threshold}  Base: {base_speed}  Gain: {steering_gain:.1f}",
        (20, frame_height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "M arm/disarm  +/- threshold  [/ ] base  ,/. gain  Q quit",
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        action="store_true",
        help="Start in ARMED mode so the car begins controlling immediately.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    black_threshold = BLACK_THRESHOLD
    base_speed = BASE_SPEED
    steering_gain = STEERING_GAIN
    motor_armed = ARM_ON_START or args.arm
    fps = 0.0
    last_frame_time = time.perf_counter()

    try:
        controller = CarMotorController(SERIAL_BAUD_RATE, SERIAL_PORT_CANDIDATES)
        controller_status = f"{controller.port_name} @ {SERIAL_BAUD_RATE}"
    except Exception as exc:
        controller = None
        motor_armed = False
        controller_status = f"unavailable ({exc})"
    install_stop_handlers(controller)
    control_state = ControlState(motor_armed, base_speed, steering_gain)
    control_thread = threading.Thread(
        target=motor_control_loop,
        args=(controller, control_state),
        daemon=True,
    )
    control_thread.start()
    telemetry_logger = TelemetryLogger()
    print(f"Telemetry log: {telemetry_logger.path}", flush=True)
    debug_recorder = FrameDebugRecorder()
    print(f"Debug capture dir: {DEBUG_CAPTURE_DIR}", flush=True)
    previous_recovery_mode = "init"

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            now = time.perf_counter()
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            event_reason = None
            center, mask, roi_top, roi_bottom, contour = find_line_center(
                frame,
                black_threshold,
            )

            if center is not None:
                accepted_center = control_state.update_detection(center[0], frame.shape[1])
                if not accepted_center:
                    event_reason = "center_jump_rejected"
                    center = None
                    contour = None
                    control_state.mark_line_lost()
            else:
                control_state.mark_line_lost()

            state = control_state.snapshot()
            motor_armed = state["motor_armed"]
            base_speed = state["base_speed"]
            steering_gain = state["steering_gain"]
            left_speed = state["left_speed"]
            right_speed = state["right_speed"]
            error_pixels = state["error_pixels"]
            recovery_turn_direction = state["recovery_turn_direction"]
            time_since_line = time.perf_counter() - state["last_line_time"]
            if center is not None:
                recovery_mode = "track"
            elif abs(state["last_nonzero_error"]) >= CORNER_ERROR_THRESHOLD and time_since_line <= CORNER_PIVOT_TIMEOUT_SECONDS:
                recovery_mode = "corner_pivot"
            elif time_since_line <= RECOVERY_HOLD_SECONDS:
                recovery_mode = "hold"
            elif time_since_line <= RECOVERY_TURN_TIMEOUT_SECONDS:
                recovery_mode = "search_turn"
            else:
                recovery_mode = "stop"

            if event_reason is None and recovery_mode in ("corner_pivot", "search_turn", "stop") and recovery_mode != previous_recovery_mode:
                event_reason = recovery_mode

            draw_overlay(
                frame,
                center,
                contour,
                roi_top,
                roi_bottom,
                error_pixels,
                left_speed,
                right_speed,
                motor_armed,
                controller_status,
                black_threshold,
                base_speed,
                steering_gain,
                fps,
            )
            debug_metadata = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fps": round(fps, 2),
                "motor_armed": motor_armed,
                "center_x": None if center is None else center[0],
                "center_y": None if center is None else center[1],
                "error_pixels": error_pixels,
                "left_speed": left_speed,
                "right_speed": right_speed,
                "recovery_mode": recovery_mode,
                "recovery_turn_direction": recovery_turn_direction,
                "black_threshold": black_threshold,
                "base_speed": base_speed,
                "steering_gain": steering_gain,
            }
            telemetry_logger.log(
                fps,
                motor_armed,
                center,
                error_pixels,
                left_speed,
                right_speed,
                recovery_mode,
                recovery_turn_direction,
                black_threshold,
                base_speed,
                steering_gain,
            )
            debug_recorder.push(frame, debug_metadata)
            if event_reason is not None:
                saved_dir = debug_recorder.capture_event(event_reason, frame, mask, debug_metadata)
                if saved_dir is not None:
                    print(f"Captured debug event: {saved_dir}", flush=True)

            cv2.imshow(WINDOW_NAME, frame)
            if SHOW_MASK_WINDOW:
                cv2.imshow(MASK_WINDOW_NAME, mask)
            previous_recovery_mode = recovery_mode

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")) and controller is not None:
                motor_armed = not motor_armed
                control_state.set_motor_armed(motor_armed)
            if key in (ord("-"), ord("_")):
                black_threshold = max(0, black_threshold - 5)
            if key in (ord("="), ord("+")):
                black_threshold = min(255, black_threshold + 5)
            if key == ord("["):
                base_speed = max(0, base_speed - 2)
                control_state.update_base_speed(base_speed)
            if key == ord("]"):
                base_speed = min(MAX_SPEED, base_speed + 2)
                control_state.update_base_speed(base_speed)
            if key == ord(","):
                steering_gain = max(0.0, steering_gain - 2.0)
                control_state.update_steering_gain(steering_gain)
            if key == ord("."):
                steering_gain = min(200.0, steering_gain + 2.0)
                control_state.update_steering_gain(steering_gain)
    finally:
        control_state.stop()
        control_thread.join(timeout=0.3)
        telemetry_logger.close()
        if controller is not None:
            controller.close()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
