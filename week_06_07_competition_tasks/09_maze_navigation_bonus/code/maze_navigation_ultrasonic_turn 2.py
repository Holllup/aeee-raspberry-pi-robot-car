from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

DEFAULT_SERIAL_PORT_CANDIDATES = (
    "/dev/ttyAMA0",
    "/dev/serial0",
    "/dev/ttyAMA10",
    "/dev/ttyS0",
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class _BaseCompat:
    MAX_SPEED = 70


base = _BaseCompat()
RIGHT_REAR_BOOST = 0

MAZE_TEMPLATE_FILENAME = "Maze.png"
MAZE_TEMPLATE_CANDIDATES = ("maze.PNG", "Maze.png", "maze.png", "maze_template.png")
MAZE_TEMPLATE_MATCH_THRESHOLD = 0.52
MAZE_TEMPLATE_MARGIN = 0.03
MAZE_CONFIRMATION_HITS = 2
MAZE_COUNTDOWN_SECONDS = 5
MAZE_TEMPLATE_SIZE = (240, 240)
TEMPLATE_LABEL_CANDIDATES = {
    "MAZE": ("Maze .png", "Maze.png", "maze.PNG", "maze.png", "maze_template.png"),
    "ALARM": ("Alarm.png", "alarm_template.png"),
    "TRAFFIC": ("TrafficLight.PNG", "traffic_light_template.png"),
    "MUSIC": ("PlayAudio.png", "music_template.png"),
    "OBSTACLE": ("Obstacle Detour.png", "obstacle_template.png"),
    "FOOTBALL": ("Football.PNG", "football_template.png"),
    "SORT2": ("Color Shape Sorting 2 (1).png",),
    "SORT3": ("Color Shape Sorting 3.png",),
}
DISPLAY_LABELS = {
    "MAZE": "Maze",
    "ALARM": "Alarm",
    "TRAFFIC": "TrafficLight",
    "MUSIC": "PlayAudio",
    "OBSTACLE": "Obstacle",
    "FOOTBALL": "Football",
    "SORT2": "Sort2",
    "SORT3": "Sort3",
    "UNKNOWN": "Unknown",
}
MAZE_CAMERA_SIZE = (640, 480)
MAZE_FRAME_SLEEP_SECONDS = 0.04
SIGN_PINK_HUE_LOW = 100
SIGN_PINK_HUE_HIGH = 179
SIGN_PINK_SAT_LOW = 12
SIGN_PINK_VAL_LOW = 90
SIGN_MIN_BORDER_AREA = 5000
SIGN_MIN_ASPECT_RATIO = 0.65
SIGN_MAX_ASPECT_RATIO = 1.8
SIGN_INNER_MARGIN_RATIO = 0.08
SIGN_SYMBOL_SAT_THRESHOLD = 45
SIGN_SYMBOL_VAL_MAX = 242
TEMPLATE_CACHE: dict[str, Optional[np.ndarray]] = {}


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"quad points must be shape (4,2), got {pts.shape}")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def ensure_default_maze_template() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in MAZE_TEMPLATE_CANDIDATES:
        candidate_path = script_dir / candidate
        if candidate_path.exists():
            print(f"[template] using local maze template: {candidate_path.name}")
            return candidate_path

    template_path = script_dir / MAZE_TEMPLATE_FILENAME

    canvas = np.zeros(MAZE_TEMPLATE_SIZE, dtype=np.uint8)

    cv2.rectangle(canvas, (20, 20), (220, 220), 255, 12)
    cv2.line(canvas, (64, 74), (182, 74), 255, 8)
    cv2.line(canvas, (64, 74), (64, 90), 255, 8)
    cv2.line(canvas, (126, 74), (126, 170), 255, 8)
    cv2.line(canvas, (182, 74), (182, 148), 255, 8)
    cv2.line(canvas, (64, 108), (64, 170), 255, 8)
    cv2.line(canvas, (64, 170), (104, 170), 255, 8)
    cv2.line(canvas, (126, 170), (182, 170), 255, 8)
    cv2.rectangle(canvas, (88, 98), (118, 145), 255, -1)
    cv2.rectangle(canvas, (138, 96), (168, 145), 255, -1)
    cv2.rectangle(canvas, (104, 141), (118, 170), 255, -1)

    if not cv2.imwrite(str(template_path), canvas):
        raise RuntimeError(f"unable to write default maze template to {template_path}")
    print(f"[template] default maze template created: {template_path}")
    return template_path


def iter_template_search_dirs() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir / "templates", Path.home() / "template"]
    for parent in script_dir.parents:
        candidates.append(parent / "template")
        candidates.append(parent / "templates")

    unique_dirs = []
    seen = set()
    for directory in candidates:
        try:
            key = str(directory.resolve())
        except Exception:
            key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        if directory.exists() and directory.is_dir():
            unique_dirs.append(directory)
    return unique_dirs


def load_named_template(label: str) -> Optional[np.ndarray]:
    if label in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[label]

    if label not in TEMPLATE_LABEL_CANDIDATES:
        TEMPLATE_CACHE[label] = None
        return None

    for search_dir in iter_template_search_dirs():
        for candidate_name in TEMPLATE_LABEL_CANDIDATES[label]:
            template_path = search_dir / candidate_name
            if not template_path.exists():
                continue
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is None or template.size == 0:
                continue
            TEMPLATE_CACHE[label] = template
            print(f"[template] using {label}: {template_path}")
            return template

    if label == "MAZE":
        template_path = ensure_default_maze_template()
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is not None and template.size > 0:
            TEMPLATE_CACHE[label] = template
            return template

    searched_names = ", ".join(TEMPLATE_LABEL_CANDIDATES[label])
    print(f"[template][warn] missing {label} template in known template folders: {searched_names}")
    TEMPLATE_CACHE[label] = None
    return None


def detect_sign_roi(frame_bgr: np.ndarray) -> Optional[dict]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    magenta_mask = cv2.inRange(
        hsv,
        np.array([SIGN_PINK_HUE_LOW, SIGN_PINK_SAT_LOW, SIGN_PINK_VAL_LOW], dtype=np.uint8),
        np.array([SIGN_PINK_HUE_HIGH, 255, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    magenta_mask = cv2.morphologyEx(magenta_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
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
        if w < frame_w * 0.18 or h < frame_h * 0.18:
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
    inner_x0 = int(clamp(x + margin_x, 0, frame_w - 1))
    inner_y0 = int(clamp(y + margin_y, 0, frame_h - 1))
    inner_x1 = int(clamp(x + w - margin_x, inner_x0 + 1, frame_w))
    inner_y1 = int(clamp(y + h - margin_y, inner_y0 + 1, frame_h))
    inner_rect = (inner_x0, inner_y0, inner_x1 - inner_x0, inner_y1 - inner_y0)

    ordered_quad = _order_quad_points(best["box"])
    dst = np.array(
        [
            [0, 0],
            [MAZE_TEMPLATE_SIZE[0] - 1, 0],
            [MAZE_TEMPLATE_SIZE[0] - 1, MAZE_TEMPLATE_SIZE[1] - 1],
            [0, MAZE_TEMPLATE_SIZE[1] - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered_quad, dst)
    warped = cv2.warpPerspective(frame_bgr, matrix, MAZE_TEMPLATE_SIZE)
    return {
        "outer_rect": (x, y, w, h),
        "inner_rect": inner_rect,
        "warped": warped,
        "quad": ordered_quad.astype(np.int32),
    }


def preprocess_sign_symbol(warped_sign_bgr: np.ndarray) -> Optional[np.ndarray]:
    if warped_sign_bgr is None or warped_sign_bgr.size == 0:
        return None
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
    pink_hue_mask = ((hsv[:, :, 0] >= 112) | (hsv[:, :, 0] <= 12))
    dark_mask = gray <= SIGN_SYMBOL_VAL_MAX
    binary = np.where((sat_mask | pink_hue_mask) & dark_mask, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def compute_template_scores(binary: np.ndarray) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scores: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}

    sample = binary
    for label in TEMPLATE_LABEL_CANDIDATES:
        template_mask = load_named_template(label)
        if template_mask is None:
            continue
        sample_resized = sample
        if sample_resized.shape != template_mask.shape:
            sample_resized = cv2.resize(
                sample_resized,
                (template_mask.shape[1], template_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        xor_frame = cv2.bitwise_xor(sample_resized, template_mask)
        difference_pixels = cv2.countNonZero(xor_frame)
        match_score = 1.0 - difference_pixels / float(xor_frame.shape[0] * xor_frame.shape[1])
        scores[label] = float(match_score)
        metrics[label] = {"difference_pixels": float(difference_pixels)}

    return scores, metrics


def classify_maze_sign(sign_roi: Optional[dict]) -> dict:
    if sign_roi is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "no_sign",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": False,
        }

    roi = sign_roi.get("warped")
    if roi is None or roi.size == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "empty_roi",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": True,
        }

    binary = preprocess_sign_symbol(roi)
    if binary is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "empty_binary",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": True,
        }

    scores, metrics = compute_template_scores(binary)
    if not scores:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "missing_templates",
            "scores": {},
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": True,
        }

    best_label = "UNKNOWN"
    best_score = -1.0
    for label, score in scores.items():
        if score > best_score:
            best_score = score
            best_label = label

    maze_score = scores.get("MAZE", -1.0)
    best_other_score = max(
        [score for label, score in scores.items() if label != "MAZE"],
        default=-1.0,
    )
    best_other_label = "NONE"
    if best_other_score >= 0.0:
        for label, score in scores.items():
            if label != "MAZE" and score == best_other_score:
                best_other_label = label
                break

    display_label = best_label if best_score >= 0.35 else "UNKNOWN"
    maze_confident = (
        best_label == "MAZE"
        and maze_score >= MAZE_TEMPLATE_MATCH_THRESHOLD
        and maze_score >= best_other_score + MAZE_TEMPLATE_MARGIN
    )

    return {
        "label": "MAZE" if maze_confident else display_label,
        "confidence": max(0.0, best_score),
        "reason": "maze_confirmed" if maze_confident else f"best_match_{display_label.lower()}",
        "scores": scores,
        "metrics": metrics,
        "binary": binary,
        "maze_confident": maze_confident,
        "maze_score": maze_score,
        "best_other_label": best_other_label,
        "best_other_score": best_other_score,
        "sign_found": True,
    }


class CameraSource(Protocol):
    def capture_bgr(self) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class Picamera2Source:
    def __init__(self, size: tuple[int, int] = MAZE_CAMERA_SIZE):
        from picamera2 import Picamera2

        self._size = (int(size[0]), int(size[1]))
        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": self._size, "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(0.25)

    def capture_bgr(self) -> np.ndarray:
        frame_rgb = self._picam2.capture_array()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self._picam2.stop()


class OpenCVCameraSource:
    def __init__(self, size: tuple[int, int] = MAZE_CAMERA_SIZE):
        self._size = (int(size[0]), int(size[1]))
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        if not self._cap.isOpened():
            raise RuntimeError("unable to open camera by cv2.VideoCapture(0)")
        time.sleep(0.2)

    def capture_bgr(self) -> np.ndarray:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("camera read failed")
        return frame_bgr

    def close(self) -> None:
        self._cap.release()


def create_camera_source() -> CameraSource:
    try:
        camera = Picamera2Source(size=MAZE_CAMERA_SIZE)
        print("[template] camera source: Picamera2")
        return camera
    except Exception as exc:
        print(f"[template][warn] Picamera2 unavailable: {exc}")

    camera = OpenCVCameraSource(size=MAZE_CAMERA_SIZE)
    print("[template] camera source: OpenCV VideoCapture")
    return camera


def wait_for_maze_template_start(
    display: DisplayModule,
    hold_stop: Callable[[], None],
    stop: Callable[[], None],
) -> None:
    ensure_default_maze_template()
    camera = create_camera_source()
    maze_hits = 0
    last_score = -1.0
    last_hit_report = -1

    print("[template] waiting for MAZE template...")
    display.show_status("Wait Template")
    try:
        while True:
            frame_bgr = camera.capture_bgr()
            sign_roi = detect_sign_roi(frame_bgr)
            sign_result = classify_maze_sign(sign_roi)
            score = float(sign_result.get("maze_score", -1.0))

            if sign_result.get("maze_confident", False):
                maze_hits += 1
            else:
                maze_hits = 0

            hold_stop()

            if abs(score - last_score) >= 0.02 or maze_hits != last_hit_report:
                current_label = str(sign_result.get("label", "UNKNOWN"))
                best_other_label = str(sign_result.get("best_other_label", "NONE"))
                best_other_score = float(sign_result.get("best_other_score", -1.0))
                print(
                    "[template] "
                    f"label={current_label} "
                    f"score={score:.3f} confident={int(bool(sign_result.get('maze_confident', False)))} "
                    f"other={best_other_label}:{best_other_score:.3f} "
                    f"hits={maze_hits}/{MAZE_CONFIRMATION_HITS}"
                )
                last_score = score
                last_hit_report = maze_hits

            if maze_hits > 0:
                display.show_status(f"Maze {maze_hits}/{MAZE_CONFIRMATION_HITS}")
            else:
                display_label = str(sign_result.get("label", "UNKNOWN"))
                display.show_status(DISPLAY_LABELS.get(display_label, display_label[:16]))

            if maze_hits >= MAZE_CONFIRMATION_HITS:
                stop()
                deadline = time.perf_counter() + MAZE_COUNTDOWN_SECONDS
                while True:
                    now = time.perf_counter()
                    remaining = max(0, int(math.ceil(deadline - now)))
                    display.show_status(f"Maze {remaining}s")
                    hold_stop()
                    if now >= deadline:
                        break
                    time.sleep(0.10)
                stop()
                print("[template] maze template confirmed, start navigation")
                return

            time.sleep(MAZE_FRAME_SLEEP_SECONDS)
    finally:
        camera.close()


class TurnModule(Protocol):
    def turn_left_90(self) -> None:
        ...

    def turn_right_90(self) -> None:
        ...

    def turn_back_180(self) -> None:
        ...

    def close(self) -> None:
        ...


class DistanceModule(Protocol):
    def measure_distance_cm(self) -> Optional[float]:
        ...

    def measure_distance_filtered(
        self,
        sample_count: int,
        sample_interval_s: float,
        min_valid_cm: float,
        max_valid_cm: float,
    ) -> Optional[float]:
        ...

    def close(self) -> None:
        ...


class MotionModule(Protocol):
    def move_one_cell(self, left_speed: int, right_speed: int, seconds: float) -> None:
        ...

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...


class DisplayModule(Protocol):
    def show_maze_distance(
        self, direction: str, distance_cm: Optional[float], tag: str = ""
    ) -> None:
        ...

    def show_status(self, status_text: str) -> None:
        ...

    def close(self) -> None:
        ...


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
LCD_REFRESH_SECONDS = 0.10


@dataclass
class MazeConfig:
    cell_move_seconds: float = 1.50
    cell_left_speed: int = 36
    cell_right_speed: int = 32
    pre_front_forward_cooldown_moves: int = 3
    start_forward_seconds: float = 1.1
    start_forward_left_speed: int = 22
    start_forward_right_speed: int = 20
    scan_settle_seconds: float = 0.60
    turn_settle_seconds: float = 1.5
    stop_between_stages_seconds: float = 0.60
    d_open_cm: float = 22.0
    d_block_cm: float = 22.0
    d_emergency_cm: float = 10.0
    exit_front_min_cm: float = 55.0
    exit_confirm_count: int = 3
    max_steps: int = 300
    scan_samples: int = 9
    scan_interval_seconds: float = 0.06
    valid_distance_min_cm: float = 2.0
    valid_distance_max_cm: float = 300.0
    start_delay_seconds: float = 1.0


@dataclass
class ScanResult:
    left_cm: Optional[float] = None
    right_cm: Optional[float] = None
    front_cm: Optional[float] = None


@dataclass
class ScanOutcome:
    action: str
    interrupted: bool
    scan: ScanResult


@dataclass
class MazeRunSummary:
    status: str
    steps: int
    exit_open_streak: int
    finished_reason: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "steps": self.steps,
            "exit_open_streak": self.exit_open_streak,
            "finished_reason": self.finished_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


class UartDriveController:
    def __init__(
        self,
        baud_rate: int = 57600,
        timeout_seconds: float = 0.1,
        serial_port_candidates: tuple[str, ...] = DEFAULT_SERIAL_PORT_CANDIDATES,
        stop_burst_repeats: int = 4,
        stop_burst_pause_s: float = 0.01,
    ):
        import serial

        self._serial_module = serial
        self.baud_rate = int(baud_rate)
        self.timeout_seconds = float(timeout_seconds)
        self.serial_port_candidates = tuple(serial_port_candidates)
        self.stop_burst_repeats = max(1, int(stop_burst_repeats))
        self.stop_burst_pause_s = max(0.0, float(stop_burst_pause_s))
        self.serial = self._open_serial_port()
        self.last_payload: Optional[bytes] = None

    def _open_serial_port(self):
        attempted_ports = []
        for port in self.serial_port_candidates:
            if not Path(port).exists():
                continue
            attempted_ports.append(port)
            try:
                conn = self._serial_module.Serial(
                    port=port,
                    baudrate=self.baud_rate,
                    timeout=self.timeout_seconds,
                    write_timeout=self.timeout_seconds,
                )
                conn.reset_input_buffer()
                conn.reset_output_buffer()
                return conn
            except Exception:
                continue
        if attempted_ports:
            raise RuntimeError(f"unable to open serial port from {', '.join(attempted_ports)}")
        raise RuntimeError(
            "no serial port found in candidates: " + ", ".join(self.serial_port_candidates)
        )

    def _write_payload(self, payload: bytes, force: bool = False) -> None:
        if not force and payload == self.last_payload:
            return
        self.serial.write(payload)
        self.serial.flush()
        self.last_payload = bytes(payload)

    def hold_stop(self) -> None:
        self._write_payload(b"#ha", force=True)
        self.last_payload = b"#ha"

    def stop(self) -> None:
        for _ in range(self.stop_burst_repeats):
            self._write_payload(b"#ha", force=True)
            time.sleep(self.stop_burst_pause_s)
        self.last_payload = b"#ha"

    def set_tank_drive(self, left_speed: int, right_speed: int, straight_mode: bool = True) -> None:
        _ = straight_mode
        per_motor = [int(right_speed), int(right_speed), int(left_speed), int(left_speed)]
        payload = bytearray(b"#ba")
        for speed in per_motor:
            payload.extend((b"f" if speed >= 0 else b"r"))
        for speed in per_motor:
            magnitude = int(clamp(abs(speed), 0, 65535))
            payload.append(magnitude & 0xFF)
            payload.append((magnitude >> 8) & 0xFF)
        self._write_payload(bytes(payload))

    def move_one_cell(self, left_speed: int, right_speed: int, seconds: float) -> None:
        self.stop()
        self.set_tank_drive(left_speed=left_speed, right_speed=right_speed, straight_mode=False)
        time.sleep(max(0.0, float(seconds)))
        self.stop()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.serial.close()


class TrimmedCarMotorController:
    def __init__(self, right_rear_boost: int = 0, right_rear_spin_boost: int = 0):
        self.drive = UartDriveController()
        self.right_rear_boost = int(right_rear_boost)
        self.right_rear_spin_boost = int(right_rear_spin_boost)

    @staticmethod
    def _norm(value: int) -> int:
        return int(clamp(int(value), -base.MAX_SPEED, base.MAX_SPEED))

    def _apply_trim(self, left_speed: int, right_speed: int) -> tuple[int, int]:
        left = self._norm(left_speed)
        right = self._norm(right_speed)
        if right > 0:
            right = self._norm(right + self.right_rear_boost)
        elif right < 0:
            right = self._norm(right - self.right_rear_spin_boost)
        return left, right

    def set_tank_drive(self, left_speed: int, right_speed: int, straight_mode: bool = True) -> None:
        left, right = self._apply_trim(left_speed, right_speed)
        self.drive.set_tank_drive(left_speed=left, right_speed=right, straight_mode=straight_mode)

    def hold_stop(self) -> None:
        self.drive.hold_stop()

    def stop(self) -> None:
        self.drive.stop()

    def close(self) -> None:
        self.drive.close()


def install_stop_handlers(controller, _display) -> None:
    def _handler(signum, _frame):
        print(f"[STOP] signal={signum}, stopping vehicle")
        try:
            controller.stop()
        finally:
            controller.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class UartTurnAdapter:
    def __init__(
        self,
        drive_controller: UartDriveController,
        config_path: Path,
        turn_settle_seconds: float,
    ):
        self.turn_settle_seconds = max(0.0, float(turn_settle_seconds))
        config = load_or_create_config(config_path)
        self.turn_controller = InPlaceTurnController(config=config, serial_conn=drive_controller.serial)

    def _settle(self) -> None:
        if self.turn_settle_seconds > 0.0:
            time.sleep(self.turn_settle_seconds)

    def turn_left_90(self) -> None:
        self.turn_controller.turn_left_90()
        self._settle()

    def turn_right_90(self) -> None:
        self.turn_controller.turn_right_90()
        self._settle()

    def turn_back_180(self) -> None:
        self.turn_controller.turn_right_90()
        self._settle()
        self.turn_controller.turn_right_90()
        self._settle()

    def close(self) -> None:
        self.turn_controller.close()


class Hcsr04DistanceAdapter:
    def __init__(
        self,
        trig_pin: int = 21,
        echo_pin: int = 20,
        echo_wait_timeout_s: float = 0.03,
        pulse_timeout_s: float = 0.03,
    ):
        import RPi.GPIO as GPIO

        self.GPIO = GPIO
        self.trig_pin = int(trig_pin)
        self.echo_pin = int(echo_pin)
        self.echo_wait_timeout_s = float(echo_wait_timeout_s)
        self.pulse_timeout_s = float(pulse_timeout_s)

        self.GPIO.setwarnings(False)
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setup(self.trig_pin, self.GPIO.OUT)
        self.GPIO.setup(self.echo_pin, self.GPIO.IN)
        self.GPIO.output(self.trig_pin, False)
        time.sleep(0.05)

    def measure_distance_cm(self) -> Optional[float]:
        self.GPIO.output(self.trig_pin, False)
        time.sleep(0.000002)
        self.GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        self.GPIO.output(self.trig_pin, False)

        wait_start = time.monotonic()
        while self.GPIO.input(self.echo_pin) == 0:
            if (time.monotonic() - wait_start) > self.echo_wait_timeout_s:
                return None
        pulse_start = time.monotonic()

        while self.GPIO.input(self.echo_pin) == 1:
            if (time.monotonic() - pulse_start) > self.pulse_timeout_s:
                return None
        pulse_end = time.monotonic()

        pulse_seconds = pulse_end - pulse_start
        if pulse_seconds <= 0:
            return None
        return pulse_seconds * 17150.0

    def measure_distance_filtered(
        self,
        sample_count: int,
        sample_interval_s: float,
        min_valid_cm: float,
        max_valid_cm: float,
    ) -> Optional[float]:
        values = []
        repeats = max(1, int(sample_count))
        for _ in range(repeats):
            value = self.measure_distance_cm()
            if value is not None and min_valid_cm <= value <= max_valid_cm:
                values.append(float(value))
            time.sleep(max(0.0, float(sample_interval_s)))
        if not values:
            return None
        return float(statistics.median(values))

    def close(self) -> None:
        self.GPIO.cleanup((self.trig_pin, self.echo_pin))


class SimTurnAdapter:
    def turn_left_90(self) -> None:
        print("[SIM] turn_left_90")

    def turn_right_90(self) -> None:
        print("[SIM] turn_right_90")

    def turn_back_180(self) -> None:
        print("[SIM] turn_back_180")

    def close(self) -> None:
        return None


class SimDistanceAdapter:
    def __init__(self, scripted_values: list[float]):
        self.scripted_values = scripted_values[:]
        self._cursor = 0

    def _next_value(self) -> float:
        if not self.scripted_values:
            return 40.0
        value = self.scripted_values[self._cursor % len(self.scripted_values)]
        self._cursor += 1
        return float(value)

    def measure_distance_cm(self) -> Optional[float]:
        return self._next_value()

    def measure_distance_filtered(
        self,
        sample_count: int,
        sample_interval_s: float,
        min_valid_cm: float,
        max_valid_cm: float,
    ) -> Optional[float]:
        _ = sample_interval_s
        values = []
        for _ in range(max(1, int(sample_count))):
            value = self._next_value()
            if min_valid_cm <= value <= max_valid_cm:
                values.append(value)
        if not values:
            return None
        return float(statistics.median(values))

    def close(self) -> None:
        return None


class SimMotionAdapter:
    def stop(self) -> None:
        print("[SIM] stop")

    def move_one_cell(self, left_speed: int, right_speed: int, seconds: float) -> None:
        print(
            f"[SIM] move_one_cell left_speed={left_speed} "
            f"right_speed={right_speed} seconds={seconds:.2f}"
        )
        time.sleep(min(0.1, max(0.0, float(seconds))))

    def close(self) -> None:
        return None


class ForwardTurnBasicController:
    # Keep this table consistent with:
    # 小车使用文档/车辆控制使用文档/uart_forward_then_turn_basic.py
    ACTION_TABLE = {
        "left": {
            "seconds": 0.44,
            "left_speed": -52,
            "right_speed": 52,
        },
        "back": {
            "seconds": 1.00,
            "left_speed": 52,
            "right_speed": -52,
        },
    }

    def __init__(self, controller: TrimmedCarMotorController, stop_between_stages_seconds: float):
        self.controller = controller
        self.stop_between_stages_seconds = max(0.0, float(stop_between_stages_seconds))
        self.pause_hook: Optional[Callable[[float], None]] = None

    @staticmethod
    def normalize_speed(value: int) -> int:
        return int(clamp(int(value), -base.MAX_SPEED, base.MAX_SPEED))

    def run_tank_motion(self, left_speed: int, right_speed: int, seconds: float) -> None:
        left_speed = self.normalize_speed(left_speed)
        right_speed = self.normalize_speed(right_speed)
        if seconds <= 0.0:
            return
        self.controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
        time.sleep(seconds)

    def stop_for(self, seconds: float) -> None:
        self.controller.hold_stop()
        if seconds > 0.0:
            if self.pause_hook is not None:
                self.pause_hook(seconds)
            else:
                time.sleep(seconds)

    def set_pause_hook(self, pause_hook: Optional[Callable[[float], None]]) -> None:
        self.pause_hook = pause_hook

    def load_turn_action(self, action_name: str) -> tuple[float, int, int]:
        if action_name not in ("left", "back"):
            raise ValueError(f"unsupported action: {action_name}")
        if action_name not in self.ACTION_TABLE:
            raise ValueError(f"missing action config: {action_name}")
        action_config = self.ACTION_TABLE[action_name]
        seconds = float(action_config["seconds"])
        left_speed = self.normalize_speed(action_config["left_speed"])
        right_speed = self.normalize_speed(action_config["right_speed"])
        return seconds, left_speed, right_speed

    def run_named_turn(self, action_name: str) -> None:
        turn_seconds, turn_left_speed, turn_right_speed = self.load_turn_action(action_name)
        print(
            f"[forward-turn-loop] action={action_name} seconds={turn_seconds:.3f} "
            f"left_speed={turn_left_speed} right_speed={turn_right_speed}"
        )
        self.run_tank_motion(turn_left_speed, turn_right_speed, turn_seconds)
        self.controller.stop()
        self.stop_for(self.stop_between_stages_seconds)

    # TurnModule
    def turn_left_90(self) -> None:
        self.run_named_turn("left")

    def turn_back_180(self) -> None:
        self.run_named_turn("back")

    def turn_right_90(self) -> None:
        # In forward_then_turn_basic mapping: right ~= left + back.
        self.run_named_turn("left")
        self.run_named_turn("back")

    # MotionModule
    def move_one_cell(self, left_speed: int, right_speed: int, seconds: float) -> None:
        left_speed = self.normalize_speed(left_speed)
        right_speed = self.normalize_speed(right_speed)
        print(
            f"[forward-turn-loop] forward_seconds={seconds:.3f} "
            f"forward_left={left_speed} forward_right={right_speed}"
        )
        self.run_tank_motion(left_speed, right_speed, seconds)
        self.controller.stop()
        self.stop_for(self.stop_between_stages_seconds)

    def stop(self) -> None:
        self.controller.stop()

    def close(self) -> None:
        self.controller.close()


class NullDisplay:
    def show_maze_distance(
        self, direction: str, distance_cm: Optional[float], tag: str = ""
    ) -> None:
        _ = (direction, distance_cm, tag)

    def show_status(self, status_text: str) -> None:
        _ = status_text

    def close(self) -> None:
        return None


class SimDisplay:
    def show_maze_distance(
        self, direction: str, distance_cm: Optional[float], tag: str = ""
    ) -> None:
        suffix = f" {tag}" if tag else ""
        if distance_cm is None:
            print(f"[SIM_LCD] Maze Module | {direction}:NoEcho{suffix}")
        else:
            print(f"[SIM_LCD] Maze Module | {direction}:{distance_cm:.1f}cm{suffix}")

    def show_status(self, status_text: str) -> None:
        print(f"[SIM_LCD] Maze Module | {status_text}")

    def close(self) -> None:
        print("[SIM_LCD] Maze Module | Closed")


class LCD1602MazeDisplay:
    def __init__(self):
        import RPi.GPIO as GPIO

        self.GPIO = GPIO
        self.last_lines = ("", "")
        self.last_update = 0.0

        self.GPIO.setwarnings(False)
        self.GPIO.setmode(self.GPIO.BCM)
        for pin in (LCD_E, LCD_RS, LCD_D4, LCD_D5, LCD_D6, LCD_D7):
            self.GPIO.setup(pin, self.GPIO.OUT)
            self.GPIO.output(pin, False)

        self._send_byte(0x33, LCD_CMD)
        self._send_byte(0x32, LCD_CMD)
        self._send_byte(0x28, LCD_CMD)
        self._send_byte(0x0C, LCD_CMD)
        self._send_byte(0x06, LCD_CMD)
        self._send_byte(0x01, LCD_CMD)
        time.sleep(LCD_E_DELAY)
        self.show_status("Init")

    def _toggle_enable(self) -> None:
        time.sleep(LCD_E_DELAY)
        self.GPIO.output(LCD_E, True)
        time.sleep(LCD_E_PULSE)
        self.GPIO.output(LCD_E, False)
        time.sleep(LCD_E_DELAY)

    def _send_byte(self, bits: int, mode: bool) -> None:
        self.GPIO.output(LCD_RS, mode)

        self.GPIO.output(LCD_D4, bool(bits & 0x10))
        self.GPIO.output(LCD_D5, bool(bits & 0x20))
        self.GPIO.output(LCD_D6, bool(bits & 0x40))
        self.GPIO.output(LCD_D7, bool(bits & 0x80))
        self._toggle_enable()

        self.GPIO.output(LCD_D4, bool(bits & 0x01))
        self.GPIO.output(LCD_D5, bool(bits & 0x02))
        self.GPIO.output(LCD_D6, bool(bits & 0x04))
        self.GPIO.output(LCD_D7, bool(bits & 0x08))
        self._toggle_enable()

    def _write_line(self, message: str, line_address: int) -> None:
        self._send_byte(line_address, LCD_CMD)
        text = message.ljust(LCD_WIDTH)[:LCD_WIDTH]
        for ch in text:
            self._send_byte(ord(ch), LCD_CHR)

    def _update(self, line_1: str, line_2: str, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self.last_update) < LCD_REFRESH_SECONDS:
            return
        lines = (line_1[:LCD_WIDTH], line_2[:LCD_WIDTH])
        if force or lines != self.last_lines:
            self._write_line(lines[0], LCD_LINE_1)
            self._write_line(lines[1], LCD_LINE_2)
            self.last_lines = lines
        self.last_update = now

    def show_maze_distance(
        self, direction: str, distance_cm: Optional[float], tag: str = ""
    ) -> None:
        if distance_cm is None:
            line_2 = f"{direction}: ---- {tag}".strip()
        else:
            line_2 = f"{direction}:{distance_cm:5.1f} {tag}".strip()
        self._update("Maze Dist", line_2, force=(tag == "LIVE"))

    def show_status(self, status_text: str) -> None:
        self._update("Maze Dist", status_text, force=True)

    def close(self) -> None:
        try:
            self.show_status("Stopped")
            time.sleep(0.2)
        finally:
            self.GPIO.cleanup((LCD_E, LCD_RS, LCD_D4, LCD_D5, LCD_D6, LCD_D7))


class MazeNavigator:
    def __init__(
        self,
        config: MazeConfig,
        turn: TurnModule,
        distance: DistanceModule,
        motion: MotionModule,
        display: DisplayModule,
    ):
        self.config = config
        self.turn = turn
        self.distance = distance
        self.motion = motion
        self.display = display
        self.state = "INIT"
        self.steps = 0
        self.exit_open_streak = 0
        self.pre_front_cooldown_remaining = 0
        self.started_at = time.perf_counter()
        if hasattr(self.motion, "set_pause_hook"):
            self.motion.set_pause_hook(self._live_pause_measurement)

    def _log(self, message: str) -> None:
        elapsed = time.perf_counter() - self.started_at
        print(f"[{elapsed:7.3f}s][{self.state}] {message}")

    @staticmethod
    def _fmt_cm(value: Optional[float]) -> str:
        if value is None:
            return "None"
        return f"{value:.1f}cm"

    def _distance_label(self, value_cm: Optional[float]) -> str:
        if value_cm is None:
            return "open"
        if value_cm <= self.config.d_block_cm:
            return "blocked"
        if value_cm >= self.config.d_open_cm:
            return "open"
        return "unknown"

    def _is_open(self, value_cm: Optional[float]) -> bool:
        # Treat "no echo" as far enough / open in this maze setup.
        return value_cm is None or value_cm >= self.config.d_open_cm

    def _scan_one(self, direction: str) -> Optional[float]:
        values = []
        repeats = max(1, int(self.config.scan_samples))
        for _ in range(repeats):
            value = self.distance.measure_distance_cm()
            if value is None:
                self.display.show_maze_distance(direction, None, "LIVE")
            else:
                self.display.show_maze_distance(direction, float(value), "LIVE")
            if (
                value is not None
                and self.config.valid_distance_min_cm <= value <= self.config.valid_distance_max_cm
            ):
                values.append(float(value))
            time.sleep(max(0.0, float(self.config.scan_interval_seconds)))
        if not values:
            self.display.show_maze_distance(direction, None, "OPEN")
            return None
        filtered = float(statistics.median(values))
        label = self._distance_label(filtered)
        if label == "open":
            tag = "OPEN"
        elif label == "blocked":
            tag = "BLK"
        else:
            tag = "UNK"
        self.display.show_maze_distance(direction, filtered, tag)
        return filtered

    def _live_pause_measurement(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        poll_interval = max(0.03, min(0.08, float(self.config.scan_interval_seconds)))
        end_time = time.perf_counter() + seconds
        while True:
            remaining = end_time - time.perf_counter()
            if remaining <= 0.0:
                break
            value = self.distance.measure_distance_cm()
            if value is None:
                self.display.show_maze_distance("F", None, "LIVE")
            else:
                self.display.show_maze_distance("F", float(value), "LIVE")
            time.sleep(min(poll_interval, max(0.0, remaining)))

    def _pause_between_stage(self) -> None:
        if self.config.stop_between_stages_seconds > 0.0:
            self._live_pause_measurement(self.config.stop_between_stages_seconds)

    def _move_one_cell(self, action: str) -> None:
        self.state = "EXECUTE"
        self.motion.move_one_cell(
            left_speed=self.config.cell_left_speed,
            right_speed=self.config.cell_right_speed,
            seconds=self.config.cell_move_seconds,
        )
        self.steps += 1
        if self.pre_front_cooldown_remaining > 0:
            self.pre_front_cooldown_remaining -= 1
        self._log(f"action={action} steps={self.steps}")

    def _scan_with_interrupt(self) -> ScanOutcome:
        self.state = "AT_CELL_SCAN"
        self.motion.stop()
        self._live_pause_measurement(self.config.scan_settle_seconds)
        scan = ScanResult()

        # 0) 每次准备左转前，先直接看前方；但直行直接过触发后，接下来若干步禁用此捷径。
        if self.pre_front_cooldown_remaining <= 0:
            scan.front_cm = self._scan_one("F")
            self._log(
                f"pre_front={self._fmt_cm(scan.front_cm)} label={self._distance_label(scan.front_cm)}"
            )
            if self._is_open(scan.front_cm):
                self.pre_front_cooldown_remaining = max(
                    0, int(self.config.pre_front_forward_cooldown_moves)
                )
                self._move_one_cell(action="forward")
                return ScanOutcome(action="forward", interrupted=True, scan=scan)
        else:
            self._log(
                "pre_front_skipped "
                f"cooldown_remaining={self.pre_front_cooldown_remaining}"
            )

        # 1) 左转检测左边，若可走立即打断扫描并前进。
        self.turn.turn_left_90()
        scan.left_cm = self._scan_one("L")
        self._log(
            f"left={self._fmt_cm(scan.left_cm)} label={self._distance_label(scan.left_cm)}"
        )
        if self._is_open(scan.left_cm):
            self._move_one_cell(action="left")
            return ScanOutcome(action="left", interrupted=True, scan=scan)

        # 2) 再 180 度检测右边，若可走立即打断扫描并前进。
        self.turn.turn_back_180()
        scan.right_cm = self._scan_one("R")
        self._log(
            f"right={self._fmt_cm(scan.right_cm)} label={self._distance_label(scan.right_cm)}"
        )
        if self._is_open(scan.right_cm):
            self._move_one_cell(action="right")
            return ScanOutcome(action="right", interrupted=True, scan=scan)

        # 3) 回 90 度回到前方，检测前方，若可走立即前进。
        self.turn.turn_left_90()
        scan.front_cm = self._scan_one("F")
        self._log(
            f"front={self._fmt_cm(scan.front_cm)} label={self._distance_label(scan.front_cm)}"
        )
        if self._is_open(scan.front_cm):
            self._move_one_cell(action="forward")
            return ScanOutcome(action="forward", interrupted=True, scan=scan)

        return ScanOutcome(action="blocked", interrupted=False, scan=scan)

    def _update_exit_streak(self, outcome: ScanOutcome) -> None:
        if (
            outcome.action == "forward"
            and (
                outcome.scan.front_cm is None
                or outcome.scan.front_cm >= self.config.exit_front_min_cm
            )
        ):
            self.exit_open_streak += 1
        else:
            self.exit_open_streak = 0

    def run(self) -> MazeRunSummary:
        self.state = "INIT"
        self.motion.stop()
        self.display.show_status("Maze Ready")
        if self.config.start_delay_seconds > 0:
            self._log(f"start_delay={self.config.start_delay_seconds:.2f}s")
            time.sleep(self.config.start_delay_seconds)
        if self.config.start_forward_seconds > 0:
            self._log(
                "start_forward "
                f"seconds={self.config.start_forward_seconds:.2f} "
                f"left={self.config.start_forward_left_speed} "
                f"right={self.config.start_forward_right_speed}"
            )
            self.motion.move_one_cell(
                self.config.start_forward_left_speed,
                self.config.start_forward_right_speed,
                self.config.start_forward_seconds,
            )

        status = "FAILSAFE"
        reason = "unknown"
        try:
            while True:
                if self.steps >= self.config.max_steps:
                    reason = f"max_steps_reached({self.config.max_steps})"
                    break

                outcome = self._scan_with_interrupt()
                self.state = "CHECK_EXIT"
                self._update_exit_streak(outcome)
                self._log(
                    "scan(left,right,front)=("
                    + f"{self._fmt_cm(outcome.scan.left_cm)},"
                    + f"{self._fmt_cm(outcome.scan.right_cm)},"
                    + f"{self._fmt_cm(outcome.scan.front_cm)}) "
                    + f"action={outcome.action} interrupted={int(outcome.interrupted)} "
                    + f"exit_streak={self.exit_open_streak}/{self.config.exit_confirm_count}"
                )

                if self.exit_open_streak >= self.config.exit_confirm_count:
                    status = "DONE"
                    reason = f"exit_confirmed({self.exit_open_streak})"
                    break

                if outcome.action == "blocked":
                    self.state = "DECIDE"
                    self._log("all directions blocked, execute backtrack")
                    self.turn.turn_back_180()
                    self._move_one_cell(action="backtrack")

        except KeyboardInterrupt:
            reason = "keyboard_interrupt"
            status = "FAILSAFE"
        except Exception as exc:
            reason = f"exception:{exc}"
            status = "FAILSAFE"
            self._log(f"exception={exc}")
        finally:
            self.state = "DONE" if status == "DONE" else "FAILSAFE"
            self.motion.stop()
            self.display.show_status("Maze Done" if status == "DONE" else "Maze Stop")

        summary = MazeRunSummary(
            status=status,
            steps=self.steps,
            exit_open_streak=self.exit_open_streak,
            finished_reason=reason,
            elapsed_seconds=(time.perf_counter() - self.started_at),
        )
        self._log("summary " + json.dumps(summary.to_dict(), ensure_ascii=True, sort_keys=True))
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Task9 maze navigation using fixed cell length + ultrasonic scan "
            "(left->right->front with interrupt move)."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Run without hardware using simulation adapters.")
    parser.add_argument("--cell-move-seconds", type=float, default=1.16)
    parser.add_argument("--cell-speed", type=int, default=None, help="Set both left/right speed together.")
    parser.add_argument("--cell-left-speed", type=int, default=37)
    parser.add_argument("--cell-right-speed", type=int, default=32)
    parser.add_argument("--pre-front-forward-cooldown-moves", type=int, default=3)
    parser.add_argument("--scan-settle-seconds", type=float, default=0.60)
    parser.add_argument("--turn-settle-seconds", type=float, default=0.35)
    parser.add_argument("--stop-between-stages-seconds", type=float, default=0.60)
    parser.add_argument("--d-open-cm", type=float, default=25.0)
    parser.add_argument("--d-block-cm", type=float, default=24.0)
    parser.add_argument("--d-emergency-cm", type=float, default=10.0)
    parser.add_argument("--exit-front-min-cm", type=float, default=55.0)
    parser.add_argument("--exit-confirm-count", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--scan-samples", type=int, default=9)
    parser.add_argument("--scan-interval-seconds", type=float, default=0.06)
    parser.add_argument("--valid-distance-min-cm", type=float, default=2.0)
    parser.add_argument("--valid-distance-max-cm", type=float, default=300.0)
    parser.add_argument("--start-delay-seconds", type=float, default=1.0)
    parser.add_argument("--trig-pin", type=int, default=21, help="HC-SR04 trig GPIO in BCM numbering.")
    parser.add_argument("--echo-pin", type=int, default=20, help="HC-SR04 echo GPIO in BCM numbering.")
    parser.add_argument(
        "--turn-config",
        type=Path,
        default=Path("turn_controller_config.json"),
        help="In-place turn config path used by in_place_turn_controller.py",
    )
    parser.add_argument(
        "--sim-distances",
        type=str,
        default="40,40,40",
        help="Comma-separated distance stream for dry-run simulation (cm).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to save run summary json.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> MazeConfig:
    if args.cell_speed is not None:
        left_speed = int(args.cell_speed)
        right_speed = int(args.cell_speed)
    else:
        left_speed = 34
        right_speed = 31
    if args.cell_left_speed is not None:
        left_speed = int(args.cell_left_speed)
    if args.cell_right_speed is not None:
        right_speed = int(args.cell_right_speed)

    return MazeConfig(
        cell_move_seconds=float(args.cell_move_seconds),
        cell_left_speed=int(clamp(left_speed, -base.MAX_SPEED, base.MAX_SPEED)),
        cell_right_speed=int(clamp(right_speed, -base.MAX_SPEED, base.MAX_SPEED)),
        pre_front_forward_cooldown_moves=max(0, int(args.pre_front_forward_cooldown_moves)),
        scan_settle_seconds=float(args.scan_settle_seconds),
        turn_settle_seconds=float(args.turn_settle_seconds),
        stop_between_stages_seconds=max(0.0, float(args.stop_between_stages_seconds)),
        d_open_cm=float(args.d_open_cm),
        d_block_cm=float(args.d_block_cm),
        d_emergency_cm=float(args.d_emergency_cm),
        exit_front_min_cm=float(args.exit_front_min_cm),
        exit_confirm_count=max(1, int(args.exit_confirm_count)),
        max_steps=max(1, int(args.max_steps)),
        scan_samples=max(1, int(args.scan_samples)),
        scan_interval_seconds=max(0.0, float(args.scan_interval_seconds)),
        valid_distance_min_cm=float(args.valid_distance_min_cm),
        valid_distance_max_cm=float(args.valid_distance_max_cm),
        start_delay_seconds=max(0.0, float(args.start_delay_seconds)),
    )


def main() -> int:
    args = parse_args()
    config = make_config(args)

    turn: TurnModule
    distance: DistanceModule
    motion: MotionModule
    display: DisplayModule
    controller: Optional[TrimmedCarMotorController] = None

    if args.dry_run:
        sim_values = []
        for token in args.sim_distances.split(","):
            token = token.strip()
            if token:
                sim_values.append(float(token))
        turn = SimTurnAdapter()
        distance = SimDistanceAdapter(sim_values)
        motion = SimMotionAdapter()
        display = SimDisplay()
    else:
        controller = TrimmedCarMotorController(
            right_rear_boost=RIGHT_REAR_BOOST,
            right_rear_spin_boost=0,
        )
        install_stop_handlers(controller, None)

        turn_motion = ForwardTurnBasicController(
            controller=controller,
            stop_between_stages_seconds=config.stop_between_stages_seconds,
        )
        turn = turn_motion
        distance = Hcsr04DistanceAdapter(
            trig_pin=args.trig_pin,
            echo_pin=args.echo_pin,
        )
        motion = turn_motion
        try:
            display = LCD1602MazeDisplay()
        except Exception as exc:
            print(f"[WARN] LCD init failed, continue without LCD: {exc}")
            display = NullDisplay()

    summary: Optional[MazeRunSummary] = None
    try:
        if (not args.dry_run) and (controller is not None):
            wait_for_maze_template_start(
                display=display,
                hold_stop=controller.hold_stop,
                stop=controller.stop,
            )
            # Startup countdown already gives placement time.
            config.start_delay_seconds = 0.0

        navigator = MazeNavigator(
            config=config,
            turn=turn,
            distance=distance,
            motion=motion,
            display=display,
        )
        summary = navigator.run()

        if args.summary_json is not None:
            args.summary_json.write_text(
                json.dumps(summary.to_dict(), ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        try:
            turn.close()
        except Exception:
            pass
        try:
            distance.close()
        except Exception:
            pass
        try:
            motion.close()
        except Exception:
            pass
        try:
            display.close()
        except Exception:
            pass

    if summary is None:
        return 1
    return 0 if summary.status == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
