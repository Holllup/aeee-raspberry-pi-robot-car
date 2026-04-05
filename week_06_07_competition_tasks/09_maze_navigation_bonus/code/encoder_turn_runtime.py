import importlib.util
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


ENCODER_I2C_ADDRESS = 42
ENCODER_SHIFT = 0x80000000
ENCODER_COMMAND_FRONT = b"i0"
ENCODER_COMMAND_REAR = b"i5"
ENCODER_READ_BYTES = 8

DEFAULT_I2C_BUS = 1
DEFAULT_POLL_HZ = 20.0
DEFAULT_SPEED_SMOOTHING_WINDOW = 3

COUNTER_RESET_JUMP_THRESHOLD = 200000

TURN_SLOWDOWN_RATIO = 0.85
TURN_SLOW_SPEED_SCALE = 0.55
TURN_MIN_SPEED = 12
TURN_BRAKE_SPEED_SCALE = 0.35
TURN_BRAKE_SECONDS = 0.04
TURN_BALANCE_DEADBAND_COUNTS_PER_SEC = 4.0
TURN_BALANCE_COUNTS_PER_SEC_STEP = 8.0
TURN_MAX_BALANCE_CORRECTION = 16


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


maze = load_local_module("maze_encoder_turn_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


def clamp(value, low, high):
    return max(low, min(high, value))


def average(values):
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def mean_abs(values):
    return average([abs(value) for value in values])


def signed_encoder_count(raw_value):
    return int(int(raw_value) - ENCODER_SHIFT)


def safe_counter_delta(current, previous):
    delta = int(current) - int(previous)
    if abs(delta) >= COUNTER_RESET_JUMP_THRESHOLD:
        return 0
    return delta


def command_turn(direction, speed):
    speed = int(clamp(speed, 0, base.MAX_SPEED))
    if direction == "right":
        return speed, -speed
    return -speed, speed


def reverse_direction(direction):
    return "left" if direction == "right" else "right"


@dataclass(frozen=True)
class EncoderSample:
    timestamp: float
    counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class WheelSpeedEstimate:
    left_counts_per_sec: float
    right_counts_per_sec: float
    per_wheel_counts_per_sec: tuple[float, float, float, float]


class SMBus2Backend:
    def __init__(self, bus_index):
        try:
            from smbus2 import SMBus, i2c_msg
        except ImportError as exc:
            raise RuntimeError("smbus2 is not installed") from exc
        self._bus = SMBus(int(bus_index))
        self._i2c_msg = i2c_msg

    def transfer(self, address, write_bytes, read_len):
        write_msg = self._i2c_msg.write(int(address), list(write_bytes))
        read_msg = self._i2c_msg.read(int(address), int(read_len))
        self._bus.i2c_rdwr(write_msg, read_msg)
        return bytes(read_msg)

    def close(self):
        self._bus.close()


class I2CTransferBackend:
    def __init__(self, bus_index):
        executable = shutil.which("i2ctransfer")
        if executable is None:
            raise RuntimeError("i2ctransfer executable not found")
        self._executable = executable
        self._bus_index = int(bus_index)

    def transfer(self, address, write_bytes, read_len):
        addr_hex = f"0x{int(address):02x}"
        write_command = [
            self._executable,
            "-y",
            str(self._bus_index),
            f"w{len(write_bytes)}@{addr_hex}",
            *[f"0x{byte_value:02x}" for byte_value in write_bytes],
        ]
        subprocess.run(
            write_command,
            check=True,
            capture_output=True,
            text=True,
        )
        time.sleep(0.001)
        read_command = [
            self._executable,
            "-y",
            str(self._bus_index),
            f"r{int(read_len)}@{addr_hex}",
        ]
        result = subprocess.run(
            read_command,
            check=True,
            capture_output=True,
            text=True,
        )
        tokens = [token.strip() for token in result.stdout.split() if token.strip()]
        if len(tokens) != int(read_len):
            raise RuntimeError(
                f"unexpected i2ctransfer response length: expected {read_len}, got {len(tokens)}"
            )
        return bytes(int(token, 16) for token in tokens)

    def close(self):
        return None


class EncoderReader:
    def __init__(self, bus_index=DEFAULT_I2C_BUS, address=ENCODER_I2C_ADDRESS):
        self.address = int(address)
        self.backend_name = ""
        self._backend = self._create_backend(bus_index)

    def _create_backend(self, bus_index):
        backend_errors = []
        for name, backend_cls in (
            ("smbus2", SMBus2Backend),
            ("i2ctransfer", I2CTransferBackend),
        ):
            try:
                backend = backend_cls(bus_index)
                self.backend_name = name
                return backend
            except Exception as exc:
                backend_errors.append(f"{name}: {exc}")
        raise RuntimeError(
            "unable to initialize encoder I2C backend. "
            "Tried " + "; ".join(backend_errors)
        )

    def _read_pair(self, command_bytes):
        payload = self._backend.transfer(self.address, command_bytes, ENCODER_READ_BYTES)
        if len(payload) != ENCODER_READ_BYTES:
            raise RuntimeError(
                f"unexpected encoder payload length for command {command_bytes!r}: {len(payload)}"
            )
        first = signed_encoder_count(int.from_bytes(payload[0:4], byteorder="little", signed=False))
        second = signed_encoder_count(int.from_bytes(payload[4:8], byteorder="little", signed=False))
        return first, second

    def read_sample(self):
        front = self._read_pair(ENCODER_COMMAND_FRONT)
        rear = self._read_pair(ENCODER_COMMAND_REAR)
        return EncoderSample(
            timestamp=time.perf_counter(),
            counts=(front[0], front[1], rear[0], rear[1]),
        )

    def close(self):
        self._backend.close()


class WheelSpeedEstimator:
    def __init__(self, smoothing_window=DEFAULT_SPEED_SMOOTHING_WINDOW):
        self.smoothing_window = max(1, int(smoothing_window))
        self._last_sample = None
        self._speed_history = deque(maxlen=self.smoothing_window)

    def update(self, sample):
        if self._last_sample is None:
            self._last_sample = sample
            estimate = WheelSpeedEstimate(
                left_counts_per_sec=0.0,
                right_counts_per_sec=0.0,
                per_wheel_counts_per_sec=(0.0, 0.0, 0.0, 0.0),
            )
            self._speed_history.append(estimate.per_wheel_counts_per_sec)
            return estimate

        dt = max(1e-6, float(sample.timestamp - self._last_sample.timestamp))
        per_wheel = []
        for current_count, previous_count in zip(sample.counts, self._last_sample.counts):
            delta = safe_counter_delta(current_count, previous_count)
            per_wheel.append(float(delta) / dt)
        self._last_sample = sample
        self._speed_history.append(tuple(per_wheel))

        smoothed = []
        for index in range(4):
            smoothed.append(average([history[index] for history in self._speed_history]))

        left_speed = average([smoothed[index] for index in base.LEFT_MOTOR_INDEXES])
        right_speed = average([smoothed[index] for index in base.RIGHT_MOTOR_INDEXES])
        return WheelSpeedEstimate(
            left_counts_per_sec=float(left_speed),
            right_counts_per_sec=float(right_speed),
            per_wheel_counts_per_sec=tuple(float(value) for value in smoothed),
        )


class ClosedLoopInplaceTurnController:
    def __init__(self, action, base_speed, target_counts_right_90, target_counts_left_90):
        if action not in ("right", "left", "back"):
            raise ValueError(f"unsupported action: {action}")
        self.action = action
        self.direction = "right" if action in ("right", "back") else "left"
        ninety_target = (
            int(target_counts_right_90)
            if self.direction == "right"
            else int(target_counts_left_90)
        )
        multiplier = 2.0 if action == "back" else 1.0
        self.target_counts = max(1, int(round(float(ninety_target) * multiplier)))
        self.base_speed = int(clamp(base_speed, TURN_MIN_SPEED, base.MAX_SPEED))

    def measure_progress(self, start_sample, current_sample):
        left_progress = mean_abs(
            [
                safe_counter_delta(current_sample.counts[index], start_sample.counts[index])
                for index in base.LEFT_MOTOR_INDEXES
            ]
        )
        right_progress = mean_abs(
            [
                safe_counter_delta(current_sample.counts[index], start_sample.counts[index])
                for index in base.RIGHT_MOTOR_INDEXES
            ]
        )
        mean_progress = average((left_progress, right_progress))
        return float(left_progress), float(right_progress), float(mean_progress)

    def compute_drive(self, progress_counts, estimate):
        progress_ratio = float(progress_counts) / float(max(1, self.target_counts))
        active_speed = self.base_speed
        if progress_ratio >= TURN_SLOWDOWN_RATIO:
            active_speed = max(TURN_MIN_SPEED, int(round(self.base_speed * TURN_SLOW_SPEED_SCALE)))

        left_command, right_command = command_turn(self.direction, active_speed)
        left_speed_abs = abs(float(estimate.left_counts_per_sec))
        right_speed_abs = abs(float(estimate.right_counts_per_sec))
        speed_delta = left_speed_abs - right_speed_abs

        if abs(speed_delta) >= TURN_BALANCE_DEADBAND_COUNTS_PER_SEC:
            correction = int(
                clamp(
                    round(abs(speed_delta) / TURN_BALANCE_COUNTS_PER_SEC_STEP),
                    1,
                    TURN_MAX_BALANCE_CORRECTION,
                )
            )
            if speed_delta < 0.0:
                left_command += correction if left_command >= 0 else -correction
            else:
                right_command += correction if right_command >= 0 else -correction

        left_command = int(clamp(left_command, -base.MAX_SPEED, base.MAX_SPEED))
        right_command = int(clamp(right_command, -base.MAX_SPEED, base.MAX_SPEED))
        return left_command, right_command

    def brake_command(self):
        brake_speed = max(8, int(round(self.base_speed * TURN_BRAKE_SPEED_SCALE)))
        return command_turn(reverse_direction(self.direction), brake_speed)


def format_counts(counts):
    return " ".join(f"{count:>8d}" for count in counts)

