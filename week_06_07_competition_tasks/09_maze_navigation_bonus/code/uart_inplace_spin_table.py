import argparse
import importlib.util
import time
from pathlib import Path


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


maze = load_local_module("maze_uart_table_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


# ----------------------------
# 调参区：以后只改这里
# 轮子顺序固定为：[右前, 右后, 左前, 左后]
# 正数=前进，负数=后退，0=该轮停转
# 如果右后轮慢，就改 speeds[1]
# 如果右转过头，就改 right 的 seconds
# 如果不是原地转，就分别改四个轮子的速度值
# ----------------------------

START_DELAY = 1.0

ACTION_TABLE = {
    "right": {
        "seconds": 0.80,
        "speeds": [-48, -56, 48, 48],
    },
    "left": {
        "seconds": 0.80,
        "speeds": [48, 48, -48, -48],
    },
    "back": {
        "seconds": 1.60,
        "speeds": [-48, -56, 48, 48],
    },
}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_wheel_speeds(raw_speeds):
    if len(raw_speeds) != 4:
        raise ValueError(f"expected 4 wheel speeds, got {len(raw_speeds)}")
    normalized = []
    for speed in raw_speeds:
        normalized.append(int(clamp(int(speed), -base.MAX_SPEED, base.MAX_SPEED)))
    return normalized


def set_raw_wheel_speeds(controller, speeds):
    speeds = normalize_wheel_speeds(speeds)

    payload = bytearray(b"#ba")
    for index, speed in enumerate(speeds):
        direction = base.MOTOR_FORWARD_DIRS[index]
        if speed < 0:
            direction = base.reverse_direction(direction)
        payload.extend(direction.encode("ascii"))

    for speed in speeds:
        magnitude = int(clamp(abs(speed), 0, 65535))
        payload.append(magnitude & 0xFF)
        payload.append((magnitude >> 8) & 0xFF)

    controller._write_payload(payload)


def stop_for(controller, seconds):
    controller.hold_stop()
    if seconds > 0.0:
        time.sleep(seconds)


def validate_action_table():
    for action_name in ("right", "left", "back"):
        if action_name not in ACTION_TABLE:
            raise ValueError(f"missing action config: {action_name}")
        action_config = ACTION_TABLE[action_name]
        if "seconds" not in action_config:
            raise ValueError(f"missing seconds for action: {action_name}")
        if "speeds" not in action_config:
            raise ValueError(f"missing speeds for action: {action_name}")
        if float(action_config["seconds"]) < 0.0:
            raise ValueError(f"seconds must be >= 0 for action: {action_name}")
        normalize_wheel_speeds(action_config["speeds"])


def main():
    parser = argparse.ArgumentParser(
        description="Very simple UART in-place spin script with a fully manual 4-wheel action table."
    )
    parser.add_argument(
        "--action",
        choices=("right", "left", "back"),
        default="right",
        help="Choose which pre-tuned action table entry to run.",
    )
    args = parser.parse_args()

    validate_action_table()
    action_config = ACTION_TABLE[args.action]
    seconds = float(action_config["seconds"])
    speeds = normalize_wheel_speeds(action_config["speeds"])

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=0,
    )
    maze.install_stop_handlers(controller, None)

    try:
        stop_for(controller, START_DELAY)
        print(
            f"[spin-table] action={args.action} seconds={seconds:.3f} "
            f"speeds={speeds}"
        )
        set_raw_wheel_speeds(controller, speeds)
        time.sleep(seconds)
        controller.stop()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
