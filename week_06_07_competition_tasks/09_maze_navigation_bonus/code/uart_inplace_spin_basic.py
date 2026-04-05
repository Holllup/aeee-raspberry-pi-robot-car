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


maze = load_local_module("maze_uart_basic_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


# ----------------------------
# 调参区：以后只改这里
# left_speed: 左侧两个轮子的统一速度
# right_speed: 右侧两个轮子的统一速度
# 正数=前进，负数=后退，0=停
# 如果右转过头，就减小 right.seconds
# 如果右转不够，就增大 right.seconds
# 如果不是原地转，就手动改 left_speed / right_speed
# ----------------------------

START_DELAY = 1.0

ACTION_TABLE = {
    "right": {
        "seconds": 0.46,
        "left_speed": 52,
        "right_speed": -52,
    },
    "left": {
        "seconds": 0.51,
        "left_speed": -52,
        "right_speed": 52,
    },
    "back": {
        "seconds": 0.92,
        "left_speed": 52,
        "right_speed": -52,
    },
}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_speed(value):
    return int(clamp(int(value), -base.MAX_SPEED, base.MAX_SPEED))


def validate_action_table():
    for action_name in ("right", "left", "back"):
        if action_name not in ACTION_TABLE:
            raise ValueError(f"missing action config: {action_name}")
        action_config = ACTION_TABLE[action_name]
        for field_name in ("seconds", "left_speed", "right_speed"):
            if field_name not in action_config:
                raise ValueError(f"missing {field_name} for action: {action_name}")
        if float(action_config["seconds"]) < 0.0:
            raise ValueError(f"seconds must be >= 0 for action: {action_name}")
        normalize_speed(action_config["left_speed"])
        normalize_speed(action_config["right_speed"])


def stop_for(controller, seconds):
    controller.hold_stop()
    if seconds > 0.0:
        time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Very simple timed UART spin script using only left/right side speeds."
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
    left_speed = normalize_speed(action_config["left_speed"])
    right_speed = normalize_speed(action_config["right_speed"])

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=0,
    )
    maze.install_stop_handlers(controller, None)

    try:
        stop_for(controller, START_DELAY)
        print(
            f"[spin-basic] action={args.action} seconds={seconds:.3f} "
            f"left_speed={left_speed} right_speed={right_speed}"
        )
        controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
        time.sleep(seconds)
        controller.stop()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
