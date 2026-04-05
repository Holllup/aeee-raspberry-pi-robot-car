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


maze = load_local_module("maze_uart_forward_turn_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


# ----------------------------
# 调参区：以后主要改这里
# 这里只改这一份文件就够了
# 如果直线距离不够，就增大 FORWARD_SECONDS
# 如果直线太远，就减小 FORWARD_SECONDS
# 如果直线跑偏，就分别改 FORWARD_LEFT_SPEED / FORWARD_RIGHT_SPEED
# 左转 / 向后转 也直接在下面 ACTION_TABLE 里改
# ----------------------------

FORWARD_SECONDS = 1.11
FORWARD_LEFT_SPEED = 34
FORWARD_RIGHT_SPEED = 34
START_DELAY = 1.0
STOP_BETWEEN_STAGES_SECONDS = 08

ACTION_TABLE = {
    "left": {
        "seconds": 0.41,
        "left_speed": -52,
        "right_speed": 52,
    },
    "back": {
        "seconds": 0.89,
        "left_speed": 52,
        "right_speed": -52,
    },
}


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_speed(value):
    return int(clamp(int(value), -base.MAX_SPEED, base.MAX_SPEED))


def run_tank_motion(controller, left_speed, right_speed, seconds):
    left_speed = normalize_speed(left_speed)
    right_speed = normalize_speed(right_speed)
    if seconds <= 0.0:
        return
    controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
    time.sleep(seconds)


def stop_for(controller, seconds):
    controller.hold_stop()
    if seconds > 0.0:
        time.sleep(seconds)


def load_turn_action(action_name):
    if action_name not in ("left", "back"):
        raise ValueError(f"unsupported action: {action_name}")
    if action_name not in ACTION_TABLE:
        raise ValueError(f"missing action config: {action_name}")

    action_config = ACTION_TABLE[action_name]
    seconds = float(action_config["seconds"])
    left_speed = normalize_speed(action_config["left_speed"])
    right_speed = normalize_speed(action_config["right_speed"])
    return seconds, left_speed, right_speed


def run_named_turn(controller, action_name):
    turn_seconds, turn_left_speed, turn_right_speed = load_turn_action(action_name)
    print(
        f"[forward-turn-loop] action={action_name} seconds={turn_seconds:.3f} "
        f"left_speed={turn_left_speed} right_speed={turn_right_speed}"
    )
    run_tank_motion(controller, turn_left_speed, turn_right_speed, turn_seconds)
    controller.stop()
    stop_for(controller, STOP_BETWEEN_STAGES_SECONDS)


def main():
    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=0,
    )
    maze.install_stop_handlers(controller, None)

    try:
        stop_for(controller, START_DELAY)
        while True:
            print(
                f"[forward-turn-loop] forward_seconds={FORWARD_SECONDS:.3f} "
                f"forward_left={normalize_speed(FORWARD_LEFT_SPEED)} "
                f"forward_right={normalize_speed(FORWARD_RIGHT_SPEED)}"
            )
            run_tank_motion(controller, FORWARD_LEFT_SPEED, FORWARD_RIGHT_SPEED, FORWARD_SECONDS)
            controller.stop()
            stop_for(controller, STOP_BETWEEN_STAGES_SECONDS)

            run_named_turn(controller, "left")
            run_named_turn(controller, "back")
            run_named_turn(controller, "left")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
