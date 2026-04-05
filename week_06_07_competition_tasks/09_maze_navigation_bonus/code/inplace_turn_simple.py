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


maze = load_local_module("maze_turn_core_simple", "maze_navigation_ultrasonic_turn.py")
base = maze.base


def clamp(value, low, high):
    return max(low, min(high, value))


def spin_command(direction, speed):
    speed = int(clamp(speed, 0, base.MAX_SPEED))
    if direction == "right":
        return speed, -speed
    return -speed, speed


def stop_for(controller, seconds):
    controller.hold_stop()
    if seconds > 0.0:
        time.sleep(seconds)


def run_stage(controller, direction, speed, seconds):
    if seconds <= 0.0 or speed <= 0:
        return
    left_speed, right_speed = spin_command(direction, speed)
    controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
    time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Simple precise in-place turn tool with a staged timing profile."
    )
    parser.add_argument(
        "--action",
        choices=("right", "left", "back"),
        default="right",
        help="Preset in-place turn action: right=90 deg, left=90 deg, back=180 deg.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Optional custom angle in degrees. If omitted, action presets are used.",
    )
    parser.add_argument("--right-90-seconds", type=float, default=0.78, help="Calibrated time for a 90 degree right turn.")
    parser.add_argument("--left-90-seconds", type=float, default=0.82, help="Calibrated time for a 90 degree left turn.")
    parser.add_argument("--spin-speed", type=int, default=26, help="Main turn speed.")
    parser.add_argument("--boost-speed", type=int, default=32, help="Short breakaway speed at the start.")
    parser.add_argument("--boost-seconds", type=float, default=0.05, help="Duration of the start boost.")
    parser.add_argument("--finish-speed", type=int, default=16, help="Slow speed used near the end of the turn.")
    parser.add_argument("--finish-ratio", type=float, default=0.22, help="Fraction of total turn time reserved for the slow finish stage.")
    parser.add_argument("--brake-speed", type=int, default=12, help="Short reverse spin used to reduce overshoot.")
    parser.add_argument("--brake-seconds", type=float, default=0.03, help="Duration of the reverse brake pulse.")
    parser.add_argument("--settle-seconds", type=float, default=0.04, help="Pause after the turn finishes.")
    parser.add_argument("--start-delay", type=float, default=1.0, help="Delay before the turn starts.")
    parser.add_argument("--right-rear-spin-boost", type=int, default=0, help="Extra boost for the right rear wheel if the car rotates unevenly.")
    args = parser.parse_args()

    direction = "right" if args.action in ("right", "back") else "left"
    target_angle = 180.0 if args.action == "back" else 90.0
    angle = target_angle if args.angle is None else abs(float(args.angle))
    if angle < 1e-3:
        print("[info] angle is zero, nothing to do")
        return

    ninety_seconds = (
        max(0.05, float(args.right_90_seconds))
        if direction == "right"
        else max(0.05, float(args.left_90_seconds))
    )
    total_seconds = ninety_seconds * (angle / 90.0)

    boost_seconds = min(max(0.0, float(args.boost_seconds)), total_seconds * 0.45)
    finish_seconds = min(
        max(0.0, total_seconds * clamp(float(args.finish_ratio), 0.0, 0.60)),
        max(0.0, total_seconds - boost_seconds),
    )
    main_seconds = max(0.0, total_seconds - boost_seconds - finish_seconds)
    reverse_direction = "left" if direction == "right" else "right"

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
    )
    maze.install_stop_handlers(controller, None)

    try:
        stop_for(controller, max(0.0, float(args.start_delay)))
        print(
            f"[turn] action={args.action} dir={direction} angle={angle:.1f} total={total_seconds:.3f}s "
            f"boost={boost_seconds:.3f}s main={main_seconds:.3f}s finish={finish_seconds:.3f}s"
        )

        run_stage(controller, direction, max(0, int(args.boost_speed)), boost_seconds)
        run_stage(controller, direction, max(0, int(args.spin_speed)), main_seconds)
        run_stage(controller, direction, max(0, int(args.finish_speed)), finish_seconds)
        run_stage(controller, reverse_direction, max(0, int(args.brake_speed)), max(0.0, float(args.brake_seconds)))
        stop_for(controller, max(0.0, float(args.settle_seconds)))
    finally:
        controller.close()


if __name__ == "__main__":
    main()
