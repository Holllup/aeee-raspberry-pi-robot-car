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


maze = load_local_module("maze_uart_spin_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


# Main tuning block. Keep your command line fixed and tune these defaults in one place.
DEFAULT_MAIN_SPIN_SPEED = 52
DEFAULT_START_BOOST_SPEED = 70
DEFAULT_START_BOOST_SECONDS = 0.06
DEFAULT_RIGHT_REAR_KICK_SECONDS = 0.12
DEFAULT_RIGHT_REAR_KICK_EXTRA = 28
DEFAULT_RIGHT_REAR_SPIN_BOOST = 14

DEFAULT_RIGHT_90_SECONDS = 0.80
DEFAULT_LEFT_90_SECONDS = 0.80
DEFAULT_BACK_180_SECONDS = 1.60


def clamp(value, low, high):
    return max(low, min(high, value))


def resolve_action(action):
    if action == "right":
        return "right", 90.0
    if action == "left":
        return "left", 90.0
    return "right", 180.0


def spin_command(direction, speed):
    speed = int(clamp(speed, 0, base.MAX_SPEED))
    if direction == "right":
        return speed, -speed
    return -speed, speed


def set_tank_drive_with_right_rear_extra(controller, left_speed, right_speed, right_rear_extra=0):
    per_motor_speeds = [0, 0, 0, 0]
    for index in base.LEFT_MOTOR_INDEXES:
        per_motor_speeds[index] = int(left_speed)
    for index in base.RIGHT_MOTOR_INDEXES:
        per_motor_speeds[index] = int(right_speed)

    if int(right_rear_extra) != 0:
        rear_index = maze.TrimmedCarMotorController.RIGHT_REAR_INDEX
        if per_motor_speeds[rear_index] >= 0:
            per_motor_speeds[rear_index] += int(right_rear_extra)
        else:
            per_motor_speeds[rear_index] -= int(right_rear_extra)

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
    controller._write_payload(payload)


def run_spin_stage(controller, direction, speed, seconds, right_rear_extra=0):
    if seconds <= 0.0 or speed <= 0:
        return
    left_speed, right_speed = spin_command(direction, speed)
    if int(right_rear_extra) != 0:
        set_tank_drive_with_right_rear_extra(
            controller,
            left_speed,
            right_speed,
            right_rear_extra=right_rear_extra,
        )
    else:
        controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
    time.sleep(seconds)


def stop_for(controller, seconds):
    controller.hold_stop()
    if seconds > 0.0:
        time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Strong UART in-place spin tool. Pure open-loop, no vision, no encoder."
    )
    parser.add_argument(
        "--action",
        choices=("right", "left", "back"),
        default="right",
        help="right=90 deg right spin, left=90 deg left spin, back=180 deg U-turn.",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Optional custom angle in degrees. If omitted, the action preset is used.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=DEFAULT_MAIN_SPIN_SPEED,
        help="Main in-place spin speed. Higher means more force.",
    )
    parser.add_argument(
        "--boost-speed",
        type=int,
        default=DEFAULT_START_BOOST_SPEED,
        help="Short startup boost used to overcome static friction.",
    )
    parser.add_argument(
        "--boost-seconds",
        type=float,
        default=DEFAULT_START_BOOST_SECONDS,
        help="How long the startup boost lasts.",
    )
    parser.add_argument(
        "--kick-seconds",
        type=float,
        default=DEFAULT_RIGHT_REAR_KICK_SECONDS,
        help="Extra startup window used to specifically wake up the weak right rear wheel.",
    )
    parser.add_argument(
        "--kick-right-rear-extra",
        type=int,
        default=DEFAULT_RIGHT_REAR_KICK_EXTRA,
        help="Extra magnitude applied only to the right rear wheel during the startup kick stage.",
    )
    parser.add_argument(
        "--right-90-seconds",
        type=float,
        default=DEFAULT_RIGHT_90_SECONDS,
        help="Open-loop duration for a 90 degree right spin.",
    )
    parser.add_argument(
        "--left-90-seconds",
        type=float,
        default=DEFAULT_LEFT_90_SECONDS,
        help="Open-loop duration for a 90 degree left spin.",
    )
    parser.add_argument(
        "--back-180-seconds",
        type=float,
        default=DEFAULT_BACK_180_SECONDS,
        help="Open-loop duration for a 180 degree U-turn.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.05,
        help="Pause after the spin finishes.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=1.0,
        help="Delay before motion starts, to give you time to place the car.",
    )
    parser.add_argument(
        "--right-rear-spin-boost",
        type=int,
        default=DEFAULT_RIGHT_REAR_SPIN_BOOST,
        help="Extra magnitude added to the right rear wheel during the whole spin if that wheel is weak.",
    )
    args = parser.parse_args()

    direction, preset_angle = resolve_action(args.action)
    angle = abs(float(args.angle)) if args.angle is not None else preset_angle
    if angle < 1e-3:
        print("[info] angle is zero, nothing to do")
        return

    if args.action == "back" and args.angle is None:
        total_seconds = max(0.05, float(args.back_180_seconds))
    else:
        ninety_seconds = (
            max(0.05, float(args.right_90_seconds))
            if direction == "right"
            else max(0.05, float(args.left_90_seconds))
        )
        total_seconds = ninety_seconds * (angle / 90.0)

    boost_seconds = min(max(0.0, float(args.boost_seconds)), total_seconds)
    kick_seconds = min(max(0.0, float(args.kick_seconds)), total_seconds - boost_seconds)
    main_seconds = max(0.0, total_seconds - boost_seconds - kick_seconds)

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
    )
    maze.install_stop_handlers(controller, None)

    try:
        stop_for(controller, max(0.0, float(args.start_delay)))
        print(
            f"[spin] action={args.action} dir={direction} angle={angle:.1f} "
            f"boost_speed={int(args.boost_speed)} speed={int(args.speed)} "
            f"boost={boost_seconds:.3f}s kick={kick_seconds:.3f}s "
            f"main={main_seconds:.3f}s total={total_seconds:.3f}s "
            f"rr_extra={int(args.kick_right_rear_extra)}"
        )
        run_spin_stage(controller, direction, max(0, int(args.boost_speed)), boost_seconds)
        run_spin_stage(
            controller,
            direction,
            max(0, int(args.speed)),
            kick_seconds,
            right_rear_extra=max(0, int(args.kick_right_rear_extra)),
        )
        run_spin_stage(controller, direction, max(0, int(args.speed)), main_seconds)
        stop_for(controller, max(0.0, float(args.settle_seconds)))
    finally:
        controller.close()


if __name__ == "__main__":
    main()
