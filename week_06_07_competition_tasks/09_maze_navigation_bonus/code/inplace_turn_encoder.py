import argparse
import time

from encoder_turn_runtime import (
    DEFAULT_I2C_BUS,
    DEFAULT_POLL_HZ,
    EncoderReader,
    WheelSpeedEstimator,
    ClosedLoopInplaceTurnController,
    TURN_BRAKE_SECONDS,
    TURN_SLOWDOWN_RATIO,
    average,
    maze,
)


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop in-place turning using wheel encoder feedback."
    )
    parser.add_argument(
        "--action",
        choices=("right", "left", "back"),
        default="right",
        help="Closed-loop in-place action.",
    )
    parser.add_argument("--speed", type=int, default=30, help="Base spin speed before slowdown.")
    parser.add_argument(
        "--target-counts-right-90",
        type=int,
        default=120,
        help="Average encoder counts needed for a 90 degree right turn.",
    )
    parser.add_argument(
        "--target-counts-left-90",
        type=int,
        default=120,
        help="Average encoder counts needed for a 90 degree left turn.",
    )
    parser.add_argument("--right-rear-spin-boost", type=int, default=0, help="Extra boost for the right rear wheel during spin.")
    parser.add_argument("--i2c-bus", type=int, default=DEFAULT_I2C_BUS, help="Linux I2C bus index.")
    parser.add_argument("--poll-hz", type=float, default=DEFAULT_POLL_HZ, help="Control loop rate.")
    parser.add_argument("--start-delay", type=float, default=1.0, help="Delay before the closed-loop turn starts.")
    args = parser.parse_args()

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
    )
    reader = EncoderReader(bus_index=args.i2c_bus)
    estimator = WheelSpeedEstimator()
    turn_controller = ClosedLoopInplaceTurnController(
        action=args.action,
        base_speed=args.speed,
        target_counts_right_90=args.target_counts_right_90,
        target_counts_left_90=args.target_counts_left_90,
    )
    maze.install_stop_handlers(controller, None)

    poll_interval = 1.0 / max(1.0, float(args.poll_hz))

    try:
        print(
            f"[turn] action={args.action} target_counts={turn_controller.target_counts} "
            f"speed={turn_controller.base_speed} backend={reader.backend_name}"
        )
        controller.hold_stop()
        time.sleep(max(0.0, float(args.start_delay)))

        start_sample = reader.read_sample()
        estimate = estimator.update(start_sample)
        print(
            f"[turn] start counts={start_sample.counts} "
            f"left_cps={estimate.left_counts_per_sec:7.2f} "
            f"right_cps={estimate.right_counts_per_sec:7.2f}"
        )

        while True:
            loop_started_at = time.perf_counter()
            sample = reader.read_sample()
            estimate = estimator.update(sample)
            left_progress, right_progress, mean_progress = turn_controller.measure_progress(
                start_sample,
                sample,
            )
            progress_ratio = mean_progress / float(max(1, turn_controller.target_counts))
            if mean_progress >= turn_controller.target_counts:
                break

            left_command, right_command = turn_controller.compute_drive(mean_progress, estimate)
            controller.set_tank_drive(left_command, right_command, straight_mode=False)

            phase = "slow" if progress_ratio >= TURN_SLOWDOWN_RATIO else "main"
            print(
                f"[turn] phase={phase} progress={mean_progress:7.1f}/{turn_controller.target_counts:<4d} "
                f"left_prog={left_progress:7.1f} right_prog={right_progress:7.1f} "
                f"left_cps={estimate.left_counts_per_sec:8.2f} right_cps={estimate.right_counts_per_sec:8.2f} "
                f"cmd=({left_command:>3d},{right_command:>3d})"
            )

            sleep_seconds = poll_interval - (time.perf_counter() - loop_started_at)
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)

        brake_left, brake_right = turn_controller.brake_command()
        controller.set_tank_drive(brake_left, brake_right, straight_mode=False)
        time.sleep(TURN_BRAKE_SECONDS)
        controller.stop()
        time.sleep(0.05)

        end_sample = reader.read_sample()
        end_estimate = estimator.update(end_sample)
        left_progress, right_progress, mean_progress = turn_controller.measure_progress(start_sample, end_sample)
        progress_error = mean_progress - float(turn_controller.target_counts)
        print(
            f"[turn] done progress={mean_progress:7.1f} target={turn_controller.target_counts:<4d} "
            f"error={progress_error:+7.1f} "
            f"left_prog={left_progress:7.1f} right_prog={right_progress:7.1f} "
            f"avg_cps={average((abs(end_estimate.left_counts_per_sec), abs(end_estimate.right_counts_per_sec))):7.2f}"
        )
    finally:
        controller.close()
        reader.close()


if __name__ == "__main__":
    main()
