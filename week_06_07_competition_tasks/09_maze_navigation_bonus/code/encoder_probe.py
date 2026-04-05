import argparse
import time

from encoder_turn_runtime import (
    DEFAULT_I2C_BUS,
    DEFAULT_POLL_HZ,
    EncoderReader,
    WheelSpeedEstimator,
    base,
    command_turn,
    format_counts,
    maze,
)


def print_snapshot(prefix, sample, estimate):
    print(
        f"{prefix} counts=[{format_counts(sample.counts)}] "
        f"left_cps={estimate.left_counts_per_sec:8.2f} "
        f"right_cps={estimate.right_counts_per_sec:8.2f}"
    )


def observe_samples(reader, estimator, duration_seconds, poll_interval, label, print_every=0.5):
    end_time = time.perf_counter() + max(0.0, float(duration_seconds))
    next_print_time = 0.0
    first_sample = None
    last_sample = None
    while time.perf_counter() < end_time:
        sample = reader.read_sample()
        estimate = estimator.update(sample)
        if first_sample is None:
            first_sample = sample
        last_sample = sample
        now = time.perf_counter()
        if now >= next_print_time:
            print_snapshot(label, sample, estimate)
            next_print_time = now + print_every
        time.sleep(poll_interval)
    return first_sample, last_sample


def counts_changed(sample_a, sample_b):
    if sample_a is None or sample_b is None:
        return False
    return any(current != previous for current, previous in zip(sample_b.counts, sample_a.counts))


def main():
    parser = argparse.ArgumentParser(description="Probe encoder availability and direction symmetry.")
    parser.add_argument("--i2c-bus", type=int, default=DEFAULT_I2C_BUS, help="Linux I2C bus index.")
    parser.add_argument("--poll-hz", type=float, default=DEFAULT_POLL_HZ, help="Encoder polling rate.")
    parser.add_argument("--idle-seconds", type=float, default=5.0, help="How long to observe counts while stationary.")
    parser.add_argument("--manual-check-seconds", type=float, default=5.0, help="How long to watch while you manually spin a wheel.")
    parser.add_argument("--spin-speed", type=int, default=14, help="Low in-place spin speed for direction verification.")
    parser.add_argument("--spin-seconds", type=float, default=0.30, help="How long the low-speed right spin lasts.")
    parser.add_argument("--right-rear-spin-boost", type=int, default=0, help="Extra boost for the right rear wheel during spin.")
    args = parser.parse_args()

    poll_interval = 1.0 / max(1.0, float(args.poll_hz))
    reader = EncoderReader(bus_index=args.i2c_bus)
    estimator = WheelSpeedEstimator()
    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
    )
    maze.install_stop_handlers(controller, None)

    print(f"[probe] encoder backend={reader.backend_name} i2c_bus={args.i2c_bus}")

    try:
        print("[probe] step 1/3: keep the car still, checking that counts stay stable")
        idle_start, idle_end = observe_samples(
            reader,
            estimator,
            args.idle_seconds,
            poll_interval,
            label="[idle]",
        )
        if idle_start is None or idle_end is None:
            raise RuntimeError("failed to capture idle encoder samples")

        print("[probe] step 2/3: lightly spin any wheel by hand now")
        print("[probe] press Enter to start the manual observation window")
        input()
        manual_start, manual_end = observe_samples(
            reader,
            estimator,
            args.manual_check_seconds,
            poll_interval,
            label="[hand]",
        )
        if not counts_changed(manual_start, manual_end):
            raise RuntimeError(
                "manual wheel movement did not change encoder counts. "
                "This suggests the encoder chain is not connected or not readable."
            )

        print("[probe] step 3/3: low-speed right spin, verifying left/right directions oppose each other")
        spin_start = reader.read_sample()
        estimator.update(spin_start)
        left_speed, right_speed = command_turn("right", args.spin_speed)
        controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
        time.sleep(max(0.0, float(args.spin_seconds)))
        controller.stop()
        time.sleep(0.10)
        spin_end = reader.read_sample()
        spin_estimate = estimator.update(spin_end)
        print_snapshot("[spin]", spin_end, spin_estimate)

        left_delta = sum(
            spin_end.counts[index] - spin_start.counts[index]
            for index in base.LEFT_MOTOR_INDEXES
        )
        right_delta = sum(
            spin_end.counts[index] - spin_start.counts[index]
            for index in base.RIGHT_MOTOR_INDEXES
        )
        if left_delta == 0 and right_delta == 0:
            raise RuntimeError("low-speed spin did not move encoder counts at all")
        if left_delta == 0 or right_delta == 0 or (left_delta > 0) == (right_delta > 0):
            raise RuntimeError(
                "left/right encoder deltas did not show opposite directions during in-place spin"
            )

        print("[probe] success: encoder counts are readable, react to manual motion, and show opposite directions while spinning")
    finally:
        controller.close()
        reader.close()


if __name__ == "__main__":
    main()
