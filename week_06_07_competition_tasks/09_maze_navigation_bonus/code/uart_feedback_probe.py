import argparse
import binascii
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


maze = load_local_module("maze_uart_probe_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


READ_CHUNK_BYTES = 256
PRINT_IDLE_FRAME_GAP_SECONDS = 0.08


def clamp(value, low, high):
    return max(low, min(high, value))


def spin_command(direction, speed):
    speed = int(clamp(speed, 0, base.MAX_SPEED))
    if direction == "right":
        return speed, -speed
    return -speed, speed


def drive_command(action, speed):
    speed = int(clamp(speed, 0, base.MAX_SPEED))
    if action == "forward":
        return speed, speed
    if action == "backward":
        return -speed, -speed
    if action == "right":
        return spin_command("right", speed)
    if action == "left":
        return spin_command("left", speed)
    return 0, 0


def bytes_to_ascii(data):
    chars = []
    for byte_value in data:
        if 32 <= byte_value <= 126:
            chars.append(chr(byte_value))
        else:
            chars.append(".")
    return "".join(chars)


class UARTFeedbackProbe:
    def __init__(self, controller):
        self.controller = controller
        self.serial = controller.serial
        self.port_name = controller.port_name
        self.last_rx_time = None
        self.total_bytes = 0
        self.frame_count = 0

    def read_available(self):
        available = int(getattr(self.serial, "in_waiting", 0))
        if available <= 0:
            return b""
        chunk = self.serial.read(min(available, READ_CHUNK_BYTES))
        if chunk:
            self.total_bytes += len(chunk)
            now = time.perf_counter()
            if self.last_rx_time is None or (now - self.last_rx_time) >= PRINT_IDLE_FRAME_GAP_SECONDS:
                self.frame_count += 1
                print(f"[rx frame {self.frame_count}]")
            self.last_rx_time = now
            print(f"  hex   {binascii.hexlify(chunk).decode('ascii')}")
            print(f"  ascii {bytes_to_ascii(chunk)}")
        return chunk

    def drain_for(self, seconds, poll_seconds):
        end_time = time.perf_counter() + max(0.0, float(seconds))
        any_rx = False
        while time.perf_counter() < end_time:
            chunk = self.read_available()
            if chunk:
                any_rx = True
            time.sleep(poll_seconds)
        return any_rx


def main():
    parser = argparse.ArgumentParser(
        description="Probe whether the baseboard returns any UART feedback while the car is moving."
    )
    parser.add_argument(
        "--action",
        choices=("idle", "forward", "backward", "left", "right"),
        default="forward",
        help="Motion command to send while probing UART feedback.",
    )
    parser.add_argument("--speed", type=int, default=36, help="Drive speed used during the probe action.")
    parser.add_argument("--duration", type=float, default=1.5, help="How long to keep the probe action active.")
    parser.add_argument("--pre-read-seconds", type=float, default=1.0, help="How long to listen before sending motion.")
    parser.add_argument("--post-read-seconds", type=float, default=1.0, help="How long to listen after stopping.")
    parser.add_argument("--poll-seconds", type=float, default=0.01, help="Polling interval while reading UART.")
    parser.add_argument("--right-rear-spin-boost", type=int, default=0, help="Optional spin boost for the right rear wheel.")
    args = parser.parse_args()

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
    )
    maze.install_stop_handlers(controller, None)
    probe = UARTFeedbackProbe(controller)

    print(f"[probe] port={probe.port_name} action={args.action} speed={args.speed}")

    try:
        controller.stop()
        controller.serial.reset_input_buffer()
        controller.serial.reset_output_buffer()

        print(f"[probe] listening before motion for {args.pre_read_seconds:.2f}s")
        had_pre_rx = probe.drain_for(args.pre_read_seconds, args.poll_seconds)

        left_speed, right_speed = drive_command(args.action, args.speed)
        if args.action != "idle":
            print(
                f"[probe] driving action={args.action} left_speed={left_speed} "
                f"right_speed={right_speed} for {args.duration:.2f}s"
            )
            controller.set_tank_drive(left_speed, right_speed, straight_mode=False)
        else:
            print(f"[probe] idle action for {args.duration:.2f}s")
            controller.hold_stop()
        had_motion_rx = probe.drain_for(args.duration, args.poll_seconds)

        controller.stop()
        print(f"[probe] listening after stop for {args.post_read_seconds:.2f}s")
        had_post_rx = probe.drain_for(args.post_read_seconds, args.poll_seconds)

        print(
            f"[probe] done total_rx_bytes={probe.total_bytes} "
            f"frames={probe.frame_count} "
            f"pre_rx={had_pre_rx} motion_rx={had_motion_rx} post_rx={had_post_rx}"
        )
        if probe.total_bytes == 0:
            print(
                "[probe] no UART bytes were received. "
                "With current evidence, the baseboard still looks write-only from the Pi side."
            )
        else:
            print(
                "[probe] UART feedback exists. Save the hex/ascii output above; "
                "it is the starting point for reverse-engineering a read protocol and enabling real closed-loop control."
            )
    finally:
        controller.close()


if __name__ == "__main__":
    main()
