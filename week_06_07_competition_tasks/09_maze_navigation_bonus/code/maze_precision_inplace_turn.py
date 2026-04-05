import argparse
import importlib.util
import os
import time
from pathlib import Path

import cv2
import numpy as np


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


maze = load_local_module("maze_turn_core", "maze_navigation_ultrasonic_turn.py")
base = maze.base


def clamp(value, low, high):
    return max(low, min(high, value))


class PreciseInPlaceTurnController:
    def __init__(self, args):
        self.args = args
        self.lock_hits = 0
        self.last_direction = 0
        self.spin_started_at = 0.0
        self.brake_until = 0.0
        self.search_direction = self._initial_search_direction()
        self.search_step_started_at = 0.0

    def _initial_search_direction(self):
        if self.args.search_direction == "right":
            return 1
        return -1

    def _spin_command(self, direction, speed, mode):
        speed = int(clamp(speed, 0, base.MAX_SPEED))
        if direction >= 0:
            return maze.DriveCommand(speed, -speed, mode)
        return maze.DriveCommand(-speed, speed, mode)

    def _compute_spin_speed(self, abs_offset):
        if abs_offset <= self.args.close_offset:
            return int(self.args.near_speed)
        blend = clamp(
            (abs_offset - self.args.close_offset)
            / max(1e-6, self.args.far_offset - self.args.close_offset),
            0.0,
            1.0,
        )
        blend = blend * blend
        speed = self.args.near_speed + ((self.args.spin_speed - self.args.near_speed) * blend)
        return int(round(speed))

    def _update_search_direction(self, now):
        if self.args.search_direction != "alternate":
            return
        if (now - self.search_step_started_at) >= self.args.search_switch_seconds:
            self.search_direction *= -1
            self.search_step_started_at = now

    def update(self, visual, now):
        if now < self.brake_until:
            return maze.DriveCommand(0, 0, "BRAKE")

        if visual.gate_visible:
            offset = float(visual.gate_offset)
            abs_offset = abs(offset)

            if abs_offset <= self.args.stop_offset:
                self.lock_hits += 1
                self.last_direction = 0
                done = self.lock_hits >= self.args.stop_hits
                return maze.DriveCommand(0, 0, "LOCKED", done=done)

            self.lock_hits = 0
            direction = 1 if offset >= 0.0 else -1

            if (
                self.last_direction != 0
                and direction != self.last_direction
                and abs_offset <= self.args.brake_offset
            ):
                self.last_direction = 0
                self.spin_started_at = now
                self.brake_until = now + self.args.brake_seconds
                return maze.DriveCommand(0, 0, "BRAKE")

            if direction != self.last_direction:
                self.spin_started_at = now

            spin_speed = self._compute_spin_speed(abs_offset)
            if (now - self.spin_started_at) <= self.args.boost_seconds:
                spin_speed = max(spin_speed, self.args.boost_speed)

            self.last_direction = direction
            return self._spin_command(direction, spin_speed, "ALIGN")

        self.lock_hits = 0
        self.last_direction = 0
        self._update_search_direction(now)
        return self._spin_command(self.search_direction, self.args.search_speed, "SEARCH")


def build_overlay(frame, visual, command, controller, fps):
    view = frame.copy()
    frame_h, frame_w = view.shape[:2]
    center_x = frame_w // 2
    cv2.line(view, (center_x, 0), (center_x, frame_h), (255, 255, 255), 1)

    if visual.gate_visible:
        gate_x = int(clamp(visual.gate_center_x, 0, frame_w - 1))
        cv2.line(view, (gate_x, 0), (gate_x, frame_h), (0, 255, 255), 2)
        cv2.putText(
            view,
            f"offset={visual.gate_offset:+.3f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            view,
            "gate=lost",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        view,
        f"mode={command.mode} lock_hits={controller.lock_hits} fps={fps:.1f}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        f"L={command.left_speed:>3} R={command.right_speed:>3}",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return view


def main():
    parser = argparse.ArgumentParser(
        description="Minimal precise in-place turning tool for the maze car."
    )
    parser.add_argument("--spin-speed", type=int, default=26, help="Max in-place turn speed.")
    parser.add_argument("--near-speed", type=int, default=14, help="Slow speed used near the target center.")
    parser.add_argument("--boost-speed", type=int, default=32, help="Short breakaway boost when starting a turn.")
    parser.add_argument("--boost-seconds", type=float, default=0.06, help="How long the breakaway boost lasts.")
    parser.add_argument("--close-offset", type=float, default=0.16, help="Offset below which the controller slows down.")
    parser.add_argument("--far-offset", type=float, default=0.52, help="Offset above which the controller uses max spin speed.")
    parser.add_argument("--stop-offset", type=float, default=0.05, help="Offset considered centered.")
    parser.add_argument("--stop-hits", type=int, default=3, help="Consecutive centered frames required before stopping.")
    parser.add_argument("--brake-offset", type=float, default=0.07, help="When direction flips inside this offset, briefly brake first.")
    parser.add_argument("--brake-seconds", type=float, default=0.05, help="Brake pause used to suppress overshoot.")
    parser.add_argument("--search-speed", type=int, default=16, help="Slow search speed when the gate is lost.")
    parser.add_argument(
        "--search-direction",
        choices=("left", "right", "alternate"),
        default="left",
        help="Fallback search direction when the gate is not visible.",
    )
    parser.add_argument("--search-switch-seconds", type=float, default=0.65, help="Direction switch period when using alternate search.")
    parser.add_argument("--right-rear-spin-boost", type=int, default=0, help="Extra boost for the right rear wheel during spin if your car rotates unevenly.")
    parser.add_argument("--gate-confidence-threshold", type=float, default=maze.GATE_CONFIDENCE_THRESHOLD, help="Minimum gate confidence before the controller trusts the detection.")
    parser.add_argument("--start-delay", type=float, default=0.0, help="Optional delay before motion starts.")
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument("--continuous", action="store_true", help="Keep running after centered instead of stopping once.")
    parser.add_argument("--debug-video-output", default="", help="Optional mp4 path for the debug overlay.")
    args = parser.parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

    vision_config = maze.MazeConfig(
        mode=maze.MODE_RUN_MAZE,
        gate_confidence_threshold=clamp(float(args.gate_confidence_threshold), 0.02, 0.99),
    )

    controller = maze.TrimmedCarMotorController(
        right_rear_boost=maze.RIGHT_REAR_BOOST,
        right_rear_spin_boost=max(0, int(args.right_rear_spin_boost)),
    )
    maze.install_stop_handlers(controller, None)

    picam2 = base.Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": base.CAPTURE_SIZE, "format": "BGR888"}
    )
    picam2.configure(camera_config)
    picam2.start()

    debug_writer = None
    aligner = PreciseInPlaceTurnController(args)
    deadline = time.perf_counter() + max(0.0, float(args.start_delay))
    last_frame_time = time.perf_counter()
    fps = 0.0

    try:
        if args.debug_video_output:
            debug_path = Path(args.debug_video_output)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(str(debug_path), fourcc, 20.0, base.PROCESS_SIZE)
            if not debug_writer.isOpened():
                raise RuntimeError(f"unable to open debug video writer: {debug_path}")

        while True:
            now = time.perf_counter()
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0.0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            full_frame = picam2.capture_array()
            frame = cv2.resize(full_frame, base.PROCESS_SIZE, interpolation=cv2.INTER_AREA)
            visual = maze.build_black_wall_observation(frame, vision_config)

            if now < deadline:
                command = maze.DriveCommand(0, 0, "COUNTDOWN")
                controller.hold_stop()
            else:
                command = aligner.update(visual, now)
                controller.set_tank_drive(command.left_speed, command.right_speed, straight_mode=False)
                if command.done and not args.continuous:
                    controller.stop()

            overlay = build_overlay(frame, visual, command, aligner, fps)

            if now < deadline:
                remaining = max(0, int(np.ceil(deadline - now)))
                cv2.putText(
                    overlay,
                    f"Start in {remaining}",
                    (12, overlay.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if debug_writer is not None:
                debug_writer.write(overlay)

            if not args.headless:
                cv2.imshow("Precise In-Place Turn", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            else:
                time.sleep(0.001)

            if command.done and not args.continuous:
                break

    finally:
        controller.close()
        if debug_writer is not None:
            debug_writer.release()
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()
