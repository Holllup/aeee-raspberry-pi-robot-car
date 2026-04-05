import argparse
import csv
import os
import time
from pathlib import Path

import cv2

import line_following_v1_8_maze_navigation as maze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before starting maze navigation.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument(
        "--record-output",
        default="",
        help="Optional output h264 path from the Pi camera.",
    )
    parser.add_argument(
        "--debug-video-output",
        default="maze_nav_debug.mp4",
        help="Optional mp4 path for the processed overlay video.",
    )
    parser.add_argument(
        "--metrics-output",
        default="maze_nav_metrics.csv",
        help="Optional CSV path for frame-by-frame perception and command metrics.",
    )
    args = parser.parse_args()
    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

    maze.base.GPIO.setwarnings(False)
    maze.base.GPIO.setmode(maze.base.GPIO.BCM)

    controller = maze.base.CarMotorController()
    lcd_display = maze.base.LCD1602Display()
    maze.install_stop_handlers(controller)

    picam2 = maze.base.Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": maze.base.CAPTURE_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)

    h264_recording = False
    if args.record_output:
        encoder = maze.base.H264Encoder(bitrate=8_000_000)
        picam2.start_recording(encoder, maze.base.FileOutput(args.record_output))
        h264_recording = True
    else:
        picam2.start()

    debug_writer = None
    metrics_file = None
    metrics_writer = None
    navigator = maze.MazeNavigator(detect_exit=False)
    state = "COUNTDOWN"
    countdown_deadline = time.perf_counter() + max(0.0, args.start_delay)
    last_frame_time = time.perf_counter()
    fps = 0.0

    try:
        if args.debug_video_output:
            debug_path = Path(args.debug_video_output)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(
                str(debug_path),
                fourcc,
                20.0,
                maze.base.PROCESS_SIZE,
            )
            if not debug_writer.isOpened():
                raise RuntimeError(f"unable to open debug video writer: {debug_path}")

        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
            metrics_writer = csv.DictWriter(
                metrics_file,
                fieldnames=[
                    "t",
                    "state",
                    "mode",
                    "left_speed",
                    "right_speed",
                    "centerline_x",
                    "corridor_width",
                    "heading_error",
                    "lookahead_error",
                    "forward_depth",
                    "left_branch_score",
                    "right_branch_score",
                    "front_block_score",
                    "width_gain",
                    "confidence",
                    "exit_candidate",
                ],
            )
            metrics_writer.writeheader()

        while True:
            now = time.perf_counter()
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            full_frame = picam2.capture_array()
            full_frame = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(full_frame, maze.base.PROCESS_SIZE, interpolation=cv2.INTER_AREA)

            perception = None
            drive_command = None

            if state == "COUNTDOWN":
                controller.hold_stop()
                remaining = countdown_deadline - now
                lcd_display.update("Maze Debug", f"Start {max(0, int(remaining + 0.99))}")
                if remaining <= 0:
                    state = "RUNNING"
                    navigator = maze.MazeNavigator(detect_exit=False)
                    controller.stop()
            elif state == "RUNNING":
                perception = maze.analyze_maze_frame(frame, navigator.previous_center_x)
                drive_command = navigator.update(perception, now)
                if drive_command["done"]:
                    state = "DONE"
                    controller.stop()
                else:
                    controller.set_tank_drive(drive_command["left_speed"], drive_command["right_speed"])
                    lcd_display.update("Maze Debug", drive_command["mode"][:16])
            else:
                controller.hold_stop()
                lcd_display.update("Maze Debug", "Stopped")

            if metrics_writer is not None:
                row = {
                    "t": f"{now:.3f}",
                    "state": state,
                    "mode": "" if drive_command is None else drive_command.get("mode", ""),
                    "left_speed": 0 if drive_command is None else drive_command.get("left_speed", 0),
                    "right_speed": 0 if drive_command is None else drive_command.get("right_speed", 0),
                    "centerline_x": "" if perception is None else perception.get("centerline_x", ""),
                    "corridor_width": "" if perception is None else perception.get("corridor_width", ""),
                    "heading_error": "" if perception is None else f"{perception.get('heading_error', 0.0):.2f}",
                    "lookahead_error": "" if perception is None else f"{perception.get('lookahead_error', 0.0):.2f}",
                    "forward_depth": "" if perception is None else f"{perception.get('forward_depth', 0.0):.4f}",
                    "left_branch_score": "" if perception is None else f"{perception.get('left_branch_score', 0.0):.4f}",
                    "right_branch_score": "" if perception is None else f"{perception.get('right_branch_score', 0.0):.4f}",
                    "front_block_score": "" if perception is None else f"{perception.get('front_block_score', 0.0):.4f}",
                    "width_gain": "" if perception is None else f"{perception.get('width_gain', 0.0):.4f}",
                    "confidence": "" if perception is None else f"{perception.get('confidence', 0.0):.4f}",
                    "exit_candidate": "" if perception is None else int(bool(perception.get("exit_candidate", False))),
                }
                metrics_writer.writerow(row)
                metrics_file.flush()

            overlay = maze.draw_overlay(
                frame,
                maze.MAZE_NAVIGATION_ACTIVE if state != "COUNTDOWN" else maze.MAZE_CONFIRMED_COUNTDOWN,
                None,
                None,
                fps,
                0,
                countdown_deadline,
                now,
                perception,
                drive_command,
            )
            if state == "COUNTDOWN":
                cv2.putText(
                    overlay,
                    f"Debug start in {max(0, int(countdown_deadline - now + 0.99))}",
                    (12, overlay.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if debug_writer is not None:
                debug_writer.write(overlay)

            if not args.headless:
                cv2.imshow("Maze Navigation Debug", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if key in (ord("f"), ord("F")):
                    state = "DONE"
                    controller.stop()
            else:
                time.sleep(0.001)

    finally:
        controller.close()
        lcd_display.close()
        if debug_writer is not None:
            debug_writer.release()
        if metrics_file is not None:
            metrics_file.close()
        maze.base.GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()


if __name__ == "__main__":
    main()
