from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional


def load_local_module(module_name: str, filename: str):
    module_path = Path(__file__).resolve().with_name(filename)
    if not module_path.exists():
        raise RuntimeError(f"unable to find local module: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


motion = load_local_module("maze_ultrasonic_turn_2_motion", "maze_navigation_ultrasonic_turn 2.py")
template = load_local_module(
    "maze_template_start_v18",
    "line_following_v1_8_maze_navigation.py",
)


def classify_maze_only_sign(sign_roi: Optional[dict]) -> dict:
    if sign_roi is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "no_sign",
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": False,
        }

    roi = sign_roi.get("warped")
    if roi is None or roi.size == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "empty_roi",
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": True,
        }

    binary = template.base.preprocess_sign_symbol(roi)
    if binary is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "empty_inner",
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": True,
        }

    maze_template = template.load_local_template(template.MAZE_TEMPLATE_FILENAME)
    if maze_template is None:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "reason": "maze_template_missing",
            "maze_confident": False,
            "maze_score": -1.0,
            "sign_found": True,
        }

    sample = binary
    if sample.shape != maze_template.shape:
        sample = template.cv2.resize(
            sample,
            (maze_template.shape[1], maze_template.shape[0]),
            interpolation=template.cv2.INTER_NEAREST,
        )
    xor_frame = template.cv2.bitwise_xor(sample, maze_template)
    difference_pixels = template.cv2.countNonZero(xor_frame)
    maze_score = 1.0 - difference_pixels / float(xor_frame.shape[0] * xor_frame.shape[1])
    maze_confident = maze_score >= float(template.MAZE_TEMPLATE_MATCH_THRESHOLD)

    return {
        "label": "MAZE" if maze_confident else "UNKNOWN",
        "confidence": max(0.0, maze_score),
        "reason": "maze_only_template_match" if maze_confident else "maze_only_not_confident",
        "maze_confident": maze_confident,
        "maze_score": maze_score,
        "sign_found": True,
    }


def wait_for_maze_template_start(
    display,
    hold_stop: Callable[[], None],
    stop: Callable[[], None],
) -> None:
    template.ensure_default_maze_template()
    camera = motion.create_camera_source()
    maze_hits = 0
    last_reported_label = None
    last_reported_reason = None

    print("[template] waiting for MAZE template from line_following_v1_8_maze_navigation.py ...")
    display.show_status("Wait Template")
    try:
        while True:
            frame_bgr = camera.capture_bgr()
            sign_roi = template.base.detect_sign_roi(frame_bgr)
            sign_result = classify_maze_only_sign(sign_roi)

            current_label = sign_result.get("label", "UNKNOWN")
            current_conf = float(sign_result.get("confidence", 0.0))
            current_reason = sign_result.get("reason", "")
            if current_label != last_reported_label or current_reason != last_reported_reason:
                print(
                    "[template] "
                    f"label={current_label} confidence={current_conf:.3f} "
                    f"reason={current_reason} sign_found={sign_result.get('sign_found', False)} "
                    f"maze_score={sign_result.get('maze_score', -1.0):.3f}"
                )
                last_reported_label = current_label
                last_reported_reason = current_reason

            if sign_result.get("maze_confident", False):
                maze_hits += 1
            else:
                maze_hits = 0

            hold_stop()

            if maze_hits > 0:
                display.show_status(f"Maze {maze_hits}/{template.MAZE_CONFIRMATION_HITS}")
            else:
                display.show_status("Wait Template")

            if maze_hits >= template.MAZE_CONFIRMATION_HITS:
                stop()
                deadline = time.perf_counter() + float(template.MAZE_COUNTDOWN_SECONDS)
                while True:
                    now = time.perf_counter()
                    remaining = max(0, int(math.ceil(deadline - now)))
                    display.show_status(f"Maze {remaining}s")
                    hold_stop()
                    if now >= deadline:
                        break
                    time.sleep(0.10)
                stop()
                print("[template] maze template confirmed, start navigation")
                return

            time.sleep(float(motion.MAZE_FRAME_SLEEP_SECONDS))
    finally:
        camera.close()


def main() -> int:
    args = motion.parse_args()
    config = motion.make_config(args)

    turn = None
    distance = None
    motion_module = None
    display = None
    controller: Optional[object] = None

    if args.dry_run:
        sim_values = []
        for token in args.sim_distances.split(","):
            token = token.strip()
            if token:
                sim_values.append(float(token))
        turn = motion.SimTurnAdapter()
        distance = motion.SimDistanceAdapter(sim_values)
        motion_module = motion.SimMotionAdapter()
        display = motion.SimDisplay()
    else:
        controller = motion.TrimmedCarMotorController(
            right_rear_boost=motion.RIGHT_REAR_BOOST,
            right_rear_spin_boost=0,
        )
        motion.install_stop_handlers(controller, None)

        turn_motion = motion.ForwardTurnBasicController(
            controller=controller,
            stop_between_stages_seconds=config.stop_between_stages_seconds,
        )
        turn = turn_motion
        distance = motion.Hcsr04DistanceAdapter(
            trig_pin=args.trig_pin,
            echo_pin=args.echo_pin,
        )
        motion_module = turn_motion
        try:
            display = motion.LCD1602MazeDisplay()
        except Exception as exc:
            print(f"[WARN] LCD init failed, continue without LCD: {exc}")
            display = motion.NullDisplay()

    summary = None
    try:
        if (not args.dry_run) and (controller is not None):
            wait_for_maze_template_start(
                display=display,
                hold_stop=controller.hold_stop,
                stop=controller.stop,
            )
            config.start_delay_seconds = 0.0

        navigator = motion.MazeNavigator(
            config=config,
            turn=turn,
            distance=distance,
            motion=motion_module,
            display=display,
        )
        summary = navigator.run()

        if args.summary_json is not None:
            args.summary_json.write_text(
                json.dumps(summary.to_dict(), ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        try:
            if turn is not None:
                turn.close()
        except Exception:
            pass
        try:
            if distance is not None:
                distance.close()
        except Exception:
            pass
        try:
            if motion_module is not None:
                motion_module.close()
        except Exception:
            pass
        try:
            if display is not None:
                display.close()
        except Exception:
            pass

    if summary is None:
        return 1
    return 0 if summary.status == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
