from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2


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


motion = load_local_module("maze_ultrasonic_turn_2_motion_real_photo", "maze_navigation_ultrasonic_turn 2.py")

REAL_TEMPLATE_CANDIDATES = (
    Path(__file__).resolve().with_name("maze_real_001.jpg"),
    Path(__file__).resolve().with_name("maze_real_001.png"),
    Path(__file__).resolve().parent.parent / "templates" / "maze_real_001.jpg",
    Path(__file__).resolve().parent.parent / "templates" / "maze_real_001.png",
    Path.home() / "maze" / "maze_real_001.jpg",
    Path.home() / "maze" / "maze_real_001.png",
    Path.home() / "template_photos" / "maze_real_001.jpg",
    Path.home() / "template_photos" / "maze_real_001.png",
)
REAL_TEMPLATE_XOR_THRESHOLD = 0.45
REAL_TEMPLATE_EDGE_THRESHOLD = 0.82
REAL_TEMPLATE_ORB_GOOD_MATCHES_THRESHOLD = 30
REAL_TEMPLATE_CONFIRM_HITS = 3
REAL_TEMPLATE_COUNTDOWN_SECONDS = 5.0
REAL_TEMPLATE_FRAME_SLEEP_SECONDS = 0.04
REAL_TEMPLATE_DEBUG_DIRNAME = "real_photo_debug"
ORB_DETECTOR = cv2.ORB_create(nfeatures=500)
ORB_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def resolve_real_template_path() -> Path:
    for candidate in REAL_TEMPLATE_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in REAL_TEMPLATE_CANDIDATES)
    raise RuntimeError(f"unable to find real maze photo template, searched:\n{searched}")


def build_real_maze_template(template_path: Path):
    image_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError(f"unable to read real maze template image: {template_path}")

    sign_roi = motion.detect_sign_roi(image_bgr)
    if sign_roi is None:
        raise RuntimeError(f"unable to detect sign roi from real maze photo: {template_path}")

    warped = sign_roi.get("warped")
    binary = motion.preprocess_sign_symbol(warped)
    if binary is None or binary.size == 0:
        raise RuntimeError(f"unable to preprocess sign symbol from real maze photo: {template_path}")

    print(f"[template] using real maze photo template: {template_path}")
    return {
        "binary": binary,
        "warped": warped,
    }


def get_debug_dir() -> Path:
    debug_dir = Path(__file__).resolve().parent / REAL_TEMPLATE_DEBUG_DIRNAME
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def save_debug_images(
    frame_bgr,
    sign_roi,
    binary,
    real_template,
    reason: str,
    xor_score: float,
    edge_score: float,
    orb_good_matches: int,
) -> None:
    debug_dir = get_debug_dir()
    if frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
        cv2.imwrite(str(debug_dir / "latest_frame.jpg"), frame_bgr)
    if sign_roi is not None:
        warped = sign_roi.get("warped")
        if warped is not None and getattr(warped, "size", 0) > 0:
            cv2.imwrite(str(debug_dir / "latest_warped.jpg"), warped)
    if binary is not None and getattr(binary, "size", 0) > 0:
        cv2.imwrite(str(debug_dir / "latest_binary.png"), binary)
    if isinstance(real_template, dict):
        template_binary = real_template.get("binary")
        template_warped = real_template.get("warped")
        if template_binary is not None and getattr(template_binary, "size", 0) > 0:
            cv2.imwrite(str(debug_dir / "real_template_binary.png"), template_binary)
        if template_warped is not None and getattr(template_warped, "size", 0) > 0:
            cv2.imwrite(str(debug_dir / "real_template_warped.jpg"), template_warped)
    (debug_dir / "latest_metrics.txt").write_text(
        (
            f"reason={reason}\n"
            f"xor_score={xor_score:.6f}\n"
            f"edge_score={edge_score:.6f}\n"
            f"orb_good_matches={orb_good_matches}\n"
        ),
        encoding="utf-8",
    )


def compute_edge_similarity(sample_warped, template_warped) -> float:
    sample_gray = cv2.cvtColor(sample_warped, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_warped, cv2.COLOR_BGR2GRAY)
    if sample_gray.shape != template_gray.shape:
        sample_gray = cv2.resize(
            sample_gray,
            (template_gray.shape[1], template_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    sample_edges = cv2.Canny(sample_gray, 60, 150)
    template_edges = cv2.Canny(template_gray, 60, 150)
    xor_edges = cv2.bitwise_xor(sample_edges, template_edges)
    difference_pixels = cv2.countNonZero(xor_edges)
    return 1.0 - difference_pixels / float(xor_edges.shape[0] * xor_edges.shape[1])


def compute_orb_good_matches(sample_warped, template_warped) -> int:
    sample_gray = cv2.cvtColor(sample_warped, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_warped, cv2.COLOR_BGR2GRAY)
    sample_keypoints, sample_desc = ORB_DETECTOR.detectAndCompute(sample_gray, None)
    template_keypoints, template_desc = ORB_DETECTOR.detectAndCompute(template_gray, None)
    if sample_desc is None or template_desc is None:
        return 0
    matches = ORB_MATCHER.match(sample_desc, template_desc)
    return sum(1 for match in matches if match.distance < 55)


def compare_with_real_maze_template(binary, sample_warped, real_template):
    template_binary = real_template["binary"]
    sample = binary
    if sample.shape != template_binary.shape:
        sample = cv2.resize(
            sample,
            (template_binary.shape[1], template_binary.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    xor_frame = cv2.bitwise_xor(sample, template_binary)
    difference_pixels = cv2.countNonZero(xor_frame)
    xor_score = 1.0 - difference_pixels / float(xor_frame.shape[0] * xor_frame.shape[1])
    edge_score = compute_edge_similarity(sample_warped, real_template["warped"])
    orb_good_matches = compute_orb_good_matches(sample_warped, real_template["warped"])
    confident = (
        xor_score >= REAL_TEMPLATE_XOR_THRESHOLD
        and edge_score >= REAL_TEMPLATE_EDGE_THRESHOLD
        and orb_good_matches >= REAL_TEMPLATE_ORB_GOOD_MATCHES_THRESHOLD
    )
    return {
        "xor_score": float(xor_score),
        "edge_score": float(edge_score),
        "orb_good_matches": int(orb_good_matches),
        "maze_confident": bool(confident),
    }


def wait_for_real_maze_start(
    display,
    hold_stop: Callable[[], None],
    stop: Callable[[], None],
) -> None:
    template_path = resolve_real_template_path()
    real_template = build_real_maze_template(template_path)
    camera = motion.create_camera_source()
    maze_hits = 0
    last_report = ""

    print("[template] waiting for real MAZE photo template...")
    display.show_status("Wait Maze")
    try:
        while True:
            frame_bgr = camera.capture_bgr()
            sign_roi = motion.detect_sign_roi(frame_bgr)
            binary = None

            if sign_roi is None:
                result = {
                    "maze_confident": False,
                    "xor_score": -1.0,
                    "edge_score": -1.0,
                    "orb_good_matches": 0,
                    "label": "UNKNOWN",
                    "reason": "no_sign",
                }
            else:
                binary = motion.preprocess_sign_symbol(sign_roi["warped"])
                if binary is None:
                    result = {
                        "maze_confident": False,
                        "xor_score": -1.0,
                        "edge_score": -1.0,
                        "orb_good_matches": 0,
                        "label": "UNKNOWN",
                        "reason": "empty_binary",
                    }
                else:
                    comparison = compare_with_real_maze_template(binary, sign_roi["warped"], real_template)
                    result = {
                        "maze_confident": comparison["maze_confident"],
                        "xor_score": comparison["xor_score"],
                        "edge_score": comparison["edge_score"],
                        "orb_good_matches": comparison["orb_good_matches"],
                        "label": "MAZE" if comparison["maze_confident"] else "UNKNOWN",
                        "reason": "real_photo_match" if comparison["maze_confident"] else "real_photo_not_confident",
                    }

            save_debug_images(
                frame_bgr=frame_bgr,
                sign_roi=sign_roi,
                binary=binary,
                real_template=real_template,
                reason=result["reason"],
                xor_score=result["xor_score"],
                edge_score=result["edge_score"],
                orb_good_matches=result["orb_good_matches"],
            )

            if result["maze_confident"]:
                maze_hits += 1
            else:
                maze_hits = 0

            hold_stop()

            report = (
                f"label={result['label']} "
                f"xor={result['xor_score']:.3f} "
                f"edge={result['edge_score']:.3f} "
                f"orb={result['orb_good_matches']} "
                f"hits={maze_hits}/{REAL_TEMPLATE_CONFIRM_HITS}"
            )
            if report != last_report:
                print(f"[template] {report}")
                last_report = report

            if maze_hits > 0:
                display.show_status(f"Maze {maze_hits}/{REAL_TEMPLATE_CONFIRM_HITS}")
            else:
                display.show_status("Unknown")

            if maze_hits >= REAL_TEMPLATE_CONFIRM_HITS:
                stop()
                deadline = time.perf_counter() + REAL_TEMPLATE_COUNTDOWN_SECONDS
                while True:
                    now = time.perf_counter()
                    remaining = max(0, int(math.ceil(deadline - now)))
                    display.show_status(f"Maze {remaining}s")
                    hold_stop()
                    if now >= deadline:
                        break
                    time.sleep(0.10)
                stop()
                print("[template] real maze template confirmed, start navigation")
                return

            time.sleep(REAL_TEMPLATE_FRAME_SLEEP_SECONDS)
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
            wait_for_real_maze_start(
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
