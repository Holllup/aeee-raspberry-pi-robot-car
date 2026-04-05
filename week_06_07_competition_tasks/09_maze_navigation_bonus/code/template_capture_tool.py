import argparse
import os
import time
from pathlib import Path

import cv2


CAPTURE_SIZE = (1280, 720)
PREVIEW_SIZE = (640, 360)


class Picamera2Source:
    def __init__(self, capture_size):
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": capture_size, "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(0.25)

    def capture_bgr(self):
        frame_rgb = self._picam2.capture_array()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        self._picam2.stop()


class OpenCVCameraSource:
    def __init__(self, capture_size):
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        if not self._cap.isOpened():
            raise RuntimeError("unable to open camera by cv2.VideoCapture(0)")
        time.sleep(0.2)

    def capture_bgr(self):
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("camera read failed")
        return frame_bgr

    def close(self):
        self._cap.release()


def create_camera_source():
    try:
        camera = Picamera2Source(CAPTURE_SIZE)
        print("[capture] camera source: Picamera2")
        return camera
    except Exception as exc:
        print(f"[capture][warn] Picamera2 unavailable: {exc}")

    camera = OpenCVCameraSource(CAPTURE_SIZE)
    print("[capture] camera source: OpenCV VideoCapture")
    return camera


def next_output_path(output_dir: Path, prefix: str) -> Path:
    existing = sorted(output_dir.glob(f"{prefix}_*.png"))
    if not existing:
        return output_dir / f"{prefix}_001.png"
    last_name = existing[-1].stem
    try:
        index = int(last_name.split("_")[-1]) + 1
    except Exception:
        index = len(existing) + 1
    return output_dir / f"{prefix}_{index:03d}.png"


def draw_overlay(frame, save_dir: Path, prefix: str, shot_count: int):
    view = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
    lines = [
        "Template Capture Tool",
        "Put the real sign/template in front of camera",
        "Press S to save current frame",
        "Press Q or ESC to quit",
        f"save_dir: {save_dir}",
        f"prefix: {prefix}",
        f"saved: {shot_count}",
    ]
    y = 28
    for text in lines:
        cv2.putText(
            view,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return view


def main():
    parser = argparse.ArgumentParser(description="Preview the camera and save template reference photos.")
    parser.add_argument(
        "--output-dir",
        default="~/maze_captures",
        help="Directory used to save captured png files.",
    )
    parser.add_argument(
        "--prefix",
        default="template",
        help="Filename prefix for saved images.",
    )
    parser.add_argument(
        "--headless-shot",
        action="store_true",
        help="Capture a single frame without opening any preview window.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.5,
        help="Seconds to wait before capturing in headless mode.",
    )
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = create_camera_source()
    shot_count = 0
    last_save_text = ""
    last_save_until = 0.0

    try:
        if args.headless_shot:
            warmup_seconds = max(0.0, float(args.warmup_seconds))
            if warmup_seconds > 0:
                print(f"[capture] warming up camera for {warmup_seconds:.1f}s")
                time.sleep(warmup_seconds)
            frame = camera.capture_bgr()
            output_path = next_output_path(output_dir, args.prefix)
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"failed to save headless capture: {output_path}")
            print(f"[capture] saved: {output_path}")
            return

        while True:
            frame = camera.capture_bgr()
            preview = draw_overlay(frame, output_dir, args.prefix, shot_count)

            now = time.perf_counter()
            if now < last_save_until and last_save_text:
                cv2.putText(
                    preview,
                    last_save_text,
                    (14, preview.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Template Capture Tool", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                output_path = next_output_path(output_dir, args.prefix)
                if not cv2.imwrite(str(output_path), frame):
                    print(f"[capture][error] failed to save: {output_path}")
                    last_save_text = "save failed"
                else:
                    shot_count += 1
                    print(f"[capture] saved: {output_path}")
                    last_save_text = f"saved: {output_path.name}"
                last_save_until = time.perf_counter() + 1.5
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
