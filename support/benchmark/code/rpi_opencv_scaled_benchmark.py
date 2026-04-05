import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


def measure_temperature_c():
    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(temp_path.read_text().strip()) / 1000.0, 2)
    except Exception:
        return None


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for contour in contours if cv2.contourArea(contour) >= 80)


def run_benchmark(capture_size, process_size, duration_seconds, warmup_seconds):
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": capture_size, "format": "BGR888"},
        buffer_count=3,
        queue=False,
    )
    camera.configure(config)
    camera.start()

    try:
        time.sleep(warmup_seconds)
        frames = 0
        frame_times_ms = []
        capture_times_ms = []
        resize_times_ms = []
        processing_times_ms = []
        contour_counts = []
        start = time.perf_counter()

        while time.perf_counter() - start < duration_seconds:
            t0 = time.perf_counter()
            frame = camera.capture_array()
            t1 = time.perf_counter()
            resized = cv2.resize(frame, process_size, interpolation=cv2.INTER_AREA)
            t2 = time.perf_counter()
            contour_count = process_frame(resized)
            t3 = time.perf_counter()

            frames += 1
            capture_times_ms.append((t1 - t0) * 1000.0)
            resize_times_ms.append((t2 - t1) * 1000.0)
            processing_times_ms.append((t3 - t2) * 1000.0)
            frame_times_ms.append((t3 - t0) * 1000.0)
            contour_counts.append(contour_count)

        elapsed = time.perf_counter() - start
    finally:
        camera.stop()
        camera.close()

    p95_frame_ms = statistics.quantiles(frame_times_ms, n=20)[18] if len(frame_times_ms) >= 20 else max(frame_times_ms)
    return {
        "capture_resolution": f"{capture_size[0]}x{capture_size[1]}",
        "process_resolution": f"{process_size[0]}x{process_size[1]}",
        "frames": frames,
        "elapsed_s": round(elapsed, 3),
        "avg_fps": round(frames / elapsed, 2),
        "avg_frame_ms": round(statistics.fmean(frame_times_ms), 2),
        "p95_frame_ms": round(p95_frame_ms, 2),
        "avg_capture_ms": round(statistics.fmean(capture_times_ms), 2),
        "avg_resize_ms": round(statistics.fmean(resize_times_ms), 2),
        "avg_processing_ms": round(statistics.fmean(processing_times_ms), 2),
        "avg_contours": round(statistics.fmean(contour_counts), 2),
        "temperature_c": measure_temperature_c(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark 1280x720 capture with lower internal processing resolution.")
    parser.add_argument("--capture-width", type=int, default=1280)
    parser.add_argument("--capture-height", type=int, default=720)
    parser.add_argument("--process-width", type=int, default=640)
    parser.add_argument("--process-height", type=int, default=360)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=1.5)
    args = parser.parse_args()

    result = run_benchmark(
        (args.capture_width, args.capture_height),
        (args.process_width, args.process_height),
        args.duration,
        args.warmup,
    )
    summary = {
        "mode": "full_fov_scaled_processing",
        "description": "Capture keeps full field of view, processing runs on resized frame.",
        "result": result,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
