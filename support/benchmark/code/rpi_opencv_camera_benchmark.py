import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


DEFAULT_RESOLUTIONS = [
    (320, 240),
    (640, 480),
    (960, 720),
    (1280, 720),
    (1640, 1232),
    (1920, 1080),
]


def parse_resolution(text):
    width_text, height_text = text.lower().split("x", 1)
    return int(width_text), int(height_text)


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
    contour_count = sum(1 for contour in contours if cv2.contourArea(contour) >= 80)
    return contour_count


def benchmark_resolution(width, height, duration_seconds, warmup_seconds):
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (width, height), "format": "BGR888"},
        buffer_count=3,
        queue=False,
    )
    camera.configure(config)
    camera.start()

    try:
        time.sleep(warmup_seconds)
        frame_times_ms = []
        processing_times_ms = []
        capture_times_ms = []
        contour_counts = []
        start = time.perf_counter()
        frames = 0

        while time.perf_counter() - start < duration_seconds:
            loop_start = time.perf_counter()
            frame = camera.capture_array()
            capture_done = time.perf_counter()
            contour_count = process_frame(frame)
            loop_end = time.perf_counter()

            frames += 1
            contour_counts.append(contour_count)
            capture_times_ms.append((capture_done - loop_start) * 1000.0)
            processing_times_ms.append((loop_end - capture_done) * 1000.0)
            frame_times_ms.append((loop_end - loop_start) * 1000.0)

        elapsed = time.perf_counter() - start
    finally:
        camera.stop()
        camera.close()

    average_frame_ms = statistics.fmean(frame_times_ms)
    average_fps = frames / elapsed if elapsed > 0 else 0.0
    p95_frame_ms = statistics.quantiles(frame_times_ms, n=20)[18] if len(frame_times_ms) >= 20 else max(frame_times_ms)

    return {
        "resolution": f"{width}x{height}",
        "frames": frames,
        "elapsed_s": round(elapsed, 3),
        "avg_fps": round(average_fps, 2),
        "avg_frame_ms": round(average_frame_ms, 2),
        "p95_frame_ms": round(p95_frame_ms, 2),
        "avg_capture_ms": round(statistics.fmean(capture_times_ms), 2),
        "avg_processing_ms": round(statistics.fmean(processing_times_ms), 2),
        "avg_contours": round(statistics.fmean(contour_counts), 2),
        "temperature_c": measure_temperature_c(),
    }


def choose_recommendation(results):
    realtime_candidates = [item for item in results if item["avg_fps"] >= 20]
    if realtime_candidates:
        return max(
            realtime_candidates,
            key=lambda item: (
                int(item["resolution"].split("x")[0]) * int(item["resolution"].split("x")[1]),
                item["avg_fps"],
            ),
        )

    usable_candidates = [item for item in results if item["avg_fps"] >= 10]
    if usable_candidates:
        return max(
            usable_candidates,
            key=lambda item: (
                int(item["resolution"].split("x")[0]) * int(item["resolution"].split("x")[1]),
                item["avg_fps"],
            ),
        )

    return max(results, key=lambda item: item["avg_fps"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=1.5)
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=[f"{w}x{h}" for w, h in DEFAULT_RESOLUTIONS],
    )
    args = parser.parse_args()

    resolutions = [parse_resolution(text) for text in args.resolutions]
    results = []
    for width, height in resolutions:
        print(f"Running benchmark for {width}x{height} ...", flush=True)
        result = benchmark_resolution(width, height, args.duration, args.warmup)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    recommendation = choose_recommendation(results)
    summary = {
        "results": results,
        "recommended": recommendation,
        "notes": [
            "recommended 优先选择可维持实时性的最高分辨率",
            "如果你的算法更重，比如 DNN/YOLO，建议在本结果基础上再降一档",
        ],
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
