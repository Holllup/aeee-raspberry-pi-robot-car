import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


CAPTURE_SIZE = (1280, 720)
PROCESS_SIZE = (640, 360)

BLACK_THRESHOLD = 80
MIN_CONTOUR_AREA = 180
ROI_CENTER_RATIO = 0.58
ROI_HEIGHT_RATIO = 0.24
RECOVERY_ROI_CENTER_RATIO = 0.74
RECOVERY_ROI_HEIGHT_RATIO = 0.40

PURPLE_MIN_CONTOUR_AREA = 160
PURPLE_MIN_SOLIDITY = 0.60
PURPLE_MIN_FILL_RATIO = 0.45
PURPLE_MAX_ASPECT_RATIO_ERROR = 0.60
PURPLE_MIN_SATURATION = 35
PURPLE_MIN_VALUE = 80
PURPLE_POLY_EPSILON_RATIO = 0.05
PURPLE_LOWER_HSV = np.array([145, 50, 80], dtype=np.uint8)
PURPLE_UPPER_HSV = np.array([170, 255, 255], dtype=np.uint8)
PURPLE_RIGHT_REGION_RATIO = 0.42


def measure_temperature_c():
    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return round(int(temp_path.read_text().strip()) / 1000.0, 2)
    except Exception:
        return None


def find_line_center_in_roi(frame_bgr, black_threshold, roi_center_ratio, roi_height_ratio):
    frame_height = frame_bgr.shape[0]
    roi_height = max(1, int(frame_height * roi_height_ratio))
    roi_center = int(frame_height * roi_center_ratio)
    roi_top = max(0, roi_center - roi_height // 2)
    roi_bottom = min(frame_height, roi_top + roi_height)
    roi_bgr = frame_bgr[roi_top:roi_bottom, :]

    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, black_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"]) + roi_top
    return center_x, center_y


def find_line_center(frame_bgr):
    center = find_line_center_in_roi(frame_bgr, BLACK_THRESHOLD, ROI_CENTER_RATIO, ROI_HEIGHT_RATIO)
    if center is not None:
        return center, "primary"
    center = find_line_center_in_roi(frame_bgr, BLACK_THRESHOLD, RECOVERY_ROI_CENTER_RATIO, RECOVERY_ROI_HEIGHT_RATIO)
    if center is not None:
        return center, "recovery"
    return None, "lost"


def detect_right_purple_square(frame_bgr):
    frame_height, frame_width = frame_bgr.shape[:2]
    roi_left = int(frame_width * (1.0 - PURPLE_RIGHT_REGION_RATIO))
    roi = frame_bgr[:, roi_left:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation_mask = cv2.inRange(hsv[:, :, 1], PURPLE_MIN_SATURATION, 255)
    value_mask = cv2.inRange(hsv[:, :, 2], PURPLE_MIN_VALUE, 255)
    hue_mask = cv2.inRange(hsv, PURPLE_LOWER_HSV, PURPLE_UPPER_HSV)
    mask = cv2.bitwise_and(hue_mask, cv2.bitwise_and(saturation_mask, value_mask))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= PURPLE_MIN_CONTOUR_AREA]
    if not contours:
        return None

    best = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, PURPLE_POLY_EPSILON_RATIO * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(approx)
        if w <= 0 or h <= 0:
            continue

        aspect_error = abs(1.0 - (w / float(h)))
        fill_ratio = area / float(w * h)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if aspect_error > PURPLE_MAX_ASPECT_RATIO_ERROR:
            continue
        if fill_ratio < PURPLE_MIN_FILL_RATIO:
            continue
        if solidity < PURPLE_MIN_SOLIDITY:
            continue

        candidate = {
            "rect": (x + roi_left, y, w, h),
            "area": area,
            "fill_ratio": fill_ratio,
            "solidity": solidity,
        }
        if best is None or candidate["area"] > best["area"]:
            best = candidate
    return best


def benchmark(duration_s=10.0, warmup_s=1.5):
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": CAPTURE_SIZE, "format": "BGR888"},
        buffer_count=3,
        queue=False,
    )
    camera.configure(config)
    camera.start()
    try:
        time.sleep(warmup_s)
        frames = 0
        capture_ms = []
        resize_ms = []
        line_ms = []
        purple_ms = []
        total_ms = []
        line_hits = 0
        recovery_hits = 0
        purple_hits = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_s:
            t0 = time.perf_counter()
            frame = camera.capture_array()
            t1 = time.perf_counter()
            small = cv2.resize(frame, PROCESS_SIZE, interpolation=cv2.INTER_AREA)
            t2 = time.perf_counter()
            line_center, line_mode = find_line_center(small)
            t3 = time.perf_counter()
            purple = detect_right_purple_square(small)
            t4 = time.perf_counter()

            frames += 1
            capture_ms.append((t1 - t0) * 1000.0)
            resize_ms.append((t2 - t1) * 1000.0)
            line_ms.append((t3 - t2) * 1000.0)
            purple_ms.append((t4 - t3) * 1000.0)
            total_ms.append((t4 - t0) * 1000.0)

            if line_center is not None:
                line_hits += 1
                if line_mode == "recovery":
                    recovery_hits += 1
            if purple is not None:
                purple_hits += 1

        elapsed = time.perf_counter() - start
    finally:
        camera.stop()
        camera.close()

    p95_total = statistics.quantiles(total_ms, n=20)[18] if len(total_ms) >= 20 else max(total_ms)
    return {
        "capture_resolution": f"{CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]}",
        "process_resolution": f"{PROCESS_SIZE[0]}x{PROCESS_SIZE[1]}",
        "duration_s": round(elapsed, 3),
        "frames": frames,
        "avg_fps": round(frames / elapsed, 2),
        "avg_total_ms": round(statistics.fmean(total_ms), 2),
        "p95_total_ms": round(p95_total, 2),
        "avg_capture_ms": round(statistics.fmean(capture_ms), 2),
        "avg_resize_ms": round(statistics.fmean(resize_ms), 2),
        "avg_line_ms": round(statistics.fmean(line_ms), 2),
        "avg_purple_ms": round(statistics.fmean(purple_ms), 2),
        "line_detect_rate": round(line_hits / frames, 3) if frames else 0.0,
        "recovery_rate": round(recovery_hits / frames, 3) if frames else 0.0,
        "purple_detect_rate": round(purple_hits / frames, 3) if frames else 0.0,
        "temperature_c": measure_temperature_c(),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
