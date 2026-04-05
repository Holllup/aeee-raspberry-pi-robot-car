import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


BLACK_THRESHOLD = 80
MIN_LINE_AREA = 180
PRIMARY_ROI_CENTER_RATIO = 0.58
PRIMARY_ROI_HEIGHT_RATIO = 0.24
RECOVERY_ROI_CENTER_RATIO = 0.74
RECOVERY_ROI_HEIGHT_RATIO = 0.40

PURPLE_RIGHT_REGION_RATIO = 0.42
PURPLE_MIN_AREA = 90
PURPLE_MIN_SOLIDITY = 0.35
PURPLE_MIN_FILL_RATIO = 0.20
PURPLE_MAX_ASPECT_ERROR = 1.20
PURPLE_MIN_SATURATION = 20
PURPLE_MIN_VALUE = 60
PURPLE_POLY_EPSILON_RATIO = 0.05
PURPLE_LOWER_HSV = np.array([140, 20, 60], dtype=np.uint8)
PURPLE_UPPER_HSV = np.array([179, 255, 255], dtype=np.uint8)
RED_LOWER_HSV = np.array([0, 35, 60], dtype=np.uint8)
RED_UPPER_HSV = np.array([14, 255, 255], dtype=np.uint8)

CORNER_ERROR_THRESHOLD = 34
CORNER_PIVOT_TIMEOUT_SECONDS = 0.72
RECOVERY_HOLD_SECONDS = 0.05
RECOVERY_TURN_TIMEOUT_SECONDS = 1.15
TURN_DIRECTION_LOCK_THRESHOLD = 22
TURN_DIRECTION_MEMORY_SECONDS = 0.80
EDGE_DIRECTION_MARGIN_RATIO = 0.18
CURVE_SAMPLE_ROI_CENTER_RATIOS = (0.52, 0.62, 0.72, 0.82)
CURVE_SAMPLE_ROI_HEIGHT_RATIO = 0.12
ENTRY_MIN_VALID_POINTS = 2
ENTRY_MIN_SPREAD_PX = 14
ENTRY_MIN_SLOPE_PX_PER_ROW = 0.10
ENTRY_SHARP_SPREAD_PX = 46
ENTRY_SHARP_CURVATURE_PX = 24
ENTRY_MEMORY_SECONDS = 0.50
ENTRY_ASSIST_SCORE_THRESHOLD = 0.42


def find_line_center_in_roi(frame_bgr, roi_center_ratio, roi_height_ratio):
    frame_height = frame_bgr.shape[0]
    roi_height = max(1, int(frame_height * roi_height_ratio))
    roi_center = int(frame_height * roi_center_ratio)
    roi_top = max(0, roi_center - roi_height // 2)
    roi_bottom = min(frame_height, roi_top + roi_height)

    roi_bgr = frame_bgr[roi_top:roi_bottom, :]
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_LINE_AREA]
    if not contours:
        return None, mask, roi_top, roi_bottom, None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, mask, roi_top, roi_bottom, contour

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"]) + roi_top
    return (cx, cy), mask, roi_top, roi_bottom, contour


def find_line_center(frame_bgr):
    result = find_line_center_in_roi(frame_bgr, PRIMARY_ROI_CENTER_RATIO, PRIMARY_ROI_HEIGHT_RATIO)
    if result[0] is not None:
        return result, "primary"
    return find_line_center_in_roi(frame_bgr, RECOVERY_ROI_CENTER_RATIO, RECOVERY_ROI_HEIGHT_RATIO), "recovery"


def sample_curve_points(frame_bgr):
    points = []
    for roi_center_ratio in CURVE_SAMPLE_ROI_CENTER_RATIOS:
        center, _, _, _, _ = find_line_center_in_roi(frame_bgr, roi_center_ratio, CURVE_SAMPLE_ROI_HEIGHT_RATIO)
        if center is not None:
            points.append(center)
    return points


def infer_curve_entry(points):
    if len(points) < ENTRY_MIN_VALID_POINTS:
        return {
            "direction": 0,
            "kind": "lost",
            "score": 0.0,
            "spread_px": None,
            "slope": None,
            "curvature_px": None,
        }

    ordered = sorted(points, key=lambda p: p[1])
    xs = np.array([p[0] for p in ordered], dtype=np.float32)
    ys = np.array([p[1] for p in ordered], dtype=np.float32)
    x_top, x_bottom = xs[0], xs[-1]
    y_top, y_bottom = ys[0], ys[-1]
    dy = max(1.0, float(y_bottom - y_top))
    slope = float((x_bottom - x_top) / dy)
    spread = float(np.max(xs) - np.min(xs))

    curvature = 0.0
    if len(xs) >= 3:
        mid_idx = len(xs) // 2
        curvature = float(x_bottom - 2.0 * xs[mid_idx] + x_top)

    direction = 0
    if abs(slope) >= ENTRY_MIN_SLOPE_PX_PER_ROW or spread >= ENTRY_MIN_SPREAD_PX:
        direction = 1 if slope >= 0 else -1

    if abs(slope) < ENTRY_MIN_SLOPE_PX_PER_ROW and spread < ENTRY_MIN_SPREAD_PX:
        curve_kind = "straight"
    elif spread >= ENTRY_SHARP_SPREAD_PX or abs(curvature) >= ENTRY_SHARP_CURVATURE_PX:
        curve_kind = "sharp"
    else:
        curve_kind = "gentle"

    score = min(
        1.0,
        0.60 * (spread / ENTRY_SHARP_SPREAD_PX)
        + 0.25 * (abs(slope) / ENTRY_MIN_SLOPE_PX_PER_ROW)
        + 0.15 * (abs(curvature) / ENTRY_SHARP_CURVATURE_PX),
    )
    return {
        "direction": direction,
        "kind": curve_kind,
        "score": round(score, 4),
        "spread_px": round(spread, 3),
        "slope": round(slope, 4),
        "curvature_px": round(curvature, 3),
    }


def detect_right_purple_square(frame_bgr):
    h, w = frame_bgr.shape[:2]
    roi_left = int(w * (1.0 - PURPLE_RIGHT_REGION_RATIO))
    roi = frame_bgr[:, roi_left:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation_mask = cv2.inRange(hsv[:, :, 1], PURPLE_MIN_SATURATION, 255)
    value_mask = cv2.inRange(hsv[:, :, 2], PURPLE_MIN_VALUE, 255)
    hue_mask_high = cv2.inRange(hsv, PURPLE_LOWER_HSV, PURPLE_UPPER_HSV)
    hue_mask_low = cv2.inRange(hsv, RED_LOWER_HSV, RED_UPPER_HSV)
    hue_mask = cv2.bitwise_or(hue_mask_high, hue_mask_low)
    mask = cv2.bitwise_and(hue_mask, cv2.bitwise_and(saturation_mask, value_mask))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= PURPLE_MIN_AREA]
    if not contours:
        return None

    best = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, PURPLE_POLY_EPSILON_RATIO * perimeter, True)
        if len(approx) < 4 or len(approx) > 8 or not cv2.isContourConvex(approx):
            continue

        area = float(cv2.contourArea(contour))
        x, y, ww, hh = cv2.boundingRect(approx)
        if ww <= 0 or hh <= 0:
            continue
        aspect_error = abs(1.0 - (ww / float(hh)))
        fill_ratio = area / float(ww * hh)
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if aspect_error > PURPLE_MAX_ASPECT_ERROR:
            continue
        if fill_ratio < PURPLE_MIN_FILL_RATIO:
            continue
        if solidity < PURPLE_MIN_SOLIDITY:
            continue

        cand = {"rect": (x + roi_left, y, ww, hh), "area": area}
        if best is None or cand["area"] > best["area"]:
            best = cand
    return best


def analyze_video(video_path, output_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    stem = video_path.stem
    csv_path = output_dir / f"{stem}_analysis.csv"
    out_video_path = output_dir / f"{stem}_annotated.mp4"
    summary_path = output_dir / f"{stem}_summary.json"

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    line_hits = 0
    purple_hits = 0
    abs_errors = []
    frame_index = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "frame",
                "time_s",
                "line_found",
                "line_mode",
                "line_center_x",
                "line_error_px",
                "drive_mode",
                "turn_direction",
                "entry_direction",
                "entry_kind",
                "entry_score",
                "entry_spread_px",
                "entry_slope",
                "entry_curvature_px",
                "purple_found",
                "purple_x",
                "purple_y",
                "purple_w",
                "purple_h",
                "purple_area",
            ]
        )

        last_line_time = 0.0
        last_nonzero_error = 0
        recovery_turn_direction = 1
        recovery_turn_time = -999.0
        last_entry_direction = 0
        last_entry_kind = "lost"
        last_entry_score = 0.0
        last_entry_time = -999.0
        mode_counter = {
            "track": 0,
            "hold": 0,
            "corner_pivot": 0,
            "search_turn": 0,
            "stop": 0,
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            t = frame_index / fps

            curve_points = sample_curve_points(frame)
            entry = infer_curve_entry(curve_points)
            if entry["direction"] != 0 and entry["kind"] != "straight":
                last_entry_direction = entry["direction"]
                last_entry_kind = entry["kind"]
                last_entry_score = entry["score"]
                last_entry_time = t

            (center, mask, roi_top, roi_bottom, contour), line_mode = find_line_center(frame)
            frame_center_x = width // 2
            line_error = None
            if center is not None:
                line_hits += 1
                line_error = int(center[0] - frame_center_x)
                abs_errors.append(abs(line_error))
                last_line_time = t
                if line_error != 0:
                    last_nonzero_error = line_error
                edge_margin = width * EDGE_DIRECTION_MARGIN_RATIO
                if center[0] <= edge_margin:
                    recovery_turn_direction = -1
                    recovery_turn_time = t
                elif center[0] >= (width - edge_margin):
                    recovery_turn_direction = 1
                    recovery_turn_time = t
                elif line_error <= -TURN_DIRECTION_LOCK_THRESHOLD:
                    recovery_turn_direction = -1
                    recovery_turn_time = t
                elif line_error >= TURN_DIRECTION_LOCK_THRESHOLD:
                    recovery_turn_direction = 1
                    recovery_turn_time = t

            # Infer drive mode transitions using the same state-machine logic as on-car controller.
            if center is not None:
                drive_mode = "track"
            else:
                time_since_line = t - last_line_time
                entry_assist_active = (
                    (t - last_entry_time) <= ENTRY_MEMORY_SECONDS
                    and last_entry_kind in ("gentle", "sharp")
                    and last_entry_score >= ENTRY_ASSIST_SCORE_THRESHOLD
                )
                remembered_direction = (
                    recovery_turn_direction
                    if (t - recovery_turn_time) <= TURN_DIRECTION_MEMORY_SECONDS
                    else (-1 if last_nonzero_error < 0 else 1)
                )
                if entry_assist_active and last_entry_direction != 0:
                    remembered_direction = last_entry_direction
                if (abs(last_nonzero_error) >= CORNER_ERROR_THRESHOLD or entry_assist_active) and time_since_line <= CORNER_PIVOT_TIMEOUT_SECONDS:
                    drive_mode = "corner_pivot"
                elif time_since_line <= RECOVERY_HOLD_SECONDS:
                    drive_mode = "hold"
                elif time_since_line <= RECOVERY_TURN_TIMEOUT_SECONDS:
                    drive_mode = "search_turn"
                else:
                    drive_mode = "stop"
                recovery_turn_direction = remembered_direction
            mode_counter[drive_mode] += 1

            purple = detect_right_purple_square(frame)
            if purple is not None:
                purple_hits += 1
                x, y, ww, hh = purple["rect"]
                area = int(purple["area"])
            else:
                x = y = ww = hh = area = None

            csv_writer.writerow(
                [
                    frame_index,
                    round(t, 4),
                    int(center is not None),
                    line_mode,
                    "" if center is None else center[0],
                    "" if line_error is None else line_error,
                    drive_mode,
                    recovery_turn_direction,
                    entry["direction"],
                    entry["kind"],
                    entry["score"],
                    "" if entry["spread_px"] is None else entry["spread_px"],
                    "" if entry["slope"] is None else entry["slope"],
                    "" if entry["curvature_px"] is None else entry["curvature_px"],
                    int(purple is not None),
                    "" if x is None else x,
                    "" if y is None else y,
                    "" if ww is None else ww,
                    "" if hh is None else hh,
                    "" if area is None else area,
                ]
            )

            cv2.rectangle(frame, (0, roi_top), (width - 1, roi_bottom - 1), (255, 200, 0), 2)
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, height - 1), (255, 0, 0), 2)
            if contour is not None:
                contour_on_frame = contour.copy()
                contour_on_frame[:, 0, 1] += roi_top
                cv2.drawContours(frame, [contour_on_frame], -1, (0, 255, 255), 2)
            if center is not None:
                cv2.circle(frame, center, 7, (0, 0, 255), -1)
            if purple is not None:
                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (255, 0, 255), 3)
                cv2.putText(frame, "Purple", (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

            text1 = f"line:{'Y' if center is not None else 'N'} mode:{line_mode} err:{line_error if line_error is not None else 'NA'}"
            text2 = f"drive:{drive_mode} turn:{recovery_turn_direction:+d} purple:{'Y' if purple is not None else 'N'}"
            text3 = f"entry:{entry['kind']} dir:{entry['direction']:+d} score:{entry['score']:.2f}"
            text4 = f"frame:{frame_index}/{total_frames}"
            cv2.putText(frame, text1, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text3, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text4, (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)

    cap.release()
    writer.release()

    summary = {
        "video": str(video_path),
        "frames": frame_index,
        "fps": round(fps, 3),
        "line_detect_rate": round(line_hits / frame_index, 4) if frame_index else 0.0,
        "purple_detect_rate": round(purple_hits / frame_index, 4) if frame_index else 0.0,
        "line_error_abs_mean": round(float(np.mean(abs_errors)), 3) if abs_errors else None,
        "line_error_abs_p95": round(float(np.percentile(abs_errors, 95)), 3) if abs_errors else None,
        "mode_frame_count": mode_counter,
        "csv": str(csv_path),
        "annotated_video": str(out_video_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline detection for line error and right-side purple square.")
    parser.add_argument("video", help="Input video path.")
    parser.add_argument("--output-dir", default="", help="Output directory. Default: same folder as video.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir) if args.output_dir else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze_video(video_path, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
