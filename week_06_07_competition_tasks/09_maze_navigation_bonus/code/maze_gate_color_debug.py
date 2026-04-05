import argparse
import importlib.util
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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


smooth = load_local_module("maze_navigation_smooth_local", "maze_navigation_smooth.py")
base = smooth.base


def clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class ColorStripCandidate:
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    height: float
    width: float
    area: float
    score: float


@dataclass
class GateCandidate:
    red: ColorStripCandidate
    green: ColorStripCandidate
    center_x: float
    width_px: float
    offset: float
    confidence: float


@dataclass
class ColorObservation:
    raw_red_mask: np.ndarray = field(repr=False)
    raw_green_mask: np.ndarray = field(repr=False)
    red_mask: np.ndarray = field(repr=False)
    green_mask: np.ndarray = field(repr=False)
    red_candidates: List[ColorStripCandidate] = field(default_factory=list, repr=False)
    green_candidates: List[ColorStripCandidate] = field(default_factory=list, repr=False)
    best_gate: Optional[GateCandidate] = field(default=None, repr=False)
    left_red_ratio: float = 0.0
    right_green_ratio: float = 0.0
    gate_visible: bool = False
    gate_center_x: float = 0.0
    gate_offset: float = 0.0
    gate_confidence: float = 0.0


def filter_marker_mask(
    mask,
    frame_shape,
    min_area_ratio=0.00025,
    min_aspect=0.95,
    min_height_ratio=0.05,
    max_keep=6,
):
    frame_h, frame_w = frame_shape[:2]
    min_area = float(frame_h * frame_w) * float(min_area_ratio)
    min_height = float(frame_h) * float(min_height_ratio)
    filtered = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0 or h < min_height:
            continue
        aspect = float(h) / float(w)
        fill = area / float(w * h)
        if aspect < min_aspect and fill < 0.56:
            continue
        cv2.drawContours(filtered, [contour], -1, 255, thickness=-1)
        kept += 1
        if kept >= max_keep:
            break
    return filtered


def extract_color_strip_candidates(mask, frame_shape, max_candidates=8):
    frame_h, frame_w = frame_shape[:2]
    frame_area = float(frame_h * frame_w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = float(h) / float(w)
        fill = area / float(w * h)
        area_ratio = area / max(1.0, frame_area)
        score = (
            clamp(area_ratio / 0.015, 0.0, 1.0) * 0.30
            + clamp((aspect - 0.8) / 4.5, 0.0, 1.0) * 0.45
            + clamp((fill - 0.20) / 0.70, 0.0, 1.0) * 0.25
        )
        candidates.append(
            ColorStripCandidate(
                bbox=(int(x), int(y), int(w), int(h)),
                center_x=float(x + (w * 0.5)),
                center_y=float(y + (h * 0.5)),
                height=float(h),
                width=float(w),
                area=float(area),
                score=float(score),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def select_best_gate(red_candidates, green_candidates, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    min_gate_width = frame_w * 0.04
    max_gate_width = frame_w * 0.85
    best_gate = None
    best_score = -1.0

    for red in red_candidates:
        red_bottom = red.bbox[1] + red.bbox[3]
        for green in green_candidates:
            green_bottom = green.bbox[1] + green.bbox[3]
            if red.center_x >= green.center_x:
                continue
            width_px = green.center_x - red.center_x
            if not (min_gate_width <= width_px <= max_gate_width):
                continue
            height_similarity = 1.0 - min(1.0, abs(red.height - green.height) / max(red.height, green.height, 1.0))
            bottom_similarity = 1.0 - min(1.0, abs(red_bottom - green_bottom) / max(red.height, green.height, 1.0))
            center_x = 0.5 * (red.center_x + green.center_x)
            offset = clamp((center_x - (frame_w * 0.5)) / max(1.0, frame_w * 0.5), -1.0, 1.0)
            center_bias = 1.0 - min(1.0, abs(offset))
            width_ratio = width_px / max(1.0, frame_w)
            width_bias = clamp((width_ratio - 0.08) / 0.42, 0.0, 1.0)
            red_outer_bias = clamp(((frame_w * 0.5) - red.center_x) / max(1.0, frame_w * 0.5), 0.0, 1.0)
            green_outer_bias = clamp((green.center_x - (frame_w * 0.5)) / max(1.0, frame_w * 0.5), 0.0, 1.0)
            outer_bias = 0.5 * (red_outer_bias + green_outer_bias)
            confidence = (
                red.score * 0.16
                + green.score * 0.16
                + height_similarity * 0.17
                + bottom_similarity * 0.17
                + width_bias * 0.22
                + outer_bias * 0.08
                + center_bias * 0.04
            )
            if confidence > best_score:
                best_score = confidence
                best_gate = GateCandidate(
                    red=red,
                    green=green,
                    center_x=float(center_x),
                    width_px=float(width_px),
                    offset=float(offset),
                    confidence=float(confidence),
                )
    return best_gate


def detect_gate_colors(frame_bgr, gate_conf_threshold=0.18):
    frame_h, frame_w = frame_bgr.shape[:2]
    color_y0 = int(frame_h * 0.20)
    color_roi_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    color_roi_mask[color_y0:, :] = 255
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    b_channel, g_channel, r_channel = cv2.split(frame_bgr)
    _, a_channel, _ = cv2.split(lab)
    _, cr_channel, cb_channel = cv2.split(ycrcb)

    sat_mask = cv2.inRange(hsv, (0, 38, 0), (180, 255, 255))
    value_mask = cv2.inRange(hsv, (0, 0, 18), (180, 255, 255))
    white_suppress = cv2.inRange(hsv, (0, 0, 165), (180, 68, 255))
    color_gate = cv2.bitwise_and(cv2.bitwise_and(sat_mask, value_mask), cv2.bitwise_not(white_suppress))

    red_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 62, 38), (18, 255, 255)),
        cv2.inRange(hsv, (162, 62, 38), (180, 255, 255)),
    )
    red_lowlight_hsv = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 36, 8), (18, 255, 150)),
        cv2.inRange(hsv, (162, 36, 8), (180, 255, 150)),
    )
    red_rgb = np.where(
        (r_channel.astype(np.int16) >= g_channel.astype(np.int16) + 10)
        & (r_channel.astype(np.int16) >= b_channel.astype(np.int16) + 10)
        & (r_channel >= 34),
        255,
        0,
    ).astype(np.uint8)
    red_lab = np.where(
        (a_channel >= 130)
        & (r_channel >= 30),
        255,
        0,
    ).astype(np.uint8)
    red_ycrcb = np.where(
        (cr_channel >= 132)
        & (cr_channel.astype(np.int16) >= cb_channel.astype(np.int16) + 4)
        & (r_channel >= 30),
        255,
        0,
    ).astype(np.uint8)

    green_hsv = cv2.inRange(hsv, (46, 96, 45), (84, 255, 255))
    green_rgb = np.where(
        (g_channel.astype(np.int16) >= r_channel.astype(np.int16) + 24)
        & (g_channel.astype(np.int16) >= b_channel.astype(np.int16) + 10)
        & (g_channel >= 78),
        255,
        0,
    ).astype(np.uint8)
    green_lab = np.where(
        (a_channel <= 118)
        & (g_channel >= 62),
        255,
        0,
    ).astype(np.uint8)

    raw_red_mask = cv2.bitwise_and(
        cv2.bitwise_or(cv2.bitwise_or(red_hsv, red_lowlight_hsv), cv2.bitwise_or(red_rgb, cv2.bitwise_or(red_lab, red_ycrcb))),
        color_gate,
    )
    raw_green_mask = cv2.bitwise_and(
        cv2.bitwise_or(cv2.bitwise_or(green_hsv, green_rgb), green_lab),
        color_gate,
    )
    raw_red_mask = cv2.bitwise_and(raw_red_mask, color_roi_mask)
    raw_green_mask = cv2.bitwise_and(raw_green_mask, color_roi_mask)

    close_kernel_red = np.ones((7, 7), dtype=np.uint8)
    close_kernel_green = np.ones((5, 5), dtype=np.uint8)
    open_kernel = np.ones((5, 5), dtype=np.uint8)

    red_mask = cv2.morphologyEx(raw_red_mask, cv2.MORPH_CLOSE, close_kernel_red, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    green_mask = cv2.morphologyEx(raw_green_mask, cv2.MORPH_CLOSE, close_kernel_green, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    red_mask = filter_marker_mask(
        red_mask,
        frame_bgr.shape,
        min_area_ratio=0.00030,
        min_aspect=0.90,
        min_height_ratio=0.05,
        max_keep=8,
    )
    green_mask = filter_marker_mask(
        green_mask,
        frame_bgr.shape,
        min_area_ratio=0.00045,
        min_aspect=1.05,
        min_height_ratio=0.07,
        max_keep=8,
    )

    red_candidates = extract_color_strip_candidates(red_mask, frame_bgr.shape)
    green_candidates = extract_color_strip_candidates(green_mask, frame_bgr.shape)
    best_gate = select_best_gate(red_candidates, green_candidates, frame_bgr.shape)

    marker_left = smooth.roi_rect(frame_bgr.shape, 0.00, 0.20, 0.50, 0.80)
    marker_right = smooth.roi_rect(frame_bgr.shape, 0.50, 0.20, 0.50, 0.80)
    left_red_ratio = smooth.occupancy_ratio(red_mask, marker_left)
    right_green_ratio = smooth.occupancy_ratio(green_mask, marker_right)
    gate_visible = bool(best_gate is not None and best_gate.confidence >= gate_conf_threshold)
    gate_center_x = (frame_w * 0.5) if best_gate is None else best_gate.center_x
    gate_offset = 0.0 if best_gate is None else best_gate.offset
    gate_confidence = 0.0 if best_gate is None else best_gate.confidence

    return ColorObservation(
        raw_red_mask=raw_red_mask,
        raw_green_mask=raw_green_mask,
        red_mask=red_mask,
        green_mask=green_mask,
        red_candidates=red_candidates,
        green_candidates=green_candidates,
        best_gate=best_gate,
        left_red_ratio=float(left_red_ratio),
        right_green_ratio=float(right_green_ratio),
        gate_visible=bool(gate_visible),
        gate_center_x=float(gate_center_x),
        gate_offset=float(gate_offset),
        gate_confidence=float(gate_confidence),
    )


def mask_to_panel(mask, color, size=(150, 90)):
    small = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    canvas[small > 0] = color
    return canvas


def draw_overlay(frame, obs, fps):
    view = frame.copy()
    frame_h, frame_w = view.shape[:2]
    cv2.line(view, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 255, 255), 1)
    if obs.gate_visible and obs.best_gate is not None:
        gate_x = int(clamp(obs.gate_center_x, 0, frame_w - 1))
        cv2.line(view, (gate_x, 0), (gate_x, frame_h), (0, 255, 255), 2)
        rx, ry, rw, rh = obs.best_gate.red.bbox
        gx, gy, gw, gh = obs.best_gate.green.bbox
        cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
        cv2.rectangle(view, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    lines = [
        f"Gate {'YES' if obs.gate_visible else 'NO'}  conf {obs.gate_confidence:.2f}  offset {obs.gate_offset:+.2f}  FPS {fps:4.1f}",
        f"left red {obs.left_red_ratio:.3f}  right green {obs.right_green_ratio:.3f}",
    ]
    y = 24
    for text in lines:
        cv2.putText(
            view,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22

    panels = [
        ("raw red", mask_to_panel(obs.raw_red_mask, (0, 0, 255))),
        ("final red", mask_to_panel(obs.red_mask, (0, 0, 255))),
        ("final green", mask_to_panel(obs.green_mask, (0, 255, 0))),
    ]
    x0 = 12
    y0 = frame_h - 90 - 28
    positions = [
        (x0, y0),
        (x0 + 162, y0),
        (x0 + 324, y0),
    ]
    for (label, panel), (px, py) in zip(panels, positions):
        view[py : py + panel.shape[0], px : px + panel.shape[1]] = panel
        cv2.rectangle(view, (px, py), (px + panel.shape[1], py + panel.shape[0]), (255, 255, 255), 2)
        cv2.putText(
            view,
            label,
            (px, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument("--record-output", default="gate_color_raw.h264", help="Raw h264 output path.")
    parser.add_argument("--debug-video-output", default="gate_color_debug.mp4", help="Debug mp4 output path.")
    parser.add_argument("--gate-confidence-threshold", type=float, default=0.18, help="Minimum paired gate confidence for Gate YES.")
    args = parser.parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

    picam2 = base.Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": base.CAPTURE_SIZE, "format": "BGR888"}
    )
    picam2.configure(camera_config)

    record_path = Path(args.record_output)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path = Path(args.debug_video_output)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = base.H264Encoder(bitrate=8_000_000)
    picam2.start_recording(encoder, base.FileOutput(str(record_path)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    debug_writer = cv2.VideoWriter(str(debug_path), fourcc, 20.0, base.PROCESS_SIZE)
    if not debug_writer.isOpened():
        raise RuntimeError(f"unable to open debug video writer: {debug_path}")

    last_frame_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            now = time.perf_counter()
            frame_dt = now - last_frame_time
            last_frame_time = now
            if frame_dt > 0:
                instant_fps = 1.0 / frame_dt
                fps = instant_fps if fps == 0.0 else (fps * 0.85 + instant_fps * 0.15)

            full_frame = picam2.capture_array()
            frame = cv2.resize(full_frame, base.PROCESS_SIZE, interpolation=cv2.INTER_AREA)
            obs = detect_gate_colors(frame, gate_conf_threshold=float(args.gate_confidence_threshold))
            overlay = draw_overlay(frame, obs, fps)
            debug_writer.write(overlay)

            if not args.headless:
                cv2.imshow("Gate Color Debug", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            else:
                time.sleep(0.001)
    finally:
        debug_writer.release()
        cv2.destroyAllWindows()
        picam2.stop_recording()


if __name__ == "__main__":
    main()
