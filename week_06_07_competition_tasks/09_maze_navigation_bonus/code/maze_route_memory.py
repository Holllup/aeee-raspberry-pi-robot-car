import argparse
import atexit
import csv
import json
import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np

import maze_navigation_smooth as maze


ALIGN_START = "ALIGN_START"
FOLLOW_ROUTE = "FOLLOW_ROUTE"
EXECUTE_TURN = "EXECUTE_TURN"
REACQUIRE_ROUTE = "REACQUIRE_ROUTE"
FINISH = "FINISH"

ROUTE_NONE = "NONE"
ROUTE_RIGHT = "RIGHT"
ROUTE_LEFT = "LEFT"
ROUTE_UTURN = "UTURN"
ROUTE_FINISH = "FINISH"

ROUTE_SCANLINE_Y_RATIOS = (0.80, 0.70, 0.60, 0.50, 0.40)
ROUTE_SCANLINE_VALID_RATIO = 0.35
ROUTE_SCANLINE_MIN_WIDTH = 16

ROUTE_ANCHOR_MIN_TIME_GAP = 0.16
ROUTE_ANCHOR_MIN_FEATURE_DELTA = 0.085
ROUTE_TURN_MIN_ANCHOR_GAP = 5
ROUTE_MIN_ROUTE_ANCHORS = 8

ROUTE_TURN_SCORE_MIN = 0.34
ROUTE_TURN_BLOCKED_SCORE_MIN = 0.75
ROUTE_TURN_FRONT_CLEAR_MIN = 0.18

ROUTE_MATCH_LOCAL_BACK = 6
ROUTE_MATCH_LOCAL_FORWARD = 22
ROUTE_MATCH_REACQUIRE_BACK = 12
ROUTE_MATCH_REACQUIRE_FORWARD = 48
ROUTE_MATCH_ACCEPT_DISTANCE = 0.70
ROUTE_MATCH_REACQUIRE_DISTANCE = 0.98
ROUTE_START_ALIGN_DISTANCE = 0.78
ROUTE_START_ALIGN_MAX_INDEX = 4
ROUTE_FINISH_HOLD_SECONDS = 0.60
ROUTE_REACQUIRE_HITS_REQUIRED = 3
ROUTE_LOST_MATCH_HITS = 4

ROUTE_TRACK_BASE_SPEED = 28
ROUTE_TRACK_MIN_SPEED = 20
ROUTE_TRACK_MAX_STEER = 18
ROUTE_TRACK_STEER_GAIN = 34.0
ROUTE_TRACK_ROUTE_CENTER_GAIN = 0.28
ROUTE_TRACK_WALL_GAIN = 15.0
ROUTE_REACQUIRE_SPEED_SCALE = 0.72

ROUTE_RIGHT_TURN_SPEED = 34
ROUTE_LEFT_TURN_SPEED = 34
ROUTE_UTURN_SPEED = 36
ROUTE_RIGHT_TURN_MAX_SECONDS = 1.35
ROUTE_LEFT_TURN_MAX_SECONDS = 1.55
ROUTE_UTURN_MAX_SECONDS = 2.40
ROUTE_TURN_ALIGN_HITS_REQUIRED = 3


def clamp(value, low, high):
    return max(low, min(high, value))


def spans_from_columns(active_columns):
    spans = []
    start = None
    for idx, active in enumerate(active_columns):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(active_columns) - 1))
    return spans


def sample_route_scanlines(mask, target_x=None):
    frame_h, frame_w = mask.shape[:2]
    if target_x is None:
        target_x = frame_w / 2.0

    scanlines = []
    guided_target_x = float(target_x)
    for ratio in ROUTE_SCANLINE_Y_RATIOS:
        y = int(frame_h * ratio)
        y0 = clamp(y - 2, 0, frame_h - 1)
        y1 = clamp(y + 3, y0 + 1, frame_h)
        band = mask[y0:y1, :]
        if band.size == 0:
            scanlines.append(None)
            continue
        active_columns = (np.mean(band > 0, axis=0) >= ROUTE_SCANLINE_VALID_RATIO)
        spans = spans_from_columns(active_columns)
        best_span = maze.pick_best_free_span(active_columns, guided_target_x)
        if best_span is None:
            scanlines.append(None)
            continue
        left, right = best_span
        width = int(right - left + 1)
        if width < ROUTE_SCANLINE_MIN_WIDTH:
            scanlines.append(None)
            continue
        center = (left + right) / 2.0
        sample = {
            "y": int((y0 + y1 - 1) / 2),
            "left": int(left),
            "right": int(right),
            "center": float(center),
            "width": width,
            "span_count": len(spans),
        }
        scanlines.append(sample)
        guided_target_x = sample["center"]
    return scanlines


def classify_turn_hint(perception):
    if (
        perception["front_blocked"]
        and perception["front_open_score"] <= 0.12
        and perception["right_open_score"] < 0.22
        and perception["left_open_score"] < 0.22
    ):
        return ROUTE_UTURN
    if (
        perception["right_open_score"] >= ROUTE_TURN_SCORE_MIN
        and perception["front_open_score"] >= ROUTE_TURN_FRONT_CLEAR_MIN
    ):
        return ROUTE_RIGHT
    if (
        perception["front_blocked"]
        and perception["left_open_score"] >= ROUTE_TURN_SCORE_MIN
    ):
        return ROUTE_LEFT
    if (
        perception["front_blocked"]
        and perception["right_open_score"] >= ROUTE_TURN_SCORE_MIN
    ):
        return ROUTE_RIGHT
    if perception["front_open_score"] <= 0.10 and perception["front_blocked"]:
        return ROUTE_UTURN
    return ROUTE_NONE


def scanline_feature_arrays(scanlines, frame_width):
    center_values = []
    width_values = []
    valid_values = []
    span_values = []
    frame_half = frame_width / 2.0
    for sample in scanlines:
        if sample is None:
            center_values.append(0.0)
            width_values.append(0.0)
            valid_values.append(0.0)
            span_values.append(0.0)
            continue
        center_values.append((sample["center"] - frame_half) / max(1.0, frame_half))
        width_values.append(sample["width"] / float(max(1, frame_width)))
        valid_values.append(1.0)
        span_values.append(sample["span_count"] / 4.0)
    return center_values, width_values, valid_values, span_values


def record_route_scan(frame_bgr, previous_center_x=None, t_s=None):
    perception = maze.analyze_maze_perception(frame_bgr, previous_center_x)
    frame_w = frame_bgr.shape[1]
    scanlines = sample_route_scanlines(
        perception["ground_connected_free_mask"],
        target_x=perception["corridor_center_x"] if perception["corridor_center_x"] is not None else previous_center_x,
    )
    scanline_centers, scanline_widths, scanline_valid, scanline_spans = scanline_feature_arrays(
        scanlines,
        frame_w,
    )
    observation = {
        "t": 0.0 if t_s is None else float(t_s),
        "corridor_center_x": None if perception["corridor_center_x"] is None else int(perception["corridor_center_x"]),
        "corridor_width": int(perception["corridor_width"]),
        "corridor_confidence": float(perception["corridor_confidence"]),
        "corridor_error": float(perception["corridor_error"]),
        "front_open_score": float(perception["front_open_score"]),
        "right_open_score": float(perception["right_open_score"]),
        "left_open_score": float(perception["left_open_score"]),
        "front_blocked": bool(perception["front_blocked"]),
        "right_wall_pressure": float(perception["right_wall_pressure"]),
        "left_wall_pressure": float(perception["left_wall_pressure"]),
        "wall_balance": float(perception["wall_balance"]),
        "scanline_centers": scanline_centers,
        "scanline_widths": scanline_widths,
        "scanline_valid": scanline_valid,
        "scanline_spans": scanline_spans,
        "turn_hint": classify_turn_hint(perception),
        "perception": perception,
        "scanlines": scanlines,
    }
    return observation


def observation_feature_vector(observation, frame_width=maze.base.PROCESS_SIZE[0]):
    frame_half = frame_width / 2.0
    center_norm = 0.0
    if observation.get("corridor_center_x") is not None:
        center_norm = (float(observation["corridor_center_x"]) - frame_half) / max(1.0, frame_half)
    width_norm = float(observation.get("corridor_width", 0.0)) / float(max(1, frame_width))
    vector = [
        center_norm,
        width_norm,
        float(observation.get("corridor_confidence", 0.0)),
        float(observation.get("front_open_score", 0.0)),
        float(observation.get("right_open_score", 0.0)),
        float(observation.get("left_open_score", 0.0)),
        float(observation.get("right_wall_pressure", 0.0)),
        float(observation.get("left_wall_pressure", 0.0)),
        float(observation.get("wall_balance", 0.0)),
    ]
    vector.extend(float(v) for v in observation.get("scanline_centers", ()))
    vector.extend(float(v) for v in observation.get("scanline_widths", ()))
    vector.extend(float(v) for v in observation.get("scanline_valid", ()))
    vector.extend(float(v) for v in observation.get("scanline_spans", ()))
    return np.asarray(vector, dtype=np.float32)


def observation_feature_distance(obs_a, obs_b):
    vec_a = observation_feature_vector(obs_a)
    vec_b = observation_feature_vector(obs_b)
    if vec_a.shape != vec_b.shape:
        raise ValueError("feature vectors must have the same shape")
    weight = np.ones_like(vec_a, dtype=np.float32)
    weight[0] = 1.6
    weight[1] = 1.4
    weight[2] = 1.2
    weight[3:6] = 1.4
    weight[6:9] = 1.1
    return float(np.mean(np.abs(vec_a - vec_b) * weight))


def make_anchor_from_observation(observation, source_index):
    anchor = {
        "source_index": int(source_index),
        "t": float(observation["t"]),
        "corridor_center_x": observation["corridor_center_x"],
        "corridor_width": int(observation["corridor_width"]),
        "corridor_confidence": float(observation["corridor_confidence"]),
        "corridor_error": float(observation["corridor_error"]),
        "front_open_score": float(observation["front_open_score"]),
        "right_open_score": float(observation["right_open_score"]),
        "left_open_score": float(observation["left_open_score"]),
        "front_blocked": bool(observation["front_blocked"]),
        "right_wall_pressure": float(observation["right_wall_pressure"]),
        "left_wall_pressure": float(observation["left_wall_pressure"]),
        "wall_balance": float(observation["wall_balance"]),
        "scanline_centers": [float(v) for v in observation["scanline_centers"]],
        "scanline_widths": [float(v) for v in observation["scanline_widths"]],
        "scanline_valid": [float(v) for v in observation["scanline_valid"]],
        "scanline_spans": [float(v) for v in observation["scanline_spans"]],
        "turn_hint": observation["turn_hint"],
        "action": ROUTE_NONE,
    }
    return anchor


def _anchor_is_worth_keeping(observation):
    return observation["corridor_confidence"] >= 0.08 or observation["turn_hint"] != ROUTE_NONE


def build_route_model(scan_observations):
    if not scan_observations:
        raise ValueError("no scan observations provided")

    anchors = []
    last_anchor_obs = None
    last_anchor_t = -1e9
    for index, observation in enumerate(scan_observations):
        if not _anchor_is_worth_keeping(observation):
            continue
        keep = False
        if last_anchor_obs is None:
            keep = True
        else:
            time_delta = observation["t"] - last_anchor_t
            feature_delta = observation_feature_distance(observation, last_anchor_obs)
            keep = (
                time_delta >= ROUTE_ANCHOR_MIN_TIME_GAP
                and (
                    feature_delta >= ROUTE_ANCHOR_MIN_FEATURE_DELTA
                    or observation["turn_hint"] != ROUTE_NONE
                )
            )
        if keep:
            anchors.append(make_anchor_from_observation(observation, index))
            last_anchor_obs = observation
            last_anchor_t = observation["t"]

    if len(anchors) < ROUTE_MIN_ROUTE_ANCHORS:
        sample_indexes = np.linspace(0, len(scan_observations) - 1, num=min(len(scan_observations), ROUTE_MIN_ROUTE_ANCHORS))
        anchors = []
        for index in sorted({int(round(v)) for v in sample_indexes}):
            anchors.append(make_anchor_from_observation(scan_observations[index], index))

    if anchors[-1]["source_index"] != len(scan_observations) - 1:
        anchors.append(make_anchor_from_observation(scan_observations[-1], len(scan_observations) - 1))

    turns = []
    last_turn_anchor_idx = -999
    for anchor_idx, anchor in enumerate(anchors):
        hint = anchor["turn_hint"]
        if hint == ROUTE_NONE:
            continue
        if anchor_idx - last_turn_anchor_idx < ROUTE_TURN_MIN_ANCHOR_GAP:
            continue
        anchor["action"] = hint
        turns.append(
            {
                "anchor_index": anchor_idx,
                "source_index": int(anchor["source_index"]),
                "t": float(anchor["t"]),
                "action": hint,
            }
        )
        last_turn_anchor_idx = anchor_idx

    if not turns or int(turns[-1]["anchor_index"]) != len(anchors) - 1:
        anchors[-1]["action"] = ROUTE_FINISH
        turns.append(
            {
                "anchor_index": len(anchors) - 1,
                "source_index": int(anchors[-1]["source_index"]),
                "t": float(anchors[-1]["t"]),
                "action": ROUTE_FINISH,
            }
        )

    segments = []
    segment_start = 0
    for turn in turns:
        segments.append(
            {
                "start_anchor_index": segment_start,
                "end_anchor_index": int(turn["anchor_index"]),
                "action_at_end": turn["action"],
            }
        )
        segment_start = int(turn["anchor_index"])

    adjacent_distances = []
    for idx in range(1, len(anchors)):
        adjacent_distances.append(observation_feature_distance(anchors[idx - 1], anchors[idx]))

    route_model = {
        "version": 1,
        "frame_size": list(maze.base.PROCESS_SIZE),
        "scanline_y_ratios": list(ROUTE_SCANLINE_Y_RATIOS),
        "observation_count": len(scan_observations),
        "anchor_count": len(anchors),
        "anchors": anchors,
        "turn_points": turns,
        "segments": segments,
        "start_reference": {
            "anchor_index": 0,
            "corridor_center_x": anchors[0]["corridor_center_x"],
            "corridor_width": anchors[0]["corridor_width"],
        },
        "finish_reference": {
            "anchor_index": len(anchors) - 1,
            "corridor_center_x": anchors[-1]["corridor_center_x"],
            "corridor_width": anchors[-1]["corridor_width"],
        },
        "summary": {
            "duration_seconds": float(scan_observations[-1]["t"] - scan_observations[0]["t"]),
            "mean_adjacent_anchor_distance": float(np.mean(adjacent_distances)) if adjacent_distances else 0.0,
            "max_adjacent_anchor_distance": float(np.max(adjacent_distances)) if adjacent_distances else 0.0,
        },
    }
    return route_model


def save_route_model(route_model, path):
    route_path = Path(path)
    route_path.parent.mkdir(parents=True, exist_ok=True)
    route_path.write_text(json.dumps(route_model, ensure_ascii=False, indent=2), encoding="utf-8")


def load_route_model(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_next_turn(route_model, after_turn_anchor_index=-1):
    for turn in route_model["turn_points"]:
        if turn["action"] == ROUTE_FINISH:
            continue
        if int(turn["anchor_index"]) > int(after_turn_anchor_index):
            return turn
    return None


def match_route_progress(frame_bgr, route_model, progress_hint):
    progress_hint = {} if progress_hint is None else dict(progress_hint)
    previous_center_x = progress_hint.get("previous_center_x")
    observation = record_route_scan(frame_bgr, previous_center_x=previous_center_x)
    anchors = route_model["anchors"]

    current_idx = int(progress_hint.get("anchor_index", 0))
    reacquiring = bool(progress_hint.get("reacquiring", False))
    if reacquiring:
        start_idx = max(0, current_idx - ROUTE_MATCH_REACQUIRE_BACK)
        end_idx = min(len(anchors), current_idx + ROUTE_MATCH_REACQUIRE_FORWARD)
    else:
        start_idx = max(0, current_idx - ROUTE_MATCH_LOCAL_BACK)
        end_idx = min(len(anchors), current_idx + ROUTE_MATCH_LOCAL_FORWARD)
    if current_idx <= ROUTE_START_ALIGN_MAX_INDEX:
        start_idx = 0
        end_idx = min(len(anchors), ROUTE_MATCH_LOCAL_FORWARD)

    best_index = start_idx
    best_distance = 1e9
    candidate_distances = []
    for idx in range(start_idx, end_idx):
        anchor = anchors[idx]
        distance = observation_feature_distance(observation, anchor)
        if idx < current_idx - 1:
            distance += 0.12 * float((current_idx - 1) - idx)
        candidate_distances.append((idx, distance))
        if distance < best_distance:
            best_distance = distance
            best_index = idx

    current_anchor = anchors[best_index]
    last_committed_turn_anchor_index = int(progress_hint.get("last_committed_turn_anchor_index", -1))
    next_turn = find_next_turn(route_model, last_committed_turn_anchor_index)
    progress_ok = best_distance <= (
        ROUTE_MATCH_REACQUIRE_DISTANCE if reacquiring else ROUTE_MATCH_ACCEPT_DISTANCE
    )

    expected_center_x = current_anchor.get("corridor_center_x")
    route_center_error = 0.0
    if expected_center_x is not None and observation["corridor_center_x"] is not None:
        route_center_error = float(observation["corridor_center_x"] - float(expected_center_x))

    return {
        "observation": observation,
        "perception": observation["perception"],
        "best_anchor_index": int(best_index),
        "best_distance": float(best_distance),
        "progress_ok": bool(progress_ok),
        "current_anchor": current_anchor,
        "next_turn": next_turn,
        "search_start_index": int(start_idx),
        "search_end_index": int(max(start_idx, end_idx - 1)),
        "candidate_distances": candidate_distances,
        "route_center_error": float(route_center_error),
        "route_progress_ratio": (
            float(best_index) / float(max(1, len(anchors) - 1))
        ),
    }


class RouteReplayNavigator:
    def __init__(self, route_model):
        self.route_model = route_model
        self.state = ALIGN_START
        self.anchor_index = 0
        self.reacquiring = False
        self.previous_center_x = None
        self.match_hits = 0
        self.lost_hits = 0
        self.turn_action = ROUTE_NONE
        self.turn_anchor_index = -1
        self.last_committed_turn_anchor_index = -1
        self.turn_until = -999.0
        self.turn_align_hits = 0
        self.finish_since = None

    def _follow_command(self, localization, speed_scale=1.0):
        perception = localization["perception"]
        frame_half = maze.base.PROCESS_SIZE[0] / 2.0
        current_center_x = perception["corridor_center_x"]
        target_center_x = current_center_x
        if target_center_x is None:
            target_center_x = self.previous_center_x
        if target_center_x is None:
            target_center_x = frame_half

        anchor_center_x = localization["current_anchor"].get("corridor_center_x")
        if anchor_center_x is not None:
            target_center_x = (
                (1.0 - ROUTE_TRACK_ROUTE_CENTER_GAIN) * float(target_center_x)
                + ROUTE_TRACK_ROUTE_CENTER_GAIN * float(anchor_center_x)
            )

        error = float(target_center_x) - frame_half
        normalized_error = 0.0 if frame_half <= 0 else error / frame_half
        steer = normalized_error * ROUTE_TRACK_STEER_GAIN
        steer += -perception["wall_balance"] * ROUTE_TRACK_WALL_GAIN
        steer = clamp(steer, -ROUTE_TRACK_MAX_STEER, ROUTE_TRACK_MAX_STEER)

        confidence = perception["corridor_confidence"]
        speed_penalty = min(7.0, abs(error) * 0.028) + (1.0 - clamp(confidence, 0.0, 1.0)) * 4.5
        base_speed = clamp(ROUTE_TRACK_BASE_SPEED - speed_penalty, ROUTE_TRACK_MIN_SPEED, ROUTE_TRACK_BASE_SPEED)
        base_speed *= speed_scale

        left_speed = int(clamp(base_speed + steer, -maze.base.MAX_SPEED, maze.base.MAX_SPEED))
        right_speed = int(clamp(base_speed - steer, -maze.base.MAX_SPEED, maze.base.MAX_SPEED))
        return {"left_speed": left_speed, "right_speed": right_speed, "mode": self.state, "done": False}

    def _start_turn(self, action, turn_anchor_index, now):
        self.state = EXECUTE_TURN
        self.turn_action = action
        self.turn_anchor_index = int(turn_anchor_index)
        self.last_committed_turn_anchor_index = int(turn_anchor_index)
        if action == ROUTE_RIGHT:
            self.turn_until = now + ROUTE_RIGHT_TURN_MAX_SECONDS
        elif action == ROUTE_LEFT:
            self.turn_until = now + ROUTE_LEFT_TURN_MAX_SECONDS
        else:
            self.turn_until = now + ROUTE_UTURN_MAX_SECONDS
        self.turn_align_hits = 0

    def _turn_command(self):
        if self.turn_action == ROUTE_RIGHT:
            return {
                "left_speed": ROUTE_RIGHT_TURN_SPEED,
                "right_speed": -ROUTE_RIGHT_TURN_SPEED,
                "mode": "EXEC_TURN_R",
                "done": False,
            }
        if self.turn_action == ROUTE_LEFT:
            return {
                "left_speed": -ROUTE_LEFT_TURN_SPEED,
                "right_speed": ROUTE_LEFT_TURN_SPEED,
                "mode": "EXEC_TURN_L",
                "done": False,
            }
        return {
            "left_speed": ROUTE_UTURN_SPEED,
            "right_speed": -ROUTE_UTURN_SPEED,
            "mode": "EXEC_UTURN",
            "done": False,
        }

    def update(self, localization, now):
        perception = localization["perception"]
        self.previous_center_x = perception["corridor_center_x"] if perception["corridor_center_x"] is not None else self.previous_center_x

        if localization["progress_ok"]:
            self.match_hits += 1
            self.lost_hits = 0
            self.anchor_index = max(self.anchor_index, localization["best_anchor_index"])
        else:
            self.match_hits = 0
            self.lost_hits += 1

        if self.state == ALIGN_START:
            if (
                localization["progress_ok"]
                and localization["best_anchor_index"] <= ROUTE_START_ALIGN_MAX_INDEX
                and localization["best_distance"] <= ROUTE_START_ALIGN_DISTANCE
            ):
                self.state = FOLLOW_ROUTE
                return self._follow_command(localization)
            return {"left_speed": 0, "right_speed": 0, "mode": ALIGN_START, "done": False}

        if self.state == EXECUTE_TURN:
            aligned = (
                localization["progress_ok"]
                and localization["best_anchor_index"] >= self.turn_anchor_index + 1
                and perception["corridor_confidence"] >= maze.MAZE_CORRIDOR_REACQUIRE_CONFIDENCE
                and abs(perception["corridor_error"]) <= maze.MAZE_CENTERED_ERROR_PIXELS
            )
            if aligned:
                self.turn_align_hits += 1
            else:
                self.turn_align_hits = 0
            if self.turn_align_hits >= ROUTE_TURN_ALIGN_HITS_REQUIRED:
                self.state = FOLLOW_ROUTE
                self.turn_action = ROUTE_NONE
                self.reacquiring = False
                return self._follow_command(localization)
            if now >= self.turn_until:
                self.state = REACQUIRE_ROUTE
                self.reacquiring = True
                return self._follow_command(localization, speed_scale=ROUTE_REACQUIRE_SPEED_SCALE)
            return self._turn_command()

        if self.state == REACQUIRE_ROUTE:
            if localization["progress_ok"]:
                self.reacquiring = False
                if self.match_hits >= ROUTE_REACQUIRE_HITS_REQUIRED:
                    self.state = FOLLOW_ROUTE
                return self._follow_command(localization, speed_scale=ROUTE_REACQUIRE_SPEED_SCALE)
            if perception["corridor_confidence"] >= maze.MAZE_CORRIDOR_CONFIDENCE_MIN:
                return self._follow_command(localization, speed_scale=ROUTE_REACQUIRE_SPEED_SCALE)
            return {"left_speed": 0, "right_speed": 0, "mode": REACQUIRE_ROUTE, "done": False}

        if self.lost_hits >= ROUTE_LOST_MATCH_HITS:
            self.state = REACQUIRE_ROUTE
            self.reacquiring = True
            return self._follow_command(localization, speed_scale=ROUTE_REACQUIRE_SPEED_SCALE)

        if localization["best_anchor_index"] >= len(self.route_model["anchors"]) - 1:
            if self.finish_since is None:
                self.finish_since = now
            if (now - self.finish_since) >= ROUTE_FINISH_HOLD_SECONDS:
                self.state = FINISH
                return {"left_speed": 0, "right_speed": 0, "mode": FINISH, "done": True}
        else:
            self.finish_since = None

        next_turn = localization["next_turn"]
        if next_turn is not None and next_turn["action"] != ROUTE_FINISH:
            if localization["best_anchor_index"] >= int(next_turn["anchor_index"]) - 1:
                self._start_turn(next_turn["action"], next_turn["anchor_index"], now)
                return self._turn_command()

        self.state = FOLLOW_ROUTE
        return self._follow_command(localization)


def tint_mask(image, mask, color, alpha):
    if not np.any(mask):
        return image
    overlay = image.copy()
    overlay[mask] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    image[mask] = blended[mask]
    return image


def draw_route_overlay(frame, localization, command, navigator, fps):
    perception = localization["perception"]
    view = frame.copy()
    trusted = perception["ground_connected_free_mask"] > 0
    ignored = perception["ignored_free_mask"] > 0
    walls = perception["wall_mask"] > 0

    view = tint_mask(view, trusted, (0, 200, 0), 0.65)
    view = tint_mask(view, ignored, (0, 180, 255), 0.65)
    view = tint_mask(view, walls, (40, 40, 40), 0.50)

    cv2.polylines(view, [perception["decision_polygon"]], True, (255, 255, 0), 2)
    cv2.line(view, (0, perception["horizon_y"]), (view.shape[1] - 1, perception["horizon_y"]), (255, 180, 0), 1)

    frame_center_x = view.shape[1] // 2
    cv2.line(view, (frame_center_x, 0), (frame_center_x, view.shape[0] - 1), (255, 0, 0), 1)
    if perception["corridor_center_x"] is not None:
        corridor_x = int(perception["corridor_center_x"])
        cv2.line(view, (corridor_x, 0), (corridor_x, view.shape[0] - 1), (0, 255, 255), 2)
    anchor_center_x = localization["current_anchor"].get("corridor_center_x")
    if anchor_center_x is not None:
        cv2.line(view, (int(anchor_center_x), 0), (int(anchor_center_x), view.shape[0] - 1), (255, 0, 255), 2)

    top_lines = [
        f"State {navigator.state}  anchor {localization['best_anchor_index']}/{len(navigator.route_model['anchors']) - 1}",
        f"Match dist {localization['best_distance']:.3f}  ok {int(localization['progress_ok'])}  FPS {fps:4.1f}",
        (
            f"Corridor conf {perception['corridor_confidence']:.2f} "
            f"err {perception['corridor_error']:+5.1f} "
            f"route {localization['route_center_error']:+5.1f}"
        ),
        (
            f"Open R {perception['right_open_score']:.2f}  "
            f"F {perception['front_open_score']:.2f}  "
            f"L {perception['left_open_score']:.2f}"
        ),
        (
            f"Wall R {perception['right_wall_pressure']:.2f}  "
            f"L {perception['left_wall_pressure']:.2f}  "
            f"bal {perception['wall_balance']:+.2f}"
        ),
    ]
    next_turn = localization["next_turn"]
    if next_turn is not None:
        top_lines.append(
            f"Next turn {next_turn['action']} @ anchor {next_turn['anchor_index']}"
        )
    if command is not None:
        top_lines.append(f"Drive L{command['left_speed']:>3} R{command['right_speed']:>3}  mode {command['mode']}")

    for index, line in enumerate(top_lines):
        cv2.putText(
            view,
            line,
            (12, 28 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return view


def install_stop_handlers(controller):
    def safe_stop():
        try:
            controller.stop()
        except Exception:
            pass

    def stop_and_exit(signum=None, frame=None):
        safe_stop()
        raise SystemExit(0)

    atexit.register(safe_stop)
    signal.signal(signal.SIGINT, stop_and_exit)
    signal.signal(signal.SIGTERM, stop_and_exit)


def save_scan_observations_csv(path, observations):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "t",
                "corridor_center_x",
                "corridor_width",
                "corridor_confidence",
                "corridor_error",
                "front_open_score",
                "right_open_score",
                "left_open_score",
                "front_blocked",
                "right_wall_pressure",
                "left_wall_pressure",
                "wall_balance",
                "turn_hint",
            ]
        )
        for obs in observations:
            writer.writerow(
                [
                    f"{obs['t']:.4f}",
                    "" if obs["corridor_center_x"] is None else int(obs["corridor_center_x"]),
                    int(obs["corridor_width"]),
                    f"{obs['corridor_confidence']:.4f}",
                    f"{obs['corridor_error']:.2f}",
                    f"{obs['front_open_score']:.4f}",
                    f"{obs['right_open_score']:.4f}",
                    f"{obs['left_open_score']:.4f}",
                    int(obs["front_blocked"]),
                    f"{obs['right_wall_pressure']:.4f}",
                    f"{obs['left_wall_pressure']:.4f}",
                    f"{obs['wall_balance']:.4f}",
                    obs["turn_hint"],
                ]
            )


def open_camera(record_output):
    picam2 = maze.base.Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": maze.base.CAPTURE_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)

    h264_recording = False
    if record_output:
        encoder = maze.base.H264Encoder(bitrate=8_000_000)
        picam2.start_recording(encoder, maze.base.FileOutput(record_output))
        h264_recording = True
    else:
        picam2.start()
    return picam2, h264_recording


def run_scan_route(args):
    maze.base.GPIO.setwarnings(False)
    maze.base.GPIO.setmode(maze.base.GPIO.BCM)
    controller = maze.base.CarMotorController()
    lcd_display = maze.base.LCD1602Display()
    install_stop_handlers(controller)
    picam2, h264_recording = open_camera(args.record_output)

    debug_writer = None
    observations = []
    previous_center_x = None
    started_at = time.perf_counter()
    last_frame_time = started_at
    fps = 0.0

    try:
        if args.debug_video_output:
            debug_path = Path(args.debug_video_output)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(str(debug_path), fourcc, 20.0, maze.base.PROCESS_SIZE)
            if not debug_writer.isOpened():
                raise RuntimeError(f"unable to open debug video writer: {debug_path}")

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

            controller.hold_stop()
            elapsed = now - started_at
            observation = record_route_scan(frame, previous_center_x=previous_center_x, t_s=elapsed)
            previous_center_x = observation["corridor_center_x"]
            observations.append(observation)
            lcd_display.update("Route Scan", f"Frames {len(observations)}")

            fake_localization = {
                "perception": observation["perception"],
                "best_anchor_index": max(0, len(observations) - 1),
                "best_distance": 0.0,
                "progress_ok": True,
                "current_anchor": {
                    "corridor_center_x": observation["corridor_center_x"],
                },
                "next_turn": None,
                "route_center_error": 0.0,
            }
            fake_navigator = type("ScanNav", (), {"state": "SCAN_ROUTE", "route_model": {"anchors": [1] * max(2, len(observations))}})()
            overlay = draw_route_overlay(
                frame,
                fake_localization,
                {"left_speed": 0, "right_speed": 0, "mode": "SCAN_ROUTE", "done": False},
                fake_navigator,
                fps,
            )
            cv2.putText(
                overlay,
                f"Hint {observation['turn_hint']}",
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
                cv2.imshow("Maze Route Scan", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q"), ord("f"), ord("F")):
                    break
            else:
                time.sleep(0.001)
    finally:
        route_model = build_route_model(observations) if observations else None
        if route_model is not None:
            save_route_model(route_model, args.route_model_path)
            if args.scan_observations_csv:
                save_scan_observations_csv(args.scan_observations_csv, observations)
            print(f"[scan] saved route model: {args.route_model_path}")
            if args.scan_observations_csv:
                print(f"[scan] saved observations csv: {args.scan_observations_csv}")
            print(
                f"[scan] observations={len(observations)} anchors={route_model['anchor_count']} "
                f"turns={len(route_model['turn_points'])}"
            )

        controller.close()
        lcd_display.close()
        if debug_writer is not None:
            debug_writer.release()
        maze.base.GPIO.cleanup()
        cv2.destroyAllWindows()
        if h264_recording:
            picam2.stop_recording()
        else:
            picam2.stop()


def run_replay_route(args):
    route_model = load_route_model(args.route_model_path)

    maze.base.GPIO.setwarnings(False)
    maze.base.GPIO.setmode(maze.base.GPIO.BCM)
    controller = maze.base.CarMotorController()
    lcd_display = maze.base.LCD1602Display()
    install_stop_handlers(controller)
    picam2, h264_recording = open_camera(args.record_output)

    debug_writer = None
    metrics_file = None
    metrics_writer = None
    navigator = RouteReplayNavigator(route_model)
    state = COUNTDOWN = "COUNTDOWN"
    countdown_deadline = time.perf_counter() + max(0.0, args.start_delay)
    last_frame_time = time.perf_counter()
    fps = 0.0

    try:
        if args.debug_video_output:
            debug_path = Path(args.debug_video_output)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(str(debug_path), fourcc, 20.0, maze.base.PROCESS_SIZE)
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
                    "anchor_index",
                    "best_distance",
                    "progress_ok",
                    "next_turn_index",
                    "next_turn_action",
                    "left_speed",
                    "right_speed",
                    "corridor_confidence",
                    "corridor_error",
                    "route_center_error",
                    "front_open_score",
                    "right_open_score",
                    "left_open_score",
                    "wall_balance",
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

            localization = match_route_progress(
                frame,
                route_model,
                {
                    "anchor_index": navigator.anchor_index,
                    "reacquiring": navigator.reacquiring,
                    "previous_center_x": navigator.previous_center_x,
                    "last_committed_turn_anchor_index": navigator.last_committed_turn_anchor_index,
                },
            )
            command = None

            if state == COUNTDOWN:
                controller.hold_stop()
                if now >= countdown_deadline:
                    state = FOLLOW_ROUTE
                    controller.stop()
            else:
                command = navigator.update(localization, now)
                if command["done"]:
                    state = FINISH
                    controller.stop()
                else:
                    controller.set_tank_drive(command["left_speed"], command["right_speed"])

            lcd_mode = state if command is None else command["mode"]
            lcd_display.update("Route Replay", lcd_mode[:16])

            if metrics_writer is not None:
                next_turn = localization["next_turn"]
                metrics_writer.writerow(
                    {
                        "t": f"{now:.3f}",
                        "state": navigator.state if state != COUNTDOWN else COUNTDOWN,
                        "anchor_index": localization["best_anchor_index"],
                        "best_distance": f"{localization['best_distance']:.4f}",
                        "progress_ok": int(localization["progress_ok"]),
                        "next_turn_index": "" if next_turn is None else next_turn["anchor_index"],
                        "next_turn_action": "" if next_turn is None else next_turn["action"],
                        "left_speed": 0 if command is None else command["left_speed"],
                        "right_speed": 0 if command is None else command["right_speed"],
                        "corridor_confidence": f"{localization['perception']['corridor_confidence']:.4f}",
                        "corridor_error": f"{localization['perception']['corridor_error']:.2f}",
                        "route_center_error": f"{localization['route_center_error']:.2f}",
                        "front_open_score": f"{localization['perception']['front_open_score']:.4f}",
                        "right_open_score": f"{localization['perception']['right_open_score']:.4f}",
                        "left_open_score": f"{localization['perception']['left_open_score']:.4f}",
                        "wall_balance": f"{localization['perception']['wall_balance']:.4f}",
                    }
                )
                metrics_file.flush()

            overlay = draw_route_overlay(frame, localization, command, navigator, fps)
            if state == COUNTDOWN:
                cv2.putText(
                    overlay,
                    f"Start in {max(0, int(np.ceil(countdown_deadline - now)))}",
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
                cv2.imshow("Maze Route Replay", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q"), ord("f"), ord("F")):
                    break
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("scan-route", "replay-route"),
        required=True,
        help="Route memory mode to run.",
    )
    parser.add_argument(
        "--route-model-path",
        default="maze_route_model.json",
        help="Path to the saved route model JSON.",
    )
    parser.add_argument(
        "--scan-observations-csv",
        default="",
        help="Optional CSV output for raw pre-scan observations.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before starting replay mode.",
    )
    parser.add_argument("--headless", action="store_true", help="Disable preview window.")
    parser.add_argument(
        "--record-output",
        default="",
        help="Optional output h264 path from the Pi camera.",
    )
    parser.add_argument(
        "--debug-video-output",
        default="",
        help="Optional mp4 path for the processed overlay video.",
    )
    parser.add_argument(
        "--metrics-output",
        default="",
        help="Optional CSV path for replay metrics.",
    )
    args = parser.parse_args()
    if not args.headless and not os.environ.get("DISPLAY"):
        print("[info] DISPLAY not set; switching to --headless mode automatically")
        args.headless = True

    if args.mode == "scan-route":
        run_scan_route(args)
    else:
        run_replay_route(args)


if __name__ == "__main__":
    main()
