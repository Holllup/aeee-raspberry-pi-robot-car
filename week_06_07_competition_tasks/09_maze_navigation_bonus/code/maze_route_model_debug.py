import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import maze_route_memory as route_memory


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_model(route_model):
    anchors = route_model["anchors"]
    turns = route_model["turn_points"]
    adjacent_distances = []
    for idx in range(1, len(anchors)):
        adjacent_distances.append(route_memory.observation_feature_distance(anchors[idx - 1], anchors[idx]))

    nearest_non_adjacent = []
    for idx, anchor in enumerate(anchors):
        best = None
        for other_idx, other in enumerate(anchors):
            if abs(other_idx - idx) <= 2:
                continue
            distance = route_memory.observation_feature_distance(anchor, other)
            if best is None or distance < best:
                best = distance
        if best is not None:
            nearest_non_adjacent.append(best)

    print(f"Route model: {len(anchors)} anchors, {len(turns)} turn points")
    print(f"Duration: {route_model['summary']['duration_seconds']:.2f}s")
    print(f"Mean adjacent distance: {mean(adjacent_distances):.4f}")
    print(f"Max adjacent distance: {max(adjacent_distances) if adjacent_distances else 0.0:.4f}")
    print(f"Mean nearest non-adjacent distance: {mean(nearest_non_adjacent):.4f}")

    low_sep = [idx for idx, value in enumerate(nearest_non_adjacent) if value < 0.18]
    high_jump = [idx for idx, value in enumerate(adjacent_distances) if value > 0.45]
    weak_turns = [
        turn for turn in turns
        if turn["action"] != route_memory.ROUTE_FINISH
        and anchors[turn["anchor_index"]]["corridor_confidence"] < 0.18
    ]
    if low_sep:
        print(f"Potentially too-similar anchors: {low_sep[:12]}")
    if high_jump:
        print(f"Potential continuity jumps: {high_jump[:12]}")
    if weak_turns:
        print("Weak turn anchors:")
        for turn in weak_turns[:12]:
            print(f"  anchor={turn['anchor_index']} action={turn['action']} t={turn['t']:.2f}")


def analyze_video(video_path, route_model, csv_output, debug_video_output):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")

    writer = None
    if debug_video_output:
        first_ok, first_frame = cap.read()
        if not first_ok:
            raise RuntimeError("video has no frames")
        first_frame = cv2.resize(first_frame, tuple(route_model["frame_size"]), interpolation=cv2.INTER_AREA)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(debug_video_output), fourcc, 20.0, tuple(route_model["frame_size"]))
        if not writer.isOpened():
            raise RuntimeError(f"unable to open debug writer: {debug_video_output}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    csv_file = None
    csv_writer = None
    if csv_output:
        csv_file = Path(csv_output).open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "anchor_index",
                "best_distance",
                "progress_ok",
                "route_center_error",
                "corridor_confidence",
                "next_turn_action",
            ],
        )
        csv_writer.writeheader()

    navigator = route_memory.RouteReplayNavigator(route_model)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, tuple(route_model["frame_size"]), interpolation=cv2.INTER_AREA)
            localization = route_memory.match_route_progress(
                frame,
                route_model,
                {
                    "anchor_index": navigator.anchor_index,
                    "reacquiring": navigator.reacquiring,
                    "previous_center_x": navigator.previous_center_x,
                    "last_committed_turn_anchor_index": navigator.last_committed_turn_anchor_index,
                },
            )
            command = navigator.update(localization, now=float(frame_index) / 20.0)
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "frame": frame_index,
                        "anchor_index": localization["best_anchor_index"],
                        "best_distance": f"{localization['best_distance']:.4f}",
                        "progress_ok": int(localization["progress_ok"]),
                        "route_center_error": f"{localization['route_center_error']:.2f}",
                        "corridor_confidence": f"{localization['perception']['corridor_confidence']:.4f}",
                        "next_turn_action": "" if localization["next_turn"] is None else localization["next_turn"]["action"],
                    }
                )
            if writer is not None:
                overlay = route_memory.draw_route_overlay(frame, localization, command, navigator, 20.0)
                writer.write(overlay)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()

    print(f"Video analysis frames: {frame_index}")
    if csv_output:
        print(f"Saved alignment CSV: {csv_output}")
    if debug_video_output:
        print(f"Saved debug video: {debug_video_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_model_path", help="Path to route model JSON.")
    parser.add_argument("--video", default="", help="Optional recorded video to align against the route model.")
    parser.add_argument("--csv-output", default="", help="Optional CSV output for alignment results.")
    parser.add_argument("--debug-video-output", default="", help="Optional debug video output path.")
    args = parser.parse_args()

    route_model = route_memory.load_route_model(args.route_model_path)
    summarize_model(route_model)
    if args.video:
        analyze_video(args.video, route_model, args.csv_output, args.debug_video_output)


if __name__ == "__main__":
    main()
