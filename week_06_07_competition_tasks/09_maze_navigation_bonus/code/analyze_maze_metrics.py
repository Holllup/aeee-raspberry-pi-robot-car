import argparse
import csv
from collections import Counter
from pathlib import Path


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(row, key, default=0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to maze_nav_metrics.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit("no rows found in metrics csv")

    running_rows = [row for row in rows if row.get("state") == "RUNNING"]
    mode_counter = Counter(row.get("mode", "") for row in running_rows if row.get("mode"))
    confidences = [as_float(row, "confidence") for row in running_rows]
    heading_errors = [abs(as_float(row, "heading_error")) for row in running_rows]
    lookahead_errors = [abs(as_float(row, "lookahead_error")) for row in running_rows]
    forward_depths = [as_float(row, "forward_depth") for row in running_rows]
    left_scores = [as_float(row, "left_branch_score") for row in running_rows]
    right_scores = [as_float(row, "right_branch_score") for row in running_rows]
    front_blocks = [as_float(row, "front_block_score") for row in running_rows]
    width_gains = [as_float(row, "width_gain") for row in running_rows]

    strong_left_frames = sum(1 for value in left_scores if value >= 0.22)
    strong_right_frames = sum(1 for value in right_scores if value >= 0.22)
    strong_block_frames = sum(1 for value in front_blocks if value >= 0.62)
    low_conf_frames = sum(1 for value in confidences if value < 0.18)

    print(f"CSV: {csv_path}")
    print(f"Total frames: {len(rows)}")
    print(f"Running frames: {len(running_rows)}")
    print("Mode counts:")
    for mode, count in mode_counter.most_common():
        print(f"  {mode}: {count}")

    print(f"Mean confidence: {mean(confidences):.3f}")
    print(f"Mean abs(heading_error): {mean(heading_errors):.2f}")
    print(f"Mean abs(lookahead_error): {mean(lookahead_errors):.2f}")
    print(f"Mean forward_depth: {mean(forward_depths):.3f}")
    print(f"Mean left_branch_score: {mean(left_scores):.3f}")
    print(f"Mean right_branch_score: {mean(right_scores):.3f}")
    print(f"Mean front_block_score: {mean(front_blocks):.3f}")
    print(f"Mean width_gain: {mean(width_gains):.3f}")
    print(f"Low-confidence frames: {low_conf_frames}")
    print(f"Strong left-branch frames: {strong_left_frames}")
    print(f"Strong right-branch frames: {strong_right_frames}")
    print(f"Strong front-block frames: {strong_block_frames}")

    if running_rows and mean(heading_errors) > 45:
        print("Hint: heading_error is still large on average; steering gain may be too low or centerline extraction too jumpy.")
    if running_rows and mode_counter.get("RECOVER", 0) + mode_counter.get("RECOVER_REVERSE", 0) + mode_counter.get("RECOVER_SEARCH_R", 0) + mode_counter.get("RECOVER_SEARCH_L", 0) > len(running_rows) * 0.20:
        print("Hint: RECOVER is still too frequent; white-floor mask may be breaking or confidence threshold is too strict.")
    if running_rows and mode_counter.get("APPROACH_JUNCTION", 0) > len(running_rows) * 0.35:
        print("Hint: APPROACH_JUNCTION occupies a large share of frames; junction trigger may be too sensitive on normal bends.")
    if running_rows and strong_right_frames < len(running_rows) * 0.02 and strong_left_frames < len(running_rows) * 0.02:
        print("Hint: branch scores rarely go high; side expansion detection may still be too conservative for your maze.")
    if running_rows and strong_block_frames > len(running_rows) * 0.30:
        print("Hint: front_block_score is high very often; forward-depth estimate may be underestimating available corridor.")


if __name__ == "__main__":
    main()
