import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def build_output_path(output, width, height, duration):
    if output:
        return Path(output)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{width}x{height}_{duration}s_{stamp}.mp4"
    return Path.home() / "Desktop" / filename


def main():
    parser = argparse.ArgumentParser(description="Record a video on Raspberry Pi with proper MP4 output.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--duration", type=int, default=5, help="Duration in seconds")
    parser.add_argument("--output", type=str, default="", help="Output MP4 path")
    args = parser.parse_args()

    output_path = build_output_path(args.output, args.width, args.height, args.duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rpicam-vid",
        "-t",
        str(args.duration * 1000),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--framerate",
        str(args.framerate),
        "--codec",
        "libav",
        "--libav-format",
        "mp4",
        "-o",
        str(output_path),
    ]

    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
