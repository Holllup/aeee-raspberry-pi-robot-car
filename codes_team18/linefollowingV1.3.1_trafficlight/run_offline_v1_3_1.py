import argparse
import importlib.util
import sys
import types
from pathlib import Path

import cv2


def install_stub_modules():
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name):
            return self

    picamera2_mod = types.ModuleType("picamera2")
    picamera2_mod.Picamera2 = _Dummy
    enc_mod = types.ModuleType("picamera2.encoders")
    enc_mod.H264Encoder = _Dummy
    out_mod = types.ModuleType("picamera2.outputs")
    out_mod.FileOutput = _Dummy
    picamera2_mod.encoders = enc_mod
    picamera2_mod.outputs = out_mod
    sys.modules["picamera2"] = picamera2_mod
    sys.modules["picamera2.encoders"] = enc_mod
    sys.modules["picamera2.outputs"] = out_mod

    gpio_mod = types.ModuleType("RPi.GPIO")
    gpio_mod.BCM = 0
    gpio_mod.OUT = 0
    gpio_mod.HIGH = 1
    gpio_mod.LOW = 0
    gpio_mod.setwarnings = lambda *_args, **_kwargs: None
    gpio_mod.setmode = lambda *_args, **_kwargs: None
    gpio_mod.setup = lambda *_args, **_kwargs: None
    gpio_mod.output = lambda *_args, **_kwargs: None
    gpio_mod.cleanup = lambda *_args, **_kwargs: None
    gpio_mod.PWM = lambda *_args, **_kwargs: _Dummy()
    rpi_mod = types.ModuleType("RPi")
    rpi_mod.GPIO = gpio_mod
    sys.modules["RPi"] = rpi_mod
    sys.modules["RPi.GPIO"] = gpio_mod

    serial_mod = types.ModuleType("serial")

    class _SerialException(Exception):
        pass

    serial_mod.SerialException = _SerialException
    serial_mod.Serial = _Dummy
    sys.modules["serial"] = serial_mod


def load_v131_module(script_path: Path):
    install_stub_modules()
    spec = importlib.util.spec_from_file_location("v131", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_video(module, input_path: Path, output_path: Path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 20.0
    width, height = module.PROCESS_SIZE
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open output: {output_path}")

    total_frames = 0
    sign_hits = {"ALARM": 0, "TRAFFIC": 0, "NONE": 0}
    light_hits = {"RED": 0, "GREEN": 0, "YELLOW": 0, "NONE": 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        frame = cv2.resize(frame, module.PROCESS_SIZE, interpolation=cv2.INTER_AREA)

        sign_roi = module.detect_sign_roi(frame)
        sign_result = module.classify_sign_symbol(frame, sign_roi)
        label = "NONE" if sign_result is None else sign_result.get("label", "NONE")
        if label not in sign_hits:
            label = "NONE"
        sign_hits[label] += 1

        if label == "TRAFFIC" and sign_roi is not None:
            traffic_light_result = module.detect_traffic_light_state(frame, sign_roi)
            state = traffic_light_result.get("state", "NONE")
        else:
            traffic_light_result = None
            state = "NONE"
        if state not in light_hits:
            state = "NONE"
        light_hits[state] += 1

        if sign_roi is not None:
            x, y, w, h = sign_roi["outer_rect"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 255), 2)
        if traffic_light_result is not None and traffic_light_result.get("rect") is not None:
            x, y, w, h = traffic_light_result["rect"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        score = 0.0 if sign_result is None else float(sign_result.get("match_score", 0.0))
        cv2.putText(
            frame,
            f"SIGN {label} {score:.2f}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"LIGHT {state}",
            (18, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    return {
        "input": str(input_path),
        "output": str(output_path),
        "frames": total_frames,
        "sign_hits": sign_hits,
        "light_hits": light_hits,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Path to line_following_minimal_v1_3_1.py")
    parser.add_argument("--videos", nargs="+", required=True, help="Input video files")
    parser.add_argument("--output-dir", required=True, help="Output directory for debug videos")
    args = parser.parse_args()

    script_path = Path(args.script)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_v131_module(script_path)

    for video in args.videos:
        input_path = Path(video)
        stem = input_path.stem
        output_path = output_dir / f"{stem}_v1_3_1_debug.mp4"
        result = run_video(module, input_path, output_path)
        print(f"[DONE] {result['input']}")
        print(f"  output: {result['output']}")
        print(f"  frames: {result['frames']}")
        print(f"  sign_hits: {result['sign_hits']}")
        print(f"  light_hits: {result['light_hits']}")


if __name__ == "__main__":
    main()
