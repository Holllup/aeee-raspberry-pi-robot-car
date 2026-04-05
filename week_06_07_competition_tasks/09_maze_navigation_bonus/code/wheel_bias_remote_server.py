from __future__ import annotations

import argparse
import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import time


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


class ControlState:
    def __init__(self, max_speed: int):
        self.lock = threading.Lock()
        self.running = False
        self.base_speed = 24
        self.bias = 0
        self.left_speed = 24
        self.right_speed = 24
        self.max_speed = int(max_speed)
        self.last_update = time.time()

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "base_speed": self.base_speed,
                "bias": self.bias,
                "left_speed": self.left_speed,
                "right_speed": self.right_speed,
                "max_speed": self.max_speed,
                "last_update": self.last_update,
            }

    def update_from_payload(self, payload: dict) -> None:
        with self.lock:
            if "running" in payload:
                self.running = bool(payload["running"])
            if "base_speed" in payload:
                self.base_speed = clamp(int(payload["base_speed"]), -self.max_speed, self.max_speed)
            if "bias" in payload:
                self.bias = clamp(int(payload["bias"]), -10, 10)
            if "left_speed" in payload:
                self.left_speed = clamp(int(payload["left_speed"]), -self.max_speed, self.max_speed)
            if "right_speed" in payload:
                self.right_speed = clamp(int(payload["right_speed"]), -self.max_speed, self.max_speed)

            use_direct = bool(payload.get("use_direct_lr", False))
            if not use_direct:
                self.left_speed = clamp(self.base_speed - self.bias, -self.max_speed, self.max_speed)
                self.right_speed = clamp(self.base_speed + self.bias, -self.max_speed, self.max_speed)
            self.last_update = time.time()


def build_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wheel Bias Remote</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 16px; }
    .card { max-width: 560px; margin: 0 auto; border: 1px solid #ddd; border-radius: 12px; padding: 16px; }
    .row { margin: 14px 0; }
    input[type=range] { width: 100%; }
    button { padding: 10px 16px; border-radius: 8px; border: 1px solid #999; margin-right: 8px; }
    .on { background: #d7ffd7; }
    .off { background: #ffd7d7; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  </style>
</head>
<body>
  <div class="card">
    <h3>Wheel Bias Remote</h3>
    <div class="row">
      <button id="runBtn" class="off">Start Output</button>
      <button id="stopBtn">Stop Car</button>
    </div>
    <div class="row">
      <label>Base Speed: <span id="baseVal" class="mono">24</span></label>
      <input id="base" type="range" min="-70" max="70" step="1" value="24" />
    </div>
    <div class="row">
      <label>Bias (Right - Left): <span id="biasVal" class="mono">0</span></label>
      <input id="bias" type="range" min="-10" max="10" step="1" value="0" />
    </div>
    <div class="row mono" id="stateLine">left=24 right=24 running=false</div>
  </div>
  <script>
    const baseEl = document.getElementById('base');
    const biasEl = document.getElementById('bias');
    const baseVal = document.getElementById('baseVal');
    const biasVal = document.getElementById('biasVal');
    const runBtn = document.getElementById('runBtn');
    const stopBtn = document.getElementById('stopBtn');
    const stateLine = document.getElementById('stateLine');
    let running = false;

    async function postState(extra={}) {
      const payload = {
        running,
        base_speed: Number(baseEl.value),
        bias: Number(biasEl.value),
        ...extra
      };
      await fetch('/api/state', {
        method: 'POST',
        headers: {'content-type':'application/json'},
        body: JSON.stringify(payload)
      });
      await refresh();
    }

    async function refresh() {
      const res = await fetch('/api/state');
      const s = await res.json();
      running = !!s.running;
      baseEl.value = s.base_speed;
      biasEl.value = s.bias;
      baseVal.textContent = s.base_speed;
      biasVal.textContent = s.bias;
      runBtn.textContent = running ? 'Pause Output' : 'Start Output';
      runBtn.className = running ? 'on' : 'off';
      stateLine.textContent = `left=${s.left_speed} right=${s.right_speed} running=${s.running}`;
    }

    baseEl.oninput = () => { baseVal.textContent = baseEl.value; postState(); };
    biasEl.oninput = () => { biasVal.textContent = biasEl.value; postState(); };
    runBtn.onclick = async () => { running = !running; await postState(); };
    stopBtn.onclick = async () => {
      running = false;
      await postState({running:false, base_speed:0, bias:0});
    };

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


def write_state_file(state_file: Path, payload: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def make_handler(state: ControlState, state_file: Path):
    html = build_html().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def _send_json(self, payload: dict, code: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if self.path == "/api/state":
                self._send_json(state.as_dict())
                return
            self._send_json({"error": "not_found"}, code=404)

        def do_POST(self):
            if self.path != "/api/state":
                self._send_json({"error": "not_found"}, code=404)
                return
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("json payload must be object")
                state.update_from_payload(payload)
                snapshot = state.as_dict()
                write_state_file(state_file, snapshot)
                self._send_json(snapshot)
            except Exception as exc:
                self._send_json({"error": str(exc)}, code=400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phone web control for wheel bias over local network (parameter server mode)."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-speed", type=int, default=70)
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/wheel_bias_state.json"))
    args = parser.parse_args()

    state = ControlState(max_speed=int(args.max_speed))
    write_state_file(args.state_file, state.as_dict())

    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state, args.state_file))

    def _shutdown(_sig, _frame):
        print("\n[STOP] shutting down remote server")
        state.update_from_payload({"running": False, "base_speed": 0, "bias": 0})
        write_state_file(args.state_file, state.as_dict())
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[remote] open in phone browser: http://<raspberrypi-ip>:{args.port}")
    print(f"[remote] state file: {args.state_file}")
    try:
        server.serve_forever()
    finally:
        state.update_from_payload({"running": False, "base_speed": 0, "bias": 0})
        write_state_file(args.state_file, state.as_dict())
        server.server_close()


if __name__ == "__main__":
    main()
