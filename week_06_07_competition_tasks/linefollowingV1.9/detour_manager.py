import threading
import socket
import json
import time

class DetourManager:
    def __init__(self, control_state, script_path):
        self.control_state = control_state
        self.script_path = script_path
        self.recording = False
        self.playing = False
        self.script = []
        self.last_cmd_time = 0
        self.current_cmd = {"left": 0, "right": 0}
        self.play_index = 0
        self.play_until = 0

    def start_record(self, now):
        self.recording = True
        self.script = []
        self.last_cmd_time = now
        self.current_cmd = {"left": 0, "right": 0}

    def stop_record(self, now, save=True):
        if self.recording:
            duration = now - self.last_cmd_time
            if duration > 0.05:
                self.script.append({"d": duration, "l": self.current_cmd["left"], "r": self.current_cmd["right"]})
            self.recording = False
            if save:
                try:
                    with open(self.script_path, "w") as f:
                        json.dump(self.script, f, indent=2)
                    print(f"Saved detour script with {len(self.script)} steps.")
                except Exception as e:
                    print(f"Save script error: {e}")

    def handle_udp(self, cmd_dict):
        now = time.perf_counter()
        cmd = cmd_dict.get("cmd", "")
        left = cmd_dict.get("left", 0)
        right = cmd_dict.get("right", 0)

        if cmd == "force_record":
            self.start_record(now)
            # 标记 control_state 状态，使主循环也知道正在强制录制
            self.control_state.set_manual_drive(0, 0, now + 10.0, mode="rc_recording")
        elif cmd == "save":
            self.stop_record(now, save=True)
            self.control_state.clear_manual_drive()
        elif cmd == "quit_record":
            self.stop_record(now, save=False)
            self.control_state.clear_manual_drive()
        elif cmd == "playback":
            def run_test_play():
                try:
                    with open(self.script_path, "r") as f:
                        scr = json.load(f)
                    for step in scr:
                        if not self.control_state.snapshot()["running"]:
                            break
                        end_t = time.perf_counter() + step["d"]
                        while time.perf_counter() < end_t:
                            self.control_state.set_manual_drive(step["l"], step["r"], time.perf_counter() + 0.5, mode="test_playback")
                            import time as tmp_time
                            tmp_time.sleep(0.02)
                    self.control_state.clear_manual_drive()
                except Exception as e:
                    print(f"Playback error: {e}")
            import threading
            threading.Thread(target=run_test_play, daemon=True).start()
        elif cmd == "drive":
            if self.recording:
                duration = now - self.last_cmd_time
                if left != self.current_cmd["left"] or right != self.current_cmd["right"]:
                    if duration > 0.05:
                        self.script.append({"d": duration, "l": self.current_cmd["left"], "r": self.current_cmd["right"]})
                    self.last_cmd_time = now
                    self.current_cmd = {"left": left, "right": right}
            # 持续刷新 manual_until 的超时时间 
            self.control_state.set_manual_drive(left, right, now + 0.4, mode="rc_recording")

    def start_playback(self, now):
        try:
            with open(self.script_path, "r") as f:
                self.script = json.load(f)
            self.playing = True
            self.play_index = 0
            if self.script:
                self.play_until = now + self.script[0]["d"]
                return True
        except Exception as e:
            print(f"Failed to load script: {e}")
        return False

    def update_playback(self, now):
        if not self.playing or self.play_index >= len(self.script):
            self.playing = False
            return False, 0, 0
        if now > self.play_until:
            self.play_index += 1
            if self.play_index >= len(self.script):
                self.playing = False
                return False, 0, 0
            self.play_until = now + self.script[self.play_index]["d"]
        step = self.script[self.play_index]
        return True, step["l"], step["r"]

def udp_listener(detour_mgr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 9999))
    sock.settimeout(1.0)
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            msg = json.loads(data.decode("utf-8"))
            detour_mgr.handle_udp(msg)
        except socket.timeout:
            continue
        except Exception:
            pass
