import socket
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>V1.9 Web Remote</title>
<style>
body { 
    text-align: center; font-family: -apple-system, sans-serif; 
    background: #1e1e1e; color: white; touch-action: none; 
    user-select: none; -webkit-user-select: none; margin: 0; padding: 20px;
}
.btn { 
    width: 90px; height: 90px; font-size: 32px; font-weight: bold; 
    margin: 15px; border-radius: 45px; border: none; 
    background: #444; color: white; cursor: pointer;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3); outline: none;
}
.btn:active, .btn.active { 
    background: #007bff; box-shadow: inset 0 3px 5px rgba(0,0,0,0.5); 
}
.row { display: flex; justify-content: center; align-items: center; }
#controls { margin-top: 40px; margin-bottom: 50px; }
.action-btn { 
    width: 140px; height: 60px; font-size: 18px; font-weight: bold; 
    margin: 10px; border-radius: 12px; background: #d9534f; 
    color: white; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.action-btn:active { opacity: 0.8; }
.status { color: #aaa; margin-top: 10px; font-size: 14px; }
</style>
</head>
<body>
<h2>🏎️ V1.9 手机遥控 (Teach)</h2>
<div class="status" id="status-text">已连接 - 按住方向键录制轨迹</div>

<div id="controls">
  <div class="row">
    <button id="btn-w" class="btn">W</button>
  </div>
  <div class="row">
    <button id="btn-a" class="btn">A</button>
    <button id="btn-s" class="btn">S</button>
    <button id="btn-d" class="btn">D</button>
  </div>
</div>

<div>
  <button id="btn-save" class="action-btn" style="background:#5cb85c;">✅ 结束并保存</button>
  <button id="btn-playback" class="action-btn" style="background:#5bc0de;">▶ 单独试跑</button>
  <button id="btn-quit" class="action-btn">❌ 不保存退出</button>
</div>

<script>
const base_speed = 38;
const turn_speed = 42;
const statusEl = document.getElementById('status-text');
const HEARTBEAT_MS = 90;
const STOP_DEBOUNCE_MS = 120;
let activeCommand = null;
let heartbeatTimer = null;
let stopTimer = null;

function sendCmd(cmd, left, right) {
    fetch(`/cmd?cmd=${cmd}&left=${left}&right=${right}`).catch(e => console.error(e));
}

function clearHeartbeat() {
    if (heartbeatTimer !== null) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

function clearStopTimer() {
    if (stopTimer !== null) {
        clearTimeout(stopTimer);
        stopTimer = null;
    }
}

function setButtonState(activeId) {
    for (const id of ['btn-w', 'btn-a', 'btn-s', 'btn-d']) {
        const el = document.getElementById(id);
        if (el) {
            el.classList.toggle('active', id === activeId);
        }
    }
}

function startDrive(buttonId, left, right) {
    clearStopTimer();
    const next = { buttonId, left, right };
    const changed = !activeCommand || activeCommand.left !== left || activeCommand.right !== right || activeCommand.buttonId !== buttonId;
    activeCommand = next;
    setButtonState(buttonId);
    statusEl.innerText = `发送指令: [drive] L:${left} R:${right}`;
    if (changed) {
        sendCmd('drive', left, right);
    }
    clearHeartbeat();
    heartbeatTimer = setInterval(() => {
        if (activeCommand) {
            sendCmd('drive', activeCommand.left, activeCommand.right);
        }
    }, HEARTBEAT_MS);
}

function scheduleStop() {
    clearStopTimer();
    stopTimer = setTimeout(() => {
        clearHeartbeat();
        activeCommand = null;
        setButtonState(null);
        statusEl.innerText = '停止 (STOP)';
        sendCmd('drive', 0, 0);
    }, STOP_DEBOUNCE_MS);
}

function bindBtn(id, left, right) {
    const el = document.getElementById(id);
    const start = (e) => {
        e.preventDefault();
        startDrive(id, left, right);
    };
    const end = (e) => {
        e.preventDefault();
        if (activeCommand && activeCommand.buttonId === id) {
            scheduleStop();
        }
    };

    el.addEventListener('pointerdown', start);
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', end);
}

// 绑定四个方向
bindBtn('btn-w', base_speed, base_speed);
bindBtn('btn-s', -base_speed, -base_speed);
bindBtn('btn-a', -turn_speed, turn_speed);
bindBtn('btn-d', turn_speed, -turn_speed);

document.getElementById('btn-save').onclick = () => { 
    clearHeartbeat();
    clearStopTimer();
    activeCommand = null;
    setButtonState(null);
    sendCmd('save', 0, 0); 
    statusEl.innerText = "已保存剧本！您可以关闭该网页。";
    statusEl.style.color = "#5cb85c";
};

document.getElementById('btn-playback').onclick = () => { 
    clearHeartbeat();
    clearStopTimer();
    activeCommand = null;
    setButtonState(null);
    sendCmd('playback', 0, 0); 
    statusEl.innerText = "🚀 正在回放刚才录制的剧本...";
    statusEl.style.color = "#5bc0de";
};

document.getElementById('btn-quit').onclick = () => {
    clearHeartbeat();
    clearStopTimer();
    activeCommand = null;
    setButtonState(null);
    sendCmd('quit_record', 0, 0); 
    statusEl.innerText = "已清空退出，主程序恢复巡线。";
    statusEl.style.color = "#d9534f";
};

// 页面加载自动抢夺主视觉线程的控制权 (Force Record)
window.onload = () => {
    sendCmd('force_record', 0, 0);
};
</script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif parsed.path == "/cmd":
            qs = parse_qs(parsed.query)
            cmd = qs.get("cmd", [""])[0]
            left = int(qs.get("left", ["0"])[0])
            right = int(qs.get("right", ["0"])[0])
            
            # Send UDP to main Logic
            payload = {"cmd": cmd, "left": left, "right": right}
            sock.sendto(json.dumps(payload).encode('utf-8'), (UDP_IP, UDP_PORT))
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # 禁用默认的长篇大论的请求打印，以便在 SSH 端保持干净
        pass

if __name__ == "__main__":
    port = 8080
    server = HTTPServer(('0.0.0.0', port), RequestHandler)
    print("==================================================")
    print(f" 🚀 Web Remote Server 正在运行!")
    print(f" 👉 请打开手机或电脑浏览器访问: http://10.176.123.119:{port}")
    print("==================================================")
    print("按 Ctrl+C 关闭 Web 服务...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExiting")
        # 安全退出
        sock.sendto(json.dumps({"cmd": "quit_record"}).encode('utf-8'), (UDP_IP, UDP_PORT))
