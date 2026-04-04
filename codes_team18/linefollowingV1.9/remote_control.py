import socket
import sys
import tty
import termios
import select
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def getch_timeout(timeout=0.15):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            ch = sys.stdin.read(1)
        else:
            ch = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("======================================")
print("  V1.9 Remote Control (Teach & Repeat)")
print("======================================")
print("  *** 键盘操作说明(已升级为流式点动控制) ***")
print("  按住 W/S/A/D 才移动，松开按键立刻刹车！")
print("  回车(ENTER)  : 保存剧本并退出录制，交还给主程序控制")
print("  Q            : 不保存，强制退出遥控")
print("--------------------------------------")

# 发送强制录制/接管信号
sock.sendto(json.dumps({"cmd": "force_record"}).encode('utf-8'), (UDP_IP, UDP_PORT))

base_speed = 35
turn_speed = 35

last_sent = None
def send_cmd(cmd):
    global last_sent
    if list(cmd.items()) != last_sent:
        sock.sendto(json.dumps(cmd).encode('utf-8'), (UDP_IP, UDP_PORT))
        last_sent = list(cmd.items())

try:
    while True:
        char = getch_timeout(0.15)
        cmd = {"cmd": "drive", "left": 0, "right": 0}
        
        if char is not None:
            if char.lower() == 'w':
                cmd["left"] = base_speed
                cmd["right"] = base_speed
                print("Action: FORWARD\r", end="")
            elif char.lower() == 's':
                cmd["left"] = -base_speed
                cmd["right"] = -base_speed
                print("Action: BACKWARD\r", end="")
            elif char.lower() == 'a':
                cmd["left"] = -turn_speed
                cmd["right"] = turn_speed
                print("Action: LEFT\r", end="")
            elif char.lower() == 'd':
                cmd["left"] = turn_speed
                cmd["right"] = -turn_speed
                print("Action: RIGHT\r", end="")
            elif char == '\r' or char == '\n':
                print("\nAction: SAVE & FINISH RECORDING\r")
                sock.sendto(json.dumps({"cmd": "save"}).encode('utf-8'), (UDP_IP, UDP_PORT))
                print("退出遥控，主程序已恢复。\r")
                break
            elif char.lower() == 'q':
                print("\nAction: QUIT RC (不保存)\r")
                sock.sendto(json.dumps({"cmd": "quit_record"}).encode('utf-8'), (UDP_IP, UDP_PORT))
                break
            elif char == ' ':
                # 空格强制覆盖为 stop
                pass
        else:
            # timeout means key released
            if last_sent is not None and last_sent[1][1] != 0: # left speed not 0
                print("Action: (Released) STOP\r", end="")
        
        send_cmd(cmd)

except KeyboardInterrupt:
    sock.sendto(json.dumps({"cmd": "quit_record"}).encode('utf-8'), (UDP_IP, UDP_PORT))
    print("\nExited.\r")
