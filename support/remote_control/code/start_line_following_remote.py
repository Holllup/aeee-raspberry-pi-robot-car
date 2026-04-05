from __future__ import annotations

import sys
import time

import paramiko


HOST = "10.176.97.136"
USERNAME = "nenene"
PASSWORD = "123"
REMOTE_COMMAND = "python3 /home/nenene/line_following_minimal.py --arm --headless"


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {USERNAME}@{HOST} ...")
    client.connect(
        hostname=HOST,
        username=USERNAME,
        password=PASSWORD,
        timeout=10,
        auth_timeout=10,
        banner_timeout=10,
    )

    launch_command = (
        "pkill -f 'python3 /home/nenene/line_following_minimal.py' 2>/dev/null || true; "
        "nohup "
        f"{REMOTE_COMMAND} "
        "> /home/nenene/line_following_launch.log 2>&1 < /dev/null & "
        "sleep 1; "
        "pgrep -af 'python3 /home/nenene/line_following_minimal.py' || true; "
        "echo '---'; "
        "tail -n 20 /home/nenene/line_following_launch.log 2>/dev/null || true"
    )

    stdin, stdout, stderr = client.exec_command(launch_command, timeout=20)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()

    if output:
        print(output)
    if error:
        print("STDERR:")
        print(error)

    client.close()
    print("")
    print("巡线程序已发送启动命令。")
    print("日志文件: /home/nenene/line_following_launch.log")
    time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
