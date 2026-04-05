from __future__ import annotations

import time

import paramiko


HOST = "10.176.97.136"
USERNAME = "nenene"
PASSWORD = "123"


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

    remote_command = (
        "pkill -f 'python3 /home/nenene/line_following_minimal.py' 2>/dev/null || true; "
        "python3 - <<'PY'\n"
        "import serial, time\n"
        "for port in ('/dev/serial0', '/dev/ttyAMA10', '/dev/ttyAMA0', '/dev/ttyS0'):\n"
        "    try:\n"
        "        ser = serial.Serial(port, 57600, timeout=0.1, write_timeout=0.1)\n"
        "        for _ in range(5):\n"
        "            ser.write(b'#ha')\n"
        "            ser.flush()\n"
        "            time.sleep(0.05)\n"
        "        ser.close()\n"
        "        print(f'STOP_SENT {port}')\n"
        "        break\n"
        "    except Exception:\n"
        "        continue\n"
        "PY\n"
        "pgrep -af 'python3 /home/nenene/line_following_minimal.py' || true"
    )

    stdin, stdout, stderr = client.exec_command(remote_command, timeout=20)
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()

    if output:
        print(output)
    if error:
        print("STDERR:")
        print(error)

    client.close()
    print("")
    print("停止命令已发送。")
    time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
