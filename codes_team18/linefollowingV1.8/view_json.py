import paramiko
import sys

host = "10.176.123.119"
username = "jacob"
password = "123"

def get_json():
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, username=username, password=password, timeout=5)
        stdin, stdout, stderr = client.exec_command("cat /home/jacob/line_following_v1_8/detour_script.json")
        output = stdout.read().decode()
        if not output.strip():
            print("File is empty or not found.")
            err = stderr.read().decode()
            if err:
                print("Error:", err)
        else:
            print("--- START JSON ---")
            print(output)
            print("--- END JSON ---")
        client.close()
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

if __name__ == "__main__":
    get_json()
