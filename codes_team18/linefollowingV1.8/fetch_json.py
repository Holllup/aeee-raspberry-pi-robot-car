import paramiko
import sys

host = "10.176.123.119"
username = "jacob"
password = "123"

def fetch_json():
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get("/home/jacob/line_following_v1_8/detour_script.json", "downloaded_script.json")
        sftp.close()
        transport.close()
        print("Download successful.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_json()
