import paramiko

def fetch_video():
    host = "10.176.123.119"
    username = "jacob"
    password = "123"
    remote_path = "/home/jacob/line_following_v1_9/debug_football.mp4"
    local_path = "debug_football.mp4"

    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print(f"Downloading {remote_path}...")
        sftp.get(remote_path, local_path)
        sftp.close()
        transport.close()
        print("Download complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_video()
