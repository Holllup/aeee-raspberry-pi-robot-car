import paramiko
import os
import sys

host = "10.176.123.119"
username = "jacob"
password = "123"
local_dir = r"d:\Onedrive\OneDrive - The University of Nottingham Ningbo China\桌面\树莓派\迭代版本\巡线V1.8"
remote_dir = "/home/jacob/line_following_v1_8"

def upload_dir_via_sftp(sftp, local_dir, remote_dir):
    try:
        sftp.mkdir(remote_dir)
        print(f"Created {remote_dir}")
    except IOError:
        print(f"Directory {remote_dir} may already exist.")

    for item in os.listdir(local_dir):
        if item in ['__pycache__', 'patch_v1_8.py', 'line_following_v1_7_obstacle_detour.py']:
            continue
        local_path = os.path.join(local_dir, item)
        remote_path = remote_dir + "/" + item
        if os.path.isfile(local_path):
            print(f"Uploading {item}...")
            sftp.put(local_path, remote_path)
            # Add executable permission for py files
            if local_path.endswith('.py'):
                sftp.chmod(remote_path, 0o755)

def main():
    try:
        transport = paramiko.Transport((host, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("SFTP connected.")
        
        upload_dir_via_sftp(sftp, local_dir, remote_dir)
        
        sftp.close()
        transport.close()
        print("Upload complete!")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
