import subprocess
import os

missing_videos = [
    "1-1f299663-4a6c-4610-90bf-e261e611cd6a-1-1",
    "1-1f5f9f17-0111-4e9c-bb82-e39d3e9020e6-1-1",
    "1-387be352-f534-42a8-959f-f820cb6eba36-1-1",
    "1-487c3fe4-8463-41e9-a1ec-01ad886c944a-1-1",
    "1-5c9e9791-6588-4fb4-b684-8a6654d70ae0-1-1",
    "1-72fe23a2-9f2a-41be-9e55-06ef1ce41788-1-1",
    "1-9a90c94c-fe7d-4d90-8d8a-689f47b14b3a-1-1",
    "1-a05c1ad6-1d53-44f6-be01-6bb8fb8b65e7-1-1",
    "1-b6ffb6c2-80e3-4e12-8cb4-6370d6a22e24-1-1",
    "1-bbafd89c-f0e2-4ac1-8ae0-05e30584a609-1-1",
    "1-c34f591e-c054-49c8-908b-f30347355b3a-1-1",
    "1-cdc47b44-92e6-44f2-848f-1b9f730a6d72-1-1",
    "1-ce2fee0b-d399-4e6e-a542-1cf2ddab2676-1-1"
]

remote_base = "yunzhewa@discovery2.carc.usc.edu:/project2/ustun_1726/CTFM/data/full/recording/video"
local_dir = r"E:\files\data\cs101\recording\video"

# Make sure local directory exists
if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

# Filter out videos that already exist locally
videos_to_transfer = []
for vid in missing_videos:
    vid_path = os.path.join(local_dir, vid)
    if os.path.exists(vid_path):
        print(f"Skipping {vid} because it already exists locally.")
    else:
        videos_to_transfer.append(vid)

if not videos_to_transfer:
    print("\nAll videos already exist locally! Nothing to transfer.")
    exit(0)

print(f"\nStarting transfer of {len(videos_to_transfer)} videos to {local_dir} in ONE connection...")

# Build a single scp command: scp -r host:/vid1 host:/vid2 ... local_dir
cmd = ["scp", "-r"]
for vid in videos_to_transfer:
    remote_path = f"{remote_base}/{vid}"
    cmd.append(remote_path)

cmd.append(local_dir)

try:
    print("Running command: scp ... (this might take a while, please wait or authenticate if prompted)")
    subprocess.run(cmd, check=True)
    print("\nSuccessfully transferred all missing videos!")
except subprocess.CalledProcessError as e:
    print(f"\nFailed to transfer videos: {e}")

print("\nAll transfers completed. You can run temp_check_video_all.py again to verify coverage.")
