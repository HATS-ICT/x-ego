import os

list_files = {
    'dust2': r'c:\Users\wangy\projects\x-ego\data\dust2\video_list.txt',
    'inferno': r'c:\Users\wangy\projects\x-ego\data\inferno\video_list.txt',
    'mirage': r'c:\Users\wangy\projects\x-ego\data\mirage\video_list.txt'
}

video_dirs = [
    r'E:\files\data\cs101\recording\video',
    r'E:\files\recordings',
    r'E:\files\recordings_0720',
    r'E:\files\recordings1-30'
]

existing_videos = set()
video_locations = {}

for vdir in video_dirs:
    if os.path.exists(vdir):
        for vid in os.listdir(vdir):
            existing_videos.add(vid)
            if vid not in video_locations:
                video_locations[vid] = []
            video_locations[vid].append(vdir)
    else:
        print(f"Warning: Directory {vdir} does not exist.")

all_listed_videos = set()
map_lists = {}

for map_name, list_file in list_files.items():
    if not os.path.exists(list_file):
        print(f"Warning: List file {list_file} does not exist.")
        map_lists[map_name] = set()
        continue
        
    with open(list_file, 'rb') as f:
        raw_data = f.read()
    
    if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
        encoding = 'utf-16'
    else:
        encoding = 'utf-8'
        
    with open(list_file, 'r', encoding=encoding) as f:
        videos = [line.strip() for line in f if line.strip()]
        map_lists[map_name] = set(videos)
        all_listed_videos.update(videos)

print("=== Coverage Report ===\n")

for map_name, videos in map_lists.items():
    print(f"--- {map_name.capitalize()} ---")
    print(f"Total videos in list: {len(videos)}")
    
    existing = videos.intersection(existing_videos)
    missing = videos - existing_videos
    
    print(f"Existing in directory: {len(existing)}")
    print(f"Missing from directory: {len(missing)}")
    
    if missing:
        print("Missing IDs:")
        for vid in sorted(list(missing)):
            print(f"  {vid}")
    print()

print("--- Overall ---")
print(f"Total videos across all lists: {len(all_listed_videos)}")
overall_existing = all_listed_videos.intersection(existing_videos)
overall_missing = all_listed_videos - existing_videos
print(f"Total Existing: {len(overall_existing)}")
print(f"Total Missing: {len(overall_missing)}")

# Optional: Find videos in directory that are not in any of the lists
unlisted_existing = existing_videos - all_listed_videos
print(f"\nVideos in directories but NOT in any list: {len(unlisted_existing)}")
# if unlisted_existing:
#     print("Unlisted existing IDs (first 10):")
#     for vid in sorted(list(unlisted_existing))[:10]:
#         print(f"  {vid}")

duplicates = {vid: dirs for vid, dirs in video_locations.items() if len(dirs) > 1}
if duplicates:
    print(f"\n--- Duplicates Found ({len(duplicates)} videos) ---")
    for vid, dirs in duplicates.items():
        if vid in all_listed_videos:
            print(f"  {vid} is in: {', '.join(dirs)}")
else:
    print("\n--- No Duplicates Found ---")

remote_list_file = r'C:\Users\wangy\Desktop\match_list.txt'
remote_videos = set()
if os.path.exists(remote_list_file):
    with open(remote_list_file, 'rb') as f:
        raw_data = f.read()
    if raw_data.startswith(b'\xff\xfe') or raw_data.startswith(b'\xfe\xff'):
        encoding = 'utf-16'
    else:
        encoding = 'utf-8'
    with open(remote_list_file, 'r', encoding=encoding) as f:
        remote_videos = set([line.strip() for line in f if line.strip()])

if remote_videos:
    print("\n--- Remote Server Coverage for Missing Videos ---")
    missing_on_remote = overall_missing - remote_videos
    found_on_remote = overall_missing.intersection(remote_videos)
    print(f"Out of {len(overall_missing)} missing videos:")
    print(f"Found on remote server: {len(found_on_remote)}")
    print(f"Still missing (not on remote): {len(missing_on_remote)}")
    
    if found_on_remote:
        print("\nCan be transferred from remote:")
        for vid in sorted(list(found_on_remote)):
            print(f"  {vid}")
            
    if missing_on_remote:
        print("\nNOT found on remote (completely missing):")
        for vid in sorted(list(missing_on_remote)):
            print(f"  {vid}")
