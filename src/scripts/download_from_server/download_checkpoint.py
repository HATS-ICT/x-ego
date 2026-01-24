import subprocess
from pathlib import Path

REMOTE_USER = "yunzhewa"
REMOTE_HOST = "discovery1"
REMOTE_BASE = "/project2/ustun_1726/x-ego/output"
LOCAL_BASE = Path(r"C:\Users\wangy\projects\x-ego\output")

FOLDERS = [
    "main_ui_cover-dinov2-ui-all-260122-035704-my8c",
    "main_ui_cover-dinov2-ui-minimap-260122-045334-demx",
    "main_ui_cover-dinov2-ui-none-260122-051419-yq1p",
    "main_ui_cover-siglip2-ui-all-260122-064933-md8t",
    "main_ui_cover-siglip2-ui-minimap-260122-064933-1z0g",
    "main_ui_cover-siglip2-ui-none-260122-071834-ct2l",
    "main_ui_cover-vjepa2-ui-all-260122-072237-nrz4",
    "main_ui_cover-vjepa2-ui-minimap-260122-072237-os7x",
    "main_ui_cover-vjepa2-ui-none-260122-101106-8h2z",
]

def get_epochs_to_download(max_epoch=50):
    epochs = [0, 1, 2, 3, 4]
    epoch = 9
    while epoch <= max_epoch:
        epochs.append(epoch)
        epoch += 5
    return epochs

def main():
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    epochs = get_epochs_to_download()
    
    for folder in FOLDERS:
        print(f"Downloading {folder}...")
        
        local_folder = LOCAL_BASE / folder
        local_checkpoint = local_folder / "checkpoint"
        local_checkpoint.mkdir(parents=True, exist_ok=True)
        
        for epoch in epochs:
            remote_file = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/{folder}/checkpoint/*-e{epoch:02d}-*.ckpt"
            result = subprocess.run(["scp", remote_file, str(local_checkpoint)], capture_output=True)
            if result.returncode == 0:
                print(f"  Downloaded epoch {epoch}")
        
        remote_hparam = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/{folder}/hparam.yaml"
        subprocess.run(["scp", remote_hparam, str(local_folder)], check=True)
        
        print(f"Completed {folder}")

if __name__ == "__main__":
    main()
