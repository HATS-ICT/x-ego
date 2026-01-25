#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

MODELS = ["siglip2", "dinov2", "vjepa2", "clip"]
SETTINGS = [
    (False, True, "nomask-recon"),
    (True, True, "mask-recon"),
]

def run_command(cmd: list[str], name: str) -> bool:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"FAILED\n{result.stderr[-2000:]}")
        return False
    print("SUCCESS")
    return True

def main():
    results = []
    for model in MODELS:
        for random_mask, recon, setting_name in SETTINGS:
            name = f"dual_head-{model}-{setting_name}"
            cmd = [
                sys.executable, "main.py",
                "--mode", "dev",
                "--task", "contrastive",
                f"model.encoder.model_type={model}",
                "data.ui_mask=all",
                f"data.random_mask.enable={str(random_mask).lower()}",
                f"model.reconstruction.enable={str(recon).lower()}",
            ]
            success = run_command(cmd, name)
            results.append((name, success))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    passed = sum(1 for _, s in results if s)
    for name, success in results:
        print(f"{'PASS' if success else 'FAIL'}: {name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    sys.exit(0 if passed == len(results) else 1)

if __name__ == "__main__":
    main()
