import os
from pathlib import Path

output_dir = Path("/project2/ustun_1726/x-ego/output/main_exp")

for exp_dir in sorted(output_dir.iterdir()):
    if exp_dir.is_dir():
        test_results = exp_dir / "test_analysis" / "enemy_location_last" / "test_results.json"
        if not test_results.exists():
            print(exp_dir.name)

