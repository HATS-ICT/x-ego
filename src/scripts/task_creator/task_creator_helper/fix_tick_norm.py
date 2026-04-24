import os
import re

directory = r'c:\Users\wangy\projects\x-ego\src\scripts\task_creator\task_creator_helper'
files = [
    'coordination_tasks.py',
    'coordination_tasks_addon.py',
    'location_tasks.py',
    'location_tasks_addon.py',
    'combat_tasks.py',
    'bomb_tasks.py',
    'bomb_tasks_addon.py',
    'spatial_tasks.py',
]

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Step 1: Update segment_info dictionary
    pattern_segment = r"('start_tick':\s*current_tick\s*-\s*global_min_tick,.*?)\n(\s*)('end_tick':\s*end_tick\s*-\s*global_min_tick,.*?)\n(\s*)('prediction_tick':\s*(middle_tick|prediction_tick)\s*-\s*global_min_tick,.*?)\n"
    
    def repl_segment(match):
        start_line = match.group(1)
        indent1 = match.group(2)
        end_line = match.group(3)
        indent2 = match.group(4)
        pred_line = match.group(5)
        pred_var = match.group(6)
        
        # We need to construct the new block
        # start_line might have a comment at the end, let's keep it if we want, but it's easier to just rebuild
        new_block = (
            f"'start_tick': current_tick,\n"
            f"{indent1}'end_tick': end_tick,\n"
            f"{indent2}'prediction_tick': {pred_var},\n"
            f"{indent2}'start_tick_norm': current_tick - global_min_tick,\n"
            f"{indent2}'end_tick_norm': end_tick - global_min_tick,\n"
            f"{indent2}'prediction_tick_norm': {pred_var} - global_min_tick,\n"
        )
        return new_block
        
    new_content = re.sub(pattern_segment, repl_segment, content)
    
    # Step 2: Update _create_output_csv dictionary
    pattern_csv = r"('start_tick':\s*segment\['start_tick'\],)\n(\s*)('end_tick':\s*segment\['end_tick'\],)\n(\s*)('prediction_tick':\s*segment\['prediction_tick'\],)\n"
    
    def repl_csv(match):
        start_line = match.group(1)
        indent1 = match.group(2)
        end_line = match.group(3)
        indent2 = match.group(4)
        pred_line = match.group(5)
        
        new_block = (
            f"{start_line}\n"
            f"{indent1}{end_line}\n"
            f"{indent2}{pred_line}\n"
            f"{indent2}'start_tick_norm': segment['start_tick_norm'],\n"
            f"{indent2}'end_tick_norm': segment['end_tick_norm'],\n"
            f"{indent2}'prediction_tick_norm': segment['prediction_tick_norm'],\n"
        )
        return new_block
        
    new_content = re.sub(pattern_csv, repl_csv, new_content)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f'Updated {os.path.basename(filepath)}')
    else:
        print(f'No changes in {os.path.basename(filepath)}')

for filename in files:
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        update_file(filepath)
    else:
        print(f'File not found: {filename}')
