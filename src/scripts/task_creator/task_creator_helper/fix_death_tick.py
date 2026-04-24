import os
import re

directory = r'c:\Users\wangy\projects\x-ego\src\scripts\task_creator\task_creator_helper'
files = ['spatial_tasks.py', 'location_tasks_addon.py', 'location_tasks.py', 'coordination_tasks_addon.py', 'combat_tasks.py', 'bomb_tasks_addon.py', 'bomb_tasks.py']

for filename in files:
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as f:
        content = f.read()

    def repl(match):
        method_call = match.group(1)
        return f"""{method_call}
            global_max_tick = round_info.get('end_tick', pov_df['tick'].max())
            if death_tick is None:
                death_tick = global_max_tick
            else:
                death_tick = min(death_tick, global_max_tick)"""

    new_content = re.sub(r"(death_tick = self\._find(?:_player)?_death_tick\(pov_df\))\s+if death_tick is None:\s+death_tick = (?:int\()?pov_df\['tick'\]\.max\(\)(?:\))?", repl, content)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f'Updated {filename}')
    else:
        print(f'No changes in {filename}')
