from pathlib import Path
import os

root_dir = Path("/home/stg/Dev/research_batch_2026")

if not root_dir.exists():
    print(f"Directory {root_dir} does not exist.")
    exit(1)

renamed_count = 0
for entry in root_dir.iterdir():
    if entry.is_dir() and entry.name.startswith("__"):
        new_name = entry.name[2:]
        new_path = entry.parent / new_name
        
        # Check if destination already exists
        if new_path.exists():
            print(f"Skipping {entry.name}: {new_name} already exists.")
            continue
            
        print(f"Renaming: {entry.name} -> {new_name}")
        entry.rename(new_path)
        renamed_count += 1

print(f"Total folders renamed: {renamed_count}")
