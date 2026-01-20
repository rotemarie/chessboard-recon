import os
import shutil
from pathlib import Path

# --- CONFIGURATION ---
# 1. Path to your REFERENCE folders (the manual split you already made)
ref_occluded_path = Path("/home/david/Downloads/dataset/val-ocluded")
ref_clean_path = Path("/home/david/Downloads/dataset/val-no-oclusions")

# 2. Path to the SOURCE folder you want to split (the padded val folder)
padded_val_source = Path("/home/david/Downloads/new_data/dataset_padded/dataset_padded/val")

# 3. Path where you want the NEW padded split folders to be created
padded_dest_root = Path("/home/david/Downloads/new_data/dataset_padded/dataset_padded") 
# This will create 'val_occluded' and 'val_clean' inside this folder

# ---------------------

def get_file_structure(root_path):
    """
    Walks through a directory and returns a set of relative paths
    (e.g., 'black_pawn/image_01.jpg').
    """
    file_map = set()
    if not root_path.exists():
        print(f"Warning: Reference path {root_path} does not exist.")
        return file_map

    for root, dirs, files in os.walk(root_path):
        for file in files:
            # We filter for common image extensions to avoid system files
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Create a relative path: class_folder/filename.jpg
                full_path = Path(root) / file
                relative_path = full_path.relative_to(root_path)
                file_map.add(relative_path)
    return file_map

def split_padded_dataset():
    print("Scanning reference folders...")
    # 1. Build the map of which files are occluded vs clean
    occluded_files = get_file_structure(ref_occluded_path)
    clean_files = get_file_structure(ref_clean_path)
    
    print(f"Found {len(occluded_files)} occluded reference images.")
    print(f"Found {len(clean_files)} clean reference images.")

    # 2. Define new destination paths
    new_occluded_dir = padded_dest_root / "val_occluded"
    new_clean_dir = padded_dest_root / "val_clean"

    # 3. Iterate through the padded source
    print("\nProcessing padded images...")
    processed_count = 0
    
    for root, dirs, files in os.walk(padded_val_source):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                
                # Get current relative path (e.g., white_king/img1.jpg)
                current_full_path = Path(root) / file
                relative_path = current_full_path.relative_to(padded_val_source)
                
                dest_path = None
                
                # Check where this file belongs
                if relative_path in occluded_files:
                    dest_path = new_occluded_dir / relative_path
                elif relative_path in clean_files:
                    dest_path = new_clean_dir / relative_path
                else:
                    # File exists in padded but wasn't in the manual split
                    print(f"Skipping {relative_path} (not found in reference split)")
                    continue

                # Create destination subfolder (e.g., val_clean/black_pawn) if it doesn't exist
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(current_full_path, dest_path)
                processed_count += 1

    print("-" * 30)
    print(f"Success! Processed {processed_count} images.")
    print(f"New folders created at:\n{new_occluded_dir}\n{new_clean_dir}")

if __name__ == "__main__":
    split_padded_dataset()