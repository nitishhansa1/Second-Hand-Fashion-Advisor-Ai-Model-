import os
import shutil
import csv
from collections import defaultdict

styles_file = "archive/styles.csv"
images_dir = "archive/images"
output_dir = "archive/categorized_images"

def organize_dataset():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Reading {styles_file}...")
    
    # Read the mapping of ID to Category
    id_to_category = {}
    with open(styles_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["id"]
            category = row["articleType"]
            
            # Clean category name (e.g., remove slashes or bad characters)
            category = category.replace("/", "_").replace("\\", "_").strip()
            id_to_category[f"{img_id}.jpg"] = category

    total_images = len(os.listdir(images_dir))
    print(f"Found {total_images} images. Starting categorization...")

    moved_count = 0
    missing_csv = 0

    # Move images to Category folders
    for filename in os.listdir(images_dir):
        if not filename.endswith(".jpg"):
            continue
            
        src_path = os.path.join(images_dir, filename)
        
        category = id_to_category.get(filename)
        if category:
            cat_dir = os.path.join(output_dir, category)
            if not os.path.exists(cat_dir):
                os.makedirs(cat_dir)
                
            dst_path = os.path.join(cat_dir, filename)
            
            # We use copy2 so the original flat folder stays safe as a backup for now
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            moved_count += 1
            if moved_count % 5000 == 0:
                print(f"  ...Categorized {moved_count} images...")
        else:
            missing_csv += 1

    print("\n[SUCCESS] Dataset Reorganized!")
    print(f"- Successfully categorized {moved_count} images.")
    if missing_csv > 0:
        print(f"- {missing_csv} images had no matching category in styles.csv and were skipped.")
    print(f"- Organized images are now located in: {os.path.abspath(output_dir)}")
    
if __name__ == "__main__":
    organize_dataset()
