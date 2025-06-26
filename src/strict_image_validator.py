import os
from PIL import Image

def is_image_valid(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def clean_directory(root_dir):
    print(f"🔍 Checking images in: {root_dir}")
    bad_files = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if not is_image_valid(filepath):
                print(f"❌ Removing invalid image: {filepath}")
                os.remove(filepath)
                bad_files += 1
    print(f"✅ Finished. Removed {bad_files} invalid images.")

if __name__ == "__main__":
    for folder in ['data/train', 'data/val', 'data/test']:
        clean_directory(folder)
