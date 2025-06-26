import os

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
ROOT_DIR = 'data'  # Base dataset directory

def remove_invalid_files(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                path = os.path.join(subdir, file)
                print(f"❌ Removing: {path}")
                os.remove(path)

if __name__ == "__main__":
    print(f"🔍 Scanning for invalid files in: {ROOT_DIR}")
    remove_invalid_files(ROOT_DIR)
    print("✅ Done: All non-image files removed by extension.")
