import os
from PIL import Image

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

def clean_directory(directory):
    print(f"📂 Scanning: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip if extension is not valid
            if not file.lower().endswith(VALID_EXTENSIONS):
                print(f"❌ Deleting unsupported file: {file_path}")
                os.remove(file_path)
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"⚠️ Corrupt image deleted: {file_path} | Error: {e}")
                os.remove(file_path)

if __name__ == "__main__":
    clean_directory("data/train")
    clean_directory("data/val")
    clean_directory("data/test")
    print("✅ All invalid/corrupt images removed.")
