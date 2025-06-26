import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, train_dir, val_dir, test_dir, val_ratio=0.15, test_ratio=0.15):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for category in categories:
        category_path = os.path.join(source_dir, category)
        images = os.listdir(category_path)

        train_imgs, temp_imgs = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

        for folder_name, image_set in zip([train_dir, val_dir, test_dir], [train_imgs, val_imgs, test_imgs]):
            category_folder = os.path.join(folder_name, category)
            os.makedirs(category_folder, exist_ok=True)
            for img in image_set:
                src = os.path.join(category_path, img)
                dst = os.path.join(category_folder, img)
                shutil.copy2(src, dst)

if __name__ == "__main__":
    source = "data/raw/garbage-dataset"
    split_dataset(source, "data/train", "data/val", "data/test")
