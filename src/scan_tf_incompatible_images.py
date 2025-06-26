import os
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

image_root_dir = "data"

def is_tf_compatible_image(path):
    try:
        img_raw = tf.io.read_file(path)
        _ = tf.image.decode_image(img_raw)
        return True
    except (InvalidArgumentError, tf.errors.InvalidArgumentError):
        return False

bad_images = []

for root, _, files in os.walk(image_root_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not is_tf_compatible_image(file_path):
            print(f"❌ Invalid TensorFlow image: {file_path}")
            bad_images.append(file_path)

print(f"\n✅ Scan complete. Found {len(bad_images)} incompatible images.")

# Optional: Delete bad files
for path in bad_images:
    os.remove(path)
    print(f"🗑️ Deleted: {path}")
