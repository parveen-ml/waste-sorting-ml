import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_datasets(train_dir, val_dir, test_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        label_mode='categorical',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_ds, val_ds, test_ds
