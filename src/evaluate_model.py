import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model
model = tf.keras.models.load_model("D:\Projects_Repo\waste-sorting-ml\models\mobilenet_waste_classifier.h5")

# Load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Get true labels
class_names = test_ds.class_names
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Predict
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluation
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n🧱 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
