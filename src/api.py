from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import logging

# Silence TensorFlow and Flask logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("models", "mobilenet_waste_classifier.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Waste class labels
CLASS_NAMES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
               'metal', 'paper', 'plastic', 'shoes', 'trash']

# Image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # MobileNet input size
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return "✅ Waste Classification API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded image temporarily
    temp_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(temp_path)

    # Preprocess and predict
    img_tensor = preprocess_image(temp_path)
    predictions = model.predict(img_tensor)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    # Clean up
    os.remove(temp_path)

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': round(confidence, 3)
    })

if __name__ == '__main__':
    app.run()  # No debug mode
