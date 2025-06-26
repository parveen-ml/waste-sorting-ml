import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("models/mobilenet_waste_classifier.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']  # Update if needed

# Preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# UI
st.title("♻️ Waste Classification Web App")
st.write("Upload an image to classify the type of waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.markdown(f"### 🏷️ Predicted: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")
