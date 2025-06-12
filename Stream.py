import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="SolarGuard Defect Detector", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("solar_defect_model.keras")

model = load_model()

class_labels = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']

st.title("🔆 SolarGuard: Solar Panel Defect Detection")

uploaded_file = st.file_uploader("📤 Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100 

    st.markdown(f"### 🧠 Predicted Class: **{predicted_class}**")
    st.progress(float(np.max(preds)))  
    st.write("📊 Confidence Scores:")

    for i, score in enumerate(preds):
        st.write(f"• {class_labels[i]}: {score:.2%}")

