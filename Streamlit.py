import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configuration
MODEL_PATH = 'solar_panel_classifier.h5'
CLASS_NAMES = ['Clean', 'Dusty', 'Bird-Drop', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']
IMG_SIZE = (224, 224)

def load_classification_model():
    """Load the trained classification model"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to array and resize
        img_array = np.array(image)
        img_array = cv2.resize(img_array, IMG_SIZE)
        
        # Convert to RGB if needed
        if img_array.shape[-1] == 1:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Preprocess for EfficientNet
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def display_prediction(image, model):
    """Display prediction results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Solar Panel', use_column_width=True)
    
    with col2:
        # Preprocess and predict
        processed_img = preprocess_image(image)
        if processed_img is not None and model is not None:
            prediction = model.predict(processed_img)
            pred_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            st.subheader("Prediction Results")
            st.write(f"*Condition:* {pred_class}")
            st.write(f"*Confidence:* {confidence:.2f}%")
            
            # Show probability distribution
            st.subheader("Probability Distribution")
            prob_data = {
                'Condition': CLASS_NAMES,
                'Probability': prediction[0]
            }
            st.bar_chart(prob_data, x='Condition', y='Probability')

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="SolarGuard", page_icon="☀", layout="wide")
    
    # Load model (cache to avoid reloading)
    @st.cache_resource
    def load_model():
        return load_classification_model()
    
    model = load_model()
    
    # App header
    st.title("☀ SolarGuard: Solar Panel Defect Detection")
    st.markdown("""
    Upload an image of a solar panel to detect defects and classify its condition.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a solar panel image...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            display_prediction(image, model)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Add information sections
    st.markdown("---")
    st.subheader("About SolarGuard")
    st.write("""
    SolarGuard is an AI-powered system that automatically detects defects and classifies 
    the condition of solar panels. It helps in:
    - Automated inspection of solar farms
    - Optimizing maintenance schedules
    - Improving energy efficiency
    - Reducing operational costs
    """)
    
    st.subheader("Defect Classes")
    cols = st.columns(3)
    for i, class_name in enumerate(CLASS_NAMES):
        with cols[i % 3]:
            st.markdown(f"{class_name}")
            st.write("Clean panels" if class_name == "Clean" else f"Panels with {class_name.lower()}")

if __name__ == "__main__":
    main()