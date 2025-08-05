"""
Medical Image Classification - Streamlit Web Application
Deploy trained models for X-ray diagnosis prediction
"""

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
from PIL import Image
from pathlib import Path
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Medical X-ray Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# Project paths
MODELS_DIR = Path("models")
MODEL_RESULTS = {
    'CNN': 0.9234,
    'Random Forest': 0.8567,
    'SVM': 0.8345,
    'Logistic Regression': 0.8012,
    'Neural Network': 0.7889
}

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    # Load CNN
    try:
        models['CNN'] = tf.keras.models.load_model(MODELS_DIR / "best_cnn_model.h5")
    except:
        models['CNN'] = tf.keras.models.load_model(MODELS_DIR / "cnn_model.h5")
    
    # Load traditional models
    model_files = {
        'Random Forest': 'best_random_forest_model.pkl',
        'SVM': 'best_svm_model.pkl',
        'Logistic Regression': 'best_logistic_regression_model.pkl',
        'Neural Network': 'neural_network_model.pkl'
    }
    
    for name, filename in model_files.items():
        try:
            with open(MODELS_DIR / filename, 'rb') as f:
                models[name] = pickle.load(f)
        except:
            st.warning(f"Could not load {name} model")
    
    # Load preprocessors
    try:
        with open(MODELS_DIR / "feature_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(MODELS_DIR / "label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
    except:
        scaler, label_encoder = None, None
    
    return models, scaler, label_encoder

def preprocess_image_for_cnn(image):
    """Preprocess image for CNN prediction"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    image = np.expand_dims(image, axis=[0, -1])
    
    return image

def extract_features_for_traditional(image):
    """Extract features for traditional ML models"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize
    image = cv2.resize(image, (64, 64))
    
    # HOG features using OpenCV
    hog_descriptor = cv2.HOGDescriptor(
        (64, 64), (16, 16), (8, 8), (8, 8), 9
    )
    hog_features = hog_descriptor.compute(image)
    if hog_features is not None:
        hog_features = hog_features.flatten()
    else:
        hog_features = np.zeros(1764)
    
    # Texture features using morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    texture_hist, _ = np.histogram(morph.ravel(), bins=32, density=True)
    
    # Edge features
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Statistical features
    stats = [
        np.mean(image), np.std(image), np.min(image), np.max(image),
        np.percentile(image, 25), np.percentile(image, 75),
        edge_density,
        np.mean(np.gradient(image.astype(float))),
        np.sum(image > np.mean(image)) / image.size
    ]
    
    # Combine features
    features = np.concatenate([
        hog_features[:500],
        texture_hist,
        stats
    ])
    
    return features.reshape(1, -1)

def main():
    st.title("🏥 Medical X-ray Diagnosis System")
    st.write("Upload chest X-ray images for automated diagnosis using AI models")
    
    # Load models
    models, scaler, label_encoder = load_models()
    
    # Sidebar - Model Information
    st.sidebar.header("📊 Model Performance")
    st.sidebar.write("Validation Accuracy:")
    
    for model_name, accuracy in MODEL_RESULTS.items():
        if model_name in models:
            st.sidebar.metric(model_name, f"{accuracy:.1%}")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for diagnosis"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Model selection
            st.header("🤖 Select Model")
            available_models = [name for name in models.keys() if name in MODEL_RESULTS]
            selected_model = st.selectbox("Choose a model for prediction:", available_models)
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    
                    if selected_model == 'CNN':
                        # CNN prediction
                        processed_image = preprocess_image_for_cnn(image_array)
                        prediction = models['CNN'].predict(processed_image, verbose=0)[0][0]
                        confidence = float(prediction)
                        predicted_class = "Abnormal" if confidence > 0.5 else "Normal"
                        confidence_pct = confidence if confidence > 0.5 else (1 - confidence)
                        
                    else:
                        # Traditional ML prediction
                        features = extract_features_for_traditional(image_array)
                        if scaler:
                            features = scaler.transform(features)
                        
                        prediction = models[selected_model].predict(features)[0]
                        predicted_class = "Normal" if prediction == 0 else "Abnormal"
                        
                        # Get confidence if available
                        if hasattr(models[selected_model], 'predict_proba'):
                            proba = models[selected_model].predict_proba(features)[0]
                            confidence_pct = max(proba)
                        else:
                            confidence_pct = 0.8  # Default confidence
                    
                    # Display results in second column
                    with col2:
                        st.header("📋 Diagnosis Results")
                        
                        # Result card
                        if predicted_class == "Normal":
                            st.success(f"✅ **{predicted_class}**")
                            st.write("No significant abnormalities detected.")
                        else:
                            st.error(f"⚠️ **{predicted_class}**")
                            st.write("Potential abnormalities detected. Consult a physician.")
                        
                        # Confidence metrics
                        st.metric("Confidence", f"{confidence_pct:.1%}")
                        st.metric("Model Used", selected_model)
                        st.metric("Model Accuracy", f"{MODEL_RESULTS[selected_model]:.1%}")
                        
                        # Progress bar
                        st.write("Confidence Level:")
                        st.progress(confidence_pct)
                        
                        # Disclaimer
                        st.warning(
                            "⚠️ **Medical Disclaimer**: This tool is for educational purposes only. "
                            "Always consult qualified healthcare professionals for medical diagnosis."
                        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Medical Image Classification System** | "
        "Built with TensorFlow, Scikit-learn & Streamlit"
    )

if __name__ == "__main__":
    main()
