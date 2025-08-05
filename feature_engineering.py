"""
Medical Image Classification - Feature Engineering
Prepare features for CNN and traditional ML models
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Project paths
PROJECT_ROOT = Path("D:/ninic/Final Project 2")
PROCESSED_DATA_DIR = PROJECT_ROOT / "data/processed"
FEATURES_DIR = PROJECT_ROOT / "data/features"
MODELS_DIR = PROJECT_ROOT / "models"

class FeatureEngineer:
    def __init__(self):
        self.df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_metadata.csv")
        self.images_dir = PROCESSED_DATA_DIR / "cleaned_images"
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    def prepare_cnn_data(self):
        """Prepare image arrays for CNN training"""
        print("Preparing CNN data...")
        
        X_images = []
        y_labels = []
        
        for _, row in self.df.iterrows():
            img_path = self.images_dir / row['Image Index']
            
            if img_path.exists():
                # Load and normalize image
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    X_images.append(img)
                    y_labels.append(row['Category'])
        
        X_cnn = np.array(X_images)
        y_cnn = np.array(y_labels)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_cnn_encoded = label_encoder.fit_transform(y_cnn)
        
        # Save label encoder
        with open(MODELS_DIR / "label_encoder.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"CNN data shape: {X_cnn.shape}")
        print(f"Labels shape: {y_cnn_encoded.shape}")
        
        return X_cnn, y_cnn_encoded, label_encoder
    
    def extract_traditional_features(self):
        """Extract features for traditional ML models"""
        print("Extracting traditional ML features...")
        
        features = []
        labels = []
        
        # Initialize HOG descriptor
        hog_descriptor = cv2.HOGDescriptor(
            (64, 64),   # winSize
            (16, 16),   # blockSize
            (8, 8),     # blockStride
            (8, 8),     # cellSize
            9           # nbins
        )
        
        for _, row in self.df.iterrows():
            img_path = self.images_dir / row['Image Index']
            
            if img_path.exists():
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize for HOG
                    img_resized = cv2.resize(img, (64, 64))
                    
                    # HOG Features using OpenCV
                    hog_features = hog_descriptor.compute(img_resized)
                    if hog_features is not None:
                        hog_features = hog_features.flatten()
                    else:
                        hog_features = np.zeros(1764)  # Default HOG size
                    
                    # Simple LBP alternative using texture analysis
                    kernel = np.ones((3,3), np.uint8)
                    morph = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
                    texture_hist, _ = np.histogram(morph.ravel(), bins=32, density=True)
                    
                    # Edge features using Canny
                    edges = cv2.Canny(img, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    
                    # Statistical features
                    stats = [
                        np.mean(img), np.std(img), np.min(img), np.max(img),
                        np.percentile(img, 25), np.percentile(img, 75),
                        edge_density,
                        np.mean(np.gradient(img.astype(float))),
                        np.sum(img > np.mean(img)) / img.size
                    ]
                    
                    # Combine all features
                    combined_features = np.concatenate([
                        hog_features[:500],  # Limit HOG features
                        texture_hist,        # Texture features
                        stats               # Statistical features
                    ])
                    
                    features.append(combined_features)
                    labels.append(row['Category'])
        
        X_traditional = np.array(features)
        y_traditional = np.array(labels)
        
        print(f"Traditional features shape: {X_traditional.shape}")
        
        return X_traditional, y_traditional
    
    def split_and_scale_data(self, X_cnn, y_cnn, X_traditional, y_traditional):
        """Split data and scale traditional features"""
        print("Splitting and scaling data...")
        
        # Split CNN data
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
            X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
        )
        
        X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
            X_train_cnn, y_train_cnn, test_size=0.2, random_state=42, stratify=y_train_cnn
        )
        
        # Split traditional data
        X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(
            X_traditional, y_traditional, test_size=0.2, random_state=42, stratify=y_traditional
        )
        
        X_train_trad, X_val_trad, y_train_trad, y_val_trad = train_test_split(
            X_train_trad, y_train_trad, test_size=0.2, random_state=42, stratify=y_train_trad
        )
        
        # Scale traditional features
        scaler = StandardScaler()
        X_train_trad_scaled = scaler.fit_transform(X_train_trad)
        X_val_trad_scaled = scaler.transform(X_val_trad)
        X_test_trad_scaled = scaler.transform(X_test_trad)
        
        # Save scaler
        with open(MODELS_DIR / "feature_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        return {
            'cnn': {
                'X_train': X_train_cnn, 'X_val': X_val_cnn, 'X_test': X_test_cnn,
                'y_train': y_train_cnn, 'y_val': y_val_cnn, 'y_test': y_test_cnn
            },
            'traditional': {
                'X_train': X_train_trad_scaled, 'X_val': X_val_trad_scaled, 'X_test': X_test_trad_scaled,
                'y_train': y_train_trad, 'y_val': y_val_trad, 'y_test': y_test_trad
            }
        }
    
    def save_processed_data(self, data_splits):
        """Save processed data for model training"""
        print("Saving processed data...")
        
        # Save CNN data
        np.save(FEATURES_DIR / "X_train_cnn.npy", data_splits['cnn']['X_train'])
        np.save(FEATURES_DIR / "X_val_cnn.npy", data_splits['cnn']['X_val'])
        np.save(FEATURES_DIR / "X_test_cnn.npy", data_splits['cnn']['X_test'])
        np.save(FEATURES_DIR / "y_train_cnn.npy", data_splits['cnn']['y_train'])
        np.save(FEATURES_DIR / "y_val_cnn.npy", data_splits['cnn']['y_val'])
        np.save(FEATURES_DIR / "y_test_cnn.npy", data_splits['cnn']['y_test'])
        
        # Save traditional ML data
        np.save(FEATURES_DIR / "X_train_traditional.npy", data_splits['traditional']['X_train'])
        np.save(FEATURES_DIR / "X_val_traditional.npy", data_splits['traditional']['X_val'])
        np.save(FEATURES_DIR / "X_test_traditional.npy", data_splits['traditional']['X_test'])
        np.save(FEATURES_DIR / "y_train_traditional.npy", data_splits['traditional']['y_train'])
        np.save(FEATURES_DIR / "y_val_traditional.npy", data_splits['traditional']['y_val'])
        np.save(FEATURES_DIR / "y_test_traditional.npy", data_splits['traditional']['y_test'])
        
        print("Data splits saved successfully!")

def main():
    print("Starting feature engineering...")
    
    fe = FeatureEngineer()
    
    # Prepare CNN data
    X_cnn, y_cnn, label_encoder = fe.prepare_cnn_data()
    
    # Extract traditional features
    X_traditional, y_traditional = fe.extract_traditional_features()
    
    # Split and scale data
    data_splits = fe.split_and_scale_data(X_cnn, y_cnn, X_traditional, y_traditional)
    
    # Save processed data
    fe.save_processed_data(data_splits)
    
    print("Feature engineering completed!")

if __name__ == "__main__":
    main()