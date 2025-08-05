"""
Medical Image Classification - Data Cleaning
Clean and preprocess chest X-ray images for model training
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import shutil

# Project paths
PROJECT_ROOT = Path("D:/ninic/Final Project 2")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

class XrayDataCleaner:
    def __init__(self):
        self.metadata_path = RAW_DATA_DIR / "Data_Entry_2017.csv"
        self.images_dir = None
        self.cleaned_images_dir = PROCESSED_DATA_DIR / "cleaned_images"
        self.cleaning_log = []
        
    def load_metadata(self):
        """Load metadata and find images directory"""
        self.df = pd.read_csv(self.metadata_path)
        
        # Find images directory
        for root, dirs, files in os.walk(RAW_DATA_DIR):
            if 'images_001' in dirs:
                self.images_dir = Path(root) / 'images_001'
                break
        
        if not self.images_dir or not self.images_dir.exists():
            raise FileNotFoundError("images_001 directory not found")
            
        print(f"Found {len(self.df)} metadata records")
        print(f"Images directory: {self.images_dir}")
        
    def check_image_integrity(self):
        """Check for corrupted or unreadable images"""
        corrupted_images = []
        valid_images = []
        
        # Sample subset for cleaning (first 6000 images for manageability)
        sample_df = self.df.head(6000).copy()
        
        for idx, row in sample_df.iterrows():
            image_path = self.images_dir / row['Image Index']
            
            try:
                # Try to open with PIL
                with Image.open(image_path) as img:
                    img.verify()
                
                # Try to load with OpenCV
                cv_img = cv2.imread(str(image_path))
                if cv_img is None:
                    raise Exception("OpenCV cannot read image")
                    
                valid_images.append(row['Image Index'])
                
            except Exception as e:
                corrupted_images.append({
                    'image': row['Image Index'],
                    'error': str(e)
                })
                
        self.cleaning_log.append(f"Corrupted images found: {len(corrupted_images)}")
        self.cleaning_log.append(f"Valid images: {len(valid_images)}")
        
        # Update dataframe to only include valid images
        self.df_clean = sample_df[sample_df['Image Index'].isin(valid_images)].copy()
        
        return corrupted_images
    
    def standardize_image_properties(self):
        """Analyze and standardize image dimensions and properties"""
        image_properties = []
        
        print("Analyzing image properties...")
        for idx, image_name in enumerate(self.df_clean['Image Index'].head(100)):
            image_path = self.images_dir / image_name
            
            try:
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w, c = img.shape
                    image_properties.append({
                        'image': image_name,
                        'height': h,
                        'width': w,
                        'channels': c,
                        'mean_intensity': np.mean(img),
                        'std_intensity': np.std(img)
                    })
            except:
                continue
                
        props_df = pd.DataFrame(image_properties)
        
        # Analyze dimensions
        target_size = (224, 224)  # Standard CNN input size
        
        self.cleaning_log.append(f"Original image sizes range: "
                                f"Height: {props_df['height'].min()}-{props_df['height'].max()}, "
                                f"Width: {props_df['width'].min()}-{props_df['width'].max()}")
        self.cleaning_log.append(f"Target size set to: {target_size}")
        
        return props_df, target_size
    
    def clean_and_preprocess_images(self, target_size=(224, 224)):
        """Clean and preprocess images"""
        self.cleaned_images_dir.mkdir(parents=True, exist_ok=True)
        
        processed_count = 0
        failed_count = 0
        
        # Process subset of images (first 5000 for project requirements)
        subset_df = self.df_clean.head(5000).copy()
        
        for idx, row in subset_df.iterrows():
            image_path = self.images_dir / row['Image Index']
            output_path = self.cleaned_images_dir / row['Image Index']
            
            try:
                # Load image
                img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    failed_count += 1
                    continue
                
                # Resize to target size
                img_resized = cv2.resize(img, target_size)
                
                # Normalize pixel values to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Apply CLAHE for contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_enhanced = clahe.apply((img_normalized * 255).astype(np.uint8))
                
                # Save processed image
                cv2.imwrite(str(output_path), img_enhanced)
                processed_count += 1
                
                if processed_count % 500 == 0:
                    print(f"Processed {processed_count} images...")
                    
            except Exception as e:
                failed_count += 1
                continue
        
        self.cleaning_log.append(f"Successfully processed: {processed_count} images")
        self.cleaning_log.append(f"Failed to process: {failed_count} images")
        
        # Update dataframe with only successfully processed images
        processed_images = [f.name for f in self.cleaned_images_dir.glob("*.png")]
        self.df_final = subset_df[subset_df['Image Index'].isin(processed_images)].copy()
        
        return processed_count
    
    def handle_class_imbalance(self):
        """Analyze and document class imbalance"""
        # Simplify labels for binary classification: Normal vs Abnormal
        def categorize_finding(finding):
            if finding == "No Finding":
                return "Normal"
            else:
                return "Abnormal"
        
        self.df_final['Category'] = self.df_final['Finding Labels'].apply(categorize_finding)
        
        class_distribution = self.df_final['Category'].value_counts()
        
        self.cleaning_log.append(f"Class distribution: {class_distribution.to_dict()}")
        
        # Save balanced subset for training
        min_class_size = min(class_distribution.values)
        balanced_df = self.df_final.groupby('Category').sample(
            n=min(min_class_size, 2000), 
            random_state=42
        )
        
        self.cleaning_log.append(f"Balanced dataset size: {len(balanced_df)}")
        
        return balanced_df
    
    def save_cleaned_data(self, balanced_df):
        """Save cleaned metadata and generate cleaning report"""
        # Save cleaned metadata
        balanced_df.to_csv(PROCESSED_DATA_DIR / "cleaned_metadata.csv", index=False)
        
        # Save cleaning log
        log_path = OUTPUTS_DIR / "data_cleaning_report.txt"
        OUTPUTS_DIR.mkdir(exist_ok=True)
        
        with open(log_path, 'w') as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("="*50 + "\n\n")
            for log_entry in self.cleaning_log:
                f.write(f"• {log_entry}\n")
        
        print(f"Cleaning report saved to: {log_path}")
        print(f"Final dataset size: {len(balanced_df)} images")

def main():
    cleaner = XrayDataCleaner()
    
    print("Starting data cleaning process...")
    
    # Load metadata
    cleaner.load_metadata()
    
    # Check image integrity
    corrupted = cleaner.check_image_integrity()
    
    # Analyze image properties
    props_df, target_size = cleaner.standardize_image_properties()
    
    # Clean and preprocess images
    processed_count = cleaner.clean_and_preprocess_images(target_size)
    
    # Handle class imbalance
    balanced_df = cleaner.handle_class_imbalance()
    
    # Save results
    cleaner.save_cleaned_data(balanced_df)
    
    print("Data cleaning completed successfully!")

if __name__ == "__main__":
    main()