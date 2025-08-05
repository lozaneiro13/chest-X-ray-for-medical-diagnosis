"""
Medical Image Classification Project - Data Download
Download and organize chest X-ray dataset for medical diagnosis
"""

import os
import pandas as pd
import numpy as np
import cv2
import kagglehub
import shutil
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path("D:/ninic/Final Project 2")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def setup_project_structure():
    """Create organized project directory structure"""
    directories = [
        PROJECT_ROOT,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "app"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def download_chest_xray_dataset():
    """Download only images_001 folder from NIH Chest X-ray dataset"""
    print("Downloading images_001 folder from NIH Chest X-ray dataset...")
    
    try:
        # Load CSV metadata first
        print("Loading dataset metadata...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "nih-chest-xrays/data",
            "Data_Entry_2017.csv"
        )
        
        print(f"Metadata loaded: {len(df)} records")
        
        # Save metadata
        metadata_path = RAW_DATA_DIR / "Data_Entry_2017.csv"
        df.to_csv(metadata_path, index=False)
        
        # Create images directory
        images_dir = RAW_DATA_DIR / "images_001"
        images_dir.mkdir(exist_ok=True)
        
        # Download specific images from images_001 folder
        # Get first 1000 image names that exist in images_001
        images_001_files = [img for img in df['Image Index'].head(1000) 
                           if img.startswith(('00000', '00001', '00002', '00003', '00004'))]
        
        print(f"Attempting to download {len(images_001_files)} images from images_001...")
        
        downloaded_count = 0
        for i, image_name in enumerate(images_001_files[:500]):  # Limit to 500 images
            try:
                # Try to download individual image file
                file_data = kagglehub.load_dataset(
                    None,  # Raw data
                    "nih-chest-xrays/data", 
                    f"images_001/{image_name}"
                )
                
                # Save image
                image_path = images_dir / image_name
                with open(image_path, 'wb') as f:
                    f.write(file_data)
                
                downloaded_count += 1
                if downloaded_count % 50 == 0:
                    print(f"Downloaded {downloaded_count} images...")
                    
            except Exception as e:
                if i == 0:  # If first image fails, likely API issue
                    print(f"Failed to download images: {e}")
                    break
                continue
        
        if downloaded_count > 0:
            print(f"Successfully downloaded {downloaded_count} images to {images_dir}")
            return df, str(images_dir)
        else:
            raise Exception("No images downloaded")
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Ensure Kaggle API is configured and dataset terms accepted")
        return None, None

def organize_data(df, images_path):
    """Organize downloaded data and create initial analysis"""
    if df is None:
        return
    
    # Create data summary
    summary = {
        'total_images': len(df),
        'unique_patients': df['Patient ID'].nunique(),
        'finding_labels': df['Finding Labels'].value_counts().to_dict(),
        'image_info': {
            'patient_age_range': (df['Patient Age'].min(), df['Patient Age'].max()),
            'gender_distribution': df['Patient Gender'].value_counts().to_dict(),
            'view_position': df['View Position'].value_counts().to_dict()
        }
    }
    
    # Save summary
    summary_path = DATA_DIR / "dataset_summary.txt"
    with open(summary_path, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print("Dataset Summary:")
    print(f"Total images: {summary['total_images']}")
    print(f"Unique patients: {summary['unique_patients']}")
    print(f"Gender distribution: {summary['image_info']['gender_distribution']}")
    
    return summary

if __name__ == "__main__":
    print("Setting up Medical Image Classification Project...")
    
    # Setup project structure
    setup_project_structure()
    
    # Download data
    df, images_path = download_chest_xray_dataset()
    
    # Organize data
    if df is not None:
        summary = organize_data(df, images_path)
        print("\nProject setup completed successfully!")
        print(f"Project root: {PROJECT_ROOT}")
    else:
        print("Failed to download dataset. Please check your Kaggle credentials.")