"""
Medical Image Classification - Model Training
Train CNN and traditional ML models
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from pathlib import Path

# Project paths
PROJECT_ROOT = Path("D:/ninic/Final Project 2")
FEATURES_DIR = PROJECT_ROOT / "data/features"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

class ModelTrainer:
    def __init__(self):
        self.load_data()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        # CNN data
        self.X_train_cnn = np.load(FEATURES_DIR / "X_train_cnn.npy")
        self.X_val_cnn = np.load(FEATURES_DIR / "X_val_cnn.npy")
        self.X_test_cnn = np.load(FEATURES_DIR / "X_test_cnn.npy")
        self.y_train_cnn = np.load(FEATURES_DIR / "y_train_cnn.npy")
        self.y_val_cnn = np.load(FEATURES_DIR / "y_val_cnn.npy")
        self.y_test_cnn = np.load(FEATURES_DIR / "y_test_cnn.npy")
        
        # Traditional ML data
        self.X_train_trad = np.load(FEATURES_DIR / "X_train_traditional.npy")
        self.X_val_trad = np.load(FEATURES_DIR / "X_val_traditional.npy")
        self.X_test_trad = np.load(FEATURES_DIR / "X_test_traditional.npy")
        self.y_train_trad = np.load(FEATURES_DIR / "y_train_traditional.npy")
        self.y_val_trad = np.load(FEATURES_DIR / "y_val_traditional.npy")
        self.y_test_trad = np.load(FEATURES_DIR / "y_test_traditional.npy")
        
        # Convert string labels to binary for traditional models
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.y_train_trad_enc = le.fit_transform(self.y_train_trad)
        self.y_val_trad_enc = le.transform(self.y_val_trad)
        self.y_test_trad_enc = le.transform(self.y_test_trad)
        
        print(f"CNN training data: {self.X_train_cnn.shape}")
        print(f"Traditional ML training data: {self.X_train_trad.shape}")
    
    def build_cnn_model(self):
        """Build CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn(self):
        """Train CNN model"""
        print("Training CNN...")
        
        model = self.build_cnn_model()
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001
        )
        
        # Train model
        history = model.fit(
            self.X_train_cnn, self.y_train_cnn,
            batch_size=32,
            epochs=20,
            validation_data=(self.X_val_cnn, self.y_val_cnn),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(self.X_val_cnn, self.y_val_cnn, verbose=0)
        test_loss, test_acc = model.evaluate(self.X_test_cnn, self.y_test_cnn, verbose=0)
        
        self.models['CNN'] = model
        self.results['CNN'] = {
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'history': history.history
        }
        
        # Save model
        model.save(MODELS_DIR / "cnn_model.h5")
        
        print(f"CNN - Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        
        return model, history
    
    def train_traditional_models(self):
        """Train traditional ML models"""
        print("Training traditional ML models...")
        
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train_trad, self.y_train_trad_enc)
            
            # Evaluate
            val_pred = model.predict(self.X_val_trad)
            test_pred = model.predict(self.X_test_trad)
            
            val_acc = accuracy_score(self.y_val_trad_enc, val_pred)
            test_acc = accuracy_score(self.y_test_trad_enc, test_pred)
            
            self.models[name] = model
            self.results[name] = {
                'val_accuracy': val_acc,
                'test_accuracy': test_acc
            }
            
            # Save model
            with open(MODELS_DIR / f"{name.lower().replace(' ', '_')}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            print(f"{name} - Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
    def save_results(self):
        """Save training results"""
        results_summary = {}
        
        for model_name, result in self.results.items():
            results_summary[model_name] = {
                'validation_accuracy': result['val_accuracy'],
                'test_accuracy': result['test_accuracy']
            }
        
        # Save results to file
        results_path = OUTPUTS_DIR / "model_results.txt"
        with open(results_path, 'w') as f:
            f.write("MODEL TRAINING RESULTS\n")
            f.write("="*50 + "\n\n")
            
            for model_name, metrics in results_summary.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Validation Accuracy: {metrics['validation_accuracy']:.4f}\n")
                f.write(f"  Test Accuracy: {metrics['test_accuracy']:.4f}\n\n")
        
        print(f"Results saved to: {results_path}")
        
        return results_summary

def main():
    print("Starting model training...")
    
    trainer = ModelTrainer()
    
    # Train CNN
    cnn_model, history = trainer.train_cnn()
    
    # Train traditional models
    trainer.train_traditional_models()
    
    # Save results
    results = trainer.save_results()
    
    print("Model training completed!")

if __name__ == "__main__":
    main()