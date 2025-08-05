"""
Medical Image Classification - Hyperparameter Tuning
Optimize model performance using GridSearchCV and Keras Tuner
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import keras_tuner as kt
import pickle
from pathlib import Path

PROJECT_ROOT = Path("D:/ninic/Final Project 2")
FEATURES_DIR = PROJECT_ROOT / "data/features"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

class HyperparameterTuner:
    def __init__(self):
        self.load_data()
        self.best_models = {}
        
    def load_data(self):
        """Load preprocessed data"""
        self.X_train_cnn = np.load(FEATURES_DIR / "X_train_cnn.npy")
        self.X_val_cnn = np.load(FEATURES_DIR / "X_val_cnn.npy")
        self.y_train_cnn = np.load(FEATURES_DIR / "y_train_cnn.npy")
        self.y_val_cnn = np.load(FEATURES_DIR / "y_val_cnn.npy")
        
        self.X_train_trad = np.load(FEATURES_DIR / "X_train_traditional.npy")
        self.X_val_trad = np.load(FEATURES_DIR / "X_val_traditional.npy")
        self.y_train_trad = np.load(FEATURES_DIR / "y_train_traditional.npy")
        self.y_val_trad = np.load(FEATURES_DIR / "y_val_traditional.npy")
        
        # Encode traditional labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.y_train_trad_enc = le.fit_transform(self.y_train_trad)
        self.y_val_trad_enc = le.transform(self.y_val_trad)
    
    def build_cnn_model(self, hp):
        """Build CNN with hyperparameter tuning"""
        model = tf.keras.Sequential()
        
        # First conv block
        model.add(tf.keras.layers.Conv2D(
            hp.Choice('conv_1_filters', [16, 32, 64]),
            (3, 3), activation='relu', input_shape=(224, 224, 1)
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        # Second conv block
        model.add(tf.keras.layers.Conv2D(
            hp.Choice('conv_2_filters', [32, 64, 128]),
            (3, 3), activation='relu'
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        # Third conv block
        model.add(tf.keras.layers.Conv2D(
            hp.Choice('conv_3_filters', [64, 128, 256]),
            (3, 3), activation='relu'
        ))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.3, 0.7)))
        
        # Dense layer
        model.add(tf.keras.layers.Dense(
            hp.Choice('dense_units', [128, 256, 512]),
            activation='relu'
        ))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def tune_cnn(self):
        """Tune CNN hyperparameters"""
        print("Tuning CNN hyperparameters...")
        
        tuner = kt.RandomSearch(
            self.build_cnn_model,
            objective='val_accuracy',
            max_trials=10,
            directory=str(MODELS_DIR / 'cnn_tuning'),
            project_name='medical_cnn'
        )
        
        # Search for best hyperparameters
        tuner.search(
            self.X_train_cnn, self.y_train_cnn,
            epochs=10,
            validation_data=(self.X_val_cnn, self.y_val_cnn),
            verbose=1
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Train best model with more epochs
        best_model.fit(
            self.X_train_cnn, self.y_train_cnn,
            epochs=20,
            validation_data=(self.X_val_cnn, self.y_val_cnn),
            verbose=1
        )
        
        self.best_models['CNN'] = best_model
        best_model.save(MODELS_DIR / "best_cnn_model.h5")
        
        # Save best hyperparameters
        hp_dict = {f"cnn_{k}": v for k, v in best_hp.values.items()}
        with open(MODELS_DIR / "best_cnn_hyperparameters.pkl", 'wb') as f:
            pickle.dump(hp_dict, f)
        
        print("CNN hyperparameter tuning completed")
        return best_model, hp_dict
    
    def tune_traditional_models(self):
        """Tune traditional ML models"""
        print("Tuning traditional ML models...")
        
        # Random Forest
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        rf_grid.fit(self.X_train_trad, self.y_train_trad_enc)
        
        # SVM
        svm_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=42, probability=True),
            svm_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        svm_grid.fit(self.X_train_trad, self.y_train_trad_enc)
        
        # Logistic Regression
        lr_params = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            lr_params, cv=3, scoring='accuracy', n_jobs=-1
        )
        lr_grid.fit(self.X_train_trad, self.y_train_trad_enc)
        
        # Store best models
        self.best_models.update({
            'Random Forest': rf_grid.best_estimator_,
            'SVM': svm_grid.best_estimator_,
            'Logistic Regression': lr_grid.best_estimator_
        })
        
        # Save models and parameters
        for name, (grid, model) in [
            ('Random Forest', (rf_grid, rf_grid.best_estimator_)),
            ('SVM', (svm_grid, svm_grid.best_estimator_)),
            ('Logistic Regression', (lr_grid, lr_grid.best_estimator_))
        ]:
            with open(MODELS_DIR / f"best_{name.lower().replace(' ', '_')}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            with open(MODELS_DIR / f"best_{name.lower().replace(' ', '_')}_params.pkl", 'wb') as f:
                pickle.dump(grid.best_params_, f)
        
        print("Traditional ML hyperparameter tuning completed")
        
        return {
            'Random Forest': rf_grid.best_params_,
            'SVM': svm_grid.best_params_,
            'Logistic Regression': lr_grid.best_params_
        }
    
    def evaluate_best_models(self):
        """Evaluate tuned models"""
        results = {}
        
        # Evaluate CNN
        if 'CNN' in self.best_models:
            val_loss, val_acc = self.best_models['CNN'].evaluate(
                self.X_val_cnn, self.y_val_cnn, verbose=0
            )
            results['CNN'] = val_acc
        
        # Evaluate traditional models
        for name, model in self.best_models.items():
            if name != 'CNN':
                val_pred = model.predict(self.X_val_trad)
                val_acc = (val_pred == self.y_val_trad_enc).mean()
                results[name] = val_acc
        
        # Save results
        with open(OUTPUTS_DIR / "hyperparameter_tuning_results.txt", 'w') as f:
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("="*50 + "\n\n")
            for model, accuracy in results.items():
                f.write(f"{model}: {accuracy:.4f}\n")
        
        print("Best model evaluation completed")
        return results

def main():
    print("Starting hyperparameter tuning...")
    
    tuner = HyperparameterTuner()
    
    # Tune CNN
    best_cnn, cnn_params = tuner.tune_cnn()
    
    # Tune traditional models
    traditional_params = tuner.tune_traditional_models()
    
    # Evaluate all tuned models
    results = tuner.evaluate_best_models()
    
    print("Hyperparameter tuning completed!")

if __name__ == "__main__":
    main()