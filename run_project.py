"""
Medical Image Classification - Main Project Runner
Execute complete pipeline from data download to model deployment
"""

import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path("D:/ninic/Final Project 2")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                               capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Show last 500 chars
        else:
            print(f"❌ Error in {description}")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run {script_name}: {e}")
        return False
    
    return True

def setup_environment():
    """Setup project environment"""
    print("🔧 Setting up project environment...")
    
    # Create virtual environment if it doesn't exist
    venv_path = PROJECT_ROOT / "venv"
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)])
    
    print("✅ Environment setup complete!")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      cwd=PROJECT_ROOT, check=True)
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    return True

def main():
    """Run complete project pipeline"""
    start_time = time.time()
    
    print("🏥 MEDICAL IMAGE CLASSIFICATION PROJECT")
    print("🚀 Starting complete pipeline execution...")
    
    # Pipeline steps
    steps = [
        ("data_download.py", "Data Download"),
        ("data_cleaning.py", "Data Cleaning & Preprocessing"),
        ("eda.py", "Exploratory Data Analysis"),
        ("feature_engineering.py", "Feature Engineering"),
        ("model_training.py", "Model Training"),
        ("hyperparameter_tuning.py", "Hyperparameter Optimization")
    ]
    
    # Setup environment
    setup_environment()
    
    # Install requirements
    if not install_requirements():
        print("❌ Pipeline failed during setup")
        return
    
    # Execute pipeline steps
    success_count = 0
    for script, description in steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"❌ Pipeline failed at: {description}")
            print("🛑 Stopping execution")
            break
        
        time.sleep(2)  # Brief pause between steps
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed Steps: {success_count}/{len(steps)}")
    print(f"⏱️  Total Time: {time.time() - start_time:.1f} seconds")
    
    if success_count == len(steps):
        print("🎉 Pipeline completed successfully!")
        print("\n📊 Generated Outputs:")
        print("  • Data cleaning report")
        print("  • EDA visualizations")
        print("  • Model training results")
        print("  • Hyperparameter tuning results")
        print("  • Trained models (CNN, RF, SVM, LR, NN)")
        
        print("\n🌐 To launch web application:")
        print("  streamlit run app.py")
        
        print("\n📁 Project files saved to:")
        print(f"  {PROJECT_ROOT}")
        
        print("\n📚 Ready for scientific article!")
    else:
        print("❌ Pipeline incomplete - check errors above")

if __name__ == "__main__":
    main()