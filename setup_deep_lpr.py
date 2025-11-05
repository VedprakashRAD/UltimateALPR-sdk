#!/usr/bin/env python3
"""
Setup script for Deep License Plate Recognition models
Downloads and sets up alpr-unconstrained repository and models
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            return True
        else:
            print(f"❌ {cmd}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {cmd}: {e}")
        return False

def download_file(url, path):
    """Download file from URL."""
    try:
        print(f"📥 Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ Downloaded to {path}")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def setup_deep_lpr():
    """Setup Deep LPR models and repository."""
    print("🚀 Setting up Deep License Plate Recognition...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Method 1: Clone alpr-unconstrained repository
    if not os.path.exists("alpr-unconstrained"):
        print("📦 Cloning alpr-unconstrained repository...")
        if run_command("git clone https://github.com/sergiomsilva/alpr-unconstrained.git"):
            print("✅ Repository cloned successfully")
        else:
            print("❌ Failed to clone repository")
            return False
    
    # Method 2: Download pre-trained models directly
    model_urls = {
        "wpod-net_update1.h5": "https://github.com/sergiomsilva/alpr-unconstrained/releases/download/v1.0.0/wpod-net_update1.h5",
        "ocr-net.h5": "https://github.com/sergiomsilva/alpr-unconstrained/releases/download/v1.0.0/ocr-net.h5"
    }
    
    for model_name, url in model_urls.items():
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            if download_file(url, model_path):
                print(f"✅ {model_name} ready")
            else:
                print(f"❌ Failed to download {model_name}")
    
    # Copy models from repository if available
    repo_models = [
        ("alpr-unconstrained/data/lp-detector/wpod-net_update1.h5", "models/wpod-net_update1.h5"),
        ("alpr-unconstrained/data/ocr/ocr-net.h5", "models/ocr-net.h5")
    ]
    
    for src, dst in repo_models:
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                import shutil
                shutil.copy2(src, dst)
                print(f"✅ Copied {src} to {dst}")
            except Exception as e:
                print(f"❌ Failed to copy {src}: {e}")
    
    # Install additional dependencies if needed
    print("📦 Installing additional dependencies...")
    dependencies = [
        "tensorflow>=2.8.0",
        "keras>=2.8.0",
        "h5py>=3.6.0"
    ]
    
    for dep in dependencies:
        if run_command(f"pip install {dep}"):
            print(f"✅ Installed {dep}")
    
    print("\n🎉 Deep LPR setup complete!")
    print("📁 Models available in ./models/ directory")
    print("📁 Repository available in ./alpr-unconstrained/ directory")
    
    return True

if __name__ == "__main__":
    try:
        setup_deep_lpr()
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted")
    except Exception as e:
        print(f"❌ Setup failed: {e}")