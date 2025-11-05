#!/usr/bin/env python3
"""Download YOLO OCR model for license plate character detection."""

import os
import urllib.request
from pathlib import Path

def download_yolo_model():
    """Download the YOLO OCR model."""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "plate_ocr_yolo.pt"
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return
    
    print("📥 Downloading YOLO OCR model (3.2MB)...")
    
    url = "https://huggingface.co/vedprakash/indian-plate-ocr-yolo/resolve/main/plate_ocr_yolo.pt"
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print(f"✅ Downloaded: {model_path}")
        print("🚀 Ready to use YOLO OCR!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Manual download:")
        print(f"wget -O {model_path} {url}")

if __name__ == "__main__":
    download_yolo_model()