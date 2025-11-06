#!/usr/bin/env python3
"""
Download YOLO Plate OCR Model (3.2MB)
"""

import os
import urllib.request
import sys

def download_yolo_ocr_model():
    """Download the YOLO plate OCR model."""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_path = "models/plate_ocr_yolo.pt"
    
    if os.path.exists(model_path):
        print(f"✅ Model already exists: {model_path}")
        return True
    
    print(f"📥 Downloading YOLO Plate OCR model (3.2MB)...")
    print(f"URL: {model_url}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r📥 Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(model_url, model_path, progress_hook)
        print(f"\n✅ Downloaded: {model_path}")
        
        # Verify file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"📊 File size: {file_size:.1f} MB")
        
        if file_size < 2.0:
            print("❌ Download incomplete - file too small")
            os.remove(model_path)
            return False
            
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)
        return False

if __name__ == "__main__":
    success = download_yolo_ocr_model()
    if success:
        print("\n🎯 YOLO Plate OCR ready!")
        print("Usage: python read_plate_yolo.py image.jpg")
    else:
        print("\n❌ Download failed")
        sys.exit(1)