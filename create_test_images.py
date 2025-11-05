#!/usr/bin/env python3
"""
Create test license plate images for testing the ALPR system
"""

import cv2
import numpy as np
import os

def create_indian_plate_image(plate_text, filename):
    """Create a synthetic Indian license plate image."""
    # Create white background (Indian plate dimensions ~520x110mm, ratio ~4.7:1)
    width, height = 400, 85
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add border
    cv2.rectangle(img, (5, 5), (width-5, height-5), (0, 0, 0), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # Calculate text size and position
    text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Add the license plate text
    cv2.putText(img, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    # Save image
    cv2.imwrite(filename, img)
    print(f"✅ Created: {filename}")

def main():
    """Create test images for different Indian license plate formats."""
    os.makedirs("test_images", exist_ok=True)
    
    # Indian license plates to test
    test_plates = [
        "MH01AB1234",  # Maharashtra
        "DL02CD5678",  # Delhi
        "KA03EF9012",  # Karnataka
        "TN09GH3456",  # Tamil Nadu
        "23BH1234AB",  # Bharat Series 2023
        "22BH5678CD",  # Bharat Series 2022
        "UP16JK7890",  # Uttar Pradesh
        "GJ01MN2345"   # Gujarat
    ]
    
    print("🖼️  Creating test license plate images...")
    
    for plate in test_plates:
        filename = f"test_images/{plate}.jpg"
        create_indian_plate_image(plate, filename)
    
    print(f"\n✅ Created {len(test_plates)} test images in 'test_images/' directory")
    print("📸 Use these images to test the ALPR system accuracy")

if __name__ == "__main__":
    main()