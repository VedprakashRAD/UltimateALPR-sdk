#!/usr/bin/env python3

"""
Demo script for ultimateALPR-SDK
This script demonstrates how to use the ultimateALPR-SDK with Python
"""

import json
import os

def main():
    print("UltimateALPR-SDK Demo")
    print("=" * 50)
    
    # Define paths
    sdk_path = "/Users/vedprakashchaubey/Desktop/ultimateALPR-sdk"
    image_path = os.path.join(sdk_path, "assets", "images", "lic_us_1280x720.jpg")
    assets_path = os.path.join(sdk_path, "assets")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    if not os.path.exists(assets_path):
        print(f"Error: Assets folder not found at {assets_path}")
        return
    
    # Show how to run the recognizer using Docker
    print("\nTo run license plate recognition on the sample image:")
    print(f"docker run --rm alpr-arm64 /app/binaries/recognizer --image /app/test_image.jpg --assets /app/assets")
    
    # Show how to run with your own image
    print(f"\nTo run with your own image:")
    print(f"docker run --rm -v $(pwd):/workspace alpr-arm64 /app/binaries/recognizer --image /workspace/your_image.jpg --assets /app/assets")
    
    # Explain the results
    print("\nSample output interpretation:")
    print("- The SDK detected a license plate with text '3PEDLM*'")
    print("- Confidence score: 89.93%")
    print("- Processing time: 180ms")
    
    print("\nFor more information, check the documentation at:")
    print("https://www.doubango.org/SDKs/anpr/docs/")

if __name__ == "__main__":
    main()