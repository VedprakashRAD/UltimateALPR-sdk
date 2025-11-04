#!/usr/bin/env python3
"""
Demo script showing how to use the Python wrapper for ultimateALPR-SDK.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python_docker_wrapper import UltimateALPRSDK

def main():
    """Main demo function."""
    print("=== ultimateALPR-SDK Python Demo ===")
    print()
    
    # Initialize SDK
    print("Initializing SDK...")
    sdk = UltimateALPRSDK()
    print("SDK initialized successfully!")
    print()
    
    # Process a sample image
    sample_image = "assets/images/lic_us_1280x720.jpg"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        print("Please make sure you're running this from the root of the ultimateALPR-sdk directory")
        return
    
    print(f"Processing image: {sample_image}")
    print()
    
    try:
        # Process the image
        result = sdk.process_image(sample_image)
        
        # Extract plate details
        plates = sdk.get_plate_details(result)
        
        if plates:
            print(f"Found {len(plates)} license plate(s):")
            print("-" * 50)
            for i, plate in enumerate(plates):
                print(f"Plate {i+1}:")
                print(f"  Text: {plate['text']}")
                print(f"  Confidence: {plate['confidence']:.2f}%")
                print(f"  Box coordinates: {plate['box']}")
                print()
        else:
            print("No license plates detected.")
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    print("=== Demo completed successfully! ===")

if __name__ == "__main__":
    main()