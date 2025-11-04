#!/usr/bin/env python3
"""
Test script for the Python Docker wrapper for ultimateALPR-SDK.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python_docker_wrapper import UltimateALPRSDK

def test_alpr_processing():
    """Test the ALPR processing with the sample image."""
    # Initialize SDK
    sdk = UltimateALPRSDK()
    
    # Use the test image that we know is in the Docker container
    # For this test, we'll copy the sample image from the assets to a temporary location
    sample_image = "assets/images/lic_us_1280x720.jpg"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        print("Please make sure you're running this from the root of the ultimateALPR-sdk directory")
        return False
    
    try:
        print("Processing sample image...")
        result = sdk.process_image(sample_image)
        
        print("ALPR Results:")
        print(f"Raw result: {result}")
        
        plates = sdk.get_plate_details(result)
        if plates:
            for i, plate in enumerate(plates):
                print(f"Plate {i+1}: {plate['text']} (Confidence: {plate['confidence']:.2f})")
        else:
            print("No plates detected")
            
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

if __name__ == "__main__":
    print("Testing Python Docker wrapper for ultimateALPR-SDK...")
    success = test_alpr_processing()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1)