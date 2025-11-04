#!/usr/bin/env python3
"""
Python wrapper for ultimateALPR-SDK that uses Docker container for processing.
This avoids the need to build native extensions and handles cross-platform compatibility.
"""

import subprocess
import json
import tempfile
import os
import re
from PIL import Image

class UltimateALPRSDK:
    def __init__(self, docker_image="alpr-arm64"):
        """
        Initialize the SDK with the Docker image name.
        
        Args:
            docker_image (str): Name of the Docker image to use
        """
        self.docker_image = docker_image
        
    def process_image(self, image_path, assets_path="/app/assets"):
        """
        Process an image to detect and recognize license plates.
        
        Args:
            image_path (str): Path to the image file to process
            assets_path (str): Path to assets directory in container
            
        Returns:
            dict: Results from the ALPR processing
        """
        # Check if Docker image exists
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True, text=True, check=True
            )
            if not result.stdout.strip():
                raise Exception(f"Docker image {self.docker_image} not found")
        except subprocess.CalledProcessError:
            raise Exception("Docker is not available or not running")
            
        # Get absolute path of the image
        image_path = os.path.abspath(image_path)
        
        try:
            # Run the recognizer in Docker container
            # We'll mount the directory containing the image and the assets directory
            image_dir = os.path.dirname(image_path)
            image_filename = os.path.basename(image_path)
            
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{image_dir}:/tmp/images",
                "-v", f"{os.path.abspath('assets')}:/app/assets",
                self.docker_image,
                "/app/binaries/recognizer",
                "--image", f"/tmp/images/{image_filename}",
                "--assets", assets_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                raise Exception(f"ALPR processing failed: {result.stderr}")
                
            # Parse JSON output from the result
            # The output goes to stderr, not stdout
            output = result.stderr.strip()
            
            # Use regex to find the JSON result
            json_match = re.search(r'\*{0,1}\[ULTALPR_SDK INFO\]: result: ({.*})', output)
            if json_match:
                try:
                    json_result = json.loads(json_match.group(1))
                    return json_result
                except json.JSONDecodeError:
                    pass
            
            # If we can't parse JSON, return the full output
            return {"text": output, "plates": []}
                
        except subprocess.TimeoutExpired:
            raise Exception("ALPR processing timed out")
                
    def get_plate_details(self, result):
        """
        Extract plate details from the result.
        
        Args:
            result (dict): Result from process_image
            
        Returns:
            list: List of detected plates with details
        """
        plates = []
        if "plates" in result:
            for plate in result["plates"]:
                # Get the maximum confidence from the confidences array
                confidence = 0
                if "confidences" in plate and plate["confidences"]:
                    confidence = max(plate["confidences"])
                
                plates.append({
                    "text": plate.get("text", ""),
                    "confidence": confidence,
                    "box": plate.get("warpedBox", [])
                })
        return plates

# Example usage
if __name__ == "__main__":
    # Initialize SDK
    sdk = UltimateALPRSDK()
    
    # Process a sample image
    # Note: You'll need to provide a valid image path
    # sample_image = "path/to/your/image.jpg"
    # result = sdk.process_image(sample_image)
    # 
    # # Print results
    # print("ALPR Results:")
    # plates = sdk.get_plate_details(result)
    # for i, plate in enumerate(plates):
    #     print(f"Plate {i+1}: {plate['text']} (Confidence: {plate['confidence']:.2f})")
    pass