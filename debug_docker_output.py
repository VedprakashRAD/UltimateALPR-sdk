#!/usr/bin/env python3
"""
Debug script to see what Docker output we're getting.
"""

import subprocess
import os
import re
import json

def debug_docker_output():
    """Debug the Docker output to see what we're getting."""
    # Get absolute path of the image
    sample_image = "assets/images/lic_us_1280x720.jpg"
    image_path = os.path.abspath(sample_image)
    
    # Run the recognizer in Docker container
    image_dir = os.path.dirname(image_path)
    image_filename = os.path.basename(image_path)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{image_dir}:/tmp/images",
        "-v", f"{os.path.abspath('assets')}:/app/assets",
        "alpr-arm64",
        "/app/binaries/recognizer",
        "--image", f"/tmp/images/{image_filename}",
        "--assets", "/app/assets"
    ]
    
    print("Running command:", " ".join(cmd))
    
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=60
    )
    
    print(f"Return code: {result.returncode}")
    print(f"Stdout length: {len(result.stdout)}")
    print(f"Stderr length: {len(result.stderr)}")
    
    if result.stdout:
        print("First 1000 characters of stdout:")
        print(result.stdout[:1000])
        print("\n" + "="*50 + "\n")
        
        # Try to find the JSON result
        output = result.stdout.strip()
        json_match = re.search(r'\*{0,1}\[ULTALPR_SDK INFO\]: result: ({.*})', output)
        if json_match:
            print("Found JSON match:")
            print(json_match.group(1))
            try:
                json_result = json.loads(json_match.group(1))
                print("Parsed JSON result:")
                print(json_result)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
        else:
            print("No JSON match found")
    
    if result.stderr:
        print("Stderr:")
        print(result.stderr)

if __name__ == "__main__":
    debug_docker_output()