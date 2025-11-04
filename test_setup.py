#!/usr/bin/env python3

"""
Test script to verify the UltimateALPR-SDK setup
"""

import os
import subprocess
import sys

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Docker is not installed or not in PATH")
            return False
    except FileNotFoundError:
        print("✗ Docker is not installed")
        return False

def check_docker_image():
    """Check if the ALPR Docker image exists"""
    try:
        result = subprocess.run(['docker', 'images', 'alpr-arm64', '--format', 'table {{.Repository}}'], 
                              capture_output=True, text=True)
        if 'alpr-arm64' in result.stdout:
            print("✓ ALPR Docker image is built")
            return True
        else:
            print("✗ ALPR Docker image is not built")
            return False
    except Exception as e:
        print(f"✗ Error checking Docker image: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'assets/images/lic_us_1280x720.jpg',
        'assets/models',
        'binaries/linux/aarch64/recognizer'
    ]
    
    all_good = True
    for file_path in required_files:
        full_path = os.path.join('/Users/vedprakashchaubey/Desktop/ultimateALPR-sdk', file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} is missing")
            all_good = False
    
    return all_good

def run_test():
    """Run a simple test of the ALPR functionality"""
    try:
        print("\nRunning ALPR test...")
        result = subprocess.run([
            'docker', 'run', '--rm', 'alpr-arm64',
            '/app/binaries/recognizer', 
            '--image', '/app/test_image.jpg',
            '--assets', '/app/assets'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and 'text' in result.stdout:
            print("✓ ALPR test successful")
            # Extract the detected plate text
            import json
            # Find the JSON result in the output
            lines = result.stdout.split('\n')
            for line in lines:
                if '"text"' in line and '"plates"' in line:
                    print(f"  Detected plate: {line}")
                    break
            return True
        else:
            print("✗ ALPR test failed")
            print(f"  Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ ALPR test timed out")
        return False
    except Exception as e:
        print(f"✗ Error running ALPR test: {e}")
        return False

def main():
    print("UltimateALPR-SDK Setup Verification")
    print("=" * 40)
    
    # Run checks
    docker_ok = check_docker()
    files_ok = check_files()
    image_ok = check_docker_image()
    
    if docker_ok and files_ok and image_ok:
        print("\n✓ All basic checks passed!")
        test_ok = run_test()
        if test_ok:
            print("\n🎉 Setup verification completed successfully!")
            print("\nYou can now use the UltimateALPR-SDK:")
            print("  - Run 'python3 demo.py' for usage instructions")
            print("  - Check 'README-SETUP.md' for detailed documentation")
        else:
            print("\n⚠ Setup is incomplete - test failed")
    else:
        print("\n✗ Setup verification failed - some checks did not pass")
        print("Please check the errors above and fix them before using the SDK")

if __name__ == "__main__":
    main()