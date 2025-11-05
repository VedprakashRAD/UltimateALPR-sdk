#!/usr/bin/env python3
"""
Complete test suite for the Enhanced ALPR System
"""

import subprocess
import sys
import time

def run_test(test_name, command):
    """Run a test and report results."""
    print(f"\n🧪 {test_name}")
    print("=" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout:
                print(result.stdout[-500:])  # Last 500 chars
        else:
            print("❌ FAILED")
            if result.stderr:
                print(result.stderr[-500:])
    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def main():
    """Run complete test suite."""
    print("🚀 Enhanced ALPR System - Complete Test Suite")
    print("=" * 60)
    
    tests = [
        ("System Initialization Test", "python test_alpr_system.py"),
        ("Indian Plate Validation Test", "python test_indian_plates.py"),
        ("Create Test Images", "python create_test_images.py"),
        ("Test Generated Images", """python -c "
import cv2, sys, os
sys.path.append('src/camera')
from working_alpr_system import WorkingALPRSystem
alpr = WorkingALPRSystem()
correct = 0
total = 0
for img_file in sorted(os.listdir('test_images')):
    if img_file.endswith('.jpg'):
        total += 1
        img_path = f'test_images/{img_file}'
        frame = cv2.imread(img_path)
        results = alpr.process_frame(frame)
        expected = img_file.replace('.jpg', '')
        if results and results[0]['text'] == expected:
            correct += 1
            print(f'✅ {expected} → {results[0][\"text\"]} ({results[0][\"confidence\"]:.1f}%)')
        else:
            detected = results[0]['text'] if results else 'No detection'
            print(f'❌ {expected} → {detected}')
print(f'\\n🎯 Accuracy: {correct}/{total} ({correct/total*100:.1f}%)')
"
""")
    ]
    
    for test_name, command in tests:
        run_test(test_name, command)
    
    print("\n" + "=" * 60)
    print("🎉 Complete Test Suite Finished!")
    print("=" * 60)
    
    print("\n🚀 To start the full system:")
    print("   python main.py")
    print("   Then open: http://localhost:8080")

if __name__ == "__main__":
    main()