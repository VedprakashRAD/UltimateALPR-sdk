#!/usr/bin/env python3
"""
OCR Comparison Tool - YOLO OCR vs PaddleOCR
Compares outputs and displays character-by-character analysis
"""

import cv2
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import OCR methods
try:
    from read_plate_yolo import read_plate
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO OCR not available")

try:
    from paddleocr import PaddleOCR
    paddle_ocr = PaddleOCR(lang='en')
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️ PaddleOCR not available")

from indian_plate_validator import validate_indian_plate

def read_with_paddleocr(image):
    """Read text using PaddleOCR."""
    if not PADDLE_AVAILABLE:
        return "", 0.0
    
    try:
        results = paddle_ocr.ocr(image, cls=True)
        if results and results[0]:
            best_result = max(results[0], key=lambda x: x[1][1])
            text = best_result[1][0].upper()
            confidence = best_result[1][1] * 100
            # Clean text
            import re
            clean_text = re.sub(r'[^A-Z0-9]', '', text)
            return clean_text, confidence
    except Exception as e:
        print(f"PaddleOCR error: {e}")
    
    return "", 0.0

def compare_character_by_character(text1, text2):
    """Compare two texts character by character."""
    max_len = max(len(text1), len(text2))
    comparison = []
    
    for i in range(max_len):
        char1 = text1[i] if i < len(text1) else '_'
        char2 = text2[i] if i < len(text2) else '_'
        match = char1 == char2
        comparison.append({
            'pos': i,
            'yolo': char1,
            'paddle': char2,
            'match': match
        })
    
    return comparison

def display_comparison_results(yolo_result, paddle_result, image_path):
    """Display detailed comparison results."""
    print("\n" + "="*60)
    print(f"📸 IMAGE: {os.path.basename(image_path)}")
    print("="*60)
    
    # OCR Results
    print("\n🔍 OCR RESULTS:")
    print(f"YOLO OCR:    {yolo_result['text']} ({yolo_result['confidence']:.1f}%)")
    print(f"PaddleOCR:   {paddle_result['text']} ({paddle_result['confidence']:.1f}%)")
    
    # Character comparison
    if yolo_result['text'] and paddle_result['text']:
        comparison = compare_character_by_character(yolo_result['text'], paddle_result['text'])
        
        print("\n📊 CHARACTER-BY-CHARACTER COMPARISON:")
        print("Pos: ", end="")
        for comp in comparison:
            print(f"{comp['pos']:2d} ", end="")
        print()
        
        print("YOLO:", end="")
        for comp in comparison:
            print(f" {comp['yolo']} ", end="")
        print()
        
        print("Paddle:", end="")
        for comp in comparison:
            print(f" {comp['paddle']} ", end="")
        print()
        
        print("Match:", end="")
        for comp in comparison:
            symbol = "✅" if comp['match'] else "❌"
            print(f" {symbol}", end="")
        print()
        
        # Count matches
        matches = sum(1 for comp in comparison if comp['match'])
        total = len(comparison)
        match_percentage = (matches / total * 100) if total > 0 else 0
        print(f"\nCharacter Match: {matches}/{total} ({match_percentage:.1f}%)")
    
    # Validation results
    print("\n🇮🇳 INDIAN PLATE VALIDATION:")
    
    if yolo_result['text']:
        yolo_validation = validate_indian_plate(yolo_result['text'])
        status = "✅ VALID" if yolo_validation['valid'] else "❌ INVALID"
        print(f"YOLO:    {status}")
        if yolo_validation['valid']:
            print(f"         Format: {yolo_validation['format']}")
            print(f"         State: {yolo_validation['state_name']}")
    
    if paddle_result['text']:
        paddle_validation = validate_indian_plate(paddle_result['text'])
        status = "✅ VALID" if paddle_validation['valid'] else "❌ INVALID"
        print(f"Paddle:  {status}")
        if paddle_validation['valid']:
            print(f"         Format: {paddle_validation['format']}")
            print(f"         State: {paddle_validation['state_name']}")
    
    # Winner determination
    print("\n🏆 WINNER DETERMINATION:")
    
    # Score calculation
    yolo_score = 0
    paddle_score = 0
    
    # Confidence score (0-40 points)
    yolo_score += min(yolo_result['confidence'] * 0.4, 40)
    paddle_score += min(paddle_result['confidence'] * 0.4, 40)
    
    # Validation bonus (30 points)
    if yolo_result['text']:
        yolo_validation = validate_indian_plate(yolo_result['text'])
        if yolo_validation['valid']:
            yolo_score += 30
    
    if paddle_result['text']:
        paddle_validation = validate_indian_plate(paddle_result['text'])
        if paddle_validation['valid']:
            paddle_score += 30
    
    # Length bonus (10 points for 8-10 chars)
    if 8 <= len(yolo_result['text']) <= 10:
        yolo_score += 10
    if 8 <= len(paddle_result['text']) <= 10:
        paddle_score += 10
    
    print(f"YOLO Score:    {yolo_score:.1f}/80")
    print(f"Paddle Score:  {paddle_score:.1f}/80")
    
    if yolo_score > paddle_score:
        print("🥇 WINNER: YOLO OCR")
        winner_text = yolo_result['text']
    elif paddle_score > yolo_score:
        print("🥇 WINNER: PaddleOCR")
        winner_text = paddle_result['text']
    else:
        print("🤝 TIE - Using YOLO OCR (default)")
        winner_text = yolo_result['text']
    
    print(f"📋 FINAL RESULT: {winner_text}")
    
    return winner_text

def process_image(image_path):
    """Process single image with both OCR methods."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Run YOLO OCR
    yolo_result = {'text': '', 'confidence': 0.0}
    if YOLO_AVAILABLE:
        try:
            text, conf = read_plate(image)
            yolo_result = {'text': text, 'confidence': conf}
        except Exception as e:
            print(f"YOLO OCR error: {e}")
    
    # Run PaddleOCR
    paddle_result = {'text': '', 'confidence': 0.0}
    if PADDLE_AVAILABLE:
        try:
            text, conf = read_with_paddleocr(image)
            paddle_result = {'text': text, 'confidence': conf}
        except Exception as e:
            print(f"PaddleOCR error: {e}")
    
    # Display comparison
    winner = display_comparison_results(yolo_result, paddle_result, image_path)
    
    return winner

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python ocr_comparison.py <image_path>")
        print("Example: python ocr_comparison.py test_plate.jpg")
        return
    
    image_path = sys.argv[1]
    
    print("🚀 OCR COMPARISON TOOL")
    print("Comparing YOLO OCR vs PaddleOCR")
    
    if not YOLO_AVAILABLE and not PADDLE_AVAILABLE:
        print("❌ No OCR methods available!")
        return
    
    process_image(image_path)

if __name__ == "__main__":
    main()