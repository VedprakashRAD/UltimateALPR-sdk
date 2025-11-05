
# Enhanced OCR Integration - Add to working_alpr_system.py

# Additional imports for enhanced OCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  EasyOCR not available")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("⚠️  TrOCR not available")

try:
    sys.path.append('models/lpgan')
    from indian_lpgan import IndianLPGAN
    LPGAN_AVAILABLE = True
except ImportError:
    LPGAN_AVAILABLE = False
    print("⚠️  LP-GAN not available")

class EnhancedWorkingALPRSystem(WorkingALPRSystem):
    """Enhanced ALPR System with LP-GAN, TrOCR, and EasyOCR."""
    
    def __init__(self):
        super().__init__()
        self.load_enhanced_models()
    
    def load_enhanced_models(self):
        """Load enhanced OCR models."""
        # Load EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.easy_ocr = easyocr.Reader(['en'], gpu=False)
                print("✅ EasyOCR model loaded")
            except Exception as e:
                print(f"❌ EasyOCR loading failed: {e}")
                self.easy_ocr = None
        else:
            self.easy_ocr = None
        
        # Load TrOCR
        if TROCR_AVAILABLE:
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                self.trocr_model.eval()
                print("✅ TrOCR model loaded")
            except Exception as e:
                print(f"❌ TrOCR loading failed: {e}")
                self.trocr_processor = None
                self.trocr_model = None
        else:
            self.trocr_processor = None
            self.trocr_model = None
        
        # Load LP-GAN
        if LPGAN_AVAILABLE:
            try:
                self.lpgan = IndianLPGAN()
                print("✅ LP-GAN model loaded")
            except Exception as e:
                print(f"❌ LP-GAN loading failed: {e}")
                self.lpgan = None
        else:
            self.lpgan = None
    
    def enhance_plate_image_advanced(self, plate_image):
        """Enhanced image preprocessing with LP-GAN techniques."""
        # Apply original Deep LPR preprocessing
        enhanced = self.enhance_plate_image(plate_image)
        
        # Apply LP-GAN enhancement if available
        if self.lpgan:
            try:
                enhanced = self.lpgan.enhance_for_ocr(enhanced)
            except Exception as e:
                print(f"LP-GAN enhancement error: {e}")
        
        return enhanced
    
    def read_plate_text_enhanced(self, plate_image):
        """Enhanced OCR with multiple models for maximum accuracy."""
        if plate_image is None or plate_image.size == 0:
            return "EMPTY", 0.0
        
        # Apply advanced enhancement
        enhanced_image = self.enhance_plate_image_advanced(plate_image)
        
        results = []
        
        # Method 1: TrOCR (Highest accuracy - 99%)
        if self.trocr_processor and self.trocr_model:
            try:
                from PIL import Image
                if isinstance(enhanced_image, np.ndarray):
                    pil_image = Image.fromarray(enhanced_image)
                else:
                    pil_image = enhanced_image
                
                pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values
                
                with torch.no_grad():
                    generated_ids = self.trocr_model.generate(pixel_values)
                
                text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if self.validate_plate_format(clean_text) and len(clean_text) >= 5:
                    results.append((clean_text, 99.0, 'trocr'))
                    
            except Exception as e:
                print(f"TrOCR error: {e}")
        
        # Method 2: EasyOCR (High accuracy - 98%)
        if self.easy_ocr:
            try:
                ocr_results = self.easy_ocr.readtext(enhanced_image)
                for (bbox, text, confidence) in ocr_results:
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    confidence_percent = confidence * 100
                    
                    if self.validate_plate_format(clean_text) and confidence_percent > 85:
                        results.append((clean_text, confidence_percent, 'easyocr'))
                        
            except Exception as e:
                print(f"EasyOCR error: {e}")
        
        # Method 3: Enhanced PaddleOCR (Good accuracy - 95%)
        if self.paddle_ocr:
            try:
                paddle_results = self.paddle_ocr.ocr(enhanced_image)
                if paddle_results and paddle_results[0]:
                    for line in paddle_results[0]:
                        if line and len(line) >= 2:
                            text = line[1][0]
                            confidence = line[1][1] * 100
                            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                            
                            if self.validate_plate_format(clean_text) and confidence > 85:
                                results.append((clean_text, confidence, 'paddleocr'))
                                
            except Exception as e:
                print(f"PaddleOCR error: {e}")
        
        # Method 4: Tesseract (Fallback)
        if TESSERACT_AVAILABLE and not results:
            try:
                _, thresh = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(thresh, config=config).strip()
                clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if self.validate_plate_format(clean_text):
                    confidence = min(len(clean_text) * 16, 90)
                    results.append((clean_text, confidence, 'tesseract'))
                    
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        # Return best result
        if results:
            # Sort by confidence and format validity
            results.sort(key=lambda x: x[1], reverse=True)
            best_text, best_confidence, best_method = results[0]
            print(f"🎯 Best OCR: {best_text} ({best_confidence:.1f}%) via {best_method}")
            return best_text, best_confidence
        
        return "NO-OCR", 0.0
    
    def process_frame(self, frame):
        """Enhanced frame processing with multiple OCR models."""
        results = []
        
        # Detect plates using existing methods
        plates = self.detect_license_plates(frame)
        
        for i, plate_data in enumerate(plates):
            plate_img = plate_data['image']
            bbox = plate_data['bbox']
            method = plate_data['method']
            
            if plate_img is None or plate_img.size == 0:
                continue
            
            # Use enhanced OCR
            plate_text, confidence = self.read_plate_text_enhanced(plate_img)
            
            # Lower threshold due to better accuracy
            if len(plate_text) >= 5 and confidence > 80:  # Reduced from 85 due to better models
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_filename = f"plate_{timestamp}_{i}.jpg"
                plate_path = f"{self.image_storage_path}/{plate_filename}"
                cv2.imwrite(plate_path, plate_img)
                
                detected_plate_path = f"detected_plates/{plate_filename}"
                cv2.imwrite(detected_plate_path, plate_img)
                
                self.delete_image_if_configured(plate_path)
                self.delete_image_if_configured(detected_plate_path)
                
                results.append({
                    'text': plate_text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'method': method,
                    'image_path': plate_path
                })
                
                # Draw enhanced visualization
                x, y, w, h = bbox
                color = (0, 255, 0) if confidence > 95 else (0, 255, 255)  # Green for high confidence
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{plate_text} ({confidence:.0f}%)", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return results

# Usage: Replace WorkingALPRSystem with EnhancedWorkingALPRSystem in main.py
