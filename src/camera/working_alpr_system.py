#!/usr/bin/env python3
"""
Working ALPR System - Real-time license plate recognition with MongoDB integration
"""

import cv2
import numpy as np
import os
import sys
import time
import re
from datetime import datetime

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

# Try to import Tesseract OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Tesseract not available - install with: pip install pytesseract")

# Import modules
from tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker
import database.vehicle_tracking_config as vehicle_tracking_config
from utils.env_loader import AUTO_DELETE_IMAGES, KEEP_PROCESSED_IMAGES


class WorkingALPRSystem:
    def __init__(self):
        """Initialize working ALPR system."""
        try:
            self.tracker = MemoryOptimizedVehicleTracker()
            print("✅ Vehicle tracker initialized")
        except Exception as e:
            print(f"❌ Tracker error: {e}")
            self.tracker = None
            
        self.camera = None
        self.running = False
        self.frame_width = 640
        self.frame_height = 480
        
        # Use the configured image storage path
        self.image_storage_path = vehicle_tracking_config.PATHS_CONFIG["image_storage"]
        os.makedirs(self.image_storage_path, exist_ok=True)
        
        # License plate cascade (if available)
        self.plate_cascade = None
        self.load_plate_detector()
        
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("detected_plates", exist_ok=True)
        
    def load_plate_detector(self):
        """Load license plate detection cascade."""
        cascade_paths = [
            '/opt/homebrew/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml',
            'haarcascade_russian_plate_number.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                self.plate_cascade = cv2.CascadeClassifier(path)
                print(f"✅ Plate detector loaded: {path}")
                return
                
        print("⚠️  No plate cascade found - using contour detection")
        
    def detect_license_plates(self, frame):
        """Detect license plates in frame."""
        plates = []
        
        # Method 1: Haar Cascade (if available)
        if self.plate_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in detected_plates:
                # Ensure coordinates are within bounds
                x = max(0, x)
                y = max(0, y)
                w = min(frame.shape[1] - x, w)
                h = min(frame.shape[0] - y, h)
                
                if w > 0 and h > 0:
                    plate_img = frame[y:y+h, x:x+w]
                    plates.append({
                        'image': plate_img,
                        'bbox': (x, y, w, h),
                        'method': 'cascade'
                    })
        
        # Method 2: Contour-based detection
        plates.extend(self.detect_plates_by_contour(frame))
        
        return plates
        
    def detect_plates_by_contour(self, frame):
        """Detect plates using contour analysis."""
        plates = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for contour in contours:
            # Approximate contour
            epsilon = 0.018 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes (potential plates)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (typical license plate ratio)
                if h > 0:  # Avoid division by zero
                    aspect_ratio = w / h
                    if 2.0 <= aspect_ratio <= 5.0 and w > 100 and h > 30:
                        # Ensure coordinates are within bounds
                        x = max(0, x)
                        y = max(0, y)
                        w = min(frame.shape[1] - x, w)
                        h = min(frame.shape[0] - y, h)
                        
                        if w > 0 and h > 0:
                            plate_img = frame[y:y+h, x:x+w]
                            plates.append({
                                'image': plate_img,
                                'bbox': (x, y, w, h),
                                'method': 'contour'
                            })
                    
        return plates
        
    def read_plate_text(self, plate_image):
        """Extract text from license plate image."""
        if not TESSERACT_AVAILABLE:
            return "NO-OCR", 0.0
            
        try:
            # Check if plate_image is valid
            if plate_image is None or plate_image.size == 0:
                return "EMPTY", 0.0
                
            # Preprocess image for better OCR
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
            
            # Resize for better OCR
            height, width = gray.shape
            if height < 50 and height > 0:
                scale = 50 / height
                new_width = int(width * scale) if width > 0 else 100
                gray = cv2.resize(gray, (new_width, 50))
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR configuration for license plates
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            # Clean up text
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Calculate confidence (simple heuristic)
            confidence = min(len(text) * 15, 95) if len(text) >= 3 else 0
            
            return text, confidence
            
        except Exception as e:
            print(f"OCR error: {e}")
            return "ERROR", 0.0
            
    def delete_image_if_configured(self, image_path):
        """Delete image if auto-delete is configured."""
        if AUTO_DELETE_IMAGES and not KEEP_PROCESSED_IMAGES:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"🗑️  Deleted image: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"❌ Error deleting image {image_path}: {e}")
                
    def process_frame(self, frame):
        """Process frame for license plates."""
        results = []
        
        # Detect plates
        plates = self.detect_license_plates(frame)
        
        for i, plate_data in enumerate(plates):
            plate_img = plate_data['image']
            bbox = plate_data['bbox']
            method = plate_data['method']
            
            # Check if plate image is valid
            if plate_img is None or plate_img.size == 0:
                continue
                
            # Read plate text
            plate_text, confidence = self.read_plate_text(plate_img)
            
            if len(plate_text) >= 3:  # Valid plate text
                # Save plate image to CCTV_photos directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plate_filename = f"plate_{timestamp}_{i}.jpg"
                plate_path = f"{self.image_storage_path}/{plate_filename}"
                cv2.imwrite(plate_path, plate_img)
                
                # Also save to detected_plates for backward compatibility
                detected_plate_path = f"detected_plates/{plate_filename}"
                cv2.imwrite(detected_plate_path, plate_img)
                
                # Delete images after processing if configured
                self.delete_image_if_configured(plate_path)
                self.delete_image_if_configured(detected_plate_path)
                
                results.append({
                    'text': plate_text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'method': method,
                    'image_path': plate_path
                })
                
                # Draw rectangle around plate
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{plate_text} ({confidence:.0f}%)", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
        return results
        
    def save_vehicle_event(self, plate_results, event_type="entry"):
        """Save vehicle event to database."""
        if not self.tracker or not plate_results:
            return
            
        # Use best plate result
        best_plate = max(plate_results, key=lambda x: x['confidence'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        event_data = {
            "front_plate_number": best_plate['text'],
            "rear_plate_number": best_plate['text'],
            "front_plate_confidence": best_plate['confidence'],
            "rear_plate_confidence": best_plate['confidence'],
            "front_plate_image_path": best_plate['image_path'],
            "rear_plate_image_path": best_plate['image_path'],
            f"{event_type}_timestamp": datetime.utcnow(),
            "vehicle_color": "detected",
            "detection_method": best_plate['method'],
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        # Save to appropriate collection
        if event_type == "entry":
            result = self.tracker.db.entry_events.insert_one(event_data)
            print(f"✅ Entry saved: {best_plate['text']} ({best_plate['confidence']:.0f}%)")
        else:
            result = self.tracker.db.exit_events.insert_one(event_data)
            print(f"✅ Exit saved: {best_plate['text']} ({best_plate['confidence']:.0f}%)")
            
        return result
        
    def run(self):
        """Main processing loop."""
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("❌ Camera not found")
            return
            
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
        print("\n🚀 WORKING ALPR SYSTEM STARTED")
        print("📸 Show license plates to camera")
        print("Controls: 'e'=Entry, 'x'=Exit, 's'=Save, 'q'=Quit")
        
        self.running = True
        last_detection = 0
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                # Process frame
                plate_results, processed_frame = self.process_frame(frame)
                
                # Show results
                if plate_results:
                    current_time = time.time()
                    if current_time - last_detection > 2:  # Cooldown
                        last_detection = current_time
                        print(f"\n🎯 PLATES DETECTED:")
                        for result in plate_results:
                            print(f"  📋 {result['text']} - {result['confidence']:.0f}% ({result['method']})")
                
                # Add instructions to frame
                cv2.putText(processed_frame, "Press 'e' for Entry, 'x' for Exit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show stats
                if self.tracker:
                    stats = self.tracker.get_system_stats()
                    stats_text = f"Entries: {stats['total_entry_events']} | Exits: {stats['total_exit_events']}"
                    cv2.putText(processed_frame, stats_text, 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Working ALPR System', processed_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e') and plate_results:
                    self.save_vehicle_event(plate_results, "entry")
                elif key == ord('x') and plate_results:
                    self.save_vehicle_event(plate_results, "exit")
                elif key == ord('s') and plate_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Save to CCTV_photos directory
                    image_path = f"{self.image_storage_path}/manual_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)
                    # Also save to captured_images for backward compatibility
                    cv2.imwrite(f"captured_images/manual_{timestamp}.jpg", frame)
                    print(f"📸 Image saved manually: {image_path}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(1)
                
        self.cleanup()
        
    def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 Cleaning up...")
        
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        if self.tracker:
            self.tracker.close()
            
        print("✅ Cleanup completed")

def main():
    """Main function."""
    try:
        alpr = WorkingALPRSystem()
        alpr.run()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted")
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()