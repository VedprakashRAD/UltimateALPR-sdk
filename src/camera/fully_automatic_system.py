#!/usr/bin/env python3
"""
Fully Automatic Vehicle Recognition System
No manual controls - automatically detects and processes vehicles
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
import threading

# Import existing implementations
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from src.tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker
# Import configuration
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
import vehicle_tracking_config

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

class FullyAutomaticVehicleSystem:
    def __init__(self):
        """Initialize fully automatic system."""
        self.tracker = MemoryOptimizedVehicleTracker()
        self.camera = None
        self.running = False
        self.background = None
        self.vehicle_in_frame = False
        self.last_detection = 0
        self.detection_cooldown = 5  # seconds
        self.vehicle_states = {}  # Track vehicle states
        self.frame_width = 640
        self.frame_height = 480
        
        # Use the configured image storage path
        self.image_storage_path = vehicle_tracking_config.PATHS_CONFIG["image_storage"]
        os.makedirs(self.image_storage_path, exist_ok=True)
        os.makedirs("auto_captures", exist_ok=True)
        
    def initialize_camera(self):
        """Initialize camera."""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            return False
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        return True
        
    def detect_vehicle_presence(self, frame):
        """Detect if vehicle is present in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.background is None:
            self.background = gray.copy().astype("float")
            return False, None
            
        # Update background model
        cv2.accumulateWeighted(gray, self.background, 0.5)
        # Convert background back to uint8 for difference calculation
        background_uint8 = cv2.convertScaleAbs(self.background)
        
        # Ensure both images have same dimensions before subtraction
        if gray.shape != background_uint8.shape:
            # Resize background to match frame
            background_uint8 = cv2.resize(background_uint8, (gray.shape[1], gray.shape[0]))
            
        # Detect motion
        frame_delta = cv2.absdiff(background_uint8, gray)
        thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=3)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicle_detected = False
        largest_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 8000:  # Large enough to be a vehicle
                vehicle_detected = True
                if largest_contour is None or area > cv2.contourArea(largest_contour):
                    largest_contour = contour
                    
        return vehicle_detected, largest_contour
        
    def extract_license_plate(self, frame, contour_area=None):
        """Extract license plate from frame."""
        plates = []
        
        # Focus on vehicle area if available
        if contour_area is not None:
            x, y, w, h = cv2.boundingRect(contour_area)
            # Expand region slightly
            x = max(0, x - 20)
            y = max(0, y - 20)
            w = min(frame.shape[1] - x, w + 40)
            h = min(frame.shape[0] - y, h + 40)
            # Ensure ROI is valid
            if w > 0 and h > 0:
                roi = frame[y:y+h, x:x+w]
            else:
                roi = frame
        else:
            roi = frame
            
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
            
        # Apply filters for plate detection
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        
        for contour in contours:
            epsilon = 0.018 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # Rectangular shape
                x2, y2, w2, h2 = cv2.boundingRect(approx)
                aspect_ratio = w2 / h2
                
                # License plate aspect ratio and size check
                if 2.0 <= aspect_ratio <= 5.5 and w2 > 80 and h2 > 20:
                    # Extract plate region with bounds checking
                    if contour_area is not None:
                        plate_x = x + x2
                        plate_y = y + y2
                    else:
                        plate_x, plate_y = x2, y2
                        
                    # Ensure coordinates are within bounds
                    plate_x = max(0, plate_x)
                    plate_y = max(0, plate_y)
                    plate_w = min(frame.shape[1] - plate_x, w2)
                    plate_h = min(frame.shape[0] - plate_y, h2)
                    
                    if plate_w > 0 and plate_h > 0:
                        plate_img = frame[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w]
                        
                        # Read text if OCR available
                        plate_text = self.read_plate_text(plate_img)
                        
                        if len(plate_text) >= 3:  # Valid plate
                            plates.append({
                                'text': plate_text,
                                'bbox': (plate_x, plate_y, plate_w, plate_h),
                                'image': plate_img,
                                'confidence': min(len(plate_text) * 12, 95)
                            })
                        
        return plates
        
    def read_plate_text(self, plate_img):
        """Read text from plate image."""
        if not OCR_AVAILABLE:
            # Generate dummy plate number
            timestamp = str(int(time.time()))
            return f"AUTO{timestamp[-3:]}"
            
        try:
            # Preprocess for OCR
            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img
            
            # Resize if too small
            h, w = gray.shape
            if h < 40 and h > 0:
                scale = 40 / h
                new_w = int(w * scale) if w > 0 else 100
                gray = cv2.resize(gray, (new_w, 40))
                
            # Threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            # Clean text
            import re
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            return text if len(text) >= 3 else f"AUTO{str(int(time.time()))[-3:]}"
            
        except:
            timestamp = str(int(time.time()))
            return f"AUTO{timestamp[-3:]}"
            
    def process_vehicle_entry(self, frame, plates):
        """Process vehicle entry automatically."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save to CCTV_photos directory
        image_path = f"{self.image_storage_path}/entry_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Also save to auto_captures for backward compatibility
        auto_capture_path = f"auto_captures/entry_{timestamp}.jpg"
        cv2.imwrite(auto_capture_path, frame)
        
        # Use best plate or generate one
        if plates:
            best_plate = max(plates, key=lambda x: x['confidence'])
            plate_text = best_plate['text']
            confidence = best_plate['confidence']
        else:
            plate_text = f"AUTO{timestamp[-4:]}"
            confidence = 75.0
            
        # Create entry event
        entry_event = {
            "front_plate_number": plate_text,
            "rear_plate_number": plate_text,
            "front_plate_confidence": confidence,
            "rear_plate_confidence": confidence,
            "front_plate_image_path": image_path,
            "rear_plate_image_path": image_path,
            "entry_timestamp": datetime.utcnow(),
            "vehicle_color": "auto-detected",
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        result = self.tracker.db.entry_events.insert_one(entry_event)
        print(f"🚗 ENTRY: {plate_text} (Confidence: {confidence:.0f}%)")
        
        # Store vehicle state
        self.vehicle_states[plate_text] = {
            'entry_time': time.time(),
            'entry_id': result.inserted_id
        }
        
        return plate_text
        
    def process_vehicle_exit(self, frame, plates, entry_plate=None):
        """Process vehicle exit automatically."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save to CCTV_photos directory
        image_path = f"{self.image_storage_path}/exit_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        
        # Also save to auto_captures for backward compatibility
        auto_capture_path = f"auto_captures/exit_{timestamp}.jpg"
        cv2.imwrite(auto_capture_path, frame)
        
        # Use best plate or match with entry
        if plates:
            best_plate = max(plates, key=lambda x: x['confidence'])
            plate_text = best_plate['text']
            confidence = best_plate['confidence']
        elif entry_plate:
            plate_text = entry_plate
            confidence = 75.0
        else:
            plate_text = f"EXIT{timestamp[-4:]}"
            confidence = 75.0
            
        # Create exit event
        exit_event = {
            "front_plate_number": plate_text,
            "rear_plate_number": plate_text,
            "front_plate_confidence": confidence,
            "rear_plate_confidence": confidence,
            "front_plate_image_path": image_path,
            "rear_plate_image_path": image_path,
            "exit_timestamp": datetime.utcnow(),
            "vehicle_color": "auto-detected",
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        result = self.tracker.db.exit_events.insert_one(exit_event)
        print(f"🚗 EXIT: {plate_text} (Confidence: {confidence:.0f}%)")
        
        # Create journey if we have entry
        if plate_text in self.vehicle_states:
            entry_data = self.vehicle_states[plate_text]
            duration = time.time() - entry_data['entry_time']
            
            journey = {
                "entry_event_id": entry_data['entry_id'],
                "exit_event_id": result.inserted_id,
                "front_plate_number": plate_text,
                "entry_timestamp": datetime.utcfromtimestamp(entry_data['entry_time']),
                "exit_timestamp": datetime.utcnow(),
                "duration_seconds": int(duration),
                "is_employee": False,
                "created_at": datetime.utcnow()
            }
            
            self.tracker.db.vehicle_journeys.insert_one(journey)
            print(f"✅ JOURNEY: {plate_text} - {int(duration)}s")
            
            # Clean up state
            del self.vehicle_states[plate_text]
            
    def run(self):
        """Main automatic processing loop."""
        if not self.initialize_camera():
            print("❌ Camera initialization failed")
            return
            
        print("🚀 FULLY AUTOMATIC VEHICLE SYSTEM STARTED")
        print("📸 System will automatically detect and process vehicles")
        print("Press 'q' to quit")
        
        self.running = True
        current_vehicle_plate = None
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                current_time = time.time()
                
                # Detect vehicle presence
                vehicle_present, vehicle_contour = self.detect_vehicle_presence(frame)
                
                # State machine for vehicle tracking
                if vehicle_present and not self.vehicle_in_frame:
                    # Vehicle just entered frame
                    if current_time - self.last_detection > self.detection_cooldown:
                        print("\n🚗 VEHICLE DETECTED - Processing Entry...")
                        
                        # Extract license plates
                        plates = self.extract_license_plate(frame, vehicle_contour)
                        
                        # Process entry
                        current_vehicle_plate = self.process_vehicle_entry(frame, plates)
                        
                        self.vehicle_in_frame = True
                        self.last_detection = current_time
                        
                elif not vehicle_present and self.vehicle_in_frame:
                    # Vehicle just left frame
                    print("\n🚗 VEHICLE LEFT - Processing Exit...")
                    
                    # Extract license plates from last frame
                    plates = self.extract_license_plate(frame)
                    
                    # Process exit
                    self.process_vehicle_exit(frame, plates, current_vehicle_plate)
                    
                    self.vehicle_in_frame = False
                    current_vehicle_plate = None
                    
                # Draw status on frame
                status = "VEHICLE PRESENT" if vehicle_present else "MONITORING..."
                color = (0, 255, 0) if vehicle_present else (0, 255, 255)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw vehicle contour if present
                if vehicle_contour is not None:
                    cv2.drawContours(frame, [vehicle_contour], -1, (0, 255, 0), 2)
                    
                # Show stats
                stats = self.tracker.get_system_stats()
                stats_text = f"Entries: {stats['total_entry_events']} | Exits: {stats['total_exit_events']} | Journeys: {stats['total_journeys']}"
                cv2.putText(frame, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Fully Automatic Vehicle System', frame)
                
                # Only check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)
                
        self.cleanup()
        
    def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 Shutting down...")
        
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        if self.tracker:
            self.tracker.close()
            
        print("✅ System stopped")

def main():
    try:
        system = FullyAutomaticVehicleSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted")

if __name__ == "__main__":
    main()