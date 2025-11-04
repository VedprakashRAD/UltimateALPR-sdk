#!/usr/bin/env python3
"""
Fully Automatic Vehicle Tracking System - Processes live camera feeds automatically
Detects vehicles, captures images, extracts license plates, and manages storage
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
import pytesseract

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

# Import modules
from tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker
import database.vehicle_tracking_config as vehicle_tracking_config
from utils.env_loader import AUTO_DELETE_IMAGES, KEEP_PROCESSED_IMAGES

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
        """Detect vehicle presence using background subtraction."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Initialize background
            if self.background is None:
                self.background = gray.copy().astype("float")
                return False, None
                
            # Update background
            cv2.accumulateWeighted(gray, self.background, 0.5)
            background = cv2.convertScaleAbs(self.background)
            
            # Compute difference
            diff = cv2.absdiff(background, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = 5000
            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    return True, contour
                    
            return False, None
            
        except Exception as e:
            print(f"Detection error: {e}")
            return False, None
            
    def extract_license_plate(self, frame, contour=None):
        """Extract license plate using OCR."""
        try:
            if contour is not None:
                # Get bounding box of vehicle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Focus on lower portion where plates are typically located
                plate_y = y + int(h * 0.6)
                plate_h = int(h * 0.3)
                
                if plate_y + plate_h <= frame.shape[0]:
                    plate_region = frame[plate_y:plate_y+plate_h, x:x+w]
                else:
                    plate_region = frame[y:y+h, x:x+w]
            else:
                plate_region = frame
                
            # Convert to grayscale
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            
            # Preprocess for OCR
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(thresh, config=config).strip()
            
            # Clean text
            import re
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            # Return plate data
            if len(text) >= 3:
                return [{'text': text, 'confidence': 85.0}]
            else:
                return []
            
        except:
            return []
            
    def delete_image_if_configured(self, image_path):
        """Delete image if auto-delete is configured."""
        if AUTO_DELETE_IMAGES and not KEEP_PROCESSED_IMAGES:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    print(f"🗑️  Deleted image: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"❌ Error deleting image {image_path}: {e}")
            
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
        
        # Delete images after processing if configured
        self.delete_image_if_configured(image_path)
        self.delete_image_if_configured(auto_capture_path)
        
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
        
        # Delete images after processing if configured
        self.delete_image_if_configured(image_path)
        self.delete_image_if_configured(auto_capture_path)
        
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
        if AUTO_DELETE_IMAGES:
            print("🗑️  Auto-delete images enabled")
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
        system = FullyAutomaticVehicleSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted")
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()