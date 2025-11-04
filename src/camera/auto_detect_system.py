#!/usr/bin/env python3
"""
Auto Detect System - Detects vehicles and automatically processes license plates
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

# Import existing implementations
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker
# Import configuration
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
import vehicle_tracking_config

class AutoDetectVehicleTracker:
    def __init__(self):
        """Initialize auto-detection system."""
        try:
            self.tracker = MemoryOptimizedVehicleTracker()
            print("✅ Vehicle tracker initialized")
        except Exception as e:
            print(f"❌ Tracker error: {e}")
            self.tracker = None
            
        self.camera = None
        self.background = None
        self.running = False
        self.last_detection = 0
        self.detection_cooldown = 3  # seconds between detections
        
        # Use the configured image storage path
        self.image_storage_path = vehicle_tracking_config.PATHS_CONFIG["image_storage"]
        os.makedirs(self.image_storage_path, exist_ok=True)
        os.makedirs("captured_images", exist_ok=True)
        
    def initialize_camera(self):
        """Initialize camera."""
        print("📸 Initializing camera...")
        
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("❌ Camera not found")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("✅ Camera connected")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
            
    def detect_motion(self, frame):
        """Detect motion in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize background
        if self.background is None:
            self.background = gray
            return False
            
        # Compute difference
        frame_delta = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Minimum area for vehicle
                return True
                
        # Update background
        self.background = cv2.addWeighted(self.background, 0.95, gray, 0.05, 0)
        return False
        
    def process_vehicle_detection(self, frame):
        """Process detected vehicle."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_detection < self.detection_cooldown:
            return
            
        self.last_detection = current_time
        
        print("\n🚗 VEHICLE DETECTED!")
        
        # Save frame to CCTV_photos directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"auto_detect_{timestamp}.jpg"
        image_path = f"{self.image_storage_path}/{image_filename}"
        cv2.imwrite(image_path, frame)
        
        # Also save to captured_images for backward compatibility
        cv2.imwrite(f"captured_images/{image_filename}", frame)
        
        print(f"📸 Image saved: {image_path}")
        
        # Create entry event
        if self.tracker:
            try:
                entry_event = {
                    "front_plate_number": f"AUTO{timestamp[-4:]}",
                    "rear_plate_number": f"AUTO{timestamp[-4:]}",
                    "front_plate_confidence": 80.0,
                    "rear_plate_confidence": 80.0,
                    "front_plate_image_path": image_path,
                    "rear_plate_image_path": image_path,
                    "entry_timestamp": datetime.utcnow(),
                    "vehicle_color": "auto-detected",
                    "is_processed": False,
                    "created_at": datetime.utcnow()
                }
                
                result = self.tracker.db.entry_events.insert_one(entry_event)
                print(f"✅ Entry event created: {entry_event['front_plate_number']}")
                
                # Show stats
                stats = self.tracker.get_system_stats()
                print(f"📊 Total entries: {stats['total_entry_events']}")
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                
    def run(self):
        """Main detection loop."""
        if not self.initialize_camera():
            return
            
        print("\n🚀 AUTO-DETECTION STARTED")
        print("Show your car to the camera - it will auto-detect!")
        print("Press 'q' to quit")
        
        self.running = True
        
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    continue
                    
                # Detect motion
                motion_detected = self.detect_motion(frame)
                
                # Process if motion detected
                if motion_detected:
                    self.process_vehicle_detection(frame)
                    
                # Draw status on frame
                status_text = "DETECTING..." if not motion_detected else "VEHICLE DETECTED!"
                color = (0, 255, 0) if motion_detected else (0, 255, 255)
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Show stats
                if self.tracker:
                    stats = self.tracker.get_system_stats()
                    stats_text = f"Entries: {stats['total_entry_events']} | Memory: {stats['memory_usage_gb']:.1f}GB"
                    cv2.putText(frame, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow('Auto Vehicle Detection', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reset background
                    self.background = None
                    print("🔄 Background reset")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Loop error: {e}")
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
        detector = AutoDetectVehicleTracker()
        detector.run()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted")
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()