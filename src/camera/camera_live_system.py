#!/usr/bin/env python3
"""
Camera Live System - Processes live camera feeds for vehicle detection
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

class LiveCameraVehicleTracker:
    def __init__(self):
        """Initialize live camera system with full integration."""
        # Initialize the optimized vehicle tracker
        self.tracker = MemoryOptimizedVehicleTracker()
        
        # Camera setup
        self.camera1 = None  # Entry front camera
        self.camera2 = None  # Entry rear camera
        self.running = False
        
        # Image storage
        os.makedirs("captured_images", exist_ok=True)
        
        print("🍓 Live Camera Vehicle Tracking System")
        print("=" * 50)
        print("✅ MongoDB integration active")
        print("✅ Memory optimization enabled")
        print("✅ Employee vehicle management ready")
        
    def setup_collections(self):
        """Collections already setup by MemoryOptimizedVehicleTracker."""
        pass
        
    def initialize_cameras(self):
        """Initialize camera connections."""
        print("📸 Initializing cameras...")
        
        try:
            # Try to connect to cameras
            self.camera1 = cv2.VideoCapture(0)  # First camera
            if not self.camera1.isOpened():
                print("⚠️  Camera 1 not found, using demo mode")
                self.camera1 = None
            else:
                print("✅ Camera 1 connected")
                
            # Try second camera
            self.camera2 = cv2.VideoCapture(1)  # Second camera
            if not self.camera2.isOpened():
                print("⚠️  Camera 2 not found, using single camera mode")
                self.camera2 = None
            else:
                print("✅ Camera 2 connected")
                
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            
    def capture_and_process(self):
        """Main camera capture and processing loop."""
        print("\n🚀 Starting live vehicle tracking...")
        print("Press 'q' to quit, 'e' for entry event, 'x' for exit event")
        
        self.running = True
        
        while self.running:
            try:
                # Show system stats
                self.show_live_stats()
                
                # Camera 1 capture
                if self.camera1 and self.camera1.isOpened():
                    ret1, frame1 = self.camera1.read()
                    if ret1:
                        cv2.imshow('Camera 1 - Entry Front', frame1)
                        
                # Camera 2 capture  
                if self.camera2 and self.camera2.isOpened():
                    ret2, frame2 = self.camera2.read()
                    if ret2:
                        cv2.imshow('Camera 2 - Entry Rear', frame2)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n🛑 Stopping system...")
                    break
                elif key == ord('e'):
                    self.simulate_entry_event()
                elif key == ord('x'):
                    self.simulate_exit_event()
                elif key == ord('m'):
                    self.match_recent_events()
                    
            except KeyboardInterrupt:
                print("\n🛑 System interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(1)
                
        self.cleanup()
        
    def simulate_entry_event(self):
        """Process real entry event with ALPR."""
        print("\n🚪 Processing Entry Event...")
        
        # Capture and save frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        front_image_path = f"captured_images/entry_front_{timestamp}.jpg"
        rear_image_path = f"captured_images/entry_rear_{timestamp}.jpg"
        
        # Save camera frames
        if self.camera1 and self.camera1.isOpened():
            ret1, frame1 = self.camera1.read()
            if ret1:
                cv2.imwrite(front_image_path, frame1)
                print(f"📸 Front image saved: {front_image_path}")
        
        if self.camera2 and self.camera2.isOpened():
            ret2, frame2 = self.camera2.read()
            if ret2:
                cv2.imwrite(rear_image_path, frame2)
                print(f"📸 Rear image saved: {rear_image_path}")
        else:
            # Use front camera for both if rear not available
            if self.camera1 and self.camera1.isOpened():
                ret1, frame1 = self.camera1.read()
                if ret1:
                    cv2.imwrite(rear_image_path, frame1)
        
        # Process with ALPR if available
        if self.alpr_sdk and os.path.exists(front_image_path):
            try:
                entry_event = self.tracker.process_entry_event(front_image_path, rear_image_path)
                if entry_event:
                    print(f"✅ ALPR Entry processed: {entry_event.get('front_plate_number', 'Unknown')}")
                    print(f"🎯 Confidence: {entry_event.get('front_plate_confidence', 0):.1f}%")
                else:
                    print("⚠️  ALPR processing failed, creating manual entry")
                    self.create_manual_entry(front_image_path, rear_image_path)
            except Exception as e:
                print(f"❌ ALPR error: {e}")
                self.create_manual_entry(front_image_path, rear_image_path)
        else:
            self.create_manual_entry(front_image_path, rear_image_path)
        
        # Show system stats
        stats = self.tracker.get_system_stats()
        print(f"💾 Memory: {stats['memory_usage_gb']:.2f}GB ({stats['memory_percent']:.1f}%)")
        print(f"📊 Total Entries: {stats['total_entry_events']}")
        
    def create_manual_entry(self, front_image_path, rear_image_path):
        """Create manual entry when ALPR is not available."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        entry_event = {
            "front_plate_number": f"MAN{timestamp[-4:]}",
            "rear_plate_number": f"MAN{timestamp[-4:]}",
            "front_plate_confidence": 85.0,
            "rear_plate_confidence": 85.0,
            "front_plate_image_path": front_image_path,
            "rear_plate_image_path": rear_image_path,
            "entry_timestamp": datetime.utcnow(),
            "vehicle_color": "manual",
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        result = self.tracker.db.entry_events.insert_one(entry_event)
        print(f"✅ Manual entry created: {entry_event['front_plate_number']}")
        
    def simulate_exit_event(self):
        """Process real exit event with ALPR."""
        print("\n🚪 Processing Exit Event...")
        
        # Capture and save frames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        front_image_path = f"captured_images/exit_front_{timestamp}.jpg"
        rear_image_path = f"captured_images/exit_rear_{timestamp}.jpg"
        
        # Save camera frames
        if self.camera1 and self.camera1.isOpened():
            ret1, frame1 = self.camera1.read()
            if ret1:
                cv2.imwrite(front_image_path, frame1)
                print(f"📸 Front image saved: {front_image_path}")
        
        if self.camera2 and self.camera2.isOpened():
            ret2, frame2 = self.camera2.read()
            if ret2:
                cv2.imwrite(rear_image_path, frame2)
        else:
            # Use front camera for both if rear not available
            if self.camera1 and self.camera1.isOpened():
                ret1, frame1 = self.camera1.read()
                if ret1:
                    cv2.imwrite(rear_image_path, frame1)
        
        # Process with ALPR if available
        if self.alpr_sdk and os.path.exists(front_image_path):
            try:
                exit_event = self.tracker.process_exit_event(front_image_path, rear_image_path)
                if exit_event:
                    print(f"✅ ALPR Exit processed: {exit_event.get('front_plate_number', 'Unknown')}")
                    print(f"🎯 Confidence: {exit_event.get('front_plate_confidence', 0):.1f}%")
                else:
                    print("⚠️  ALPR processing failed, creating manual exit")
                    self.create_manual_exit(front_image_path, rear_image_path)
            except Exception as e:
                print(f"❌ ALPR error: {e}")
                self.create_manual_exit(front_image_path, rear_image_path)
        else:
            self.create_manual_exit(front_image_path, rear_image_path)
        
        # Try to match with recent entries
        self.match_recent_events()
        
    def create_manual_exit(self, front_image_path, rear_image_path):
        """Create manual exit when ALPR is not available."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exit_event = {
            "front_plate_number": f"MAN{timestamp[-4:]}",
            "rear_plate_number": f"MAN{timestamp[-4:]}",
            "front_plate_confidence": 85.0,
            "rear_plate_confidence": 85.0,
            "front_plate_image_path": front_image_path,
            "rear_plate_image_path": rear_image_path,
            "exit_timestamp": datetime.utcnow(),
            "vehicle_color": "manual",
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        result = self.tracker.db.exit_events.insert_one(exit_event)
        print(f"✅ Manual exit created: {exit_event['front_plate_number']}")
        
    def match_recent_events(self):
        """Use the optimized matching from MemoryOptimizedVehicleTracker."""
        print("🔄 Matching recent events...")
        
        try:
            # Use the optimized matching system
            journeys = self.tracker.match_entry_exit_events(batch_size=5)
            
            if journeys:
                for journey in journeys:
                    duration = journey.get('duration_seconds', 0)
                    plate = journey.get('front_plate_number', 'Unknown')
                    employee_status = "👨💼 Employee" if journey.get('is_employee') else "👤 Visitor"
                    print(f"✅ Journey: {plate} ({duration}s) - {employee_status}")
            else:
                print("⏳ No matches found yet")
                
        except Exception as e:
            print(f"❌ Matching error: {e}")
                    
    def show_live_stats(self):
        """Show live system statistics."""
        # Clear screen and show stats
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🍓 LIVE VEHICLE TRACKING SYSTEM")
        print("=" * 50)
        
        # System stats
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        print(f"💾 Memory: {memory_info.used / (1024**3):.2f}GB ({memory_info.percent:.1f}%)")
        print(f"🖥️  CPU: {cpu_percent:.1f}%")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Database stats from tracker
        stats = self.tracker.get_system_stats()
        entry_count = stats['total_entry_events']
        exit_count = stats['total_exit_events']
        journey_count = stats['total_journeys']
        unprocessed_entries = stats['unprocessed_entries']
        unprocessed_exits = stats['unprocessed_exits']
        
        print(f"📊 Entries: {entry_count} | Exits: {exit_count} | Journeys: {journey_count}")
        print(f"⏳ Pending: {unprocessed_entries} entries, {unprocessed_exits} exits")
        
        # Camera status
        cam1_status = "🟢 Connected" if self.camera1 and self.camera1.isOpened() else "🔴 Disconnected"
        cam2_status = "🟢 Connected" if self.camera2 and self.camera2.isOpened() else "🔴 Disconnected"
        
        print(f"📸 Camera 1: {cam1_status}")
        print(f"📸 Camera 2: {cam2_status}")
        
        print("\n🎮 Controls:")
        print("  'e' = Entry Event | 'x' = Exit Event | 'm' = Match Events | 'q' = Quit")
        print("-" * 50)
        
    def cleanup(self):
        """Cleanup resources."""
        print("\n🧹 Cleaning up...")
        
        self.running = False
        
        if self.camera1:
            self.camera1.release()
        if self.camera2:
            self.camera2.release()
            
        cv2.destroyAllWindows()
        
        if self.tracker:
            self.tracker.close()
            
        print("✅ Cleanup completed")

def main():
    """Main function."""
    try:
        # Initialize system
        tracker = LiveCameraVehicleTracker()
        
        # Initialize cameras
        tracker.initialize_cameras()
        
        # Start live tracking
        tracker.capture_and_process()
        
    except KeyboardInterrupt:
        print("\n🛑 System interrupted")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        print("👋 System shutdown")

if __name__ == "__main__":
    main()