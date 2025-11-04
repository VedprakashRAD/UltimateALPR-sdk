#!/usr/bin/env python3
"""
Camera Simulator for Vehicle Tracking System
This script simulates a continuous monitoring system that processes
vehicle entry and exit events as they occur.
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
import threading
from typing import List, Dict

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vehicle_tracking_system import VehicleTrackingSystem

class CameraSimulator:
    def __init__(self, db_path: str = "camera_simulation.db"):
        """
        Initialize the camera simulator.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.tracker = VehicleTrackingSystem(db_path)
        self.is_running = False
        self.entry_queue: List[Dict] = []
        self.exit_queue: List[Dict] = []
        self.vehicle_counter = 0
        
    def simulate_entry_event(self):
        """Simulate a vehicle entry event."""
        self.vehicle_counter += 1
        
        # For simulation, we'll use the sample image but with varying confidence
        sample_image = "assets/images/lic_us_1280x720.jpg"
        
        if not os.path.exists(sample_image):
            print(f"Sample image not found: {sample_image}")
            return None
            
        # Add some randomness to simulate different vehicles
        vehicle_id = f"SIM{self.vehicle_counter:04d}"
        
        # Process entry event
        entry_event = self.tracker.process_entry_event(sample_image, sample_image)
        entry_event["vehicle_id"] = vehicle_id
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ENTRY: Vehicle {vehicle_id} detected")
        print(f"  Front plate: {entry_event['front_plate_number']} ({entry_event['front_plate_confidence']:.2f}%)")
        print(f"  Rear plate: {entry_event['rear_plate_number']} ({entry_event['rear_plate_confidence']:.2f}%)")
        
        # Add to queue for matching
        self.entry_queue.append(entry_event)
        
        return entry_event
        
    def simulate_exit_event(self):
        """Simulate a vehicle exit event."""
        if not self.entry_queue:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT: No vehicles to exit")
            return None
            
        # Take the oldest entry event
        entry_event = self.entry_queue.pop(0)
        
        # For simulation, we'll use the sample image
        sample_image = "assets/images/lic_us_1280x720.jpg"
        
        # Process exit event
        exit_event = self.tracker.process_exit_event(sample_image, sample_image)
        exit_event["vehicle_id"] = entry_event["vehicle_id"]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EXIT: Vehicle {entry_event['vehicle_id']} detected")
        print(f"  Front plate: {exit_event['front_plate_number']} ({exit_event['front_plate_confidence']:.2f}%)")
        print(f"  Rear plate: {exit_event['rear_plate_number']} ({exit_event['rear_plate_confidence']:.2f}%)")
        
        # Add to queue for matching
        self.exit_queue.append((entry_event, exit_event))
        
        return exit_event
        
    def process_matched_journeys(self):
        """Process and display matched journeys."""
        # Try to match events
        journeys = self.tracker.match_entry_exit_events()
        
        for journey in journeys:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] MATCHED: Vehicle journey completed")
            print(f"  Vehicle: {journey['front_plate_number']} / {journey['rear_plate_number']}")
            print(f"  Duration: {journey['duration_seconds']} seconds")
            print(f"  Employee: {'Yes' if journey['is_employee'] else 'No'}")
            print(f"  Review needed: {'Yes' if journey['flagged_for_review'] else 'No'}")
            print()
            
    def start_monitoring(self, duration_minutes: int = 5):
        """
        Start the continuous monitoring simulation.
        
        Args:
            duration_minutes (int): Duration to run the simulation in minutes
        """
        print("=== Camera Simulation Started ===")
        print(f"Monitoring for {duration_minutes} minutes...")
        print("Press Ctrl+C to stop early")
        print()
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            while self.is_running and datetime.now() < end_time:
                # Randomly simulate entry events (every 30-90 seconds)
                if random.random() < 0.3:  # 30% chance every cycle
                    self.simulate_entry_event()
                    
                # Randomly simulate exit events (every 45-120 seconds)
                if random.random() < 0.2 and self.entry_queue:  # 20% chance if vehicles are queued
                    self.simulate_exit_event()
                    
                # Process matched journeys every 30 seconds
                if random.random() < 0.1:  # 10% chance every cycle
                    self.process_matched_journeys()
                    
                # Wait before next cycle
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user")
        finally:
            # Process any remaining matches
            self.process_matched_journeys()
            self.show_final_status()
            
        print("=== Camera Simulation Ended ===")
        
    def show_final_status(self):
        """Show the final status of the simulation."""
        print("\n=== Final Status ===")
        print(f"Total vehicles processed: {self.vehicle_counter}")
        
        unmatched_entries = self.tracker.get_unmatched_entries()
        unmatched_exits = self.tracker.get_unmatched_exits()
        
        print(f"Unmatched entry events: {len(unmatched_entries)}")
        print(f"Unmatched exit events: {len(unmatched_exits)}")
        
        # Count total journeys in database
        import sqlite3
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vehicle_journeys")
        journey_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"Completed journeys: {journey_count}")
        print()

def main():
    """Main function to run the camera simulation."""
    print("Vehicle Tracking System - Camera Simulator")
    print("=" * 45)
    print()
    
    # Initialize simulator
    simulator = CameraSimulator("camera_simulation.db")
    
    # Show available actions
    print("Available actions:")
    print("1. Run continuous simulation (5 minutes)")
    print("2. Simulate single entry event")
    print("3. Simulate single exit event")
    print("4. Process matched journeys")
    print("5. Show system status")
    print("6. Run extended simulation (10 minutes)")
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Select action (0-6): ").strip()
            
            if choice == "0":
                print("Exiting...")
                break
            elif choice == "1":
                simulator.start_monitoring(5)
            elif choice == "2":
                simulator.simulate_entry_event()
            elif choice == "3":
                simulator.simulate_exit_event()
            elif choice == "4":
                simulator.process_matched_journeys()
            elif choice == "5":
                simulator.show_final_status()
            elif choice == "6":
                simulator.start_monitoring(10)
            else:
                print("Invalid choice. Please select 0-6.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()