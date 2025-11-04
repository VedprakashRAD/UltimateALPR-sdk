#!/usr/bin/env python3
"""
Demo script for the Vehicle Tracking System using ultimateALPR-SDK
"""

import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vehicle_tracking_system import VehicleTrackingSystem

def demo_vehicle_tracking():
    """Demonstrate the vehicle tracking system."""
    print("=== Vehicle Tracking System Demo ===")
    print()
    
    # Initialize the tracking system
    print("Initializing Vehicle Tracking System...")
    tracker = VehicleTrackingSystem("demo_vehicle_tracking.db")
    print("System initialized successfully!")
    print()
    
    # For demo purposes, we'll use the sample image from the SDK
    sample_image = "assets/images/lic_us_1280x720.jpg"
    
    if not os.path.exists(sample_image):
        print(f"Sample image not found: {sample_image}")
        print("Please make sure you're running this from the root of the ultimateALPR-sdk directory")
        return
    
    print("Processing sample vehicle entry event...")
    
    # Simulate an entry event (in a real system, these would be from different cameras)
    entry_event = tracker.process_entry_event(sample_image, sample_image)
    print(f"Entry event recorded:")
    print(f"  Front plate: {entry_event['front_plate_number']} (Confidence: {entry_event['front_plate_confidence']:.2f}%)")
    print(f"  Rear plate: {entry_event['rear_plate_number']} (Confidence: {entry_event['rear_plate_confidence']:.2f}%)")
    print(f"  Timestamp: {entry_event['entry_timestamp']}")
    print()
    
    # Wait a moment to simulate time passing
    print("Waiting 5 seconds to simulate vehicle movement...")
    time.sleep(5)
    
    # Simulate an exit event
    print("Processing sample vehicle exit event...")
    exit_event = tracker.process_exit_event(sample_image, sample_image)
    print(f"Exit event recorded:")
    print(f"  Front plate: {exit_event['front_plate_number']} (Confidence: {exit_event['front_plate_confidence']:.2f}%)")
    print(f"  Rear plate: {exit_event['rear_plate_number']} (Confidence: {exit_event['rear_plate_confidence']:.2f}%)")
    print(f"  Timestamp: {exit_event['exit_timestamp']}")
    print()
    
    # Wait for matching process
    print("Matching entry and exit events...")
    journeys = tracker.match_entry_exit_events()
    
    if journeys:
        print(f"Successfully matched {len(journeys)} vehicle journey(s):")
        for journey in journeys:
            print(f"  Vehicle: {journey['front_plate_number']} / {journey['rear_plate_number']}")
            print(f"  Entry: {journey['entry_timestamp']}")
            print(f"  Exit: {journey['exit_timestamp']}")
            print(f"  Duration: {journey['duration_seconds']} seconds")
            print(f"  Employee vehicle: {'Yes' if journey['is_employee'] else 'No'}")
            print(f"  Flagged for review: {'Yes' if journey['flagged_for_review'] else 'No'}")
            print()
    else:
        print("No journeys matched. Checking for unmatched events...")
        unmatched_entries = tracker.get_unmatched_entries()
        unmatched_exits = tracker.get_unmatched_exits()
        
        print(f"Unmatched entries: {len(unmatched_entries)}")
        print(f"Unmatched exits: {len(unmatched_exits)}")
        print()
    
    # Show database status
    print("Database status:")
    unmatched_entries = tracker.get_unmatched_entries()
    unmatched_exits = tracker.get_unmatched_exits()
    print(f"  Unmatched entry events: {len(unmatched_entries)}")
    print(f"  Unmatched exit events: {len(unmatched_exits)}")
    
    print()
    print("=== Demo completed successfully! ===")

def demo_employee_vehicle():
    """Demonstrate employee vehicle handling."""
    print("\n=== Employee Vehicle Demo ===")
    print()
    
    # Initialize the tracking system
    tracker = VehicleTrackingSystem("demo_vehicle_tracking.db")
    
    # Add an employee vehicle to the database
    conn = sqlite3.connect("demo_vehicle_tracking.db")
    cursor = conn.cursor()
    
    # For demo, we'll add a vehicle with the plate number we detected
    sample_plate = "3PEDLM*"
    
    cursor.execute("""
        INSERT OR IGNORE INTO vehicles (
            plate_number, is_employee
        ) VALUES (?, ?)
    """, (sample_plate, True))
    
    conn.commit()
    conn.close()
    
    print(f"Added vehicle {sample_plate} as employee vehicle to database.")
    
    # Process an entry event with this employee vehicle
    sample_image = "assets/images/lic_us_1280x720.jpg"
    entry_event = tracker.process_entry_event(sample_image, sample_image)
    
    print(f"Processed entry event for employee vehicle:")
    print(f"  Front plate: {entry_event['front_plate_number']}")
    print(f"  System detected as employee vehicle: {tracker.is_employee_vehicle(entry_event['front_plate_number'], entry_event['rear_plate_number'])}")
    
    print()
    print("=== Employee Vehicle Demo completed ===")

if __name__ == "__main__":
    # Run the main demo
    demo_vehicle_tracking()
    
    # Run the employee vehicle demo
    demo_employee_vehicle()