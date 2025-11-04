#!/usr/bin/env python3
"""
Utility script to populate the MongoDB database with test data.
"""

import sys
import os
import random
import string
from datetime import datetime, timedelta

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))

from tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker

def generate_random_plate():
    """Generate a random license plate number."""
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers = ''.join(random.choices(string.digits, k=4))
    more_letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    return f"{letters}{numbers}{more_letters}"

def generate_test_image_path(prefix="test"):
    """Generate a test image path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./CCTV_photos/{prefix}_{timestamp}.jpg"

def create_test_entry_event(tracker, vehicle_id):
    """Create a test entry event."""
    plate_number = generate_random_plate()
    timestamp = datetime.now() - timedelta(minutes=random.randint(1, 120))
    
    entry_event = {
        "front_plate_number": plate_number,
        "rear_plate_number": plate_number,
        "front_plate_confidence": random.uniform(80, 95),
        "rear_plate_confidence": random.uniform(80, 95),
        "front_plate_image_path": generate_test_image_path("entry"),
        "rear_plate_image_path": generate_test_image_path("entry"),
        "entry_timestamp": timestamp,
        "vehicle_color": random.choice(["Red", "Blue", "Green", "Black", "White", "Silver"]),
        "vehicle_make": random.choice(["Toyota", "Honda", "Ford", "BMW", "Audi", "Mercedes"]),
        "vehicle_model": random.choice(["Camry", "Civic", "Focus", "X3", "A4", "C-Class"]),
        "is_processed": False,
        "created_at": timestamp
    }
    
    result = tracker.db.entry_events.insert_one(entry_event)
    print(f"✅ Created entry event: {plate_number} (ID: {result.inserted_id})")
    return result.inserted_id, plate_number, timestamp

def create_test_exit_event(tracker, entry_plate, entry_timestamp):
    """Create a test exit event."""
    exit_timestamp = entry_timestamp + timedelta(minutes=random.randint(5, 60))
    
    exit_event = {
        "front_plate_number": entry_plate,
        "rear_plate_number": entry_plate,
        "front_plate_confidence": random.uniform(80, 95),
        "rear_plate_confidence": random.uniform(80, 95),
        "front_plate_image_path": generate_test_image_path("exit"),
        "rear_plate_image_path": generate_test_image_path("exit"),
        "exit_timestamp": exit_timestamp,
        "vehicle_color": random.choice(["Red", "Blue", "Green", "Black", "White", "Silver"]),
        "vehicle_make": random.choice(["Toyota", "Honda", "Ford", "BMW", "Audi", "Mercedes"]),
        "vehicle_model": random.choice(["Camry", "Civic", "Focus", "X3", "A4", "C-Class"]),
        "is_processed": False,
        "created_at": exit_timestamp
    }
    
    result = tracker.db.exit_events.insert_one(exit_event)
    print(f"✅ Created exit event: {entry_plate} (ID: {result.inserted_id})")
    return result.inserted_id

def create_test_journey(tracker, entry_id, exit_id, plate_number):
    """Create a test vehicle journey."""
    entry_event = tracker.db.entry_events.find_one({"_id": entry_id})
    exit_event = tracker.db.exit_events.find_one({"_id": exit_id})
    
    if entry_event and exit_event:
        duration = (exit_event["exit_timestamp"] - entry_event["entry_timestamp"]).total_seconds()
        
        journey = {
            "entry_event_id": entry_id,
            "exit_event_id": exit_id,
            "front_plate_number": plate_number,
            "rear_plate_number": plate_number,
            "entry_timestamp": entry_event["entry_timestamp"],
            "exit_timestamp": exit_event["exit_timestamp"],
            "duration_seconds": int(duration),
            "vehicle_color": entry_event.get("vehicle_color"),
            "vehicle_make": entry_event.get("vehicle_make"),
            "vehicle_model": entry_event.get("vehicle_model"),
            "is_employee": random.choice([True, False]),
            "flagged_for_review": False,
            "created_at": datetime.now()
        }
        
        result = tracker.db.vehicle_journeys.insert_one(journey)
        print(f"✅ Created journey: {plate_number} (ID: {result.inserted_id})")
        return result.inserted_id

def create_test_employee_vehicle(tracker, plate_number):
    """Create a test employee vehicle."""
    employee_vehicle = {
        "plate_number": plate_number,
        "employee_name": f"Employee {random.randint(1, 100)}",
        "is_active": True,
        "created_at": datetime.now()
    }
    
    try:
        result = tracker.db.employee_vehicles.insert_one(employee_vehicle)
        print(f"✅ Created employee vehicle: {plate_number} (ID: {result.inserted_id})")
    except Exception as e:
        print(f"ℹ️  Employee vehicle {plate_number} already exists")

def main():
    """Main function to populate test data."""
    print("🌱 Populating MongoDB with test data...")
    print("=" * 50)
    
    try:
        # Initialize the tracker
        tracker = MemoryOptimizedVehicleTracker()
        print("✅ Database connection established")
        
        # Create test directories if they don't exist
        os.makedirs("./CCTV_photos", exist_ok=True)
        
        # Create test data
        num_vehicles = 10
        
        print(f"\nCreating {num_vehicles} test vehicles...")
        
        for i in range(num_vehicles):
            # Create entry event
            entry_id, plate_number, entry_timestamp = create_test_entry_event(tracker, i)
            
            # Randomly create exit event and journey (not all vehicles exit)
            if random.choice([True, False, True]):  # 2/3 chance
                exit_id = create_test_exit_event(tracker, plate_number, entry_timestamp)
                create_test_journey(tracker, entry_id, exit_id, plate_number)
            
            # Randomly mark some as employee vehicles
            if random.choice([True, False, False]):  # 1/3 chance
                create_test_employee_vehicle(tracker, plate_number)
        
        # Show final statistics
        print("\n📊 Final Database Statistics:")
        print("=" * 30)
        print(f"Entry Events: {tracker.db.entry_events.count_documents({})}")
        print(f"Exit Events: {tracker.db.exit_events.count_documents({})}")
        print(f"Vehicle Journeys: {tracker.db.vehicle_journeys.count_documents({})}")
        print(f"Employee Vehicles: {tracker.db.employee_vehicles.count_documents({})}")
        
        # Close connection
        tracker.close()
        print("\n✅ Test data population completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())