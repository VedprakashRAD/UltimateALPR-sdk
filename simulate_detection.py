#!/usr/bin/env python3
"""
Simulate vehicle detection and save to database
"""

import sys
import os
from datetime import datetime, timezone
import random

# Add paths
sys.path.append('src/database')
sys.path.append('src/tracking')

from pymongo import MongoClient

def simulate_vehicle_detections():
    """Simulate vehicle detections and save to database."""
    print("🚗 Simulating Vehicle Detections")
    print("=" * 40)
    
    # Connect to database
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['vehicle_tracking']
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Sample Indian license plates
    sample_plates = [
        'KL31T3155', 'MH12AB1234', 'DL9CAQ1234', 'TN09BC5678', 'KA05NP3747',
        'GJ01CD9876', 'UP32EF5432', 'BR03GH8765', 'WB19IJ2109', 'RJ14KL6543',
        'MP20MN3456', 'HR26OP7890', 'PB03QR1234', 'AP28ST5678', 'TS07UV9012'
    ]
    
    # Vehicle attributes
    colors = ["Red", "Blue", "Black", "White", "Silver", "Gray", "Green", "Yellow"]
    makes = ["Toyota", "Honda", "Maruti", "Hyundai", "Ford", "BMW", "Audi", "Mercedes"]
    models = ["Swift", "City", "Creta", "Innova", "Verna", "Baleno", "Dzire", "Seltos"]
    
    # Generate 5 entry events
    print("\n📥 Generating Entry Events...")
    for i in range(5):
        plate = random.choice(sample_plates)
        
        entry_data = {
            "front_plate_number": plate,
            "rear_plate_number": plate,
            "front_plate_confidence": random.uniform(85, 99),
            "rear_plate_confidence": random.uniform(85, 99),
            "entry_timestamp": datetime.now(timezone.utc),
            "camera_id": 0,
            "camera_name": "Camera 1 (Entry - Simulated)",
            "vehicle_color": random.choice(colors),
            "vehicle_make": random.choice(makes),
            "vehicle_model": random.choice(models),
            "detection_method": "TEST-GENERATOR",
            "is_processed": False,
            "created_at": datetime.now(timezone.utc)
        }
        
        result = db.entry_events.insert_one(entry_data)
        print(f"  ✅ Entry {i+1}: {plate} ({entry_data['front_plate_confidence']:.0f}%) - ID: {result.inserted_id}")
    
    # Generate 3 exit events
    print("\n📤 Generating Exit Events...")
    for i in range(3):
        plate = random.choice(sample_plates)
        
        exit_data = {
            "front_plate_number": plate,
            "rear_plate_number": plate,
            "front_plate_confidence": random.uniform(85, 99),
            "rear_plate_confidence": random.uniform(85, 99),
            "exit_timestamp": datetime.now(timezone.utc),
            "camera_id": 1,
            "camera_name": "Camera 2 (Exit - Simulated)",
            "vehicle_color": random.choice(colors),
            "vehicle_make": random.choice(makes),
            "vehicle_model": random.choice(models),
            "detection_method": "TEST-GENERATOR",
            "is_processed": False,
            "created_at": datetime.now(timezone.utc)
        }
        
        result = db.exit_events.insert_one(exit_data)
        print(f"  ✅ Exit {i+1}: {plate} ({exit_data['front_plate_confidence']:.0f}%) - ID: {result.inserted_id}")
    
    # Check final counts
    entry_count = db.entry_events.count_documents({})
    exit_count = db.exit_events.count_documents({})
    
    print(f"\n📊 Database Summary:")
    print(f"  Total Entries: {entry_count}")
    print(f"  Total Exits: {exit_count}")
    
    client.close()
    print("\n✅ Simulation completed!")

if __name__ == "__main__":
    simulate_vehicle_detections()