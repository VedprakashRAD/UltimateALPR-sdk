#!/usr/bin/env python3
"""
Raspberry Pi Vehicle Tracking System Demo
Demonstrates the memory-optimized MongoDB-based vehicle tracking system.
"""

import os
import sys
import time
from datetime import datetime
import psutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker

def print_system_info():
    """Print system information."""
    print("=" * 60)
    print("🍓 RASPBERRY PI VEHICLE TRACKING SYSTEM DEMO")
    print("=" * 60)
    
    # System info
    memory_info = psutil.virtual_memory()
    print(f"💾 Total RAM: {memory_info.total / (1024**3):.2f}GB")
    print(f"💾 Available RAM: {memory_info.available / (1024**3):.2f}GB")
    print(f"💾 Used RAM: {memory_info.used / (1024**3):.2f}GB ({memory_info.percent:.1f}%)")
    
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"🖥️  CPU Usage: {cpu_percent:.1f}%")
    
    # Disk info
    disk_info = psutil.disk_usage('/')
    print(f"💿 Disk Free: {disk_info.free / (1024**3):.2f}GB")
    
    print("-" * 60)

def simulate_vehicle_events(tracker):
    """Simulate vehicle entry and exit events."""
    print("🚗 Simulating Vehicle Events...")
    
    # Sample vehicle data
    vehicles = [
        {
            "front_plate": "ABC123",
            "rear_plate": "ABC123",
            "employee": False,
            "description": "Regular visitor vehicle"
        },
        {
            "front_plate": "EMP001", 
            "rear_plate": "EMP001",
            "employee": True,
            "description": "Employee vehicle - John Doe"
        },
        {
            "front_plate": "XYZ789",
            "rear_plate": "XYZ789", 
            "employee": False,
            "description": "Delivery vehicle"
        }
    ]
    
    # Add employee vehicles to database
    print("\n📝 Adding employee vehicles...")
    for vehicle in vehicles:
        if vehicle["employee"]:
            tracker.add_employee_vehicle(
                vehicle["front_plate"], 
                vehicle["description"].split(" - ")[1] if " - " in vehicle["description"] else None
            )
    
    # Create sample image paths (in real scenario, these would be actual camera captures)
    sample_images = {
        "front_entry": "sample_images/front_entry.jpg",
        "rear_entry": "sample_images/rear_entry.jpg",
        "front_exit": "sample_images/front_exit.jpg", 
        "rear_exit": "sample_images/rear_exit.jpg"
    }
    
    # Create sample image directory and placeholder files
    os.makedirs("sample_images", exist_ok=True)
    for image_path in sample_images.values():
        if not os.path.exists(image_path):
            # Create placeholder image files
            with open(image_path, "w") as f:
                f.write("# Placeholder image file for demo")
    
    print("\n🚪 Processing Entry Events...")
    entry_events = []
    
    for i, vehicle in enumerate(vehicles):
        print(f"  📸 Processing entry for {vehicle['description']}")
        
        # Simulate entry event processing
        # In real scenario, these would be actual camera image paths
        entry_event = {
            "front_plate_number": vehicle["front_plate"],
            "rear_plate_number": vehicle["rear_plate"],
            "front_plate_confidence": 95.0 + i,
            "rear_plate_confidence": 93.0 + i,
            "front_plate_image_path": sample_images["front_entry"],
            "rear_plate_image_path": sample_images["rear_entry"],
            "entry_timestamp": datetime.utcnow(),
            "vehicle_color": ["blue", "red", "white"][i],
            "vehicle_make": ["Toyota", "Honda", "Ford"][i],
            "vehicle_model": ["Camry", "Civic", "Transit"][i],
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        # Save entry event directly to MongoDB
        result = tracker.db.entry_events.insert_one(entry_event)
        entry_event["_id"] = result.inserted_id
        entry_events.append(entry_event)
        
        # Show memory usage
        memory_info = psutil.virtual_memory()
        print(f"    💾 Memory: {memory_info.used / (1024**3):.2f}GB ({memory_info.percent:.1f}%)")
        
        time.sleep(1)  # Simulate time between vehicles
    
    print("\n🚪 Processing Exit Events...")
    
    # Wait a bit to simulate vehicles being inside
    time.sleep(2)
    
    for i, (vehicle, entry_event) in enumerate(zip(vehicles, entry_events)):
        print(f"  📸 Processing exit for {vehicle['description']}")
        
        # Simulate exit event processing
        exit_event = {
            "front_plate_number": vehicle["front_plate"],
            "rear_plate_number": vehicle["rear_plate"], 
            "front_plate_confidence": 94.0 + i,
            "rear_plate_confidence": 92.0 + i,
            "front_plate_image_path": sample_images["front_exit"],
            "rear_plate_image_path": sample_images["rear_exit"],
            "exit_timestamp": datetime.utcnow(),
            "vehicle_color": entry_event["vehicle_color"],
            "vehicle_make": entry_event["vehicle_make"],
            "vehicle_model": entry_event["vehicle_model"],
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        
        # Save exit event directly to MongoDB
        result = tracker.db.exit_events.insert_one(exit_event)
        exit_event["_id"] = result.inserted_id
        
        # Show memory usage
        memory_info = psutil.virtual_memory()
        print(f"    💾 Memory: {memory_info.used / (1024**3):.2f}GB ({memory_info.percent:.1f}%)")
        
        time.sleep(1)
    
    return len(vehicles)

def demonstrate_matching(tracker):
    """Demonstrate event matching and journey creation."""
    print("\n🔄 Matching Entry/Exit Events...")
    
    # Match events to create journeys
    journeys = tracker.match_entry_exit_events(batch_size=5)
    
    print(f"✅ Created {len(journeys)} vehicle journeys")
    
    # Display journey details
    for i, journey in enumerate(journeys):
        duration_minutes = journey["duration_seconds"] // 60
        duration_seconds = journey["duration_seconds"] % 60
        
        employee_status = "👨‍💼 Employee" if journey["is_employee"] else "👤 Visitor"
        review_status = "⚠️  Flagged" if journey["flagged_for_review"] else "✅ Clean"
        
        print(f"\n  🚗 Journey {i+1}:")
        print(f"    📋 Plate: {journey['front_plate_number']}")
        print(f"    👤 Type: {employee_status}")
        print(f"    ⏱️  Duration: {duration_minutes}m {duration_seconds}s")
        print(f"    🔍 Status: {review_status}")
    
    return len(journeys)

def show_system_statistics(tracker):
    """Display system statistics and performance metrics."""
    print("\n📊 SYSTEM STATISTICS")
    print("-" * 40)
    
    stats = tracker.get_system_stats()
    
    print(f"💾 Memory Usage: {stats['memory_usage_gb']:.2f}GB ({stats['memory_percent']:.1f}%)")
    print(f"📊 Total Entry Events: {stats['total_entry_events']}")
    print(f"📊 Total Exit Events: {stats['total_exit_events']}")
    print(f"📊 Total Journeys: {stats['total_journeys']}")
    print(f"⏳ Unprocessed Entries: {stats['unprocessed_entries']}")
    print(f"⏳ Unprocessed Exits: {stats['unprocessed_exits']}")
    print(f"👨‍💼 Employee Vehicles: {stats['employee_vehicles']}")
    
    # Memory efficiency calculation
    memory_efficiency = (4.0 - stats['memory_usage_gb']) / 4.0 * 100
    print(f"🎯 Memory Efficiency: {memory_efficiency:.1f}% (Target: >25%)")
    
    if stats['memory_usage_gb'] < 4.0:
        print("✅ Memory usage within 4GB target!")
    else:
        print("⚠️  Memory usage exceeds 4GB target")

def demonstrate_cleanup(tracker):
    """Demonstrate automatic cleanup functionality."""
    print("\n🧹 CLEANUP DEMONSTRATION")
    print("-" * 40)
    
    print("🗑️  Running cleanup of old unprocessed events...")
    
    # Clean up events older than 1 day (for demo purposes)
    tracker.cleanup_old_data(days=1)
    
    print("✅ Cleanup completed")

def performance_test(tracker):
    """Run a simple performance test."""
    print("\n⚡ PERFORMANCE TEST")
    print("-" * 40)
    
    print("🏃‍♂️ Testing batch processing performance...")
    
    start_time = time.time()
    
    # Create multiple test events
    test_events = []
    for i in range(20):
        event = {
            "front_plate_number": f"TEST{i:03d}",
            "rear_plate_number": f"TEST{i:03d}",
            "front_plate_confidence": 90.0,
            "rear_plate_confidence": 90.0,
            "front_plate_image_path": "test.jpg",
            "rear_plate_image_path": "test.jpg",
            "entry_timestamp": datetime.utcnow(),
            "is_processed": False,
            "created_at": datetime.utcnow()
        }
        test_events.append(event)
    
    # Batch insert
    tracker.db.entry_events.insert_many(test_events)
    
    end_time = time.time()
    processing_time = end_time - start_time
    events_per_second = len(test_events) / processing_time
    
    print(f"📊 Processed {len(test_events)} events in {processing_time:.2f}s")
    print(f"⚡ Performance: {events_per_second:.1f} events/second")
    
    # Memory usage during processing
    memory_info = psutil.virtual_memory()
    print(f"💾 Memory during processing: {memory_info.used / (1024**3):.2f}GB")

def main():
    """Main demo function."""
    try:
        # Print system information
        print_system_info()
        
        # Initialize the tracking system
        print("🚀 Initializing Raspberry Pi Vehicle Tracking System...")
        tracker = MemoryOptimizedVehicleTracker()
        print("✅ System initialized successfully!")
        
        # Show initial statistics
        show_system_statistics(tracker)
        
        # Simulate vehicle events
        vehicle_count = simulate_vehicle_events(tracker)
        
        # Demonstrate matching
        journey_count = demonstrate_matching(tracker)
        
        # Show updated statistics
        show_system_statistics(tracker)
        
        # Performance test
        performance_test(tracker)
        
        # Cleanup demonstration
        demonstrate_cleanup(tracker)
        
        # Final statistics
        print("\n📈 FINAL RESULTS")
        print("-" * 40)
        print(f"🚗 Vehicles Processed: {vehicle_count}")
        print(f"🛣️  Journeys Created: {journey_count}")
        
        final_stats = tracker.get_system_stats()
        print(f"💾 Final Memory Usage: {final_stats['memory_usage_gb']:.2f}GB")
        
        if final_stats['memory_usage_gb'] < 4.0:
            print("🎉 SUCCESS: Memory usage stayed within 4GB target!")
        
        print("\n✅ Demo completed successfully!")
        print("🔧 System is ready for production use on Raspberry Pi")
        
        # Close the system
        tracker.close()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo finished")

if __name__ == "__main__":
    main()