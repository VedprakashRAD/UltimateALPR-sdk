#!/usr/bin/env python3
"""
Simple dashboard to display vehicle tracking statistics.
"""

import sys
import os
from datetime import datetime, timedelta

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))

from tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker

def show_dashboard():
    """Show a dashboard with vehicle tracking statistics."""
    print("📊 Vehicle Tracking Dashboard")
    print("=" * 40)
    
    try:
        # Initialize the tracker
        tracker = MemoryOptimizedVehicleTracker()
        print("✅ Connected to MongoDB database")
        
        # Get statistics
        stats = tracker.get_system_stats()
        
        # Show overall statistics
        print(f"\n📈 Overall Statistics:")
        print(f"  Total Entry Events: {stats['total_entry_events']}")
        print(f"  Total Exit Events: {stats['total_exit_events']}")
        print(f"  Total Vehicle Journeys: {stats['total_journeys']}")
        print(f"  Registered Employee Vehicles: {stats['employee_vehicles']}")
        print(f"  Unprocessed Entries: {stats['unprocessed_entries']}")
        print(f"  Unprocessed Exits: {stats['unprocessed_exits']}")
        
        # Show recent activity (last 24 hours)
        print(f"\n⏱️  Recent Activity (Last 24 Hours):")
        recent_journeys = tracker.get_recent_journeys(hours=24)
        print(f"  Journeys in last 24h: {len(recent_journeys)}")
        
        if recent_journeys:
            print(f"\n🚗 Recent Vehicle Journeys:")
            print("-" * 50)
            for journey in recent_journeys[:10]:  # Show first 10
                plate = journey.get('front_plate_number', 'Unknown')
                entry_time = journey.get('entry_timestamp', datetime.min)
                exit_time = journey.get('exit_timestamp', datetime.min)
                duration = journey.get('duration_seconds', 0)
                is_employee = journey.get('is_employee', False)
                
                print(f"  Plate: {plate}{' (Employee)' if is_employee else ''}")
                print(f"    Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Exit:  {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Duration: {duration//60} min {duration%60} sec")
                print()
        
        # Show employee vehicles
        print(f"👨‍💼 Employee Vehicles:")
        print("-" * 20)
        employee_vehicles = list(tracker.db.employee_vehicles.find({"is_active": True}))
        if employee_vehicles:
            for vehicle in employee_vehicles:
                print(f"  {vehicle.get('plate_number', 'Unknown')} - {vehicle.get('employee_name', 'Unknown')}")
        else:
            print("  No employee vehicles registered")
        
        # Close connection
        tracker.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

def main():
    """Main function."""
    return show_dashboard()

if __name__ == "__main__":
    sys.exit(main())