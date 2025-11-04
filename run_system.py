#!/usr/bin/env python3
"""
Runner script to start different vehicle tracking systems.
"""

import sys
import os
import argparse

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'camera'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'ui'))

def main():
    """Main function to run the selected system."""
    parser = argparse.ArgumentParser(description="Vehicle Tracking System Runner")
    parser.add_argument(
        "system",
        choices=["auto", "fully", "working", "dashboard", "web", "database"],
        help="Which system to run: auto (auto-detect), fully (fully automatic), working (ALPR), dashboard (show stats), web (web dashboard), database (show database)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Vehicle Tracking System Runner")
    print("=" * 40)
    
    try:
        if args.system == "auto":
            print("Starting Auto-Detection System...")
            from src.camera.auto_detect_system import AutoDetectVehicleTracker
            system = AutoDetectVehicleTracker()
            system.run()
            
        elif args.system == "fully":
            print("Starting Fully Automatic System...")
            from src.camera.fully_automatic_system import FullyAutomaticVehicleSystem
            system = FullyAutomaticVehicleSystem()
            system.run()
            
        elif args.system == "working":
            print("Starting Working ALPR System...")
            from src.camera.working_alpr_system import WorkingALPRSystem
            system = WorkingALPRSystem()
            system.run()
            
        elif args.system == "dashboard":
            print("Showing Dashboard...")
            from src.utils.dashboard import show_dashboard
            show_dashboard()
            
        elif args.system == "web":
            print("Starting Web Dashboard...")
            print("📱 Access the dashboard at: http://localhost:5000")
            print("🔌 Press Ctrl+C to stop the server")
            from src.ui.web_dashboard import main as web_dashboard
            web_dashboard()
            
        elif args.system == "database":
            print("Showing Database Contents...")
            from src.utils.show_database import main as show_database
            show_database()
            
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ Error running system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())