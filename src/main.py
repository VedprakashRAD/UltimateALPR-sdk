#!/usr/bin/env python3
"""
Main entry point for the UltimateALPR-SDK Vehicle Tracking System
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.fully_automatic_system import FullyAutomaticVehicleSystem

def main():
    """Main function to run the vehicle tracking system"""
    print("🚀 UltimateALPR-SDK Vehicle Tracking System")
    print("==========================================")
    
    try:
        # Initialize and run the fully automatic system
        system = FullyAutomaticVehicleSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())