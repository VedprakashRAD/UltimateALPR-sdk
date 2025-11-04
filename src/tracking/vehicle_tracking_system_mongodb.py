#!/usr/bin/env python3
"""
Raspberry Pi Optimized Vehicle Tracking System using ultimateALPR-SDK with MongoDB
Memory optimized for 4GB usage on 8GB Raspberry Pi systems.
"""

import os
import sys
import time
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import psutil
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure
import threading
import queue

# Add the database directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from src.core.python_docker_wrapper import UltimateALPRSDK

class MemoryOptimizedVehicleTracker:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "vehicle_tracking"):
        """
        Initialize memory-optimized vehicle tracking system for Raspberry Pi.
        
        Args:
            mongo_uri (str): MongoDB connection URI
            db_name (str): Database name
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.sdk = None
        self.processing_queue = queue.Queue(maxsize=50)  # Limit queue size
        self.memory_threshold = 3.5 * 1024 * 1024 * 1024  # 3.5GB limit
        
        self._init_mongodb()
        self._init_alpr_sdk()
        self._setup_collections()
        
    def _init_mongodb(self):
        """Initialize MongoDB connection with optimized settings for Raspberry Pi."""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                maxPoolSize=5,  # Limit connection pool
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=20000
            )
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ping')
            print("MongoDB connection established")
        except ConnectionFailure as e:
            print(f"MongoDB connection failed: {e}")
            raise
            
    def _init_alpr_sdk(self):
        """Initialize ALPR SDK with memory constraints."""
        try:
            self.sdk = UltimateALPRSDK()
            print("ALPR SDK initialized")
        except Exception as e:
            print(f"ALPR SDK initialization failed: {e}")
            raise
            
    def _setup_collections(self):
        """Setup MongoDB collections with indexes for performance."""
        # Entry events collection
        self.db.entry_events.create_index([("entry_timestamp", DESCENDING)])
        self.db.entry_events.create_index([("is_processed", ASCENDING)])
        self.db.entry_events.create_index([("front_plate_number", ASCENDING)])
        
        # Exit events collection
        self.db.exit_events.create_index([("exit_timestamp", DESCENDING)])
        self.db.exit_events.create_index([("is_processed", ASCENDING)])
        self.db.exit_events.create_index([("front_plate_number", ASCENDING)])
        
        # Vehicle journeys collection
        self.db.vehicle_journeys.create_index([("entry_timestamp", DESCENDING)])
        self.db.vehicle_journeys.create_index([("is_employee", ASCENDING)])
        
        # Employee vehicles collection
        self.db.employee_vehicles.create_index([("plate_number", ASCENDING)], unique=True)
        
    def _check_memory_usage(self):
        """Monitor memory usage and trigger cleanup if needed."""
        memory_info = psutil.virtual_memory()
        if memory_info.used > self.memory_threshold:
            print(f"Memory usage high: {memory_info.used / (1024**3):.2f}GB, triggering cleanup")
            gc.collect()
            return True
        return False
        
    def process_entry_event(self, front_image_path: str, rear_image_path: str) -> Dict:
        """
        Process entry event with memory optimization.
        
        Args:
            front_image_path (str): Path to front camera image
            rear_image_path (str): Path to rear camera image
            
        Returns:
            dict: Entry event data
        """
        try:
            # Check memory before processing
            self._check_memory_usage()
            
            # Process images with memory cleanup
            front_result = self.sdk.process_image(front_image_path)
            front_plates = self.sdk.get_plate_details(front_result)
            
            # Clear intermediate results
            del front_result
            gc.collect()
            
            rear_result = self.sdk.process_image(rear_image_path)
            rear_plates = self.sdk.get_plate_details(rear_result)
            
            # Clear intermediate results
            del rear_result
            gc.collect()
            
            # Extract plate data
            front_plate_data = front_plates[0] if front_plates else None
            rear_plate_data = rear_plates[0] if rear_plates else None
            
            entry_event = {
                "front_plate_number": front_plate_data["text"] if front_plate_data else None,
                "rear_plate_number": rear_plate_data["text"] if rear_plate_data else None,
                "front_plate_confidence": front_plate_data["confidence"] if front_plate_data else 0,
                "rear_plate_confidence": rear_plate_data["confidence"] if rear_plate_data else 0,
                "front_plate_image_path": front_image_path,
                "rear_plate_image_path": rear_image_path,
                "entry_timestamp": datetime.utcnow(),
                "vehicle_color": None,
                "vehicle_make": None,
                "vehicle_model": None,
                "is_processed": False,
                "created_at": datetime.utcnow()
            }
            
            # Save to MongoDB
            result = self.db.entry_events.insert_one(entry_event)
            entry_event["_id"] = result.inserted_id
            
            # Clear variables
            del front_plates, rear_plates, front_plate_data, rear_plate_data
            gc.collect()
            
            return entry_event
            
        except Exception as e:
            print(f"Error processing entry event: {e}")
            return None
            
    def process_exit_event(self, front_image_path: str, rear_image_path: str) -> Dict:
        """
        Process exit event with memory optimization.
        
        Args:
            front_image_path (str): Path to front camera image
            rear_image_path (str): Path to rear camera image
            
        Returns:
            dict: Exit event data
        """
        try:
            # Check memory before processing
            self._check_memory_usage()
            
            # Process images with memory cleanup
            front_result = self.sdk.process_image(front_image_path)
            front_plates = self.sdk.get_plate_details(front_result)
            
            del front_result
            gc.collect()
            
            rear_result = self.sdk.process_image(rear_image_path)
            rear_plates = self.sdk.get_plate_details(rear_result)
            
            del rear_result
            gc.collect()
            
            # Extract plate data
            front_plate_data = front_plates[0] if front_plates else None
            rear_plate_data = rear_plates[0] if rear_plates else None
            
            exit_event = {
                "front_plate_number": front_plate_data["text"] if front_plate_data else None,
                "rear_plate_number": rear_plate_data["text"] if rear_plate_data else None,
                "front_plate_confidence": front_plate_data["confidence"] if front_plate_data else 0,
                "rear_plate_confidence": rear_plate_data["confidence"] if rear_plate_data else 0,
                "front_plate_image_path": front_image_path,
                "rear_plate_image_path": rear_image_path,
                "exit_timestamp": datetime.utcnow(),
                "vehicle_color": None,
                "vehicle_make": None,
                "vehicle_model": None,
                "is_processed": False,
                "created_at": datetime.utcnow()
            }
            
            # Save to MongoDB
            result = self.db.exit_events.insert_one(exit_event)
            exit_event["_id"] = result.inserted_id
            
            # Clear variables
            del front_plates, rear_plates, front_plate_data, rear_plate_data
            gc.collect()
            
            return exit_event
            
        except Exception as e:
            print(f"Error processing exit event: {e}")
            return None
            
    def match_entry_exit_events(self, time_window_minutes: int = 30, batch_size: int = 10) -> List[Dict]:
        """
        Match entry and exit events with memory optimization.
        
        Args:
            time_window_minutes (int): Time window for matching
            batch_size (int): Process in small batches to save memory
            
        Returns:
            list: Matched vehicle journeys
        """
        matched_journeys = []
        
        try:
            # Process in batches to manage memory
            skip = 0
            while True:
                # Get unprocessed entry events in batches
                entry_events = list(self.db.entry_events.find(
                    {"is_processed": False}
                ).sort("entry_timestamp", 1).skip(skip).limit(batch_size))
                
                if not entry_events:
                    break
                    
                for entry in entry_events:
                    entry_timestamp = entry["entry_timestamp"]
                    time_window_start = entry_timestamp
                    time_window_end = entry_timestamp + timedelta(minutes=time_window_minutes)
                    
                    # Find matching exit events
                    exit_events = list(self.db.exit_events.find({
                        "is_processed": False,
                        "exit_timestamp": {
                            "$gte": time_window_start,
                            "$lte": time_window_end
                        }
                    }).sort("exit_timestamp", 1).limit(5))  # Limit to save memory
                    
                    # Find best match
                    best_match = self._find_best_match(entry, exit_events)
                    
                    if best_match:
                        journey = self._create_journey(entry, best_match)
                        if journey:
                            matched_journeys.append(journey)
                            
                            # Mark as processed
                            self.db.entry_events.update_one(
                                {"_id": entry["_id"]},
                                {"$set": {"is_processed": True}}
                            )
                            self.db.exit_events.update_one(
                                {"_id": best_match["_id"]},
                                {"$set": {"is_processed": True}}
                            )
                
                skip += batch_size
                
                # Memory cleanup after each batch
                gc.collect()
                
                # Check if we should stop due to memory constraints
                if self._check_memory_usage():
                    break
                    
        except Exception as e:
            print(f"Error matching events: {e}")
            
        return matched_journeys
        
    def _find_best_match(self, entry_event: Dict, exit_events: List[Dict]) -> Optional[Dict]:
        """Find best matching exit event for entry event."""
        if not exit_events:
            return None
            
        entry_front = entry_event.get("front_plate_number")
        entry_rear = entry_event.get("rear_plate_number")
        
        for exit_event in exit_events:
            exit_front = exit_event.get("front_plate_number")
            exit_rear = exit_event.get("rear_plate_number")
            
            if self._plates_match(entry_front, exit_front) and \
               self._plates_match(entry_rear, exit_rear):
                return exit_event
                
        return None
        
    def _plates_match(self, plate1: str, plate2: str, threshold: float = 0.8) -> bool:
        """Check if plates match with similarity threshold."""
        if not plate1 or not plate2:
            return False
            
        if plate1 == plate2:
            return True
            
        # Simple similarity check
        max_len = max(len(plate1), len(plate2))
        if max_len == 0:
            return True
            
        matches = sum(c1 == c2 for c1, c2 in zip(plate1, plate2))
        similarity = matches / max_len
        
        return similarity >= threshold
        
    def _create_journey(self, entry_event: Dict, exit_event: Dict) -> Optional[Dict]:
        """Create vehicle journey from matched events."""
        try:
            entry_time = entry_event["entry_timestamp"]
            exit_time = exit_event["exit_timestamp"]
            duration = (exit_time - entry_time).total_seconds()
            
            journey = {
                "entry_event_id": entry_event["_id"],
                "exit_event_id": exit_event["_id"],
                "front_plate_number": entry_event.get("front_plate_number"),
                "rear_plate_number": entry_event.get("rear_plate_number"),
                "entry_timestamp": entry_time,
                "exit_timestamp": exit_time,
                "duration_seconds": int(duration),
                "vehicle_color": entry_event.get("vehicle_color"),
                "vehicle_make": entry_event.get("vehicle_make"),
                "vehicle_model": entry_event.get("vehicle_model"),
                "is_employee": self._is_employee_vehicle(
                    entry_event.get("front_plate_number"),
                    entry_event.get("rear_plate_number")
                ),
                "flagged_for_review": self._check_anomalies(entry_event, exit_event),
                "created_at": datetime.utcnow()
            }
            
            # Save journey
            result = self.db.vehicle_journeys.insert_one(journey)
            journey["_id"] = result.inserted_id
            
            return journey
            
        except Exception as e:
            print(f"Error creating journey: {e}")
            return None
            
    def _is_employee_vehicle(self, front_plate: str, rear_plate: str) -> bool:
        """Check if vehicle is employee vehicle."""
        if not front_plate and not rear_plate:
            return False
            
        count = self.db.employee_vehicles.count_documents({
            "plate_number": {"$in": [front_plate, rear_plate]},
            "is_active": True
        })
        
        return count > 0
        
    def _check_anomalies(self, entry_event: Dict, exit_event: Dict) -> bool:
        """Check for anomalies in matched events."""
        # Missing plate data
        if not entry_event.get("front_plate_number") or not entry_event.get("rear_plate_number"):
            return True
            
        # Low confidence
        if (entry_event.get("front_plate_confidence", 0) < 80 or
            entry_event.get("rear_plate_confidence", 0) < 80 or
            exit_event.get("front_plate_confidence", 0) < 80 or
            exit_event.get("rear_plate_confidence", 0) < 80):
            return True
            
        return False
        
    def add_employee_vehicle(self, plate_number: str, employee_name: str = None):
        """Add employee vehicle to database."""
        try:
            employee_vehicle = {
                "plate_number": plate_number,
                "employee_name": employee_name,
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            
            self.db.employee_vehicles.insert_one(employee_vehicle)
            print(f"Added employee vehicle: {plate_number}")
            
        except Exception as e:
            print(f"Error adding employee vehicle: {e}")
            
    def get_recent_journeys(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent vehicle journeys."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        journeys = list(self.db.vehicle_journeys.find({
            "entry_timestamp": {"$gte": since}
        }).sort("entry_timestamp", -1).limit(limit))
        
        return journeys
        
    def cleanup_old_data(self, days: int = 30):
        """Clean up old unprocessed events to save space."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Remove old unprocessed events
        entry_result = self.db.entry_events.delete_many({
            "is_processed": False,
            "created_at": {"$lt": cutoff_date}
        })
        
        exit_result = self.db.exit_events.delete_many({
            "is_processed": False,
            "created_at": {"$lt": cutoff_date}
        })
        
        print(f"Cleaned up {entry_result.deleted_count} entry events and {exit_result.deleted_count} exit events")
        
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        memory_info = psutil.virtual_memory()
        
        stats = {
            "memory_usage_gb": memory_info.used / (1024**3),
            "memory_percent": memory_info.percent,
            "total_entry_events": self.db.entry_events.count_documents({}),
            "total_exit_events": self.db.exit_events.count_documents({}),
            "total_journeys": self.db.vehicle_journeys.count_documents({}),
            "unprocessed_entries": self.db.entry_events.count_documents({"is_processed": False}),
            "unprocessed_exits": self.db.exit_events.count_documents({"is_processed": False}),
            "employee_vehicles": self.db.employee_vehicles.count_documents({"is_active": True})
        }
        
        return stats
        
    def close(self):
        """Close connections and cleanup."""
        if self.client:
            self.client.close()
        print("Vehicle tracking system closed")

# Example usage for Raspberry Pi
if __name__ == "__main__":
    # Initialize system
    tracker = MemoryOptimizedVehicleTracker()
    
    print("Raspberry Pi Vehicle Tracking System initialized")
    print("Memory optimized for 4GB usage on 8GB systems")
    
    # Show system stats
    stats = tracker.get_system_stats()
    print(f"Current memory usage: {stats['memory_usage_gb']:.2f}GB ({stats['memory_percent']:.1f}%)")
    
    # Example: Add some employee vehicles
    tracker.add_employee_vehicle("ABC123", "John Doe")
    tracker.add_employee_vehicle("XYZ789", "Jane Smith")
    
    print("\nSystem ready for processing vehicle events")
    print("Use tracker.process_entry_event() and tracker.process_exit_event()")