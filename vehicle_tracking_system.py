#!/usr/bin/env python3
"""
Vehicle Tracking System using ultimateALPR-SDK
This module implements the logic for tracking vehicle entry and exit events
using ALPR cameras as described in the requirements.
"""

import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Add the current directory to Python path to import our ALPR wrapper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from python_docker_wrapper import UltimateALPRSDK

class VehicleTrackingSystem:
    def __init__(self, db_path: str = "vehicle_tracking.db"):
        """
        Initialize the vehicle tracking system.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.sdk = UltimateALPRSDK()
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the database with the required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables using our schema
        with open("database_schema.sql", "r") as f:
            schema = f.read()
            # Split by semicolon to execute each statement separately
            statements = schema.split(";")
            for statement in statements:
                statement = statement.strip()
                if statement:
                    try:
                        cursor.execute(statement)
                    except sqlite3.Error as e:
                        print(f"Warning: {e}")
        
        conn.commit()
        conn.close()
        
    def process_entry_event(self, front_image_path: str, rear_image_path: str) -> Dict:
        """
        Process an entry event using both front and rear camera images.
        
        Args:
            front_image_path (str): Path to the front camera image
            rear_image_path (str): Path to the rear camera image
            
        Returns:
            dict: Entry event data
        """
        # Process front plate
        front_result = self.sdk.process_image(front_image_path)
        front_plates = self.sdk.get_plate_details(front_result)
        
        # Process rear plate
        rear_result = self.sdk.process_image(rear_image_path)
        rear_plates = self.sdk.get_plate_details(rear_result)
        
        # Extract best plate information
        front_plate_data = front_plates[0] if front_plates else None
        rear_plate_data = rear_plates[0] if rear_plates else None
        
        # Create entry event record
        entry_event = {
            "front_plate_number": front_plate_data["text"] if front_plate_data else None,
            "rear_plate_number": rear_plate_data["text"] if rear_plate_data else None,
            "front_plate_confidence": front_plate_data["confidence"] if front_plate_data else 0,
            "rear_plate_confidence": rear_plate_data["confidence"] if rear_plate_data else 0,
            "front_plate_image_path": front_image_path,
            "rear_plate_image_path": rear_image_path,
            "entry_timestamp": datetime.now().isoformat(),
            "vehicle_color": None,  # Would be extracted from ALPR results in a full implementation
            "vehicle_make": None,   # Would be extracted from ALPR results in a full implementation
            "vehicle_model": None,  # Would be extracted from ALPR results in a full implementation
            "is_processed": False
        }
        
        # Save to database
        self.save_entry_event(entry_event)
        
        return entry_event
        
    def process_exit_event(self, front_image_path: str, rear_image_path: str) -> Dict:
        """
        Process an exit event using both front and rear camera images.
        
        Args:
            front_image_path (str): Path to the front camera image (as car exits)
            rear_image_path (str): Path to the rear camera image (as car exits)
            
        Returns:
            dict: Exit event data
        """
        # Process front plate (car's front as it exits)
        front_result = self.sdk.process_image(front_image_path)
        front_plates = self.sdk.get_plate_details(front_result)
        
        # Process rear plate (car's rear as it exits)
        rear_result = self.sdk.process_image(rear_image_path)
        rear_plates = self.sdk.get_plate_details(rear_result)
        
        # Extract best plate information
        front_plate_data = front_plates[0] if front_plates else None
        rear_plate_data = rear_plates[0] if rear_plates else None
        
        # Create exit event record
        exit_event = {
            "front_plate_number": front_plate_data["text"] if front_plate_data else None,
            "rear_plate_number": rear_plate_data["text"] if rear_plate_data else None,
            "front_plate_confidence": front_plate_data["confidence"] if front_plate_data else 0,
            "rear_plate_confidence": rear_plate_data["confidence"] if rear_plate_data else 0,
            "front_plate_image_path": front_image_path,
            "rear_plate_image_path": rear_image_path,
            "exit_timestamp": datetime.now().isoformat(),
            "vehicle_color": None,  # Would be extracted from ALPR results in a full implementation
            "vehicle_make": None,   # Would be extracted from ALPR results in a full implementation
            "vehicle_model": None,  # Would be extracted from ALPR results in a full implementation
            "is_processed": False
        }
        
        # Save to database
        self.save_exit_event(exit_event)
        
        return exit_event
        
    def save_entry_event(self, entry_event: Dict):
        """
        Save an entry event to the database.
        
        Args:
            entry_event (dict): Entry event data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO entry_events (
                front_plate_number, rear_plate_number,
                front_plate_image_path, rear_plate_image_path,
                front_plate_confidence, rear_plate_confidence,
                entry_timestamp, vehicle_color, vehicle_make, vehicle_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry_event["front_plate_number"],
            entry_event["rear_plate_number"],
            entry_event["front_plate_image_path"],
            entry_event["rear_plate_image_path"],
            entry_event["front_plate_confidence"],
            entry_event["rear_plate_confidence"],
            entry_event["entry_timestamp"],
            entry_event["vehicle_color"],
            entry_event["vehicle_make"],
            entry_event["vehicle_model"]
        ))
        
        conn.commit()
        conn.close()
        
    def save_exit_event(self, exit_event: Dict):
        """
        Save an exit event to the database.
        
        Args:
            exit_event (dict): Exit event data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO exit_events (
                front_plate_number, rear_plate_number,
                front_plate_image_path, rear_plate_image_path,
                front_plate_confidence, rear_plate_confidence,
                exit_timestamp, vehicle_color, vehicle_make, vehicle_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exit_event["front_plate_number"],
            exit_event["rear_plate_number"],
            exit_event["front_plate_image_path"],
            exit_event["rear_plate_image_path"],
            exit_event["front_plate_confidence"],
            exit_event["rear_plate_confidence"],
            exit_event["exit_timestamp"],
            exit_event["vehicle_color"],
            exit_event["vehicle_make"],
            exit_event["vehicle_model"]
        ))
        
        conn.commit()
        conn.close()
        
    def match_entry_exit_events(self, time_window_minutes: int = 30) -> List[Dict]:
        """
        Match entry and exit events to create complete vehicle journeys.
        Uses time-window and vehicle attributes for accuracy.
        
        Args:
            time_window_minutes (int): Time window in minutes to search for matching events
            
        Returns:
            list: List of matched vehicle journeys
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get unprocessed entry events
        cursor.execute("""
            SELECT * FROM entry_events 
            WHERE is_processed = FALSE
            ORDER BY entry_timestamp
        """)
        
        entry_events = cursor.fetchall()
        matched_journeys = []
        
        for entry in entry_events:
            entry_id = entry[0]
            front_plate = entry[1]
            rear_plate = entry[2]
            entry_timestamp = datetime.fromisoformat(entry[7])
            
            # Calculate time window
            time_window_start = entry_timestamp
            time_window_end = entry_timestamp + timedelta(minutes=time_window_minutes)
            
            # Look for matching exit events
            cursor.execute("""
                SELECT * FROM exit_events 
                WHERE is_processed = FALSE
                AND exit_timestamp BETWEEN ? AND ?
                ORDER BY exit_timestamp
            """, (time_window_start.isoformat(), time_window_end.isoformat()))
            
            exit_events = cursor.fetchall()
            
            # Find the best matching exit event
            best_match = self.find_best_match(entry, exit_events)
            
            if best_match:
                exit_event = best_match
                exit_id = exit_event[0]
                exit_timestamp = datetime.fromisoformat(exit_event[7])
                
                # Calculate duration
                duration = (exit_timestamp - entry_timestamp).total_seconds()
                
                # Create journey record
                journey = {
                    "entry_event_id": entry_id,
                    "exit_event_id": exit_id,
                    "front_plate_number": front_plate,
                    "rear_plate_number": rear_plate,
                    "entry_timestamp": entry_timestamp.isoformat(),
                    "exit_timestamp": exit_timestamp.isoformat(),
                    "duration_seconds": int(duration),
                    "vehicle_color": entry[8],  # From entry event
                    "vehicle_make": entry[9],    # From entry event
                    "vehicle_model": entry[10],  # From entry event
                    "is_employee": self.is_employee_vehicle(front_plate, rear_plate),
                    "flagged_for_review": self.check_for_anomalies(entry, exit_event)
                }
                
                # Save journey
                self.save_vehicle_journey(journey)
                
                # Mark events as processed
                self.mark_event_as_processed("entry", entry_id)
                self.mark_event_as_processed("exit", exit_id)
                
                matched_journeys.append(journey)
                
        conn.close()
        return matched_journeys
        
    def find_best_match(self, entry_event, exit_events) -> Optional[Tuple]:
        """
        Find the best matching exit event for an entry event.
        
        Args:
            entry_event (tuple): Entry event data from database
            exit_events (list): List of potential exit events
            
        Returns:
            tuple: Best matching exit event or None
        """
        if not exit_events:
            return None
            
        # For simplicity, we'll match the first exit event with matching plates
        # In a real implementation, this would use more sophisticated matching
        entry_front_plate = entry_event[1]
        entry_rear_plate = entry_event[2]
        
        for exit_event in exit_events:
            exit_front_plate = exit_event[1]
            exit_rear_plate = exit_event[2]
            
            # Check if plates match (allowing for potential OCR errors)
            if self.plates_match(entry_front_plate, exit_front_plate) and \
               self.plates_match(entry_rear_plate, exit_rear_plate):
                return exit_event
                
        return None
        
    def plates_match(self, plate1: str, plate2: str, threshold: float = 0.8) -> bool:
        """
        Check if two plate numbers match (accounting for OCR errors).
        
        Args:
            plate1 (str): First plate number
            plate2 (str): Second plate number
            threshold (float): Similarity threshold (0.0 to 1.0)
            
        Returns:
            bool: True if plates match
        """
        if not plate1 or not plate2:
            return False
            
        # Simple character-by-character comparison
        if plate1 == plate2:
            return True
            
        # Calculate similarity ratio
        max_len = max(len(plate1), len(plate2))
        if max_len == 0:
            return True
            
        matches = sum(c1 == c2 for c1, c2 in zip(plate1, plate2))
        similarity = matches / max_len
        
        return similarity >= threshold
        
    def is_employee_vehicle(self, front_plate: str, rear_plate: str) -> bool:
        """
        Check if a vehicle is an employee vehicle.
        
        Args:
            front_plate (str): Front plate number
            rear_plate (str): Rear plate number
            
        Returns:
            bool: True if vehicle is an employee vehicle
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if either plate is in the employee vehicles table
        cursor.execute("""
            SELECT COUNT(*) FROM vehicles 
            WHERE plate_number IN (?, ?) AND is_employee = TRUE
        """, (front_plate, rear_plate))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
        
    def check_for_anomalies(self, entry_event, exit_event) -> bool:
        """
        Check for inconsistencies or anomalies in the entry/exit events.
        
        Args:
            entry_event (tuple): Entry event data
            exit_event (tuple): Exit event data
            
        Returns:
            bool: True if anomalies are detected
        """
        # Check for missing plate data
        if not entry_event[1] or not entry_event[2] or not exit_event[1] or not exit_event[2]:
            return True
            
        # Check for mismatched plates
        if not self.plates_match(entry_event[1], exit_event[1]) or \
           not self.plates_match(entry_event[2], exit_event[2]):
            return True
            
        # Check for low confidence readings
        if entry_event[5] < 80.0 or entry_event[6] < 80.0 or \
           exit_event[5] < 80.0 or exit_event[6] < 80.0:
            return True
            
        return False
        
    def save_vehicle_journey(self, journey: Dict):
        """
        Save a vehicle journey to the database.
        
        Args:
            journey (dict): Vehicle journey data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO vehicle_journeys (
                entry_event_id, exit_event_id,
                front_plate_number, rear_plate_number,
                entry_timestamp, exit_timestamp,
                duration_seconds, vehicle_color,
                vehicle_make, vehicle_model,
                is_employee, flagged_for_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            journey["entry_event_id"],
            journey["exit_event_id"],
            journey["front_plate_number"],
            journey["rear_plate_number"],
            journey["entry_timestamp"],
            journey["exit_timestamp"],
            journey["duration_seconds"],
            journey["vehicle_color"],
            journey["vehicle_make"],
            journey["vehicle_model"],
            journey["is_employee"],
            journey["flagged_for_review"]
        ))
        
        conn.commit()
        conn.close()
        
    def mark_event_as_processed(self, event_type: str, event_id: int):
        """
        Mark an event as processed.
        
        Args:
            event_type (str): Type of event ('entry' or 'exit')
            event_id (int): Event ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if event_type == "entry":
            cursor.execute("UPDATE entry_events SET is_processed = TRUE WHERE id = ?", (event_id,))
        elif event_type == "exit":
            cursor.execute("UPDATE exit_events SET is_processed = TRUE WHERE id = ?", (event_id,))
            
        conn.commit()
        conn.close()
        
    def get_unmatched_entries(self) -> List[Dict]:
        """
        Get entry events that haven't been matched with exit events.
        
        Returns:
            list: List of unmatched entry events
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM entry_events 
            WHERE is_processed = FALSE
            ORDER BY entry_timestamp
        """)
        
        entries = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        result = []
        for entry in entries:
            result.append({
                "id": entry[0],
                "front_plate_number": entry[1],
                "rear_plate_number": entry[2],
                "front_plate_image_path": entry[3],
                "rear_plate_image_path": entry[4],
                "front_plate_confidence": entry[5],
                "rear_plate_confidence": entry[6],
                "entry_timestamp": entry[7],
                "vehicle_color": entry[8],
                "vehicle_make": entry[9],
                "vehicle_model": entry[10]
            })
            
        return result
        
    def get_unmatched_exits(self) -> List[Dict]:
        """
        Get exit events that haven't been matched with entry events.
        
        Returns:
            list: List of unmatched exit events
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM exit_events 
            WHERE is_processed = FALSE
            ORDER BY exit_timestamp
        """)
        
        exits = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        result = []
        for exit_event in exits:
            result.append({
                "id": exit_event[0],
                "front_plate_number": exit_event[1],
                "rear_plate_number": exit_event[2],
                "front_plate_image_path": exit_event[3],
                "rear_plate_image_path": exit_event[4],
                "front_plate_confidence": exit_event[5],
                "rear_plate_confidence": exit_event[6],
                "exit_timestamp": exit_event[7],
                "vehicle_color": exit_event[8],
                "vehicle_make": exit_event[9],
                "vehicle_model": exit_event[10]
            })
            
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the tracking system
    tracker = VehicleTrackingSystem()
    
    print("Vehicle Tracking System initialized.")
    print("Database schema created.")
    print("\nSystem is ready to process vehicle entry/exit events.")
    print("\nTo process an entry event:")
    print("  tracker.process_entry_event('front_image.jpg', 'rear_image.jpg')")
    print("\nTo process an exit event:")
    print("  tracker.process_exit_event('front_image.jpg', 'rear_image.jpg')")
    print("\nTo match events and create journeys:")
    print("  journeys = tracker.match_entry_exit_events()")