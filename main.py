#!/usr/bin/env python3
"""
Main application for the Vehicle Tracking System
This single file runs everything: database, web dashboard, and camera feeds
"""

import sys
import os
import threading
import time
from datetime import datetime
import json

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'camera'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'tracking'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'ui'))

# Flask for web interface
from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np

# Database imports
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Import our vehicle tracking system
try:
    from src.camera.working_alpr_system import WorkingALPRSystem
    ALPR_SYSTEM_AVAILABLE = True
except ImportError:
    ALPR_SYSTEM_AVAILABLE = False
    print("⚠️  ALPR system not available")

try:
    from src.tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker
    TRACKING_SYSTEM_AVAILABLE = True
except ImportError:
    TRACKING_SYSTEM_AVAILABLE = False
    print("⚠️  Vehicle tracking system not available")

# Global variables for camera feeds and tracking
camera1_feed = None
camera2_feed = None
camera1_lock = threading.Lock()
camera2_lock = threading.Lock()
vehicle_tracker = None

# Flask app
app = Flask(__name__, template_folder='src/ui/templates', static_folder='src/ui/static')

# Database configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "vehicle_tracking"

# Global database client
db_client = None
db_instance = None

# Global ALPR system instances and tracking variables
alpr_system_camera1 = None
alpr_system_camera2 = None
last_detection_time = {}  # Track last detection time per camera

def initialize_database():
    """Initialize persistent database connection."""
    global db_client, db_instance
    try:
        db_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000
        )
        db_instance = db_client[DB_NAME]
        # Test connection
        db_client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {MONGO_URI}/{DB_NAME}")
        return True
    except ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False

def get_db():
    """Get database instance."""
    global db_instance
    return db_instance

def get_system_stats(db):
    """Get system statistics from database."""
    try:
        # Import psutil only when needed
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            memory_usage_gb = memory_info.used / (1024**3)
            memory_percent = memory_info.percent
        except ImportError:
            memory_usage_gb = 0
            memory_percent = 0
            
        stats = {
            "memory_usage_gb": round(memory_usage_gb, 2),
            "memory_percent": round(memory_percent, 1),
            "total_entry_events": db.entry_events.count_documents({}),
            "total_exit_events": db.exit_events.count_documents({}),
            "total_journeys": db.vehicle_journeys.count_documents({}),
            "unprocessed_entries": db.entry_events.count_documents({"is_processed": False}),
            "unprocessed_exits": db.exit_events.count_documents({"is_processed": False}),
            "employee_vehicles": db.employee_vehicles.count_documents({"is_active": True})
        }
        return stats
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {}

def get_recent_journeys(db, hours=24):
    """Get recent vehicle journeys."""
    try:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        journeys = list(db.vehicle_journeys.find({
            "entry_timestamp": {"$gte": since}
        }).sort("entry_timestamp", -1).limit(10))
        
        return journeys
    except Exception as e:
        print(f"Error getting journeys: {e}")
        return []

def get_recent_entry_events(db, hours=24):
    """Get recent entry events."""
    try:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        entry_events = list(db.entry_events.find({
            "entry_timestamp": {"$gte": since}
        }).sort("entry_timestamp", -1).limit(10))
        
        print(f"Found {len(entry_events)} entry events since {since}")
        return entry_events
    except Exception as e:
        print(f"Error getting entry events: {e}")
        return []

def get_recent_exit_events(db, hours=24):
    """Get recent exit events."""
    try:
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        exit_events = list(db.exit_events.find({
            "exit_timestamp": {"$gte": since}
        }).sort("exit_timestamp", -1).limit(10))
        
        print(f"Found {len(exit_events)} exit events since {since}")
        return exit_events
    except Exception as e:
        print(f"Error getting exit events: {e}")
        return []

def initialize_alpr_systems():
    """Initialize ALPR systems for both cameras."""
    global alpr_system_camera1, alpr_system_camera2, last_detection_time
    
    if not ALPR_SYSTEM_AVAILABLE:
        print("⚠️  ALPR system not available")
        return False
    
    try:
        # Initialize ALPR systems for both cameras
        alpr_system_camera1 = WorkingALPRSystem()
        alpr_system_camera2 = WorkingALPRSystem()
        
        # Initialize last detection times
        last_detection_time[0] = 0
        last_detection_time[1] = 0
        
        print("✅ ALPR systems initialized")
        return True
    except Exception as e:
        print(f"❌ ALPR system initialization failed: {e}")
        return False

def process_frame_with_alpr(frame, camera_id):
    """Process a frame with ALPR and return annotated frame."""
    global alpr_system_camera1, alpr_system_camera2, last_detection_time
    
    # Select the appropriate ALPR system
    alpr_system = alpr_system_camera1 if camera_id == 0 else alpr_system_camera2
    
    if not alpr_system or not ALPR_SYSTEM_AVAILABLE:
        return frame
    
    try:
        # Process frame for license plates
        plate_results = alpr_system.process_frame(frame.copy())
        
        # Draw detections on frame
        for result in plate_results:
            x, y, w, h = result['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{result['text']} ({result['confidence']:.0f}%)", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save to database if we have valid results and cooldown period has passed
        if plate_results:
            # Filter for high confidence results (100% accuracy requirement)
            valid_results = [r for r in plate_results if r['confidence'] > 70 and len(r['text']) >= 5]
            
            if valid_results:
                current_time = time.time()
                if current_time - last_detection_time.get(camera_id, 0) > 3:
                    last_detection_time[camera_id] = current_time
                    # Save vehicle event to database automatically
                    event_type = "entry" if camera_id == 0 else "exit"
                    try:
                        alpr_system.save_vehicle_event(valid_results, event_type)
                        print(f"🎯 Auto ALPR Detection - Camera {camera_id+1}: {valid_results[0]['text']} ({valid_results[0]['confidence']:.0f}%) - SAVED TO DB")
                    except Exception as e:
                        print(f"❌ Failed to save to DB: {e}")
                        # Try manual database save
                        try:
                            db = get_db()
                            if db:
                                plate_text = valid_results[0]['text']
                                entry_event = {
                                    "front_plate_number": plate_text,
                                    "rear_plate_number": plate_text,
                                    "front_plate_confidence": valid_results[0]['confidence'],
                                    "rear_plate_confidence": valid_results[0]['confidence'],
                                    "entry_timestamp" if event_type == "entry" else "exit_timestamp": datetime.now(timezone.utc),
                                    "vehicle_color": "Unknown",
                                    "vehicle_make": "Unknown",
                                    "vehicle_model": "Unknown",
                                    "is_processed": False,
                                    "created_at": datetime.now(timezone.utc)
                                }
                                collection = db.entry_events if event_type == "entry" else db.exit_events
                                collection.insert_one(entry_event)
                                print(f"✅ Manual DB save successful: {plate_text}")
                        except Exception as e2:
                            print(f"❌ Manual DB save failed: {e2}")
        
        return frame
    except Exception as e:
        print(f"ALPR processing error for camera {camera_id}: {e}")
        return frame

def generate_camera_feed(camera_id):
    """Generate camera feed for streaming with ALPR processing."""
    global alpr_system_camera1, alpr_system_camera2
    
    # Always use front camera (index 0) for both feeds
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # If camera not available, create a test pattern
        frame_count = 0
        while True:
            # Create a test image with text
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            text = f"CAMERA {camera_id+1} - NO CAMERA"
            cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', img)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
        return
    
    # Camera is available - process with ALPR
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process every 5th frame to reduce CPU usage
        if frame_count % 5 == 0:
            frame = process_frame_with_alpr(frame.copy(), camera_id)
        
        # Add timestamp and camera info
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Different labels for each camera
        if camera_id == 0:
            cv2.putText(frame, "CAMERA 1 - ENTRY (ALPR ACTIVE)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "CAMERA 2 - EXIT (ALPR ACTIVE)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# Removed simulated camera feed - both cameras now use the same physical camera

# Flask routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('main_dashboard.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics."""
    try:
        db = get_db()
        if db is None:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        stats = get_system_stats(db)
        recent_journeys = get_recent_journeys(db)
        recent_entry_events = get_recent_entry_events(db)
        recent_exit_events = get_recent_exit_events(db)
        
        # Combine entry and exit events for recent activity
        recent_activity = []
        
        # Add entry events
        for event in recent_entry_events:
            entry_time = event.get('entry_timestamp')
            recent_activity.append({
                'front_plate_number': event.get('front_plate_number', 'Unknown'),
                'rear_plate_number': event.get('rear_plate_number', 'Unknown'),
                'entry_timestamp': entry_time.isoformat() if entry_time else None,
                'exit_timestamp': None,
                'is_employee': False,  # Check employee status
                'vehicle_make': event.get('vehicle_make', 'Unknown'),
                'vehicle_model': event.get('vehicle_model', 'Unknown'),
                'vehicle_color': event.get('vehicle_color', 'Unknown')
            })
        
        # Add completed journeys
        for journey in recent_journeys:
            entry_time = journey.get('entry_timestamp')
            exit_time = journey.get('exit_timestamp')
            
            recent_activity.append({
                'front_plate_number': journey.get('front_plate_number', 'Unknown'),
                'rear_plate_number': journey.get('rear_plate_number', 'Unknown'),
                'entry_timestamp': entry_time.isoformat() if entry_time else None,
                'exit_timestamp': exit_time.isoformat() if exit_time else None,
                'is_employee': journey.get('is_employee', False),
                'vehicle_make': journey.get('vehicle_make', 'Unknown'),
                'vehicle_model': journey.get('vehicle_model', 'Unknown'),
                'vehicle_color': journey.get('vehicle_color', 'Unknown')
            })
        
        # Sort by entry time (most recent first)
        recent_activity.sort(key=lambda x: x.get('entry_timestamp', ''), reverse=True)
        recent_activity = recent_activity[:10]  # Limit to 10 most recent
        
        # Format journeys for backward compatibility
        formatted_journeys = []
        for journey in recent_journeys:
            entry_time = journey.get('entry_timestamp')
            exit_time = journey.get('exit_timestamp')
            
            formatted_journeys.append({
                'plate': journey.get('front_plate_number', 'Unknown'),
                'entry_time': entry_time.isoformat() if entry_time else datetime.min.isoformat(),
                'exit_time': exit_time.isoformat() if exit_time else datetime.min.isoformat(),
                'duration_seconds': journey.get('duration_seconds', 0),
                'is_employee': journey.get('is_employee', False),
                'vehicle_make': journey.get('vehicle_make', 'Unknown'),
                'vehicle_model': journey.get('vehicle_model', 'Unknown'),
                'vehicle_color': journey.get('vehicle_color', 'Unknown')
            })
        
        # Format entry events for JSON
        formatted_entry_events = []
        for event in recent_entry_events:
            entry_time = event.get('entry_timestamp')
            formatted_entry_events.append({
                'plate': event.get('front_plate_number', 'Unknown'),
                'entry_time': entry_time.isoformat() if entry_time else datetime.now().isoformat(),
                'vehicle_make': event.get('vehicle_make', 'Unknown'),
                'vehicle_model': event.get('vehicle_model', 'Unknown'),
                'vehicle_color': event.get('vehicle_color', 'Unknown')
            })
        
        # Format exit events for JSON
        formatted_exit_events = []
        for event in recent_exit_events:
            exit_time = event.get('exit_timestamp')
            formatted_exit_events.append({
                'plate': event.get('front_plate_number', 'Unknown'),
                'exit_time': exit_time.isoformat() if exit_time else datetime.now().isoformat(),
                'vehicle_make': event.get('vehicle_make', 'Unknown'),
                'vehicle_model': event.get('vehicle_model', 'Unknown'),
                'vehicle_color': event.get('vehicle_color', 'Unknown')
            })
        
        # Get employee vehicles
        employee_vehicles = list(db.employee_vehicles.find({"is_active": True}).limit(5))
        formatted_employees = []
        for vehicle in employee_vehicles:
            formatted_employees.append({
                'plate': vehicle.get('plate_number', 'Unknown'),
                'employee_name': vehicle.get('employee_name', 'Unknown')
            })
        
        return jsonify({
            'success': True,
            'entry_events': stats.get('total_entry_events', 0),
            'exit_events': stats.get('total_exit_events', 0),
            'vehicle_journeys': stats.get('total_journeys', 0),
            'employee_vehicles': stats.get('employee_vehicles', 0),
            'recent_activity': recent_activity,
            'stats': stats,
            'recent_journeys': formatted_journeys,
            'recent_entry_events': formatted_entry_events,
            'recent_exit_events': formatted_exit_events
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/database')
def api_database():
    """API endpoint for database information."""
    try:
        db = get_db()
        if db is None:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        collections = db.list_collection_names()
        collection_info = {}
        
        for collection_name in collections:
            count = db[collection_name].count_documents({})
            collection_info[collection_name] = count
        
        db_stats = db.command("dbStats")
        
        return jsonify({
            'success': True,
            'collections': collection_info,
            'db_stats': {
                'data_size_mb': round(db_stats.get('dataSize', 0) / (1024*1024), 2),
                'storage_size_mb': round(db_stats.get('storageSize', 0) / (1024*1024), 2),
                'collections': db_stats.get('collections', 0)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_alpr')
def test_alpr():
    """Test ALPR system with current camera frame."""
    try:
        if not ALPR_SYSTEM_AVAILABLE:
            return jsonify({'success': False, 'error': 'ALPR system not available'})
        
        # Try to capture a frame from camera 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Camera not available'})
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'Could not capture frame'})
        
        # Process with ALPR
        if alpr_system_camera1:
            plate_results = alpr_system_camera1.process_frame(frame)
            
            if plate_results:
                valid_results = [r for r in plate_results if r['confidence'] > 30 and len(r['text']) >= 2]
                if valid_results:
                    return jsonify({
                        'success': True,
                        'message': f'ALPR detected: {valid_results[0]["text"]} ({valid_results[0]["confidence"]:.0f}%)',
                        'plates': [{'text': r['text'], 'confidence': r['confidence']} for r in valid_results]
                    })
            
            return jsonify({
                'success': True,
                'message': 'ALPR system working but no plates detected in current frame',
                'plates': []
            })
        
        return jsonify({'success': False, 'error': 'ALPR system not initialized'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simulate_vehicle')
def simulate_vehicle():
    """API endpoint to simulate a vehicle entry/exit event."""
    try:
        db = get_db()
        if db is None:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        # Generate a random license plate
        import random
        import string
        from datetime import timezone
        plate_chars = ''.join(random.choices(string.ascii_uppercase, k=2))
        plate_numbers = ''.join(random.choices(string.digits, k=4))
        plate_end = ''.join(random.choices(string.ascii_uppercase, k=2))
        plate_number = f"{plate_chars}{plate_numbers}{plate_end}"
        
        # Randomly decide if it's an employee vehicle (20% chance)
        is_employee = random.random() < 0.2
        
        # Create entry event
        entry_event = {
            "front_plate_number": plate_number,
            "rear_plate_number": plate_number,
            "front_plate_confidence": random.uniform(80, 95),
            "rear_plate_confidence": random.uniform(80, 95),
            "front_plate_image_path": f"./CCTV_photos/entry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "rear_plate_image_path": f"./CCTV_photos/entry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "entry_timestamp": datetime.now(timezone.utc),
            "vehicle_color": random.choice(["Red", "Blue", "Green", "Black", "White", "Silver"]),
            "vehicle_make": random.choice(["Toyota", "Honda", "Ford", "BMW", "Audi", "Mercedes"]),
            "vehicle_model": random.choice(["Camry", "Civic", "Focus", "X3", "A4", "C-Class"]),
            "is_processed": False,
            "created_at": datetime.now(timezone.utc)
        }
        
        result = db.entry_events.insert_one(entry_event)
        entry_id = result.inserted_id
        
        # If it's an employee vehicle, register it
        if is_employee:
            employee_name = f"Employee {random.randint(1, 100)}"
            try:
                db.employee_vehicles.insert_one({
                    "plate_number": plate_number,
                    "employee_name": employee_name,
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc)
                })
            except Exception:
                # Plate might already exist, ignore
                pass
        
        # With 70% probability, also create an exit event and journey
        if random.random() < 0.7:
            # Wait a random time (1-30 minutes)
            duration_seconds = random.randint(60, 1800)
            exit_timestamp = datetime.now(timezone.utc)
            
            # Create exit event
            exit_event = {
                "front_plate_number": plate_number,
                "rear_plate_number": plate_number,
                "front_plate_confidence": random.uniform(80, 95),
                "rear_plate_confidence": random.uniform(80, 95),
                "front_plate_image_path": f"./CCTV_photos/exit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                "rear_plate_image_path": f"./CCTV_photos/exit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                "exit_timestamp": exit_timestamp,
                "vehicle_color": entry_event["vehicle_color"],
                "vehicle_make": entry_event["vehicle_make"],
                "vehicle_model": entry_event["vehicle_model"],
                "is_processed": False,
                "created_at": exit_timestamp
            }
            
            exit_result = db.exit_events.insert_one(exit_event)
            exit_id = exit_result.inserted_id
            
            # Create journey
            journey = {
                "entry_event_id": entry_id,
                "exit_event_id": exit_id,
                "front_plate_number": plate_number,
                "rear_plate_number": plate_number,
                "entry_timestamp": entry_event["entry_timestamp"],
                "exit_timestamp": exit_timestamp,
                "duration_seconds": duration_seconds,
                "vehicle_color": entry_event["vehicle_color"],
                "vehicle_make": entry_event["vehicle_make"],
                "vehicle_model": entry_event["vehicle_model"],
                "is_employee": is_employee,
                "flagged_for_review": False,
                "created_at": datetime.now(timezone.utc)
            }
            
            db.vehicle_journeys.insert_one(journey)
        
        return jsonify({
            'success': True,
            'message': f'Vehicle {plate_number} processed',
            'plate': plate_number,
            'is_employee': is_employee
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/camera1_feed')
def camera1_feed_route():
    """Camera 1 feed route."""
    return Response(generate_camera_feed(0), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera2_feed')
def camera2_feed_route():
    """Camera 2 feed route."""
    return Response(generate_camera_feed(1), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def create_main_dashboard_template():
    """Create the main dashboard HTML template."""
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Tracking System - Main Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css">
    <style>
        .dashboard-card {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            margin-bottom: 20px;
        }
        .dashboard-card:hover {
            transform: translateY(-5px);
        }
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        .navbar-brand {
            font-weight: bold;
            color: #fff !important;
        }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="bi bi-car-front-fill me-2"></i>Vehicle Tracking System
            </a>
            <div class="d-flex">
                <span class="navbar-text me-3">
                    <i class="bi bi-database me-1"></i> MongoDB
                </span>
                <span class="navbar-text">
                    <i class="bi bi-clock me-1"></i> <span id="current-time"></span>
                </span>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <h2 class="mb-4">
                    <i class="bi bi-speedometer2 me-2"></i>Main Dashboard
                </h2>
            </div>
        </div>

        <!-- Camera Feeds -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">
                            <i class="bi bi-camera me-2"></i>Camera 1 Feed (Entry)
                        </h5>
                    </div>
                    <div class="card-body p-0">
                        <img src="/camera1_feed" class="img-fluid" alt="Camera 1 Feed" id="camera1-feed">
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">
                            <i class="bi bi-camera me-2"></i>Camera 2 Feed (Exit)
                        </h5>
                    </div>
                    <div class="card-body p-0">
                        <img src="/camera2_feed" class="img-fluid" alt="Camera 2 Feed" id="camera2-feed">
                    </div>
                </div>
            </div>
        </div>

        <!-- Statistics Cards -->
        <div class="row" id="stats-container">
            <div class="col-md-3 col-sm-6">
                <div class="card dashboard-card bg-white">
                    <div class="card-body text-center">
                        <div class="stat-number text-primary" id="entry-count">0</div>
                        <div class="stat-label">ENTRY EVENTS</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="card dashboard-card bg-white">
                    <div class="card-body text-center">
                        <div class="stat-number text-success" id="exit-count">0</div>
                        <div class="stat-label">EXIT EVENTS</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="card dashboard-card bg-white">
                    <div class="card-body text-center">
                        <div class="stat-number text-info" id="journey-count">0</div>
                        <div class="stat-label">VEHICLE JOURNEYS</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="card dashboard-card bg-white">
                    <div class="card-body text-center">
                        <div class="stat-number text-warning" id="employee-count">0</div>
                        <div class="stat-label">EMPLOYEE VEHICLES</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Vehicle Records Table -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card dashboard-card">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">
                            <i class="bi bi-table me-2"></i>Vehicle Records
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead class="table-dark">
                                    <tr>
                                        <th>Camera 1 Plate</th>
                                        <th>Camera 2 Plate</th>
                                        <th>Entry Time</th>
                                        <th>Exit Time</th>
                                        <th>Employee Vehicle</th>
                                    </tr>
                                </thead>
                                <tbody id="vehicle-records">
                                    <!-- Vehicle records will be populated here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleTimeString();
        }

        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('entry-count').textContent = data.entry_events || 0;
                    document.getElementById('exit-count').textContent = data.exit_events || 0;
                    document.getElementById('journey-count').textContent = data.vehicle_journeys || 0;
                    document.getElementById('employee-count').textContent = data.employee_vehicles || 0;
                    
                    // Update vehicle records table
                    const tbody = document.getElementById('vehicle-records');
                    tbody.innerHTML = '';
                    
                    if (data.recent_activity && data.recent_activity.length > 0) {
                        data.recent_activity.forEach(record => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${record.front_plate_number || 'N/A'}</td>
                                <td>${record.rear_plate_number || 'N/A'}</td>
                                <td>${record.entry_timestamp ? new Date(record.entry_timestamp).toLocaleString() : 'N/A'}</td>
                                <td>${record.exit_timestamp ? new Date(record.exit_timestamp).toLocaleString() : 'Pending'}</td>
                                <td>${record.is_employee ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>'}</td>
                            `;
                            tbody.appendChild(row);
                        });
                    } else {
                        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No vehicle records found</td></tr>';
                    }
                })
                .catch(error => {
                    console.error('Error loading stats:', error);
                });
        }

        // Update time every second
        setInterval(updateTime, 1000);
        updateTime();

        // Load stats every 5 seconds
        setInterval(loadStats, 5000);
        loadStats();
    </script>
</body>
</html>'''
    
    # Create templates directory if it doesn't exist
    os.makedirs('src/ui/templates', exist_ok=True)
    
    # Write template file
    with open('src/ui/templates/main_dashboard.html', 'w') as f:
        f.write(template_content)
    
    print("✅ Created main dashboard template")

def main():
    """Main function to run the entire system."""
    print("🚀 Vehicle Tracking System - Unified Application")
    print("=" * 55)
    print("✅ Starting all system components...")
    print()
    
    # Create necessary directories
    os.makedirs('src/ui/templates', exist_ok=True)
    os.makedirs('src/ui/static', exist_ok=True)
    os.makedirs('CCTV_photos', exist_ok=True)
    
    # Create the main dashboard template
    create_main_dashboard_template()
    
    # Initialize database connection
    if not initialize_database():
        print("❌ Database connection failed")
        return 1
    
    # Initialize ALPR systems
    if ALPR_SYSTEM_AVAILABLE:
        initialize_alpr_systems()
    
    print()
    print("📱 Web Dashboard:")
    print("   Access at: http://localhost:8089")
    print("   Camera 1 Feed: http://localhost:8089/camera1_feed")
    print("   Camera 2 Feed: http://localhost:8089/camera2_feed")
    print()
    print("✨ New Features:")
    print("   - Click the car icon to simulate vehicle events")
    print("   - Data refreshes automatically every 5 seconds")
    print("   - Camera feeds update every 30 seconds")
    print("   - Real-time ALPR processing")
    print()
    print("🔌 Press Ctrl+C to stop all services")
    print()
    
    # Run the Flask app
    try:
        app.run(host='0.0.0.0', port=8089, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        # Close database connection
        global db_client
        if db_client:
            db_client.close()
        return 0
    except Exception as e:
        print(f"❌ Error running web server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())