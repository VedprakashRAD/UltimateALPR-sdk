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

# Global variables for camera feeds
camera1_feed = None
camera2_feed = None
camera1_lock = threading.Lock()
camera2_lock = threading.Lock()

# Flask app
app = Flask(__name__, template_folder='src/ui/templates', static_folder='src/ui/static')

# Database configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "vehicle_tracking"

def connect_to_mongodb():
    """Connect to MongoDB database."""
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000
        )
        db = client[DB_NAME]
        # Test connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {MONGO_URI}/{DB_NAME}")
        return client, db
    except ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None, None

def get_system_stats(db):
    """Get system statistics from database."""
    try:
        stats = {
            "memory_usage_gb": 0,
            "memory_percent": 0,
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
        from datetime import datetime, timedelta
        since = datetime.utcnow() - timedelta(hours=hours)
        
        journeys = list(db.vehicle_journeys.find({
            "entry_timestamp": {"$gte": since}
        }).sort("entry_timestamp", -1).limit(100))
        
        return journeys
    except Exception as e:
        print(f"Error getting journeys: {e}")
        return []

# Camera feed functions
def generate_camera_feed(camera_id):
    """Generate camera feed for streaming."""
    global camera1_feed, camera2_feed
    
    # Try to open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        # If camera not available, create a test pattern
        while True:
            # Create a test image with text
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            text = f"CAMERA {camera_id+1} - OFFLINE"
            cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', img)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.1)
        return
    
    # Camera is available
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add timestamp
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CAMERA {camera_id+1}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# Flask routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('main_dashboard.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics."""
    try:
        client, db = connect_to_mongodb()
        if db is None:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        stats = get_system_stats(db)
        recent_journeys = get_recent_journeys(db)
        
        # Format journeys for JSON
        formatted_journeys = []
        for journey in recent_journeys[:10]:  # First 10
            formatted_journeys.append({
                'plate': journey.get('front_plate_number', 'Unknown'),
                'entry_time': journey.get('entry_timestamp', datetime.min).isoformat(),
                'exit_time': journey.get('exit_timestamp', datetime.min).isoformat(),
                'duration_seconds': journey.get('duration_seconds', 0),
                'is_employee': journey.get('is_employee', False)
            })
        
        # Get employee vehicles
        employee_vehicles = list(db.employee_vehicles.find({"is_active": True}))
        formatted_employees = []
        for vehicle in employee_vehicles:
            formatted_employees.append({
                'plate': vehicle.get('plate_number', 'Unknown'),
                'employee_name': vehicle.get('employee_name', 'Unknown')
            })
        
        client.close()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_journeys': formatted_journeys,
            'employee_vehicles': formatted_employees
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/database')
def api_database():
    """API endpoint for database information."""
    try:
        client, db = connect_to_mongodb()
        if db is None:
            return jsonify({'success': False, 'error': 'Database connection failed'})
        
        collections = db.list_collection_names()
        collection_info = {}
        
        for collection_name in collections:
            count = db[collection_name].count_documents({})
            collection_info[collection_name] = count
        
        db_stats = db.command("dbStats")
        
        client.close()
        
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
        .journey-item {
            border-left: 4px solid #0d6efd;
            padding-left: 15px;
            margin-bottom: 15px;
        }
        .employee-item {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .navbar-brand {
            font-weight: bold;
            color: #fff !important;
        }
        .camera-feed {
            width: 100%;
            height: 300px;
            background-color: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        .camera-placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #fff;
            font-size: 1.2rem;
        }
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 1000;
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
                <div class="alert alert-success alert-dismissible fade show" role="alert">
                    <i class="bi bi-info-circle me-2"></i>
                    <strong>System Online!</strong> Access this dashboard at <a href="http://localhost:8080" class="alert-link">http://localhost:8080</a>
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
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
                        <img src="/camera1_feed" class="img-fluid" alt="Camera 1 Feed">
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
                        <img src="/camera2_feed" class="img-fluid" alt="Camera 2 Feed">
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

        <div class="row mt-4">
            <!-- Recent Activity -->
            <div class="col-lg-8">
                <div class="card dashboard-card">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">
                            <i class="bi bi-clock-history me-2"></i>Recent Vehicle Journeys
                        </h5>
                    </div>
                    <div class="card-body" id="journeys-container">
                        <div class="text-center py-5">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-2">Loading recent journeys...</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Employee Vehicles & Database Info -->
            <div class="col-lg-4">
                <div class="card dashboard-card mb-4">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">
                            <i class="bi bi-people me-2"></i>Employee Vehicles
                        </h5>
                    </div>
                    <div class="card-body" id="employees-container">
                        <div class="text-center py-3">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card dashboard-card">
                    <div class="card-header bg-white">
                        <h5 class="mb-0">
                            <i class="bi bi-database me-2"></i>Database Information
                        </h5>
                    </div>
                    <div class="card-body" id="database-container">
                        <div class="text-center py-3">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Refresh Button -->
    <button class="btn btn-primary btn-lg rounded-circle refresh-btn" id="refresh-btn" title="Refresh Data">
        <i class="bi bi-arrow-repeat"></i>
    </button>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Update current time
        function updateTime() {
            const now = new Date();
            document.getElementById('current-time').textContent = now.toLocaleString();
        }
        updateTime();
        setInterval(updateTime, 1000);

        // Fetch data from API
        async function fetchData() {
            try {
                // Show loading states
                document.getElementById('journeys-container').innerHTML = `
                    <div class="text-center py-5">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-2">Loading recent journeys...</p>
                    </div>
                `;
                
                document.getElementById('employees-container').innerHTML = `
                    <div class="text-center py-3">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                `;
                
                document.getElementById('database-container').innerHTML = `
                    <div class="text-center py-3">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                `;

                // Fetch stats
                const statsResponse = await fetch('/api/stats');
                const statsData = await statsResponse.json();
                
                if (statsData.success) {
                    // Update statistics
                    document.getElementById('entry-count').textContent = statsData.stats.total_entry_events;
                    document.getElementById('exit-count').textContent = statsData.stats.total_exit_events;
                    document.getElementById('journey-count').textContent = statsData.stats.total_journeys;
                    document.getElementById('employee-count').textContent = statsData.stats.employee_vehicles;

                    // Update journeys
                    let journeysHtml = '';
                    if (statsData.recent_journeys.length > 0) {
                        statsData.recent_journeys.slice(0, 10).forEach(journey => {
                            const entryTime = new Date(journey.entry_time);
                            const exitTime = new Date(journey.exit_time);
                            const durationMinutes = Math.floor(journey.duration_seconds / 60);
                            const durationSeconds = journey.duration_seconds % 60;
                            
                            journeysHtml += `
                                <div class="journey-item">
                                    <div class="d-flex justify-content-between">
                                        <h6 class="mb-1">
                                            ${journey.plate} 
                                            ${journey.is_employee ? '<span class="badge bg-success">Employee</span>' : ''}
                                        </h6>
                                        <small class="text-muted">${durationMinutes}m ${durationSeconds}s</small>
                                    </div>
                                    <p class="mb-1">
                                        <i class="bi bi-box-arrow-in-right text-success me-1"></i> 
                                        ${entryTime.toLocaleTimeString()} | 
                                        <i class="bi bi-box-arrow-right text-danger me-1"></i> 
                                        ${exitTime.toLocaleTimeString()}
                                    </p>
                                </div>
                            `;
                        });
                    } else {
                        journeysHtml = '<p class="text-muted text-center py-3">No recent journeys</p>';
                    }
                    document.getElementById('journeys-container').innerHTML = journeysHtml;

                    // Update employees
                    let employeesHtml = '';
                    if (statsData.employee_vehicles.length > 0) {
                        statsData.employee_vehicles.forEach(employee => {
                            employeesHtml += `
                                <div class="employee-item">
                                    <div class="d-flex justify-content-between">
                                        <strong>${employee.plate}</strong>
                                        <span class="badge bg-primary">Employee</span>
                                    </div>
                                    <small class="text-muted">${employee.employee_name}</small>
                                </div>
                            `;
                        });
                    } else {
                        employeesHtml = '<p class="text-muted text-center">No employee vehicles registered</p>';
                    }
                    document.getElementById('employees-container').innerHTML = employeesHtml;
                }

                // Fetch database info
                const dbResponse = await fetch('/api/database');
                const dbData = await dbResponse.json();
                
                if (dbData.success) {
                    let dbHtml = `
                        <div class="row">
                            <div class="col-6">
                                <p><strong>Collections:</strong> ${dbData.db_stats.collections}</p>
                                <p><strong>Data Size:</strong> ${dbData.db_stats.data_size_mb} MB</p>
                                <p><strong>Storage Size:</strong> ${dbData.db_stats.storage_size_mb} MB</p>
                            </div>
                            <div class="col-6">
                    `;
                    
                    Object.entries(dbData.collections).forEach(([name, count]) => {
                        dbHtml += `<p><strong>${name}:</strong> ${count}</p>`;
                    });
                    
                    dbHtml += `
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('database-container').innerHTML = dbHtml;
                }
            } catch (error) {
                console.error('Error fetching data:', error);
                document.getElementById('journeys-container').innerHTML = '<p class="text-danger text-center py-3">Error loading data</p>';
                document.getElementById('employees-container').innerHTML = '<p class="text-danger text-center py-3">Error loading data</p>';
                document.getElementById('database-container').innerHTML = '<p class="text-danger text-center py-3">Error loading data</p>';
            }
        }

        // Initial data load
        document.addEventListener('DOMContentLoaded', function() {
            fetchData();
        });

        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', function() {
            const btn = this;
            const icon = btn.querySelector('i');
            
            // Add spinning animation
            icon.classList.add('spin');
            btn.classList.add('btn-secondary');
            btn.classList.remove('btn-primary');
            
            fetchData().then(() => {
                // Remove spinning animation
                setTimeout(() => {
                    icon.classList.remove('spin');
                    btn.classList.add('btn-primary');
                    btn.classList.remove('btn-secondary');
                }, 500);
            });
        });

        // Auto-refresh every 30 seconds
        setInterval(fetchData, 30000);
        
        // Refresh camera feeds every 5 seconds
        setInterval(function() {
            document.querySelectorAll('img[src*="_feed"]').forEach(img => {
                const src = img.src;
                img.src = src.split('?')[0] + '?' + new Date().getTime();
            });
        }, 5000);
    </script>
    
    <style>
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        .spin {
            animation: spin 1s linear infinite;
        }
    </style>
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
    
    # Create the main dashboard template
    create_main_dashboard_template()
    
    # Test database connection
    client, db = connect_to_mongodb()
    if db is not None:
        print("✅ Database connection successful")
        client.close()
    else:
        print("❌ Database connection failed")
        return 1
    
    print()
    print("📱 Web Dashboard:")
    print("   Access at: http://localhost:8080")
    print("   Camera 1 Feed: http://localhost:8080/camera1_feed")
    print("   Camera 2 Feed: http://localhost:8080/camera2_feed")
    print()
    print("🔌 Press Ctrl+C to stop all services")
    print()
    
    # Run the Flask app
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        return 0
    except Exception as e:
        print(f"❌ Error running web server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())