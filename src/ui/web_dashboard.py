#!/usr/bin/env python3
"""
Web-based dashboard for the Vehicle Tracking System
"""

import sys
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import json

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tracking'))

from tracking.vehicle_tracking_system_mongodb import MemoryOptimizedVehicleTracker

# Create Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics."""
    try:
        # Initialize the tracker
        tracker = MemoryOptimizedVehicleTracker()
        
        # Get statistics
        stats = tracker.get_system_stats()
        
        # Get recent journeys
        recent_journeys = tracker.get_recent_journeys(hours=24)
        
        # Format journeys for JSON serialization
        formatted_journeys = []
        for journey in recent_journeys:
            formatted_journeys.append({
                'plate': journey.get('front_plate_number', 'Unknown'),
                'entry_time': journey.get('entry_timestamp', datetime.min).isoformat(),
                'exit_time': journey.get('exit_timestamp', datetime.min).isoformat(),
                'duration_seconds': journey.get('duration_seconds', 0),
                'is_employee': journey.get('is_employee', False),
                'vehicle_make': journey.get('vehicle_make', 'Unknown'),
                'vehicle_model': journey.get('vehicle_model', 'Unknown')
            })
        
        # Get employee vehicles
        employee_vehicles = list(tracker.db.employee_vehicles.find({"is_active": True}))
        formatted_employees = []
        for vehicle in employee_vehicles:
            formatted_employees.append({
                'plate': vehicle.get('plate_number', 'Unknown'),
                'employee_name': vehicle.get('employee_name', 'Unknown')
            })
        
        # Close connection
        tracker.close()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_journeys': formatted_journeys,
            'employee_vehicles': formatted_employees
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/database')
def api_database():
    """API endpoint for database information."""
    try:
        # Initialize the tracker
        tracker = MemoryOptimizedVehicleTracker()
        
        # Get collection information
        collections = tracker.db.list_collection_names()
        collection_info = {}
        
        for collection_name in collections:
            count = tracker.db[collection_name].count_documents({})
            collection_info[collection_name] = count
        
        # Get database stats
        db_stats = tracker.db.command("dbStats")
        
        # Close connection
        tracker.close()
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ocr_comparisons')
def api_ocr_comparisons():
    """API endpoint for recent OCR comparison results."""
    try:
        # Initialize the tracker
        tracker = MemoryOptimizedVehicleTracker()
        
        # Get recent OCR comparisons (last 50)
        comparisons = list(tracker.db.ocr_comparisons.find(
            {}, 
            {'_id': 0}
        ).sort('timestamp', -1).limit(50))
        
        # Format for JSON serialization
        formatted_comparisons = []
        for comp in comparisons:
            formatted_comp = {
                'timestamp': comp.get('timestamp', datetime.min).isoformat(),
                'ocr_methods': comp.get('ocr_methods', 0),
                'results': comp.get('results', []),
                'winner': comp.get('winner', {})
            }
            formatted_comparisons.append(formatted_comp)
        
        # Get OCR method statistics
        pipeline = [
            {'$unwind': '$results'},
            {'$group': {
                '_id': '$results.method',
                'total_detections': {'$sum': 1},
                'avg_confidence': {'$avg': '$results.confidence'},
                'valid_plates': {'$sum': {'$cond': ['$results.is_valid_indian', 1, 0]}}
            }}
        ]
        
        method_stats = list(tracker.db.ocr_comparisons.aggregate(pipeline))
        
        # Close connection
        tracker.close()
        
        return jsonify({
            'success': True,
            'recent_comparisons': formatted_comparisons,
            'method_statistics': method_stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def main():
    """Main function to run the web dashboard."""
    print("🌐 Starting Web Dashboard for Vehicle Tracking System")
    print("=" * 55)
    print("✅ MongoDB connection configured")
    print("✅ Flask web framework initialized")
    print()
    print("📱 Access the dashboard at: http://localhost:8080")
    print("🔍 OCR Comparisons API: http://localhost:8080/api/ocr_comparisons")
    print("🔌 Press Ctrl+C to stop the server")
    print()
    
    # Run the Flask app on port 8080 instead of 5000
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

if __name__ == "__main__":
    main()