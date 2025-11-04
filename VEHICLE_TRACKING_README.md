# Vehicle Tracking System using ultimateALPR-SDK

This document describes a complete vehicle entry/exit tracking system using ALPR cameras and the ultimateALPR-SDK.

## System Overview

The vehicle tracking system implements a comprehensive solution for monitoring vehicle entry and exit events using AI-based ALPR cameras. The system captures images from strategically placed cameras, processes them using the ultimateALPR-SDK, and maintains a database of vehicle journeys.

## System Architecture

### Camera Setup
- **Camera 1**: Front-facing camera at entry point
- **Camera 2**: Rear-facing camera at entry point
- Cameras operate 24/7, recording video continuously

### Entry Logic
1. Camera 1 detects vehicle entering and captures front number plate
2. OCR extracts and records front plate number
3. Vehicle passes and is captured by Camera 2 (rear-facing)
4. OCR extracts and records rear plate number
5. System pairs both plate numbers using time-window and vehicle attributes
6. Entry event is saved to database

### Exit Logic
1. Camera 2 detects vehicle exiting and records rear number plate
2. OCR extracts and records rear plate number
3. Vehicle passes and Camera 1 captures front plate
4. OCR extracts and records front plate number
5. System pairs both plate numbers (reverse matching)
6. Exit event is logged and matched with entry record

## Key Features

### 1. Dual Plate Recognition
- Captures both front and rear license plates for accuracy
- Uses time-window correlation to match entry/exit events
- Employs vehicle attributes (color, make, model) for matching accuracy

### 2. Employee Vehicle Management
- Automatic flagging and categorization of employee vehicles
- Prevents duplicate logging of employee vehicles
- Maintains employee vehicle database

### 3. Anomaly Detection
- Identifies mismatched plate numbers
- Flags missing or low-confidence data
- Routes anomalies for manual review

### 4. Database Management
- SQLite-based storage for simplicity and reliability
- Complete journey tracking from entry to exit
- Anomaly logging for quality control

## Implementation Files

### Core Components
- `vehicle_tracking_system.py` - Main tracking system implementation
- `python_docker_wrapper.py` - ALPR SDK Python wrapper (already created)
- `database_schema.sql` - Database schema definition
- `vehicle_tracking_config.py` - System configuration

### Demo and Testing
- `demo_vehicle_tracking.py` - Demonstration script
- `VEHICLE_TRACKING_README.md` - This documentation

## Database Schema

The system uses the following tables:

1. **vehicles** - Vehicle information and employee status
2. **entry_events** - Raw entry event data
3. **exit_events** - Raw exit event data
4. **vehicle_journeys** - Matched entry/exit pairs
5. **anomalies** - Events requiring manual review

## Usage

### Basic Setup
1. Ensure Docker is running (required for ALPR SDK)
2. Activate the Python virtual environment:
   ```bash
   source alpr_venv/bin/activate
   ```

### Running the Demo
```bash
python demo_vehicle_tracking.py
```

### Integration Example
```python
from vehicle_tracking_system import VehicleTrackingSystem

# Initialize the system
tracker = VehicleTrackingSystem()

# Process entry event
entry_event = tracker.process_entry_event(
    front_image_path="camera1_entry.jpg",
    rear_image_path="camera2_entry.jpg"
)

# Process exit event
exit_event = tracker.process_exit_event(
    front_image_path="camera1_exit.jpg",
    rear_image_path="camera2_exit.jpg"
)

# Match events to create journeys
journeys = tracker.match_entry_exit_events()
```

## Configuration

The system can be configured through `vehicle_tracking_config.py`:

- Database paths and connection settings
- ALPR confidence thresholds
- Camera configurations
- Matching parameters
- Anomaly detection settings

## Error Handling

The system includes comprehensive error handling for:

- Low-confidence OCR readings
- Missing plate data
- Mismatched plate numbers
- Database connection issues
- Docker container availability

## Performance Considerations

- Time-window matching for efficient event correlation
- Database indexing for fast queries
- Configurable processing timeouts
- Concurrent processing capabilities

## Extending the System

The vehicle tracking system can be extended to include:

1. Real-time video stream processing
2. Web-based dashboard for monitoring
3. Alerting system for anomalies
4. Integration with gate control systems
5. Reporting and analytics features
6. Cloud-based storage options

## Dependencies

- Python 3.6+
- ultimateALPR-SDK (via Docker)
- SQLite3 (built into Python)
- Pillow (for image handling)

## Troubleshooting

### Common Issues
1. **Docker not running**: Ensure Docker daemon is active
2. **Database locked**: Check for concurrent access issues
3. **Low confidence readings**: Adjust camera positioning or lighting
4. **Mismatched events**: Verify time synchronization between cameras

### Logs
The system maintains logs in `vehicle_tracking.log` for debugging purposes.