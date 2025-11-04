# UltimateALPR-SDK with Vehicle Tracking System

This repository contains the UltimateALPR-SDK along with a comprehensive vehicle entry/exit tracking system implementation.

## Table of Contents

1. [About UltimateALPR-SDK](#about-ultimatealpr-sdk)
2. [Vehicle Tracking System](#vehicle-tracking-system)
3. [System Architecture](#system-architecture)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Database Schema](#database-schema)
8. [Configuration](#configuration)
9. [Demos and Examples](#demos-and-examples)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [License](#license)

## About UltimateALPR-SDK

The UltimateALPR-SDK is a high-performance ANPR/ALPR implementation using deep learning. It's the fastest ANPR/ALPR implementation you'll find on the market, capable of running at 64fps on a $99 ARM device.

Key features:
- State of the art deep learning techniques for unmatched accuracy
- Runs on CPU with SIMD ARM NEON optimizations
- Supports CUDA, NVIDIA TensorRT and Intel OpenVINO for GPU acceleration
- No registration, license key or internet connection required
- Cross-platform support (Android, Windows, Raspberry Pi, Linux, NVIDIA Jetson, etc.)

For more information about the UltimateALPR-SDK, please refer to the original [README.md](README.md).

## Vehicle Tracking System

This implementation includes a complete vehicle entry/exit tracking system using ALPR cameras and the UltimateALPR-SDK.

### System Features

1. **24/7 Camera Operation**: Continuous video recording
2. **Dual Camera Setup**: Entry and exit monitoring with front/rear plate capture
3. **AI-Based Detection**: Using the UltimateALPR-SDK for accurate plate recognition
4. **Event Pairing**: Using time-window and vehicle attributes for accuracy
5. **Employee Vehicle Management**: Automatic flagging and categorization
6. **Anomaly Detection**: Identification of inconsistencies and manual review flagging
7. **Database Management**: SQLite storage for all vehicle events and journeys

### Entry Logic

1. Camera 1 (front-facing at entry) detects a vehicle entering and captures a clear image of the front number plate using AI-based motion or vehicle detection
2. OCR extracts and records the front plate number
3. Vehicle passes and is captured by Camera 2 (rear-facing), which detects and captures the rear number plate
4. OCR extracts and records the rear plate number
5. System pairs both plate numbers (using time-window, vehicle attributes like color and model for accuracy) as a single entry event
6. Entry event with timestamps, plate data, images, and vehicle attributes is saved to the database

### Exit Logic

1. When a vehicle is ready to exit, Camera 2 (now facing rear as the vehicle leaves) detects and records the rear number plate
2. OCR extracts and records the rear plate number
3. Vehicle passes and Camera 1 (entry camera, now on the car's front as it exits) captures the front plate
4. OCR extracts and records the front plate number
5. System pairs both plate numbers (reverse time-window and attribute matching) as a single exit event
6. Exit event is logged in the database, matched with the corresponding entry record to complete the journey

### Additional Features

- Employee vehicles are flagged and categorized automatically through the plate database to avoid duplicate logging
- System checks for inconsistencies or anomalies (like mismatched plate numbers or missing data) and flags for manual review

## System Architecture

### Camera Setup
- **Camera 1**: Front-facing camera at entry point
- **Camera 2**: Rear-facing camera at entry point
- Cameras operate 24/7, recording video continuously

### Core Components

1. **ALPR SDK Integration**: Uses the UltimateALPR-SDK via Docker containers for cross-platform compatibility
2. **Database Management**: SQLite-based storage for all vehicle events and journeys
3. **Event Processing**: Logic for handling entry and exit events
4. **Matching Engine**: Algorithm for pairing entry and exit events
5. **Anomaly Detection**: System for identifying inconsistencies

## Installation and Setup

### Prerequisites

1. Docker (required for running the ALPR SDK)
2. Python 3.6 or later
3. Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/VedprakashRAD/UltimateALPR-sdk.git
   cd UltimateALPR-sdk
   ```

2. Ensure Docker is running

3. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv alpr_venv
   source alpr_venv/bin/activate  # On Windows: alpr_venv\Scripts\activate
   ```

4. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Build the ARM64 Docker image (for Apple Silicon Macs):
   ```bash
   docker build -t alpr-arm64 -f arm64.Dockerfile .
   ```

## Usage

### Basic Usage

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

### Running Demos

1. Run the basic vehicle tracking demo:
   ```bash
   python demo_vehicle_tracking.py
   ```

2. Run the camera simulator:
   ```bash
   python camera_simulator.py
   ```

## API Documentation

### VehicleTrackingSystem Class

#### Initialization
```python
tracker = VehicleTrackingSystem(db_path="vehicle_tracking.db")
```

#### Methods

- `process_entry_event(front_image_path, rear_image_path)`: Process an entry event
- `process_exit_event(front_image_path, rear_image_path)`: Process an exit event
- `match_entry_exit_events(time_window_minutes=30)`: Match entry and exit events
- `get_unmatched_entries()`: Get unmatched entry events
- `get_unmatched_exits()`: Get unmatched exit events
- `is_employee_vehicle(front_plate, rear_plate)`: Check if vehicle is an employee vehicle

## Database Schema

The system uses the following tables:

1. **vehicles**: Vehicle information and employee status
2. **entry_events**: Raw entry event data
3. **exit_events**: Raw exit event data
4. **vehicle_journeys**: Matched entry/exit pairs
5. **anomalies**: Events requiring manual review

For detailed schema, see [database_schema.sql](database_schema.sql).

## Configuration

The system can be configured through [vehicle_tracking_config.py](vehicle_tracking_config.py):

- Database paths and connection settings
- ALPR confidence thresholds
- Camera configurations
- Matching parameters
- Anomaly detection settings

## Demos and Examples

### Files

1. [demo_vehicle_tracking.py](demo_vehicle_tracking.py) - Main demonstration script
2. [camera_simulator.py](camera_simulator.py) - Camera simulation for continuous monitoring
3. [demo_python_sdk.py](demo_python_sdk.py) - Demo for the ALPR SDK wrapper
4. [test_setup.py](test_setup.py) - Test script for the setup

### Running the Demos

```bash
# Run the main vehicle tracking demo
python demo_vehicle_tracking.py

# Run the camera simulator
python camera_simulator.py

# Run the ALPR SDK demo
python demo_python_sdk.py
```

## Troubleshooting

### Common Issues

1. **Docker not running**: Ensure Docker daemon is active
2. **Database locked**: Check for concurrent access issues
3. **Low confidence readings**: Adjust camera positioning or lighting
4. **Mismatched events**: Verify time synchronization between cameras

### Logs

The system maintains logs in `vehicle_tracking.log` for debugging purposes.

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project uses the same license as the UltimateALPR-SDK. See [LICENSE](LICENSE) for more information.

---

For more detailed information about the UltimateALPR-SDK, please refer to:
- Original documentation: https://www.doubango.org/SDKs/anpr/docs/
- Online demo: https://www.doubango.org/webapps/alpr/