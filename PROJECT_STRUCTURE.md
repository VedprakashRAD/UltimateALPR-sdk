# Project Structure Summary

## Overview

This document describes the modular organization of the UltimateALPR-SDK Vehicle Tracking System.

## Directory Structure

```
ultimateALPR-sdk/
├── src/                        # Source code directory
│   ├── __init__.py            # Package initializer
│   ├── main.py                # Main entry point
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   └── python_docker_wrapper.py    # Docker wrapper for ALPR SDK
│   ├── database/              # Database components
│   │   ├── __init__.py
│   │   ├── database_schema.sql         # SQL schema definition
│   │   └── vehicle_tracking_config.py  # Configuration settings
│   ├── tracking/              # Vehicle tracking logic
│   │   ├── __init__.py
│   │   ├── vehicle_tracking_system.py          # SQLite-based tracking
│   │   └── vehicle_tracking_system_mongodb.py  # MongoDB-based tracking
│   ├── camera/                # Camera processing systems
│   │   ├── __init__.py
│   │   ├── camera_live_system.py       # Live camera processing
│   │   ├── camera_simulator.py         # Camera simulation
│   │   ├── fully_automatic_system.py   # Fully automatic vehicle detection
│   │   ├── working_alpr_system.py      # Working ALPR system
│   │   └── auto_detect_system.py       # Auto-detection system
│   ├── utils/                 # Utility scripts and tools
│   │   ├── __init__.py
│   │   ├── install_macos.sh            # macOS installation script
│   │   ├── install_raspberry_pi.sh     # Raspberry Pi installation script
│   │   ├── raspberry_pi_setup.py       # Raspberry Pi setup
│   │   ├── debug_docker_output.py      # Docker output debugging
│   │   ├── test_python_wrapper.py      # Python wrapper testing
│   │   ├── test_setup.py               # Setup testing
│   │   └── demo_*.py                   # Demo scripts
│   └── ui/                    # User interface (placeholder)
│       └── __init__.py
├── assets/                    # SDK assets (models, etc.)
├── binaries/                  # SDK binaries
├── captured_images/           # Captured vehicle images
├── auto_captures/             # Automatic captures
├── detected_plates/           # Detected license plates
├── logs/                      # Log files
├── backups/                   # Database backups
├── processed_images/          # Processed images
├── sample_images/             # Sample images
├── requirements.txt           # Python dependencies
├── arm64.Dockerfile           # Docker build file for ARM64
├── simple.Dockerfile          # Simple Docker build file
├── .gitignore                 # Git ignore file
└── README.md                  # Project documentation
```

## Module Descriptions

### Core Module
- **python_docker_wrapper.py**: Provides a Python interface to the UltimateALPR-SDK through Docker containers

### Database Module
- **database_schema.sql**: Defines the MongoDB collections structure
- **vehicle_tracking_config.py**: Configuration settings for the database and tracking system

### Tracking Module
- **vehicle_tracking_system.py**: SQLite-based vehicle tracking implementation
- **vehicle_tracking_system_mongodb.py**: MongoDB-based vehicle tracking implementation with memory optimization

### Camera Module
- **camera_live_system.py**: Processes live camera feeds for vehicle detection
- **camera_simulator.py**: Simulates camera operations for testing
- **fully_automatic_system.py**: Fully automatic vehicle recognition system
- **working_alpr_system.py**: Working ALPR system with real license plate detection
- **auto_detect_system.py**: Auto-detection system for vehicles

### Utils Module
- **install_*.sh**: Installation scripts for different platforms
- **raspberry_pi_setup.py**: Raspberry Pi specific setup
- **debug_docker_output.py**: Tools for debugging Docker output
- **test_*.py**: Testing utilities
- **demo_*.py**: Demonstration scripts

## Benefits of Modular Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Maintainability**: Easier to locate and modify specific functionality
3. **Reusability**: Modules can be reused in different contexts
4. **Testability**: Individual modules can be tested independently
5. **Scalability**: New features can be added without disrupting existing code
6. **Collaboration**: Multiple developers can work on different modules simultaneously

## Usage

To run the system:
```bash
python src/main.py
```

To run specific components:
```bash
# Run the working ALPR system
python src/camera/working_alpr_system.py

# Run the camera simulator
python src/camera/camera_simulator.py
```