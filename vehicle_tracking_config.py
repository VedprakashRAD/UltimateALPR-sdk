#!/usr/bin/env python3
"""
Configuration file for the Vehicle Tracking System
"""

import os

# Database configuration
DATABASE_PATH = os.environ.get("VEHICLE_TRACKING_DB", "vehicle_tracking.db")

# ALPR SDK configuration
ALPR_DOCKER_IMAGE = "alpr-arm64"
ALPR_CONFIDENCE_THRESHOLD = 80.0  # Minimum confidence percentage for plate readings

# Camera configuration
CAMERA_CONFIG = {
    "entry": {
        "front_camera": {
            "id": 1,
            "position": "front",
            "description": "Front-facing camera at entry point"
        },
        "rear_camera": {
            "id": 2,
            "position": "rear",
            "description": "Rear-facing camera at entry point"
        }
    },
    "exit": {
        "front_camera": {
            "id": 1,
            "position": "front",
            "description": "Front-facing camera at exit point (captures front of exiting vehicles)"
        },
        "rear_camera": {
            "id": 2,
            "position": "rear",
            "description": "Rear-facing camera at exit point (captures rear of exiting vehicles)"
        }
    }
}

# Matching configuration
MATCHING_CONFIG = {
    "time_window_minutes": 30,  # Time window for matching entry/exit events
    "plate_similarity_threshold": 0.8,  # Minimum similarity for plate matching
    "enable_attribute_matching": True,  # Use vehicle attributes for matching
    "max_retry_attempts": 3  # Maximum retry attempts for failed matches
}

# Employee vehicle configuration
EMPLOYEE_VEHICLE_CONFIG = {
    "auto_categorize": True,  # Automatically categorize known employee vehicles
    "database_check": True,   # Check database for employee vehicle status
    "flag_for_review": False  # Flag employee vehicles for manual review (usually False)
}

# Anomaly detection configuration
ANOMALY_DETECTION = {
    "enable_anomaly_detection": True,
    "low_confidence_threshold": 70.0,  # Flag readings below this confidence
    "missing_plate_threshold": 1,      # Flag events with missing plates
    "mismatched_plate_threshold": 1,   # Flag events with mismatched plates
    "review_required": True            # Require manual review for anomalies
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_file": "vehicle_tracking.log",
    "max_log_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "max_concurrent_processes": 4,
    "processing_timeout_seconds": 60,
    "database_commit_interval": 10,  # Commit to database every N events
    "cleanup_interval_hours": 24     # Cleanup old unmatched events every N hours
}

# Paths configuration
PATHS_CONFIG = {
    "image_storage": "./captured_images",
    "processed_images": "./processed_images",
    "log_directory": "./logs",
    "backup_directory": "./backups"
}

# Create directories if they don't exist
for path_key, path_value in PATHS_CONFIG.items():
    if path_key.endswith("_directory") or path_key.endswith("_images"):
        os.makedirs(path_value, exist_ok=True)