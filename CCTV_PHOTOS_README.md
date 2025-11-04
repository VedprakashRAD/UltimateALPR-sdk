# CCTV Photos Directory

This directory is dedicated to storing all images captured by the CCTV camera system for vehicle tracking.

## Purpose

The CCTV_photos directory serves as the central repository for all vehicle-related images captured by the system, including:
- Entry and exit vehicle images
- Detected license plates
- Manual captures
- Simulation images

## Directory Structure

```
CCTV_photos/
├── entry_*.jpg        # Vehicle entry images
├── exit_*.jpg         # Vehicle exit images
├── plate_*.jpg        # Detected license plate images
├── manual_*.jpg       # Manually captured images
├── sim_entry_*.jpg    # Simulation entry images
├── sim_exit_*.jpg     # Simulation exit images
└── auto_detect_*.jpg  # Auto-detected vehicle images
```

## Benefits

1. **Centralized Storage**: All CCTV-related images are stored in one location
2. **Easy Access**: Images can be easily accessed and managed
3. **Organization**: Clear naming conventions for different types of images
4. **Backup Ready**: Central location makes it easy to backup all images
5. **Compliance**: Organized storage for security and compliance purposes

## Usage

The system automatically saves images to this directory when:
- A vehicle is detected entering the monitored area
- A vehicle is detected exiting the monitored area
- License plates are successfully detected and processed
- Manual captures are triggered by the user
- Simulation events are generated

## Configuration

The CCTV_photos directory path can be configured in:
`src/database/vehicle_tracking_config.py`

```python
PATHS_CONFIG = {
    "image_storage": "./CCTV_photos",  # Main CCTV photos directory
    # ... other paths
}
```

## Maintenance

- Regular cleanup of old images may be necessary to manage disk space
- Images are stored in JPEG format for optimal balance of quality and file size
- All images are timestamped for easy identification and sorting