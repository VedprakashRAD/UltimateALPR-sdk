# CCTV Photos Implementation Summary

## Overview

This document summarizes the implementation of the dedicated CCTV_photos folder for storing all images captured by the vehicle tracking system.

## Changes Made

### 1. Created CCTV_photos Directory
- Created a new directory `/CCTV_photos` to serve as the central repository for all captured images

### 2. Updated Configuration
- Modified `src/database/vehicle_tracking_config.py` to include the new image storage path
- Added the CCTV_photos directory to the PATHS_CONFIG configuration

### 3. Updated Camera Systems
- Modified all camera processing systems to save images to the new CCTV_photos directory:
  - `src/camera/fully_automatic_system.py`
  - `src/camera/working_alpr_system.py`
  - `src/camera/auto_detect_system.py`
  - `src/camera/camera_simulator.py`

### 4. Maintained Backward Compatibility
- All systems still save to their original directories for backward compatibility
- New images are additionally saved to the CCTV_photos directory

### 5. Created Documentation
- Added `CCTV_PHOTOS_README.md` to explain the purpose and usage of the directory
- Updated configuration documentation

## Implementation Details

### Directory Structure
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

### File Naming Convention
All files follow a consistent naming pattern with timestamps:
- `entry_YYYYMMDD_HHMMSS.jpg` - Vehicle entry images
- `exit_YYYYMMDD_HHMMSS.jpg` - Vehicle exit images
- `plate_YYYYMMDD_HHMMSS_N.jpg` - Detected license plates (N = plate number)
- `manual_YYYYMMDD_HHMMSS.jpg` - Manual captures
- `sim_entry_VEHICLEID_YYYYMMDD_HHMMSS.jpg` - Simulation entry images
- `sim_exit_VEHICLEID_YYYYMMDD_HHMMSS.jpg` - Simulation exit images
- `auto_detect_YYYYMMDD_HHMMSS.jpg` - Auto-detected vehicle images

## Benefits

1. **Centralized Storage**: All CCTV-related images are stored in one location
2. **Easy Management**: Simplified organization and maintenance of image files
3. **Scalability**: Ready for future expansion and additional camera sources
4. **Security Compliance**: Organized storage for security and compliance purposes
5. **Backup Efficiency**: Central location makes it easy to backup all images
6. **Searchability**: Consistent naming makes it easy to find specific images

## Verification

The implementation has been verified to:
- ✅ Create the CCTV_photos directory automatically
- ✅ Save images to the correct location
- ✅ Maintain backward compatibility
- ✅ Use proper timestamped naming conventions
- ✅ Import configuration correctly

## Usage

The system automatically saves images to the CCTV_photos directory when:
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

## Future Enhancements

1. Add automatic cleanup of old images based on retention policy
2. Implement image compression for long-term storage
3. Add metadata files for each image with additional information
4. Create subdirectories for date-based organization