# Python SDK Setup for ultimateALPR-SDK

This document describes how to set up and use the Python wrapper for the ultimateALPR-SDK.

## Overview

Due to cross-platform compatibility issues (x86_64 vs ARM64), we've created a Python wrapper that uses Docker containers to run the ultimateALPR-SDK binaries. This approach avoids the need to build native extensions and handles cross-platform compatibility seamlessly.

## Prerequisites

1. Docker installed and running
2. Python 3.6 or later
3. Virtual environment (already created as `alpr_venv`)

## Setup Instructions

### 1. Activate the Virtual Environment

```bash
source alpr_venv/bin/activate
```

### 2. Install Required Python Packages

The required packages should already be installed in the virtual environment:

- Pillow (for image handling)

To verify:
```bash
pip list
```

### 3. Ensure Docker Image is Available

The Docker image should already be built as `alpr-arm64`. To verify:

```bash
docker images | grep alpr-arm64
```

If the image is missing, rebuild it using the ARM64 Dockerfile:

```bash
docker build -t alpr-arm64 -f arm64.Dockerfile .
```

## Usage

### Basic Usage

```python
from python_docker_wrapper import UltimateALPRSDK

# Initialize the SDK
sdk = UltimateALPRSDK()

# Process an image
result = sdk.process_image("path/to/your/image.jpg")

# Extract plate details
plates = sdk.get_plate_details(result)

# Print results
for plate in plates:
    print(f"Plate: {plate['text']} (Confidence: {plate['confidence']:.2f}%)")
```

### Running the Demo

```bash
python demo_python_sdk.py
```

## How It Works

1. The Python wrapper uses `subprocess` to run Docker commands
2. It mounts the image directory and assets directory into the container
3. The recognizer processes the image and outputs results to stderr
4. The wrapper parses the JSON output to extract plate information
5. Results are returned as Python dictionaries for easy processing

## Files

- `python_docker_wrapper.py` - Main Python wrapper class
- `demo_python_sdk.py` - Demo script showing usage
- `test_python_wrapper_simple.py` - Simple test script
- `debug_docker_output.py` - Debug script for troubleshooting
- `requirements.txt` - Python package requirements

## Troubleshooting

### Docker Image Not Found

If you get an error about the Docker image not being found:

```bash
docker build -t alpr-arm64 -f arm64.Dockerfile .
```

### Permission Issues

Make sure Docker is running and you have permission to run Docker commands.

### No Plates Detected

- Ensure the image path is correct
- Check that the image contains visible license plates
- Verify that the assets directory is properly mounted

## Limitations

- Requires Docker to be installed
- Processing speed depends on Docker container startup time
- Only works with images, not real-time video streams (in this implementation)

## Extending Functionality

You can extend the wrapper by:

1. Adding support for additional recognizer options
2. Implementing batch processing for multiple images
3. Adding support for different output formats
4. Implementing video processing capabilities