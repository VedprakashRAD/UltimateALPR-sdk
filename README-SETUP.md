# UltimateALPR-SDK Setup Guide

This document explains how to set up and use the UltimateALPR-SDK that was cloned from https://github.com/DoubangoTelecom/ultimateALPR-SDK

## Overview

The UltimateALPR-SDK is a high-performance Automatic Number/License Plate Recognition engine that works on CPUs, GPUs, VPUs, and NPUs using deep learning. It supports multiple platforms including Android, Raspberry Pi, Linux, Windows, and others.

## Prerequisites

- Docker (for running the SDK in a compatible environment)
- Python 3 (for running the demo script)

## Setup Instructions

### 1. Clone the Repository

The repository has already been cloned to your system:

```bash
cd /Users/vedprakashchaubey/Desktop/ultimateALPR-sdk
```

### 2. Build the Docker Image

We've created a Docker image that works with ARM64 processors (like Apple Silicon Macs):

```bash
docker build -f arm64.Dockerfile -t alpr-arm64 .
```

### 3. Run License Plate Recognition

#### With the sample image:

```bash
docker run --rm alpr-arm64 /app/binaries/recognizer --image /app/test_image.jpg --assets /app/assets
```

#### With your own image:

```bash
docker run --rm -v $(pwd):/workspace alpr-arm64 /app/binaries/recognizer --image /workspace/your_image.jpg --assets /app/assets
```

## Understanding the Output

The SDK will output JSON-formatted results that include:

- Detected license plate text
- Confidence scores
- Processing time
- Bounding box coordinates

Example output:
```json
{
  "duration": 180,
  "frame_id": 0,
  "plates": [
    {
      "car": {
        "confidence": 99.60938,
        "warpedBox": [78.40557, 151.3893, 1095.845, 151.3893, 1095.845, 619.1255, 78.40557, 619.1255]
      },
      "confidences": [89.93234, 99.60938, 90.45452, 90.46402, 89.93234, 89.96776, 90.43755, 90.58742, 90.51079],
      "text": "3PEDLM*",
      "warpedBox": [821.1327, 337.2377, 913.8901, 337.2377, 913.8901, 397.0274, 821.1327, 397.0274]
    }
  ]
}
```

## Features

The SDK supports many advanced features beyond basic license plate recognition:

- Image Enhancement for Night-Vision (IENV)
- License Plate Country Identification (LPCI)
- Vehicle Color Recognition (VCR)
- Vehicle Make Model Recognition (VMMR)
- Vehicle Body Style Recognition (VBSR)
- Vehicle Direction Tracking (VDT)
- Vehicle Speed Estimation (VSE)

## Performance

The SDK is optimized for high performance:
- Up to 315 fps on high-end NVIDIA GPUs
- Up to 237 fps on Intel Xeon CPUs
- Up to 64 fps on ARM devices like Khadas VIM3
- Up to 12 fps on Raspberry Pi 4

## Documentation

For complete documentation, visit:
- [Main Documentation](https://www.doubango.org/SDKs/anpr/docs/)
- [Configuration Options](https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html)
- [API Reference](https://www.doubango.org/SDKs/anpr/docs/cpp-api.html)

## Support

For technical questions, you can:
- Check the [discussion group](https://groups.google.com/forum/#!forum/doubango-ai)
- Contact the developers on [Twitter](https://twitter.com/doubangotelecom?lang=en)