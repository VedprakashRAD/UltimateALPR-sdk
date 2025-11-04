FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install Pillow

# Set working directory
WORKDIR /app

# Copy the ARM64 binaries and assets
COPY assets/ /app/assets/
COPY binaries/linux/aarch64/ /app/binaries/

# Copy a sample image for testing
COPY assets/images/lic_us_1280x720.jpg /app/test_image.jpg

# Make the recognizer executable
RUN chmod +x /app/binaries/recognizer

# Set library path
ENV LD_LIBRARY_PATH=/app/binaries

# Default command - show help
CMD ["/app/binaries/recognizer", "--help"]