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

# Copy only the essential files
COPY assets/ /app/assets/
COPY binaries/linux/x86_64/recognizer /app/
COPY samples/c++/recognizer/README.md /app/

# Make the recognizer executable
RUN chmod +x /app/recognizer

# Default command
CMD ["/app/recognizer", "--help"]