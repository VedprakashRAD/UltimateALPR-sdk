#!/bin/bash

# Raspberry Pi Vehicle Tracking System Installation Script
# Optimized for 8GB Raspberry Pi using only 4GB RAM with MongoDB

set -e

echo "🍓 Raspberry Pi Vehicle Tracking System Installer"
echo "=================================================="
echo "Optimizing for 4GB RAM usage on 8GB Raspberry Pi"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    print_step "Checking if running on Raspberry Pi..."
    
    if [[ ! -f /proc/device-tree/model ]] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
        print_warning "This script is optimized for Raspberry Pi. Continuing anyway..."
    else
        PI_MODEL=$(cat /proc/device-tree/model)
        print_status "Detected: $PI_MODEL"
    fi
}

# Check system requirements
check_requirements() {
    print_step "Checking system requirements..."
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk 'NR==2{print $2}')
    if [ "$TOTAL_RAM" -lt 6 ]; then
        print_error "Insufficient RAM. Need at least 6GB, found ${TOTAL_RAM}GB"
        exit 1
    fi
    print_status "RAM: ${TOTAL_RAM}GB (✓)"
    
    # Check disk space
    AVAILABLE_SPACE=$(df / | awk 'NR==2{print int($4/1024/1024)}')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        print_error "Insufficient disk space. Need at least 10GB, found ${AVAILABLE_SPACE}GB"
        exit 1
    fi
    print_status "Disk space: ${AVAILABLE_SPACE}GB available (✓)"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        print_status "Python: $PYTHON_VERSION (✓)"
    else
        print_error "Python 3 not found"
        exit 1
    fi
}

# Update system packages
update_system() {
    print_step "Updating system packages..."
    
    sudo apt update
    sudo apt upgrade -y
    
    print_status "System updated"
}

# Install dependencies
install_dependencies() {
    print_step "Installing system dependencies..."
    
    sudo apt install -y \
        python3-pip \
        python3-venv \
        docker.io \
        mongodb \
        mongodb-server \
        git \
        curl \
        htop \
        iotop \
        build-essential \
        python3-dev
    
    print_status "Dependencies installed"
}

# Configure Docker
setup_docker() {
    print_step "Configuring Docker for Raspberry Pi..."
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Create Docker daemon configuration
    sudo mkdir -p /etc/docker
    
    cat << EOF | sudo tee /etc/docker/daemon.json
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "default-runtime": "runc",
    "storage-driver": "overlay2"
}
EOF
    
    # Enable and start Docker
    sudo systemctl enable docker
    sudo systemctl start docker
    
    print_status "Docker configured"
}

# Configure MongoDB
setup_mongodb() {
    print_step "Configuring MongoDB for memory optimization..."
    
    # Create MongoDB configuration optimized for Raspberry Pi
    cat << EOF | sudo tee /etc/mongod.conf
# MongoDB configuration for Raspberry Pi (4GB RAM limit)
storage:
  dbPath: /var/lib/mongodb
  journal:
    enabled: true
  engine: wiredTiger
  wiredTiger:
    engineConfig:
      cacheSizeGB: 1

systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log

net:
  port: 27017
  bindIp: 127.0.0.1

processManagement:
  fork: true
  pidFilePath: /var/run/mongodb/mongod.pid

# Memory optimization
setParameter:
  wiredTigerConcurrentReadTransactions: 64
  wiredTigerConcurrentWriteTransactions: 64
EOF
    
    # Enable and start MongoDB
    sudo systemctl enable mongodb
    sudo systemctl start mongodb
    
    # Wait for MongoDB to start
    sleep 5
    
    # Test MongoDB connection
    if mongo --eval "db.stats()" > /dev/null 2>&1; then
        print_status "MongoDB configured and running"
    else
        print_error "MongoDB failed to start"
        exit 1
    fi
}

# Setup Python environment
setup_python_env() {
    print_step "Setting up Python virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "alpr_venv" ]; then
        python3 -m venv alpr_venv
    fi
    
    # Activate virtual environment and install packages
    source alpr_venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    pip install -r requirements.txt
    
    print_status "Python environment configured"
}

# Configure system optimization
optimize_system() {
    print_step "Applying system optimizations..."
    
    # Memory optimization settings
    cat << EOF | sudo tee -a /etc/sysctl.conf
# Vehicle tracking system optimizations
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.dirty_background_ratio=5
vm.dirty_ratio=10
EOF
    
    # GPU memory split (reduce GPU memory for more system RAM)
    if ! grep -q "gpu_mem=64" /boot/config.txt; then
        echo "gpu_mem=64" | sudo tee -a /boot/config.txt
    fi
    
    # Apply sysctl settings
    sudo sysctl -p
    
    print_status "System optimizations applied"
}

# Create systemd service
create_service() {
    print_step "Creating systemd service..."
    
    cat << EOF | sudo tee /etc/systemd/system/vehicle-tracking.service
[Unit]
Description=Vehicle Tracking System
After=network.target mongodb.service docker.service
Requires=mongodb.service docker.service

[Service]
Type=simple
User=pi
WorkingDirectory=$(pwd)
Environment=PYTHONPATH=$(pwd)
ExecStart=$(pwd)/alpr_venv/bin/python vehicle_tracking_system_mongodb.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vehicle-tracking

# Memory limits
MemoryLimit=3G
MemoryAccounting=true

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable vehicle-tracking
    
    print_status "Systemd service created"
}

# Setup log rotation
setup_logging() {
    print_step "Setting up log rotation..."
    
    cat << EOF | sudo tee /etc/logrotate.d/vehicle-tracking
/var/log/vehicle_tracking.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pi pi
}

/var/log/mongodb/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        /bin/kill -SIGUSR1 \$(cat /var/run/mongodb/mongod.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
EOF
    
    print_status "Log rotation configured"
}

# Create directories
create_directories() {
    print_step "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p captured_images
    mkdir -p processed_images
    mkdir -p backups
    mkdir -p sample_images
    
    print_status "Directories created"
}

# Run system test
run_test() {
    print_step "Running system test..."
    
    # Activate virtual environment
    source alpr_venv/bin/activate
    
    # Run the demo script
    if python3 demo_raspberry_pi_tracking.py; then
        print_status "System test passed!"
    else
        print_warning "System test had issues, but installation completed"
    fi
}

# Main installation function
main() {
    echo "Starting installation..."
    echo ""
    
    check_raspberry_pi
    check_requirements
    update_system
    install_dependencies
    setup_docker
    setup_mongodb
    setup_python_env
    optimize_system
    create_directories
    create_service
    setup_logging
    
    echo ""
    print_status "Installation completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Reboot the system: sudo reboot"
    echo "2. After reboot, test the system: python3 demo_raspberry_pi_tracking.py"
    echo "3. Start the service: sudo systemctl start vehicle-tracking"
    echo "4. Check service status: sudo systemctl status vehicle-tracking"
    echo ""
    echo "📊 Monitoring Commands:"
    echo "- Memory usage: watch -n 1 free -h"
    echo "- System logs: journalctl -u vehicle-tracking -f"
    echo "- MongoDB status: mongo --eval \"db.stats()\""
    echo ""
    echo "🎯 Target: System should use <4GB RAM out of 8GB available"
    echo ""
    
    # Ask if user wants to run test now
    read -p "Run system test now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_test
    fi
    
    print_status "Setup complete! 🎉"
}

# Check if script is run as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

# Run main installation
main "$@"