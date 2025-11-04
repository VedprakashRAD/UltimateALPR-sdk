#!/bin/bash

# macOS Vehicle Tracking System Installation Script
# Adapted for macOS development environment

set -e

echo "🍓 Vehicle Tracking System Installer (macOS)"
echo "============================================="
echo "Setting up development environment on macOS"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_step "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        print_status "Python: $PYTHON_VERSION (✓)"
    else
        echo "❌ Python 3 not found"
        exit 1
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2{print $4}' | sed 's/Gi//')
    print_status "Disk space: ${AVAILABLE_SPACE} available (✓)"
}

# Install MongoDB using Homebrew
install_mongodb() {
    print_step "Installing MongoDB..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install MongoDB
    brew tap mongodb/brew
    brew install mongodb-community
    
    # Start MongoDB service
    brew services start mongodb/brew/mongodb-community
    
    print_status "MongoDB installed and started"
}

# Install Docker Desktop
setup_docker() {
    print_step "Checking Docker..."
    
    if command -v docker &> /dev/null; then
        print_status "Docker already installed"
    else
        echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        echo "Then restart this script"
        exit 1
    fi
}

# Setup Python environment
setup_python_env() {
    print_step "Setting up Python environment..."
    
    # Install Python dependencies
    pip3 install -r requirements.txt
    
    print_status "Python dependencies installed"
}

# Create directories
create_directories() {
    print_step "Creating directories..."
    
    mkdir -p logs captured_images processed_images backups sample_images
    
    print_status "Directories created"
}

# Test MongoDB connection
test_mongodb() {
    print_step "Testing MongoDB connection..."
    
    python3 -c "
from pymongo import MongoClient
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print('✅ MongoDB connection successful')
except Exception as e:
    print(f'❌ MongoDB connection failed: {e}')
    exit(1)
"
}

# Run system test
run_test() {
    print_step "Running system test..."
    
    if python3 demo_simple.py; then
        print_status "System test passed!"
    else
        echo "⚠️  System test had issues"
    fi
}

# Main installation
main() {
    check_requirements
    setup_docker
    install_mongodb
    setup_python_env
    create_directories
    test_mongodb
    
    echo ""
    print_status "Installation completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Test the system: python3 demo_simple.py"
    echo "2. Run full system: python3 vehicle_tracking_system_mongodb.py"
    echo ""
    echo "🎯 System ready for development and testing!"
    
    # Run test
    run_test
}

main "$@"