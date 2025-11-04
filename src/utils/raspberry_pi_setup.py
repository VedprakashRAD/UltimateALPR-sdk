#!/usr/bin/env python3
"""
Raspberry Pi Setup and Configuration Script
Optimizes system for 4GB RAM usage and sets up MongoDB
"""

import os
import sys
import subprocess
import psutil
from pathlib import Path

class RaspberryPiSetup:
    def __init__(self):
        self.memory_limit_gb = 4
        self.swap_size_gb = 2
        
    def check_system_requirements(self):
        """Check if system meets requirements."""
        print("Checking system requirements...")
        
        # Check total memory
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        
        print(f"Total system memory: {total_memory_gb:.2f}GB")
        
        if total_memory_gb < 6:
            print("WARNING: System has less than 6GB RAM. Performance may be limited.")
            return False
            
        # Check available disk space
        disk_info = psutil.disk_usage('/')
        free_space_gb = disk_info.free / (1024**3)
        
        print(f"Available disk space: {free_space_gb:.2f}GB")
        
        if free_space_gb < 10:
            print("WARNING: Less than 10GB free space available.")
            return False
            
        print("System requirements check passed ✓")
        return True
        
    def setup_memory_optimization(self):
        """Configure memory optimization settings."""
        print("Setting up memory optimization...")
        
        # Create memory optimization script
        optimization_script = """#!/bin/bash
# Memory optimization for Raspberry Pi ALPR system

# Set memory limits
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio=5' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=10' >> /etc/sysctl.conf

# GPU memory split (reduce GPU memory for more system RAM)
echo 'gpu_mem=64' >> /boot/config.txt

# Apply settings
sysctl -p
"""
        
        with open("/tmp/memory_optimization.sh", "w") as f:
            f.write(optimization_script)
            
        print("Memory optimization configured ✓")
        
    def install_mongodb(self):
        """Install and configure MongoDB for Raspberry Pi."""
        print("Installing MongoDB...")
        
        try:
            # Update package list
            subprocess.run(["sudo", "apt", "update"], check=True)
            
            # Install MongoDB
            subprocess.run([
                "sudo", "apt", "install", "-y", 
                "mongodb", "mongodb-server"
            ], check=True)
            
            # Start MongoDB service
            subprocess.run(["sudo", "systemctl", "enable", "mongodb"], check=True)
            subprocess.run(["sudo", "systemctl", "start", "mongodb"], check=True)
            
            # Configure MongoDB for low memory usage
            mongodb_config = """# MongoDB configuration for Raspberry Pi
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
"""
            
            # Write MongoDB config
            with open("/tmp/mongod.conf", "w") as f:
                f.write(mongodb_config)
                
            print("MongoDB installation completed ✓")
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing MongoDB: {e}")
            return False
            
        return True
        
    def setup_docker_optimization(self):
        """Configure Docker for memory-optimized ALPR processing."""
        print("Setting up Docker optimization...")
        
        # Docker daemon configuration for Raspberry Pi
        docker_config = {
            "log-driver": "json-file",
            "log-opts": {
                "max-size": "10m",
                "max-file": "3"
            },
            "default-runtime": "runc",
            "storage-driver": "overlay2",
            "default-ulimits": {
                "memlock": {
                    "Name": "memlock",
                    "Hard": -1,
                    "Soft": -1
                }
            }
        }
        
        # Create Docker daemon config directory
        os.makedirs("/tmp/docker", exist_ok=True)
        
        import json
        with open("/tmp/docker/daemon.json", "w") as f:
            json.dump(docker_config, f, indent=2)
            
        print("Docker optimization configured ✓")
        
    def create_systemd_service(self):
        """Create systemd service for vehicle tracking system."""
        print("Creating systemd service...")
        
        service_content = f"""[Unit]
Description=Vehicle Tracking System
After=network.target mongodb.service docker.service
Requires=mongodb.service docker.service

[Service]
Type=simple
User=pi
WorkingDirectory={os.getcwd()}
Environment=PYTHONPATH={os.getcwd()}
ExecStart=/usr/bin/python3 vehicle_tracking_system_mongodb.py
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
"""
        
        with open("/tmp/vehicle-tracking.service", "w") as f:
            f.write(service_content)
            
        print("Systemd service created ✓")
        
    def setup_log_rotation(self):
        """Setup log rotation to prevent disk space issues."""
        print("Setting up log rotation...")
        
        logrotate_config = """/var/log/vehicle_tracking.log {
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
        /bin/kill -SIGUSR1 $(cat /var/run/mongodb/mongod.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
"""
        
        with open("/tmp/vehicle-tracking-logrotate", "w") as f:
            f.write(logrotate_config)
            
        print("Log rotation configured ✓")
        
    def install_python_dependencies(self):
        """Install Python dependencies."""
        print("Installing Python dependencies...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            
            print("Python dependencies installed ✓")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing Python dependencies: {e}")
            return False
            
    def create_directories(self):
        """Create necessary directories."""
        print("Creating directories...")
        
        directories = [
            "logs",
            "captured_images",
            "processed_images",
            "backups",
            "data/mongodb"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        print("Directories created ✓")
        
    def run_setup(self):
        """Run complete setup process."""
        print("=== Raspberry Pi Vehicle Tracking System Setup ===")
        print("Optimizing for 4GB RAM usage on 8GB systems\n")
        
        if not self.check_system_requirements():
            print("System requirements not met. Exiting.")
            return False
            
        self.create_directories()
        self.setup_memory_optimization()
        
        if not self.install_mongodb():
            print("MongoDB installation failed. Exiting.")
            return False
            
        self.setup_docker_optimization()
        
        if not self.install_python_dependencies():
            print("Python dependencies installation failed. Exiting.")
            return False
            
        self.create_systemd_service()
        self.setup_log_rotation()
        
        print("\n=== Setup Complete ===")
        print("Next steps:")
        print("1. Run: sudo cp /tmp/memory_optimization.sh /usr/local/bin/ && sudo chmod +x /usr/local/bin/memory_optimization.sh")
        print("2. Run: sudo /usr/local/bin/memory_optimization.sh")
        print("3. Run: sudo cp /tmp/mongod.conf /etc/mongod.conf")
        print("4. Run: sudo cp /tmp/docker/daemon.json /etc/docker/daemon.json")
        print("5. Run: sudo cp /tmp/vehicle-tracking.service /etc/systemd/system/")
        print("6. Run: sudo cp /tmp/vehicle-tracking-logrotate /etc/logrotate.d/vehicle-tracking")
        print("7. Run: sudo systemctl daemon-reload")
        print("8. Run: sudo systemctl enable vehicle-tracking")
        print("9. Reboot the system")
        print("\nAfter reboot, the system will be ready for vehicle tracking!")
        
        return True

if __name__ == "__main__":
    setup = RaspberryPiSetup()
    setup.run_setup()