#!/bin/bash

# Setup virtual environment for Vehicle Tracking System

echo "🔧 Setting up virtual environment for Vehicle Tracking System..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "📥 Installing required packages..."
pip install opencv-python
pip install pymongo
pip install psutil
pip install pytesseract
pip install numpy

# Create requirements.txt
echo "📝 Creating requirements.txt..."
pip freeze > requirements.txt

echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo ""
echo "Requirements have been saved to requirements.txt"