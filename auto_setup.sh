#!/bin/bash

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "Please run as root (use sudo)"
  exit
fi

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Python 3.8+ and pip
echo "Installing Python 3.8+ and pip..."
sudo apt-get install -y python3 python3-pip

# Install Python dependencies using pip
echo "Installing Python dependencies from requirements.txt..."
pip3 install -r requirements.txt

echo "All dependencies installed successfully. You can now run the vPIN python files."

