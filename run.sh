#!/bin/bash

# Run script for Signature Forgery Detection System

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if directories exist
if [ ! -d "output/ensemble_models" ] || [ ! -d "output/visualizations" ] || [ ! -d "uploads" ] || [ ! -d "templates" ] || [ ! -d "test_samples" ]; then
    echo "Creating required directories..."
    mkdir -p output/ensemble_models
    mkdir -p output/visualizations
    mkdir -p uploads
    mkdir -p templates
    mkdir -p test_samples
fi

# Check if model exists
if [ ! -f "output/ensemble_models/simple_ensemble.pth" ]; then
    echo "Warning: Model file not found at output/ensemble_models/simple_ensemble.pth"
    echo "The system will run in fallback mode without model predictions."
fi

# Check if port 5001 is available
PORT=$(grep -o "port=[0-9]*" app.py | cut -d= -f2)
if command -v lsof >/dev/null 2>&1; then
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo "Warning: Port $PORT is already in use."
        echo "You can run ./fix_port.sh to resolve this issue."
        echo "Do you want to continue anyway? (y/n)"
        read -r response
        if [[ "$response" != "y" ]]; then
            echo "Exiting. Run ./fix_port.sh to fix the port issue."
            exit 1
        fi
    fi
fi

# Start the Flask application
echo "Starting the Signature Forgery Detection System on port $PORT..."
python app.py 