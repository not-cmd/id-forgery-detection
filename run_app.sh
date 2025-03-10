#!/bin/bash

# Kill any existing processes on ports 5002 and 3000-3002
echo "Checking for existing processes on ports 5002 and 3000-3002..."
EXISTING_PIDS=$(lsof -t -i :5002,3000-3002)
if [ ! -z "$EXISTING_PIDS" ]; then
    echo "Killing existing processes: $EXISTING_PIDS"
    kill -9 $EXISTING_PIDS 2>/dev/null
    sleep 1
fi

# Create necessary directories if they don't exist
echo "Creating necessary directories..."
mkdir -p uploads results

# Start the Flask backend
echo "Starting Flask backend..."
cd /Users/divyeshmedidi/Hackscript
python id_forgery_app.py &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3
if ! curl -s http://localhost:5002/health > /dev/null; then
    echo "Backend failed to start. Please check for errors."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
echo "Backend started on http://localhost:5002"

# Navigate to frontend directory and start the development server
echo "Starting React frontend..."
cd /Users/divyeshmedidi/Hackscript/frontend1/project
npm install
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5
echo "Frontend should be running now. Check the output above for the URL."

# Function to handle script termination
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Register the cleanup function for when the script is terminated
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "Both servers are running."
echo "Backend: http://localhost:5002"
echo "Frontend: Check the output above for the URL (likely http://localhost:3000)"
echo "Press Ctrl+C to stop."
wait 