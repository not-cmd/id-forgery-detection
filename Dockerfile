# Build stage for frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend1/project/package*.json ./
RUN npm install
COPY frontend1/project/ ./
RUN npm run build

# Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy frontend build from previous stage
COPY --from=frontend-build /app/frontend/dist /app/frontend1/project/dist

# Create necessary directories
RUN mkdir -p uploads results static models

# Copy model file
COPY models/yolov8n.pt /app/models/

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5002

# Start the application
CMD ["python", "app.py"] 