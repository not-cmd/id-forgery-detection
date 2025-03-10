#!/bin/bash
# Script to prepare deployment files for both frontend and backend

echo "=== Preparing Deployment Package ==="

# Create deployment directory
DEPLOY_DIR="deployment_package"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Prepare frontend for GitHub Pages
echo "Preparing frontend..."
cd Frontend/project

# Install dependencies and build
npm install
npm run build

# Copy frontend build to deployment package
cd ../..
cp -r Frontend/project/dist $DEPLOY_DIR/frontend

# Prepare backend for PythonAnywhere
echo "Preparing backend..."
python3 prepare_pythonanywhere.py

# Copy necessary backend files
mkdir -p $DEPLOY_DIR/backend
cp -r app.py app_pythonanywhere.py wsgi_app.py requirements.txt model.py visualization.py id_anomaly_detector.py $DEPLOY_DIR/backend/
cp -r deployment_models $DEPLOY_DIR/backend/
cp -r static $DEPLOY_DIR/backend/
cp PYTHONANYWHERE_DEPLOY.md $DEPLOY_DIR/

# Create deployment instructions
cat > $DEPLOY_DIR/README.md << EOL
# Signature Forgery Detection System - Deployment Package

This package contains everything needed to deploy the application using free resources:

## Frontend (GitHub Pages)
1. Create a new repository on GitHub
2. Push the contents of the \`frontend\` directory to the repository
3. Enable GitHub Pages in the repository settings
4. The frontend will be available at https://your-username.github.io/repository-name

## Backend (PythonAnywhere)
1. Follow the instructions in PYTHONANYWHERE_DEPLOY.md to set up the backend
2. Update the frontend's API URL in config.ts with your PythonAnywhere URL
3. The backend will be available at https://your-username.pythonanywhere.com

## Directory Structure
- frontend/: Built frontend files ready for GitHub Pages
- backend/: Backend files ready for PythonAnywhere
  - app.py: Main Flask application
  - model.py: ML model implementation
  - requirements.txt: Python dependencies
  - deployment_models/: ML model files
  - static/: Static files and uploads directory
EOL

echo "=== Deployment Package Created ==="
echo "Your deployment package is ready in the '$DEPLOY_DIR' directory"
echo "Follow the instructions in $DEPLOY_DIR/README.md to deploy your application" 