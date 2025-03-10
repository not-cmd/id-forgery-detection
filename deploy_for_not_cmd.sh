#!/bin/bash
# Simplified deployment script for GitHub user "not-cmd" and PythonAnywhere user "awesomeddd"

echo "=== Starting Deployment Process for not-cmd ==="

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
mkdir -p $DEPLOY_DIR/frontend
cp -r Frontend/project/dist/* $DEPLOY_DIR/frontend/

# Prepare backend for PythonAnywhere
echo "Preparing backend..."
python3 prepare_pythonanywhere.py

# Copy necessary backend files
mkdir -p $DEPLOY_DIR/backend
cp -r app.py app_pythonanywhere.py wsgi_app.py requirements.txt model.py visualization.py id_anomaly_detector.py $DEPLOY_DIR/backend/
cp -r deployment_models $DEPLOY_DIR/backend/
cp -r static $DEPLOY_DIR/backend/
cp PYTHONANYWHERE_DEPLOY.md $DEPLOY_DIR/
cp PYTHONANYWHERE_SIMPLE_GUIDE.md $DEPLOY_DIR/
cp GITHUB_PAGES_SIMPLE_GUIDE.md $DEPLOY_DIR/

# Create GitHub repository setup script
cat > $DEPLOY_DIR/setup_github_repo.sh << EOL
#!/bin/bash
# Script to set up GitHub repository for not-cmd

cd frontend
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/not-cmd/signature-forgery-frontend.git
git push -u origin main

echo "Frontend code pushed to GitHub repository"
echo "Now go to https://github.com/not-cmd/signature-forgery-frontend/settings/pages"
echo "and enable GitHub Pages with the following settings:"
echo "- Source: Deploy from a branch"
echo "- Branch: main"
echo "- Folder: / (root)"
EOL

chmod +x $DEPLOY_DIR/setup_github_repo.sh

echo "=== Deployment Package Created ==="
echo "Your deployment package is ready in the '$DEPLOY_DIR' directory"
echo ""
echo "To deploy the frontend to GitHub:"
echo "1. Create a new repository named 'signature-forgery-frontend' at https://github.com/new"
echo "2. Run the setup script: cd $DEPLOY_DIR && ./setup_github_repo.sh"
echo ""
echo "To deploy the backend to PythonAnywhere:"
echo "1. Follow the instructions in $DEPLOY_DIR/PYTHONANYWHERE_SIMPLE_GUIDE.md"
echo ""
echo "Your frontend will be available at: https://not-cmd.github.io/signature-forgery-frontend/"
echo "Your backend will be available at: https://awesomeddd.pythonanywhere.com/" 