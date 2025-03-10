import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import time
import shutil
from detect_id_forgery import detect_forgery
from detect_signature_forgery import detect_signature_forgery
import json
import threading
from flask_cors import CORS
import logging
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes with all origins

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ID_MODEL_PATH = os.environ.get('ID_MODEL_PATH', 'runs/detect/id_forgery_detection/weights/best.pt')
SIGNATURE_MODEL_PATH = os.environ.get('SIGNATURE_MODEL_PATH', 'runs/signature_detection/weights/best.pt')
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.35'))

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def convert_bools(obj):
    """Convert non-JSON-serializable types to serializable ones."""
    if isinstance(obj, dict):
        return {k: convert_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bools(elem) for elem in obj]
    elif isinstance(obj, bool) or isinstance(obj, np.bool_):
        return int(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_page():
    return render_template('test.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    document_type = request.form.get('document_type', 'id')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Generate a unique filename to avoid conflicts
            unique_id = str(uuid.uuid4())
            filename = secure_filename(f"{unique_id}_{file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            logger.info(f"Processing file: {filename}, type: {document_type}")
            
            result = None
            if document_type == 'signature':
                result = detect_signature_forgery(file_path)
            else:
                result = detect_forgery(file_path)
            
            # Convert non-serializable types to JSON-serializable ones
            result = convert_bools(result)
            
            # Add the visualization path to the result
            if 'visualization_path' in result and result['visualization_path']:
                # Convert to relative URL path
                vis_path = result['visualization_path']
                if os.path.exists(vis_path):
                    # Get just the filename from the path
                    vis_filename = os.path.basename(vis_path)
                    result['visualization_path'] = f"/results/{vis_filename}"
                    logger.info(f"Visualization path: {result['visualization_path']}")
            
            # Clean up the uploaded file to avoid filling up disk space
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {file_path}: {str(e)}")
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/results/<path:filename>')
def result_file(filename):
    logger.info(f"Serving result file: {filename}")
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    logger.info(f"Serving uploaded file: {filename}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        if 'genuine_files' not in request.files or 'forged_files' not in request.files:
            return jsonify({"error": "Missing files"}), 400
        
        # Create temporary directories for training data
        genuine_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'genuine_training')
        forged_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'forged_training')
        os.makedirs(genuine_dir, exist_ok=True)
        os.makedirs(forged_dir, exist_ok=True)
        
        # Save genuine files
        genuine_files = request.files.getlist('genuine_files')
        for file in genuine_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(genuine_dir, filename)
                file.save(filepath)
        
        # Save forged files
        forged_files = request.files.getlist('forged_files')
        for file in forged_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(forged_dir, filename)
                file.save(filepath)
        
        # Determine document type (ID card or signature)
        document_type = request.form.get('document_type', 'id')
        
        # Start training in a background thread
        def train_in_background():
            try:
                import subprocess
                
                if document_type == 'signature':
                    output_dir = 'user_signature_model'
                    cmd = [
                        'python', 'upload_training_data.py',
                        '--genuine-dir', genuine_dir,
                        '--forged-dir', forged_dir,
                        '--output-dir', output_dir,
                        '--augment'
                    ]
                    subprocess.run(cmd, check=True)
                    
                    # Train the model
                    cmd = [
                        'python', 'train_signature_detection.py',
                        '--dataset', output_dir,
                        '--epochs', '50',
                        '--batch-size', '8'
                    ]
                    subprocess.run(cmd, check=True)
                else:  # Default to ID card
                    output_dir = 'user_id_model'
                    cmd = [
                        'python', 'upload_training_data.py',
                        '--genuine-dir', genuine_dir,
                        '--forged-dir', forged_dir,
                        '--output-dir', output_dir,
                        '--augment'
                    ]
                    subprocess.run(cmd, check=True)
                    
                    # Train the model
                    cmd = [
                        'python', 'train_id_detection.py',
                        '--dataset', output_dir,
                        '--epochs', '50',
                        '--batch-size', '8'
                    ]
                    subprocess.run(cmd, check=True)
            except Exception as e:
                logger.error(f"Error during training: {str(e)}", exc_info=True)
        
        # Start the training thread
        training_thread = threading.Thread(target=train_in_background)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Training started in the background. This may take some time to complete."
        })
    
    except Exception as e:
        logger.error(f"Error processing training request: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False) 