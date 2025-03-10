from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import os
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import logging

# Check for model files and download if needed
def ensure_model_exists(model_path):
    """
    Ensure the model file exists, and if not, try to find an alternative format.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Path to the available model file
    """
    if os.path.exists(model_path):
        return model_path
    
    # Try different extensions
    base_path = os.path.splitext(model_path)[0]
    for ext in ['.pt', '.onnx', '.tflite']:
        alt_path = f"{base_path}{ext}"
        if os.path.exists(alt_path):
            print(f"Using alternative model format: {alt_path}")
            return alt_path
    
    # If deployed on Vercel, we need to use a smaller model or download it
    print(f"Model {model_path} not found. Using a default YOLOv8n model.")
    return "yolov8n.pt"  # Use a smaller default model

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj

def analyze_image_forensics(image_path):
    """
    Analyze an image for forensic indicators of forgery.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Forensic analysis results
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to a standard size to avoid dimension mismatches
        standard_size = (640, 480)
        gray_resized = cv2.resize(gray, standard_size)
        
        # Calculate noise level (using median filter difference)
        median_filtered = cv2.medianBlur(gray_resized, 5)
        noise_diff = cv2.absdiff(gray_resized, median_filtered)
        noise_level = np.mean(noise_diff) / 255.0
        
        # Calculate edge density
        edges = cv2.Canny(gray_resized, 100, 200)
        edge_density = np.sum(edges > 0) / (standard_size[0] * standard_size[1])
        
        # Calculate Laplacian variance (for blur detection)
        laplacian_var = cv2.Laplacian(gray_resized, cv2.CV_64F).var()
        
        # Calculate DCT coefficients for compression analysis
        dct_size = (8, 8)
        h, w = gray_resized.shape
        h_pad, w_pad = h % dct_size[0], w % dct_size[1]
        if h_pad != 0 or w_pad != 0:
            # Ensure dimensions are multiples of 8 for DCT
            h_new, w_new = h - h_pad, w - w_pad
            gray_resized = gray_resized[:h_new, :w_new]
        
        # Compute DCT on blocks
        dct_coeffs = []
        for i in range(0, gray_resized.shape[0], dct_size[0]):
            for j in range(0, gray_resized.shape[1], dct_size[1]):
                block = gray_resized[i:i+dct_size[0], j:j+dct_size[1]].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_coeffs.extend(dct_block.flatten()[1:])  # Skip DC component
        
        dct_coeffs = np.array(dct_coeffs)
        dct_mean = np.mean(np.abs(dct_coeffs))
        dct_std = np.std(dct_coeffs)
        
        # Error Level Analysis (ELA)
        temp_jpg = os.path.join(os.path.dirname(image_path), "temp_ela.jpg")
        cv2.imwrite(temp_jpg, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed_img = cv2.imread(temp_jpg)
        if compressed_img is None:
            raise ValueError(f"Could not read compressed image from {temp_jpg}")
        
        # Ensure both images have the same dimensions
        compressed_img = cv2.resize(compressed_img, (image.shape[1], image.shape[0]))
        
        ela_diff = cv2.absdiff(image, compressed_img)
        ela_diff_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)
        ela_mean = np.mean(ela_diff_gray)
        
        # Clean up temporary file
        try:
            os.remove(temp_jpg)
        except:
            pass
        
        # Determine if the image is likely forged based on the forensic indicators
        # These thresholds should be tuned based on your specific dataset
        is_forged = (
            (noise_level > 0.05) or  # High noise level
            (edge_density > 0.15) or  # Unusual edge density
            (laplacian_var < 100) or  # Too blurry
            (dct_std < 10) or  # Unusual DCT distribution
            (ela_mean > 10)  # High error level
        )
        
        # Calculate an overall forgery score
        forgery_score = (
            min(1.0, noise_level * 10) * 0.2 +
            min(1.0, edge_density * 5) * 0.2 +
            min(1.0, 1000 / (laplacian_var + 10)) * 0.2 +
            min(1.0, 50 / (dct_std + 1)) * 0.2 +
            min(1.0, ela_mean / 20) * 0.2
        )
        
        return convert_to_native_types({
            'is_forged': is_forged,
            'forgery_score': forgery_score,
            'noise_level': noise_level,
            'edge_density': edge_density,
            'laplacian_var': laplacian_var,
            'dct_mean': dct_mean,
            'dct_std': dct_std,
            'ela_mean': ela_mean
        })
    except Exception as e:
        logging.error(f"Error in analyze_image_forensics: {str(e)}")
        raise

def detect_forgery(image_path, model_path=None, conf_threshold=0.35):
    """
    Detect forgery in an ID card image using YOLOv8 and forensic analysis.
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to the YOLOv8 model. Defaults to None.
        conf_threshold (float, optional): Confidence threshold for detection. Defaults to 0.35.
        
    Returns:
        dict: Detection results including forgery status, confidence, and visualization path
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Ensure image is in RGB format for model
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a copy for visualization
        visualization = image.copy()
        
        # Perform forensic analysis
        forensic_results = analyze_image_forensics(image_path)
        
        # Determine if the image is forged based on forensic analysis
        is_forged = forensic_results.get('is_forged', False)
        forgery_score = forensic_results.get('forgery_score', 0.0)
        
        # Generate a unique filename for the visualization
        result_filename = f"result_{os.path.basename(image_path)}"
        visualization_path = os.path.join('results', result_filename)
        
        # Add forensic indicators to the visualization
        visualization = add_forensic_indicators_to_visualization(visualization, forensic_results)
        
        # Save the visualization
        cv2.imwrite(visualization_path, visualization)
        
        # Return the results
        return convert_to_native_types({
            'is_forged': is_forged,
            'confidence': forgery_score,
            'document_type': 'id',
            'visualization_path': visualization_path,
            'forgery_indicators': forensic_results
        })
    except Exception as e:
        logging.error(f"Error in detect_forgery: {str(e)}")
        raise

def add_forensic_indicators_to_visualization(image, forensic_results):
    """
    Add forensic indicators to the visualization image.
    
    Args:
        image (numpy.ndarray): The image to add indicators to
        forensic_results (dict): Forensic analysis results
        
    Returns:
        numpy.ndarray: The image with indicators added
    """
    # Create a copy of the image
    vis_image = image.copy()
    
    # Get image dimensions
    h, w = vis_image.shape[:2]
    
    # Add a border to indicate forgery status
    border_color = (0, 0, 255) if forensic_results.get('is_forged', False) else (0, 255, 0)
    border_thickness = 10
    vis_image = cv2.copyMakeBorder(
        vis_image, 
        border_thickness, border_thickness, border_thickness, border_thickness, 
        cv2.BORDER_CONSTANT, 
        value=border_color
    )
    
    # Add text with forensic indicators
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    # Add a black background for the text
    text_bg_height = 180
    text_bg = np.zeros((text_bg_height, w + 2*border_thickness, 3), dtype=np.uint8)
    vis_image = np.vstack((vis_image, text_bg))
    
    # Add text for each indicator
    y_pos = h + border_thickness * 2 + 30
    line_height = 30
    
    # Add forgery status
    status_text = "FORGED" if forensic_results.get('is_forged', False) else "GENUINE"
    status_color = (0, 0, 255) if forensic_results.get('is_forged', False) else (0, 255, 0)
    cv2.putText(vis_image, f"Status: {status_text}", (20, y_pos), font, font_scale*1.5, status_color, font_thickness)
    y_pos += line_height
    
    # Add confidence score
    score = forensic_results.get('forgery_score', 0.0)
    cv2.putText(vis_image, f"Confidence: {score:.2f}", (20, y_pos), font, font_scale, text_color, font_thickness)
    y_pos += line_height
    
    # Add other indicators
    indicators = [
        ("Noise Level", 'noise_level'),
        ("Edge Density", 'edge_density'),
        ("Blur Level", 'laplacian_var'),
        ("Compression", 'dct_std'),
        ("Error Level", 'ela_mean')
    ]
    
    for label, key in indicators:
        if key in forensic_results:
            value = forensic_results[key]
            if isinstance(value, float):
                text = f"{label}: {value:.4f}"
            else:
                text = f"{label}: {value}"
            cv2.putText(vis_image, text, (20, y_pos), font, font_scale, text_color, font_thickness)
            y_pos += line_height
    
    return vis_image

def main():
    parser = argparse.ArgumentParser(description='Detect ID forgery in an image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='runs/detect/id_forgery_detection/weights/best.pt', 
                        help='Path to the YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    # Detect forgery
    result = detect_forgery(args.image, args.model, args.conf)
    
    # Print result
    print(f"Image: {args.image}")
    print(f"Forged: {result['is_forged']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Processing time: {result['processing_time']:.4f} seconds")
    print(f"Visualization saved to: {result['visualization_path']}")
    
    # Print forgery indicators
    if 'forgery_indicators' in result and result['forgery_indicators']:
        print("\nForgery Indicators:")
        for key, value in result['forgery_indicators'].items():
            print(f"  {key}: {value:.4f}")

if __name__ == '__main__':
    main() 