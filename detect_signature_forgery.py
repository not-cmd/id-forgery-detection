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

def extract_signature_features(image):
    """
    Extract features from a signature image for forgery detection.
    
    Args:
        image: The signature image
        
    Returns:
        dict: Dictionary containing extracted features
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize for consistent feature extraction
    gray = cv2.resize(gray, (256, 256))
    
    # 1. Contour-based features
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate contour features
    contour_area = 0
    contour_perimeter = 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        contour_perimeter = cv2.arcLength(largest_contour, True)
    
    # 2. Texture features
    # GLCM (Gray-Level Co-occurrence Matrix) features
    glcm_energy = np.sum(gray**2) / (gray.shape[0] * gray.shape[1])
    
    # 3. Gradient features
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)
    
    # 4. DCT coefficients
    dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    dct_mean = np.mean(np.abs(dct_coeffs))
    dct_std = np.std(np.abs(dct_coeffs))
    
    # 5. Stroke width variation
    # Estimate stroke width using distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_width_mean = np.mean(dist_transform[dist_transform > 0])
    stroke_width_std = np.std(dist_transform[dist_transform > 0])
    
    # 6. Pressure points (estimated by intensity variations)
    pressure_points = np.sum(gray < 50) / np.sum(binary > 0) if np.sum(binary > 0) > 0 else 0
    
    # Return all features
    return {
        'contour_area': float(contour_area),
        'contour_perimeter': float(contour_perimeter),
        'aspect_ratio': float(binary.shape[1] / binary.shape[0]),
        'glcm_energy': float(glcm_energy),
        'gradient_mean': float(gradient_mean),
        'gradient_std': float(gradient_std),
        'dct_mean': float(dct_mean),
        'dct_std': float(dct_std),
        'stroke_width_mean': float(stroke_width_mean) if not np.isnan(stroke_width_mean) else 0.0,
        'stroke_width_std': float(stroke_width_std) if not np.isnan(stroke_width_std) else 0.0,
        'pressure_points': float(pressure_points),
        'pixel_density': float(np.sum(binary > 0) / (binary.shape[0] * binary.shape[1]))
    }

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

def detect_signature_forgery(image_path, model_path=None, conf_threshold=0.35):
    """
    Detect forgery in a signature image.
    
    Args:
        image_path (str): Path to the image file
        model_path (str, optional): Path to the model. Defaults to None.
        conf_threshold (float, optional): Confidence threshold. Defaults to 0.35.
        
    Returns:
        dict: Detection results
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Create a copy for visualization
        visualization = image.copy()
        
        # Perform forensic analysis
        forensic_results = analyze_signature_forensics(image_path)
        
        # Determine if the signature is forged based on forensic analysis
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
            'document_type': 'signature',
            'visualization_path': visualization_path,
            'forgery_indicators': forensic_results
        })
    except Exception as e:
        logging.error(f"Error in detect_signature_forgery: {str(e)}")
        raise

def analyze_signature_forensics(image_path):
    """
    Analyze a signature image for forensic indicators of forgery.
    
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
        standard_size = (400, 200)
        gray_resized = cv2.resize(gray, standard_size)
        
        # Binarize the image
        _, binary = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate stroke width variation
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        stroke_width = cv2.subtract(binary, eroded)
        stroke_width_mean = np.mean(stroke_width[stroke_width > 0]) if np.sum(stroke_width > 0) > 0 else 0
        stroke_width_std = np.std(stroke_width[stroke_width > 0]) if np.sum(stroke_width > 0) > 0 else 0
        
        # Calculate contour features
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate contour complexity
        contour_complexity = 0
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(main_contour, True)
            area = cv2.contourArea(main_contour)
            contour_complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
        
        # Calculate pressure points (darker regions in the signature)
        pressure_points = np.sum(gray_resized < 50) / np.sum(binary > 0) if np.sum(binary > 0) > 0 else 0
        
        # Calculate tremor features
        gradient_x = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        tremor_level = np.std(gradient_magnitude[binary > 0]) if np.sum(binary > 0) > 0 else 0
        
        # Calculate line quality
        line_quality = cv2.Laplacian(gray_resized, cv2.CV_64F).var()
        
        # Calculate signature density
        signature_density = np.sum(binary > 0) / (standard_size[0] * standard_size[1])
        
        # Determine if the signature is likely forged based on the forensic indicators
        # These thresholds should be tuned based on your specific dataset
        is_forged = (
            (stroke_width_std > 20) or  # High stroke width variation
            (contour_complexity > 5) or  # Overly complex contour
            (pressure_points < 0.1) or  # Few pressure points
            (tremor_level > 50) or  # High tremor level
            (line_quality < 100)  # Poor line quality
        )
        
        # Calculate an overall forgery score
        forgery_score = (
            min(1.0, stroke_width_std / 30) * 0.2 +
            min(1.0, contour_complexity / 10) * 0.2 +
            min(1.0, 0.3 / (pressure_points + 0.1)) * 0.2 +
            min(1.0, tremor_level / 100) * 0.2 +
            min(1.0, 200 / (line_quality + 1)) * 0.2
        )
        
        return {
            'is_forged': is_forged,
            'forgery_score': forgery_score,
            'stroke_width_mean': float(stroke_width_mean),
            'stroke_width_std': float(stroke_width_std),
            'contour_complexity': float(contour_complexity),
            'pressure_points': float(pressure_points),
            'tremor_level': float(tremor_level),
            'line_quality': float(line_quality),
            'signature_density': float(signature_density)
        }
    except Exception as e:
        logging.error(f"Error in analyze_signature_forensics: {str(e)}")
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
        ("Stroke Width", 'stroke_width_std'),
        ("Contour Complexity", 'contour_complexity'),
        ("Pressure Points", 'pressure_points'),
        ("Tremor Level", 'tremor_level'),
        ("Line Quality", 'line_quality')
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
    parser = argparse.ArgumentParser(description='Detect signature forgery in an image')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='runs/signature_detection/weights/best.pt', 
                        help='Path to the YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    # Detect forgery
    result = detect_signature_forgery(args.image, args.model, args.conf)
    
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