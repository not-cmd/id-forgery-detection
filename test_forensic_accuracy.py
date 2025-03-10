import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from detect_id_forgery import detect_forgery

def test_forensic_accuracy(genuine_dir, forged_dir, sample_size=None, threshold=0.5):
    """
    Test the accuracy of the forensic analysis methods.
    
    Args:
        genuine_dir: Directory containing genuine images
        forged_dir: Directory containing forged images
        sample_size: Number of images to sample from each directory (None for all)
        threshold: Confidence threshold for classification
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    # Get list of image files
    genuine_files = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    forged_files = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sample if requested
    if sample_size is not None:
        genuine_files = random.sample(genuine_files, min(sample_size, len(genuine_files)))
        forged_files = random.sample(forged_files, min(sample_size, len(forged_files)))
    
    # Initialize counters
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Store detailed results for analysis
    genuine_results = []
    forged_results = []
    
    # Test genuine images
    print(f"Testing {len(genuine_files)} genuine images...")
    for image_path in tqdm(genuine_files):
        result = detect_forgery(image_path)
        genuine_results.append(result)
        
        if result["is_forged"]:
            false_positives += 1
        else:
            true_negatives += 1
    
    # Test forged images
    print(f"Testing {len(forged_files)} forged images...")
    for image_path in tqdm(forged_files):
        result = detect_forgery(image_path)
        forged_results.append(result)
        
        if result["is_forged"]:
            true_positives += 1
        else:
            false_negatives += 1
    
    # Calculate metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create visualizations
    create_visualizations(genuine_results, forged_results)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "genuine_results": genuine_results,
        "forged_results": forged_results
    }

def create_visualizations(genuine_results, forged_results):
    """Create visualizations of the forensic indicators"""
    # Extract forensic indicators
    genuine_indicators = {key: [] for key in genuine_results[0]["forgery_indicators"].keys()}
    forged_indicators = {key: [] for key in forged_results[0]["forgery_indicators"].keys()}
    
    for result in genuine_results:
        for key, value in result["forgery_indicators"].items():
            if isinstance(value, (int, float)):
                genuine_indicators[key].append(value)
    
    for result in forged_results:
        for key, value in result["forgery_indicators"].items():
            if isinstance(value, (int, float)):
                forged_indicators[key].append(value)
    
    # Create directory for plots
    os.makedirs("analysis_plots", exist_ok=True)
    
    # Create histograms for each indicator
    for key in genuine_indicators.keys():
        if key == "model_detection":
            continue
            
        plt.figure(figsize=(10, 6))
        plt.hist(genuine_indicators[key], alpha=0.5, label='Genuine', bins=20)
        plt.hist(forged_indicators[key], alpha=0.5, label='Forged', bins=20)
        plt.title(f'Distribution of {key}')
        plt.xlabel(key)
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f"analysis_plots/{key}_distribution.png")
        plt.close()
    
    # Create scatter plot of noise level vs edge density
    plt.figure(figsize=(10, 6))
    plt.scatter([r["forgery_indicators"]["noise_level"] for r in genuine_results], 
                [r["forgery_indicators"]["edge_density"] for r in genuine_results], 
                alpha=0.7, label='Genuine')
    plt.scatter([r["forgery_indicators"]["noise_level"] for r in forged_results], 
                [r["forgery_indicators"]["edge_density"] for r in forged_results], 
                alpha=0.7, label='Forged')
    plt.title('Noise Level vs Edge Density')
    plt.xlabel('Noise Level')
    plt.ylabel('Edge Density')
    plt.legend()
    plt.savefig("analysis_plots/noise_vs_edge.png")
    plt.close()
    
    # Create scatter plot of laplacian variance vs dct_std
    plt.figure(figsize=(10, 6))
    plt.scatter([r["forgery_indicators"]["laplacian_var"] for r in genuine_results], 
                [r["forgery_indicators"]["dct_std"] for r in genuine_results], 
                alpha=0.7, label='Genuine')
    plt.scatter([r["forgery_indicators"]["laplacian_var"] for r in forged_results], 
                [r["forgery_indicators"]["dct_std"] for r in forged_results], 
                alpha=0.7, label='Forged')
    plt.title('Laplacian Variance vs DCT Standard Deviation')
    plt.xlabel('Laplacian Variance')
    plt.ylabel('DCT Standard Deviation')
    plt.legend()
    plt.savefig("analysis_plots/laplacian_vs_dct.png")
    plt.close()
    
    print(f"Visualizations saved to the 'analysis_plots' directory")

def main():
    parser = argparse.ArgumentParser(description="Test the accuracy of forensic analysis methods")
    parser.add_argument("--genuine", type=str, required=True, help="Directory containing genuine images")
    parser.add_argument("--forged", type=str, required=True, help="Directory containing forged images")
    parser.add_argument("--samples", type=int, default=None, help="Number of images to sample from each directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for classification")
    args = parser.parse_args()
    
    results = test_forensic_accuracy(args.genuine, args.forged, args.samples, args.threshold)
    
    print("\nForensic Analysis Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"True Positives: {results['true_positives']}")
    print(f"True Negatives: {results['true_negatives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")

if __name__ == "__main__":
    main() 