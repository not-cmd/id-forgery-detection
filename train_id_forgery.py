from ultralytics import YOLO
import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import shutil

def prepare_dataset():
    """Prepare the dataset for training by organizing images and labels."""
    dataset_path = Path("/Users/divyeshmedidi/Hackscript/Datasets/identity_cards/forged/MASK-RCNN-Dataset")
    categories = ["IndCard", "PakCNIC", "IndPass", "PakPass"]
    
    # Create train and val directories if they don't exist
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    print("Processing dataset...")
    for category in categories:
        print(f"Processing category: {category}")
        
        # Process original (genuine) images
        orig_path = dataset_path / category / "Original"
        if orig_path.exists():
            print(f"Found {len(list(orig_path.glob('*.jpg')))} original images in {category}")
            for img_path in orig_path.glob("*.jpg"):
                # Copy image to train directory
                shutil.copy2(img_path, train_dir / img_path.name)
                # Create corresponding label file
                label_path = train_dir / img_path.with_suffix('.txt').name
                if not label_path.exists():
                    with open(label_path, 'w') as f:
                        f.write('0 0.5 0.5 1.0 1.0\n')  # Format: class x_center y_center width height
        
        # Process augmented (forged) images
        aug_path = dataset_path / category / "Augmented"
        if aug_path.exists():
            print(f"Found {len(list(aug_path.glob('*.jpg')))} augmented images in {category}")
            for img_path in aug_path.glob("*.jpg"):
                # Copy image to validation directory
                shutil.copy2(img_path, val_dir / img_path.name)
                # Create corresponding label file
                label_path = val_dir / img_path.with_suffix('.txt').name
                if not label_path.exists():
                    with open(label_path, 'w') as f:
                        f.write('1 0.5 0.5 1.0 1.0\n')  # Format: class x_center y_center width height

def train_model():
    """Train the YOLOv8 model on the ID card dataset."""
    # Load configuration
    with open('id_forgery_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("\nInitializing model...")
    # Initialize model
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8n model
    
    print("\nStarting training...")
    # Train the model
    try:
        results = model.train(
            data='id_forgery_config.yaml',
            epochs=config['epochs'],
            imgsz=config['img_size'],
            batch=config['batch_size'],
            name='id_forgery_detection',
            verbose=True
        )
        print("\nTraining completed successfully!")
        return results
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

def main():
    """Main function to run the training pipeline."""
    print("Starting the training pipeline...")
    print("\nPreparing dataset...")
    prepare_dataset()
    
    print("\nStarting model training...")
    try:
        results = train_model()
        print(f"\nModel saved at: {results.save_dir}")
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        return

if __name__ == "__main__":
    main() 