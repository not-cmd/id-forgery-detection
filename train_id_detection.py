from ultralytics import YOLO
import os
import argparse
import yaml
import shutil
from pathlib import Path

def setup_dataset_config(dataset_path, output_config_path):
    """
    Create a YAML configuration file for the dataset.
    
    Args:
        dataset_path: Path to the dataset
        output_config_path: Path to save the configuration file
    """
    # Create a new data.yaml
    config = {
        'path': dataset_path,
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {
            0: 'genuine',
            1: 'forged'
        }
    }
    
    # Create train/val/test directories if they don't exist
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Write the configuration to a YAML file
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created new data.yaml at {output_config_path}")
    return output_config_path

def prepare_dataset(dataset_path):
    """
    Prepare the dataset for training by organizing the files into train/val/test directories.
    
    Args:
        dataset_path: Path to the dataset
    """
    # Create train/val/test directories
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Create subdirectories for images and labels
    train_images = os.path.join(train_dir, 'images')
    train_labels = os.path.join(train_dir, 'labels')
    val_images = os.path.join(val_dir, 'images')
    val_labels = os.path.join(val_dir, 'labels')
    test_images = os.path.join(test_dir, 'images')
    test_labels = os.path.join(test_dir, 'labels')
    
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_labels, exist_ok=True)
    
    # Process each subdirectory in the dataset
    for subdir in ['IndCard', 'IndPass', 'PakCNIC', 'PakPass']:
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        # Process images and annotations
        for img_file in os.listdir(subdir_path):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(subdir_path, img_file)
            
            # Determine if the image is genuine or forged
            is_forged = 'forged' in img_file.lower() or 'fake' in img_file.lower()
            
            # Decide which set to put the image in (70% train, 20% val, 10% test)
            import random
            rand_val = random.random()
            
            if rand_val < 0.7:
                dest_img = os.path.join(train_images, img_file)
                dest_label = os.path.join(train_labels, img_file.rsplit('.', 1)[0] + '.txt')
            elif rand_val < 0.9:
                dest_img = os.path.join(val_images, img_file)
                dest_label = os.path.join(val_labels, img_file.rsplit('.', 1)[0] + '.txt')
            else:
                dest_img = os.path.join(test_images, img_file)
                dest_label = os.path.join(test_labels, img_file.rsplit('.', 1)[0] + '.txt')
            
            # Copy the image
            shutil.copy(img_path, dest_img)
            
            # Create a label file (YOLO format)
            with open(dest_label, 'w') as f:
                # Format: class x_center y_center width height
                # All values are normalized to [0, 1]
                # For simplicity, we'll assume the object covers the entire image
                class_id = 1 if is_forged else 0
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    print(f"Dataset prepared at {dataset_path}")

def train_model(config_path, model_name, epochs, batch_size, img_size, output_dir):
    """
    Train a YOLOv8 model on the dataset.
    
    Args:
        config_path: Path to the dataset configuration file
        model_name: Name of the model to use (e.g., 'yolov8n.pt')
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        img_size: Image size for training
        output_dir: Directory to save the trained model
    """
    # Load a pretrained model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=output_dir,
        name='id_forgery_detection'
    )
    
    print(f"Training completed. Model saved to {output_dir}/id_forgery_detection")
    return results

def main():
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model for ID forgery detection')
    parser.add_argument('--dataset', type=str, default='Datasets/sign_data/MASK-RCNN-Dataset-master', 
                        help='Path to the dataset')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='Path to the model to use')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Image size for training')
    parser.add_argument('--output-dir', type=str, default='runs', 
                        help='Directory to save the trained model')
    parser.add_argument('--prepare-dataset', action='store_true',
                        help='Prepare the dataset for training')
    
    args = parser.parse_args()
    
    # Prepare the dataset if requested
    if args.prepare_dataset:
        prepare_dataset(args.dataset)
    
    # Setup dataset configuration
    config_path = setup_dataset_config(args.dataset, 'id_forgery_config.yaml')
    
    # Train the model
    train_model(
        config_path=config_path,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main() 