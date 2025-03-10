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
    # Check if data.yaml exists in the dataset
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    if os.path.exists(data_yaml_path):
        # Copy the existing data.yaml
        shutil.copy(data_yaml_path, output_config_path)
        print(f"Using existing data.yaml from {dataset_path}")
        return output_config_path
    
    # Create a new data.yaml
    config = {
        'path': dataset_path,
        'train': 'train',
        'val': 'valid',
        'test': 'test',
        'names': {
            0: 'signature'
        }
    }
    
    # Write the configuration to a YAML file
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created new data.yaml at {output_config_path}")
    return output_config_path

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
        name='signature_detection'
    )
    
    print(f"Training completed. Model saved to {output_dir}/signature_detection")
    return results

def main():
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model for signature detection')
    parser.add_argument('--dataset', type=str, default='Datasets/signatures', 
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
    
    args = parser.parse_args()
    
    # Setup dataset configuration
    config_path = setup_dataset_config(args.dataset, 'signature_config.yaml')
    
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