import os
import shutil
import argparse
from pathlib import Path

def create_directory(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

def organize_datasets():
    """
    Organize the datasets as requested.
    """
    # Create the main datasets directory
    datasets_dir = "Datasets"
    create_directory(datasets_dir)
    
    # Create the organized datasets directory
    organized_dir = "Organized_Datasets"
    create_directory(organized_dir)
    
    # Create directories for each dataset
    signature_dir = os.path.join(organized_dir, "signatures")
    signature_detection_dir = os.path.join(organized_dir, "signature_detection")
    id_detection_dir = os.path.join(organized_dir, "id_detection")
    
    create_directory(signature_dir)
    create_directory(signature_detection_dir)
    create_directory(id_detection_dir)
    
    # Copy the signature dataset
    if os.path.exists(os.path.join(datasets_dir, "signatures")):
        print("Copying signatures dataset...")
        shutil.copytree(
            os.path.join(datasets_dir, "signatures"),
            os.path.join(signature_dir, "signatures"),
            dirs_exist_ok=True
        )
    
    # Copy the signature detection dataset
    if os.path.exists(os.path.join(datasets_dir, "Signature Detection")):
        print("Copying Signature Detection dataset...")
        shutil.copytree(
            os.path.join(datasets_dir, "Signature Detection"),
            os.path.join(signature_detection_dir, "Signature Detection"),
            dirs_exist_ok=True
        )
    
    # Copy the MASK-RCNN-Dataset
    if os.path.exists(os.path.join(datasets_dir, "sign_data", "MASK-RCNN-Dataset-master")):
        print("Copying MASK-RCNN-Dataset...")
        shutil.copytree(
            os.path.join(datasets_dir, "sign_data", "MASK-RCNN-Dataset-master"),
            os.path.join(id_detection_dir, "MASK-RCNN-Dataset-master"),
            dirs_exist_ok=True
        )
    
    print("Dataset organization complete!")
    print(f"Organized datasets are available in the '{organized_dir}' directory.")

def main():
    parser = argparse.ArgumentParser(description='Organize datasets for forgery detection')
    parser.add_argument('--clean', action='store_true', help='Remove the original datasets after organizing')
    
    args = parser.parse_args()
    
    # Organize the datasets
    organize_datasets()
    
    # Clean up if requested
    if args.clean:
        print("Cleaning up original datasets...")
        datasets_dir = "Datasets"
        
        # Remove the original datasets
        if os.path.exists(os.path.join(datasets_dir, "signatures")):
            shutil.rmtree(os.path.join(datasets_dir, "signatures"))
        
        if os.path.exists(os.path.join(datasets_dir, "Signature Detection")):
            shutil.rmtree(os.path.join(datasets_dir, "Signature Detection"))
        
        if os.path.exists(os.path.join(datasets_dir, "sign_data", "MASK-RCNN-Dataset-master")):
            shutil.rmtree(os.path.join(datasets_dir, "sign_data", "MASK-RCNN-Dataset-master"))
        
        print("Cleanup complete!")

if __name__ == "__main__":
    main() 