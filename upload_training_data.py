import os
import shutil
import argparse
from pathlib import Path
import random
import cv2
import numpy as np

def create_directory_structure(base_dir):
    """
    Create the directory structure for training data.
    
    Args:
        base_dir: Base directory for the training data
    """
    # Create train/val/test directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')
    
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
    
    return {
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

def process_image(img_path, is_forged, dest_dirs, split_ratio=(0.7, 0.2, 0.1)):
    """
    Process an image and create a label file.
    
    Args:
        img_path: Path to the image
        is_forged: Whether the image is forged
        dest_dirs: Dictionary of destination directories
        split_ratio: Ratio for train/val/test split
    """
    # Check if the file is an image
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return False
    
    # Decide which set to put the image in
    rand_val = random.random()
    
    if rand_val < split_ratio[0]:
        dest_img_dir = dest_dirs['train_images']
        dest_label_dir = dest_dirs['train_labels']
    elif rand_val < split_ratio[0] + split_ratio[1]:
        dest_img_dir = dest_dirs['val_images']
        dest_label_dir = dest_dirs['val_labels']
    else:
        dest_img_dir = dest_dirs['test_images']
        dest_label_dir = dest_dirs['test_labels']
    
    # Copy the image
    img_filename = os.path.basename(img_path)
    dest_img_path = os.path.join(dest_img_dir, img_filename)
    shutil.copy(img_path, dest_img_path)
    
    # Create a label file (YOLO format)
    label_filename = os.path.splitext(img_filename)[0] + '.txt'
    dest_label_path = os.path.join(dest_label_dir, label_filename)
    
    with open(dest_label_path, 'w') as f:
        # Format: class x_center y_center width height
        # All values are normalized to [0, 1]
        # For simplicity, we'll assume the object covers the entire image
        class_id = 1 if is_forged else 0
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    
    return True

def process_directory(dir_path, is_forged, dest_dirs, split_ratio=(0.7, 0.2, 0.1)):
    """
    Process all images in a directory.
    
    Args:
        dir_path: Path to the directory
        is_forged: Whether the images are forged
        dest_dirs: Dictionary of destination directories
        split_ratio: Ratio for train/val/test split
    """
    processed_count = 0
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        
        if os.path.isfile(file_path):
            if process_image(file_path, is_forged, dest_dirs, split_ratio):
                processed_count += 1
    
    return processed_count

def augment_data(dest_dirs, augmentation_factor=2):
    """
    Augment the training data to increase the dataset size.
    
    Args:
        dest_dirs: Dictionary of destination directories
        augmentation_factor: Factor by which to augment the data
    """
    augmented_count = 0
    
    # Augment training images
    for img_file in os.listdir(dest_dirs['train_images']):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(dest_dirs['train_images'], img_file)
        label_path = os.path.join(dest_dirs['train_labels'], os.path.splitext(img_file)[0] + '.txt')
        
        if not os.path.exists(label_path):
            continue
        
        # Read the image and label
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        with open(label_path, 'r') as f:
            label_content = f.read().strip()
        
        # Perform augmentations
        for i in range(augmentation_factor):
            # Generate a unique filename for the augmented image
            aug_img_file = f"{os.path.splitext(img_file)[0]}_aug{i}{os.path.splitext(img_file)[1]}"
            aug_img_path = os.path.join(dest_dirs['train_images'], aug_img_file)
            aug_label_path = os.path.join(dest_dirs['train_labels'], os.path.splitext(aug_img_file)[0] + '.txt')
            
            # Apply random augmentations
            aug_img = img.copy()
            
            # 1. Random brightness and contrast
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-30, 30)    # Brightness
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)
            
            # 2. Random rotation
            angle = random.uniform(-15, 15)
            h, w = aug_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # 3. Random noise
            if random.random() > 0.5:
                noise = np.random.normal(0, random.uniform(5, 15), aug_img.shape).astype(np.uint8)
                aug_img = cv2.add(aug_img, noise)
            
            # Save the augmented image and label
            cv2.imwrite(aug_img_path, aug_img)
            with open(aug_label_path, 'w') as f:
                f.write(label_content)
            
            augmented_count += 1
    
    return augmented_count

def main():
    parser = argparse.ArgumentParser(description='Process user-provided training data')
    parser.add_argument('--genuine-dir', type=str, required=True,
                        help='Directory containing genuine images')
    parser.add_argument('--forged-dir', type=str, required=True,
                        help='Directory containing forged images')
    parser.add_argument('--output-dir', type=str, default='user_training_data',
                        help='Directory to save the processed data')
    parser.add_argument('--augment', action='store_true',
                        help='Augment the training data')
    parser.add_argument('--augmentation-factor', type=int, default=2,
                        help='Factor by which to augment the data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data to use for training')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of data to use for validation')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.isdir(args.genuine_dir):
        print(f"Error: Genuine directory '{args.genuine_dir}' does not exist")
        return
    
    if not os.path.isdir(args.forged_dir):
        print(f"Error: Forged directory '{args.forged_dir}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create directory structure
    dest_dirs = create_directory_structure(args.output_dir)
    
    # Process genuine images
    genuine_count = process_directory(
        args.genuine_dir, 
        is_forged=False, 
        dest_dirs=dest_dirs, 
        split_ratio=(args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    )
    print(f"Processed {genuine_count} genuine images")
    
    # Process forged images
    forged_count = process_directory(
        args.forged_dir, 
        is_forged=True, 
        dest_dirs=dest_dirs, 
        split_ratio=(args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    )
    print(f"Processed {forged_count} forged images")
    
    # Augment data if requested
    if args.augment:
        augmented_count = augment_data(dest_dirs, args.augmentation_factor)
        print(f"Created {augmented_count} augmented images")
    
    print(f"Training data prepared at {args.output_dir}")
    print(f"You can now train a model using this data with the command:")
    print(f"python train_id_detection.py --dataset {args.output_dir}")

if __name__ == '__main__':
    main() 