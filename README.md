# Document Forgery Detection System

This system provides advanced forgery detection for ID cards and signatures using deep learning and image forensics techniques.

## Features

- **ID Card Forgery Detection**: Detect forged ID cards using YOLOv8 and image forensics.
- **Signature Forgery Detection**: Detect forged signatures using specialized signature analysis.
- **User Training**: Train custom models with your own data for improved accuracy.
- **Visual Results**: View detailed visual results with highlighted forgery indicators.

## Project Structure

```
.
├── Organized_Datasets/           # Organized datasets for training
│   ├── signatures/               # Signature forgery datasets
│   ├── signature_detection/      # Signature detection datasets
│   └── id_detection/             # ID card detection datasets
├── detect_id_forgery.py          # ID forgery detection algorithm
├── detect_signature_forgery.py   # Signature forgery detection algorithm
├── id_forgery_app.py             # Flask web application
├── organize_datasets.py          # Script to organize datasets
├── train_id_detection.py         # Script to train ID detection model
├── train_signature_detection.py  # Script to train signature detection model
├── upload_training_data.py       # Script to process user-provided training data
├── templates/                    # HTML templates for the web application
│   └── index.html                # Main web interface
├── id_forgery_config.yaml        # Configuration for ID forgery detection
├── signature_config.yaml         # Configuration for signature forgery detection
└── signature_detection_config.yaml # Configuration for signature detection
```

## Setup and Installation

1. **Organize Datasets**:
   ```
   python organize_datasets.py
   ```

2. **Train Models**:
   ```
   # Train ID forgery detection model
   python train_id_detection.py --prepare-dataset
   
   # Train signature forgery detection model
   python train_signature_detection.py
   
   # Train signature detection model
   python train_signature_detection.py --dataset Organized_Datasets/signature_detection/Signature\ Detection
   ```

3. **Run the Web Application**:
   ```
   python id_forgery_app.py
   ```

4. Open your browser and navigate to `http://localhost:5002` to use the application.

## Using the Application

### Forgery Detection

1. Select the document type (ID Card or Signature).
2. Upload an image by dragging and dropping or using the browse button.
3. View the results, including:
   - Forgery status (genuine or forged)
   - Confidence score
   - Highlighted forgery indicators
   - Forensic analysis details

### Training Custom Models

1. Go to the Training tab.
2. Select the document type (ID Card or Signature).
3. Upload at least 5 genuine and 5 forged document images.
4. Click "Start Training" to begin the training process.
5. Training will run in the background and may take several minutes.

## Forgery Detection Methods

### ID Card Forgery Detection

- **Deep Learning**: YOLOv8 model trained on ID card forgery datasets.
- **Image Forensics**:
  - Noise analysis
  - Edge detection
  - Laplacian variance (blur detection)
  - DCT coefficients (compression artifacts)
  - Error Level Analysis (ELA)
  - Copy-move forgery detection

### Signature Forgery Detection

- **Deep Learning**: YOLOv8 model trained on signature forgery datasets.
- **Signature Forensics**:
  - Contour analysis
  - Texture features
  - Gradient features
  - DCT coefficients
  - Stroke width variation
  - Pressure points analysis

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Flask
- PyTorch
- Ultralytics YOLOv8
- SciPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.