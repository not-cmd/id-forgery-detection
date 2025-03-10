from ultralytics import YOLO
import os
import sys

def optimize_model(input_model_path, output_format='onnx'):
    """
    Optimize a YOLOv8 model for deployment.
    
    Args:
        input_model_path: Path to the input model file
        output_format: Format to export to (onnx, openvino, etc.)
    """
    if not os.path.exists(input_model_path):
        print(f"Error: Model file {input_model_path} not found.")
        return False
    
    try:
        # Load the model
        model = YOLO(input_model_path)
        
        # Export the model
        success = model.export(format=output_format)
        
        if success:
            print(f"Model successfully exported to {output_format} format.")
            print(f"Output file: {os.path.splitext(input_model_path)[0]}.{output_format}")
            return True
        else:
            print(f"Failed to export model to {output_format} format.")
            return False
    
    except Exception as e:
        print(f"Error optimizing model: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python optimize_model.py <model_path> [output_format]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else 'onnx'
    
    optimize_model(model_path, output_format) 