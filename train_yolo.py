from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(annotations_dir, output_dir="dataset", val_split=0.2):
    """
    Prepare dataset from annotations for YOLOv8 training
    
    Args:
        annotations_dir: Directory containing 'images' and 'labels' folders
        output_dir: Output directory for train/val split
        val_split: Fraction of data to use for validation (default 0.2 = 20%)
    """
    print("Preparing dataset...")
    
    images_dir = os.path.join(annotations_dir, "images")
    labels_dir = os.path.join(annotations_dir, "labels")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # Split into train and validation
    train_images, val_images = train_test_split(image_files, test_size=val_split, random_state=42)
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Create directory structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Copy files to train directory
    for img_file in train_images:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_dir, 'train', 'images', img_file)
        )
        
        label_src = os.path.join(labels_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy(
                label_src,
                os.path.join(output_dir, 'train', 'labels', label_file)
            )
    
    # Copy files to val directory
    for img_file in val_images:
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_dir, 'val', 'images', img_file)
        )
        
        label_src = os.path.join(labels_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy(
                label_src,
                os.path.join(output_dir, 'val', 'labels', label_file)
            )
    
    print(f"Dataset prepared in: {output_dir}")
    return output_dir

def create_yaml_config(dataset_dir, class_names, yaml_path="data.yaml"):
    """
    Create YAML configuration file for YOLOv8
    
    Args:
        dataset_dir: Path to dataset directory
        class_names: List of class names
        yaml_path: Output path for YAML file
    """
    dataset_dir = os.path.abspath(dataset_dir)
    
    config = {
        'path': dataset_dir,
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_full_path = os.path.join(dataset_dir, yaml_path)
    with open(yaml_full_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created YAML config: {yaml_full_path}")
    print(f"Classes: {class_names}")
    return yaml_full_path

def train_yolov8(yaml_path, model_size='n', epochs=30, imgsz=640, batch=16, 
                 device='0', project='runs/detect', name='train'):
    """
    Train YOLOv8 model
    
    Args:
        yaml_path: Path to data.yaml file
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to use ('0' for GPU, 'cpu' for CPU)
        project: Project directory
        name: Experiment name
    """
    print(f"\n{'='*50}")
    print(f"Starting YOLOv8{model_size.upper()} training...")
    print(f"{'='*50}\n")
    
    # Load a pretrained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=50,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Best model saved at: {project}/{name}/weights/best.pt")
    print(f"{'='*50}\n")
    
    return model

def validate_model(model_path, yaml_path, device='0'):
    """
    Validate trained model
    
    Args:
        model_path: Path to trained model weights
        yaml_path: Path to data.yaml
        device: Device to use
    """
    print("Validating model...")
    model = YOLO(model_path)
    metrics = model.val(data=yaml_path, device=device)
    
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

def test_inference(model_path, test_image_or_video, save_dir="inference_results"):
    """
    Test model inference on an image or video
    
    Args:
        model_path: Path to trained model
        test_image_or_video: Path to test image or video
        save_dir: Directory to save results
    """
    print(f"Running inference on: {test_image_or_video}")
    model = YOLO(model_path)
    
    results = model.predict(
        source=test_image_or_video,
        save=True,
        project=save_dir,
        conf=0.25,  # Confidence threshold
        iou=0.45    # NMS IoU threshold
    )
    
    print(f"Results saved to: {save_dir}")
    return results

# Main execution
if __name__ == "__main__":
    # Configuration
    ANNOTATIONS_DIR = "annotations"  # Directory from the annotation tool
    DATASET_DIR = "dataset"
    CLASS_NAMES = ["left_stick", "right_stick"]
    
    # Model configuration
    MODEL_SIZE = 'n'  # Options: 'n', 's', 'm', 'l', 'x' (nano to xlarge)
    EPOCHS = 30
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    DEVICE = 'cpu'  # Use '0' for GPU, 'cpu' for CPU
    
    # Step 1: Prepare dataset (train/val split)
    dataset_dir = prepare_dataset(
        annotations_dir=ANNOTATIONS_DIR,
        output_dir=DATASET_DIR,
        val_split=0.2
    )
    
    # Step 2: Create YAML configuration
    yaml_path = create_yaml_config(
        dataset_dir=dataset_dir,
        class_names=CLASS_NAMES
    )
    
    # Step 3: Train the model
    model = train_yolov8(
        yaml_path=yaml_path,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='runs/detect',
        name='custom_model'
    )
    
    # Step 4: Validate the model
    best_model_path = 'runs/detect/custom_model/weights/best.pt'
    validate_model(best_model_path, yaml_path, device=DEVICE)
    
    # Step 5: Test inference (optional)
    # Uncomment to test on a video or image
    # test_inference(best_model_path, "test_video.mp4")
    
    print("\n🎉 Training pipeline complete!")