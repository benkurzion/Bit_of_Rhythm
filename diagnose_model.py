"""
Diagnostic script to check if your YOLOv8 model is working properly
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_model(model_path, test_image_or_video, output_dir="diagnosis"):
    """
    Run comprehensive diagnostics on your trained model
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    print("="*60)
    print("YOLOv8 MODEL DIAGNOSTIC")
    print("="*60)
    
    # Load model
    print(f"\n1. Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return
    
    # Check model info
    print(f"\n2. Model Information:")
    print(f"   Class names: {model.names}")
    print(f"   Number of classes: {len(model.names)}")
    
    # Test on training images first
    print(f"\n3. Testing on training data...")
    train_img_dir = Path("dataset/train/images")
    if train_img_dir.exists():
        train_images = list(train_img_dir.glob("*.jpg"))[:3]  # Test on first 3
        if train_images:
            print(f"   Testing on {len(train_images)} training images:")
            for img_path in train_images:
                results = model.predict(img_path, conf=0.001, save=False, verbose=False)
                num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
                print(f"   {img_path.name}: {num_detections} detections")
                if num_detections > 0:
                    print(f"      Confidences: {results[0].boxes.conf.cpu().numpy()}")
        else:
            print("   No training images found")
    else:
        print("   Training image directory not found")
    
    # Test with different confidence thresholds
    print(f"\n4. Testing on target video with various thresholds...")
    
    # Load first frame from video
    cap = cv2.VideoCapture(test_image_or_video)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("   ✗ Could not read video")
        return
    
    # Save test frame
    test_frame_path = Path(output_dir) / "test_frame.jpg"
    cv2.imwrite(str(test_frame_path), frame)
    print(f"   Saved test frame: {test_frame_path}")
    
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    print(f"\n   Testing confidence thresholds:")
    
    best_conf = None
    max_detections = 0
    
    for conf in thresholds:
        results = model.predict(test_frame_path, conf=conf, save=False, verbose=False)
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"   conf={conf}: {num_detections} detections", end="")
        
        if num_detections > 0:
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            print(f" (conf range: {confidences.min():.3f}-{confidences.max():.3f})")
            
            if num_detections > max_detections:
                max_detections = num_detections
                best_conf = conf
                
                # Save visualization
                annotated = results[0].plot()
                save_path = Path(output_dir) / f"detection_conf_{conf}.jpg"
                cv2.imwrite(str(save_path), annotated)
                print(f"      Saved visualization: {save_path}")
        else:
            print()
    
    # Check training results
    print(f"\n5. Checking training results...")
    results_path = Path("runs/detect/custom_model/results.png")
    if results_path.exists():
        print(f"   ✓ Training results found: {results_path}")
        print(f"   Please check this file to see:")
        print(f"   - Did the loss decrease during training?")
        print(f"   - Did mAP increase?")
        print(f"   - Are there any obvious training issues?")
    else:
        print(f"   ✗ Training results not found at {results_path}")
    
    # Check for best weights
    weights_path = Path("runs/detect/custom_model/weights")
    if weights_path.exists():
        best_pt = weights_path / "best.pt"
        last_pt = weights_path / "last.pt"
        print(f"\n6. Model weights:")
        print(f"   best.pt exists: {best_pt.exists()}")
        print(f"   last.pt exists: {last_pt.exists()}")
    
    # Summary
    print(f"\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if max_detections == 0:
        print("❌ PROBLEM: Model is not detecting anything!")
        print("\nPossible causes:")
        print("1. Model wasn't trained properly")
        print("   - Check runs/detect/custom_model/results.png")
        print("   - Look for decreasing loss and increasing mAP")
        print("   - If loss didn't decrease, model didn't learn")
        print("\n2. Training data mismatch:")
        print("   - Are objects in test video similar to training data?")
        print("   - Check class names match: ", model.names)
        print("\n3. Need more training:")
        print("   - Try training for more epochs")
        print("   - Add more annotated frames")
        print("   - Use data augmentation")
        print("\n4. Video/image quality issues:")
        print("   - Check if test_frame.jpg looks reasonable")
        print("   - Verify objects are visible")
        
        print("\n📋 RECOMMENDATIONS:")
        print("1. First, check if model works on training data")
        print("2. Review training results.png")
        print("3. If needed, re-train with more epochs or data")
    else:
        print(f"✓ Model is working!")
        print(f"  Max detections: {max_detections} at confidence {best_conf}")
        print(f"  Check saved visualizations in: {output_dir}")
        print(f"\n  Use confidence threshold: {best_conf}")
    
    print("="*60)

# Run diagnostics
if __name__ == "__main__":
    MODEL_PATH = "runs/detect/custom_model/weights/best.pt"
    TEST_VIDEO = "Video Examples/simple_rhythm_test.mp4"
    
    diagnose_model(MODEL_PATH, TEST_VIDEO)