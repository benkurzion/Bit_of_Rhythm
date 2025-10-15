import cv2
import os
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog



class VideoAnnotator:
    def __init__(self, video_path, output_dir="annotations", frame_skip=1):
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_skip = frame_skip  # Only annotate every Nth frame
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # Annotation state
        self.drawing = False
        self.start_point = None
        self.boxes = []
        self.class_id = 0
        self.temp_box = None
        
        # Class names
        self.class_names = ["object"]  # Add your classes here
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.start_point, (x, y))
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                end_point = (x, y)
                self.boxes.append({
                    'class': self.class_id,
                    'bbox': (self.start_point, end_point)
                })
                self.temp_box = None
                print(f"Box added. Total boxes: {len(self.boxes)}")
                
    def draw_boxes(self, frame):
        # Draw saved boxes
        for box in self.boxes:
            pt1, pt2 = box['bbox']
            color = (0, 255, 0)
            cv2.rectangle(frame, pt1, pt2, color, 2)
            label = self.class_names[box['class']]
            cv2.putText(frame, label, (pt1[0], pt1[1]-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw current box being drawn
        if self.temp_box:
            cv2.rectangle(frame, self.temp_box[0], self.temp_box[1], (255, 0, 0), 2)
        
        return frame
    
    def convert_to_yolo_format(self, bbox, img_w, img_h):
        """Convert bbox to YOLO format (x_center, y_center, width, height) normalized"""
        pt1, pt2 = bbox
        x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
        x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
        
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        return x_center, y_center, width, height
    
    def save_annotations(self, frame, frame_num):
        """Save frame and annotations in YOLO format"""
        img_filename = f"frame_{frame_num:06d}.jpg"
        label_filename = f"frame_{frame_num:06d}.txt"
        
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(self.labels_dir, label_filename)
        
        # Save image
        cv2.imwrite(img_path, frame)
        
        # Save labels
        h, w = frame.shape[:2]
        with open(label_path, 'w') as f:
            for box in self.boxes:
                x_c, y_c, bw, bh = self.convert_to_yolo_format(box['bbox'], w, h)
                f.write(f"{box['class']} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
        
        print(f"Saved: {img_filename} with {len(self.boxes)} annotations")
        
    def run(self):
        cv2.namedWindow('Video Annotator')
        cv2.setMouseCallback('Video Annotator', self.mouse_callback)
        
        print("\n=== Video Annotation Tool ===")
        print(f"Total frames: {self.total_frames}")
        print(f"Frame skip: Annotating every {self.frame_skip} frame(s)")
        print(f"Frames to annotate: ~{self.total_frames // self.frame_skip}")
        print("\nControls:")
        print("  Draw box: Click and drag")
        print("  Next frame: SPACE or RIGHT arrow")
        print("  Previous frame: LEFT arrow")
        print("  Skip 10 frames: UP arrow")
        print("  Back 10 frames: DOWN arrow")
        print("  Save & next: ENTER")
        print("  Delete last box: BACKSPACE")
        print("  Change class: 0-9 keys")
        print(f"  Current class: {self.class_id} ({self.class_names[self.class_id]})")
        print("  Quit: Q or ESC")
        print("=" * 30 + "\n")
        
        # Start at first frame to annotate
        self.current_frame = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        # Read the first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read video")
            return
        
        print(f"Loaded frame {self.current_frame}. Waiting for input...")
        print("Make sure the OpenCV window has focus (click on it)")
        
        while True:
            display_frame = frame.copy()
            display_frame = self.draw_boxes(display_frame)
            
            # Display info
            info = f"Frame: {self.current_frame}/{self.total_frames} | Class: {self.class_id} ({self.class_names[self.class_id]}) | Boxes: {len(self.boxes)}"
            cv2.putText(display_frame, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Video Annotator', display_frame)
            
            key = cv2.waitKey(30) & 0xFF  # Wait 30ms for key press
            
            if key == 255:  # No key pressed
                continue
                
            print(f"Key pressed: {key} (char: {chr(key) if 32 <= key <= 126 else 'N/A'})")
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("Quitting...")
                break
            elif key == ord(' ') or key == 83:  # SPACE or RIGHT
                print(f"Moving to next frame (skip={self.frame_skip})...")
                self.current_frame = min(self.current_frame + self.frame_skip, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                self.boxes = []
                print(f"Now on frame {self.current_frame}")
            elif key == 81:  # LEFT
                print("Moving to previous frame...")
                self.current_frame = max(0, self.current_frame - self.frame_skip)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.boxes = []
                print(f"Now on frame {self.current_frame}")
            elif key == 82:  # UP
                print("Skipping forward 10 intervals...")
                self.current_frame = min(self.current_frame + self.frame_skip * 10, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.boxes = []
                print(f"Now on frame {self.current_frame}")
            elif key == 84:  # DOWN
                print("Skipping backward 10 intervals...")
                self.current_frame = max(0, self.current_frame - self.frame_skip * 10)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.boxes = []
                print(f"Now on frame {self.current_frame}")
            elif key == 13:  # ENTER
                print("Saving annotations...")
                self.save_annotations(frame, self.current_frame)
                self.current_frame = min(self.current_frame + self.frame_skip, self.total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                self.boxes = []
                print(f"Now on frame {self.current_frame}")
            elif key == 8:  # BACKSPACE
                if self.boxes:
                    self.boxes.pop()
                    print(f"Deleted last box. {len(self.boxes)} boxes remaining")
            elif ord('0') <= key <= ord('9'):
                self.class_id = min(key - ord('0'), len(self.class_names) - 1)
                print(f"Changed to class {self.class_id}: {self.class_names[self.class_id]}")
                
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Create classes.txt
        with open(os.path.join(self.output_dir, "classes.txt"), 'w') as f:
            for cls in self.class_names:
                f.write(f"{cls}\n")
        
        print("\nAnnotation complete!")
        print(f"Images saved to: {self.images_dir}")
        print(f"Labels saved to: {self.labels_dir}")





def select_video_file():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
    )
    
    return file_path


# Usage
if __name__ == "__main__":

    video_path = select_video_file()  # Change this to your video path
    
    # frame_skip: Annotate every Nth frame (1=every frame, 5=every 5th frame, 30=every 30th frame)
    annotator = VideoAnnotator(video_path, frame_skip=25)
    annotator.class_names = ["left_stick", "right_stick"]
    
    annotator.run()