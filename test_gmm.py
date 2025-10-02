import GMM
import cv2 as cv
import os
import numpy as np # Needed for array manipulation outside the class
from scipy.stats import multivariate_normal as mv_norm # Needed for pdf calculation
import main

if __name__ == '__main__':
    # --- Configuration ---
    video_path = main.select_video_file()
    output_dir = r'./output_video_frames'
    train_num = 200 # Number of initial frames to use for training
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Initialize Video Capture
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()

    # 2. Initialize and Train the GMM Model
    gmm = GMM.GMM()
    frame_count = 0

    print(f"Starting GMM training on the first {train_num} frames...")

    while frame_count < train_num:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Reached end of video while training at frame {frame_count}")
            break
        
        SCALE_FACTOR = 0.5
        frame = cv.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv.INTER_LINEAR)
        # Initialization
        if frame_count == 0:
            gmm._initialize_model(frame)
        
        # Update
        gmm.update_with_frame(frame)
        
        if (frame_count + 1) % 50 == 0:
             print(f'Training: Frame {frame_count + 1} processed.')
             
        frame_count += 1
        
    print('Training finished.')

    # 3. Inference Loop (start from the frame after the last training frame)
    print('Starting inference...')
    
    # The cap pointer is already at the correct position (frame 'train_num')

    index = train_num 
    while True:
        ret, img = cap.read()
        if not ret:
            break # End of video
            
        # Perform inference
        img_segmented = gmm.infer(img)
        
        # Save the segmented frame
        cv.imwrite(os.path.join(output_dir, '%05d' % index + '.bmp'), img_segmented)
        
        if (index + 1) % 50 == 0:
            print(f'Infering: Frame {index + 1} saved.')
            
        index += 1
        
    cap.release() # Release the video capture object
    print('Inference finished and frames saved to', output_dir)