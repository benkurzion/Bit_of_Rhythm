import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os


def modified_edge_detection(img, d=9, sigma_color=75, sigma_space=75, 
                        low_thresh=20, high_thresh=50, use_fast=True):
    """
    Edge Detection
    
    Args:
        img: Input BGR image
        d: Bilateral filter diameter
        sigma_color: Bilateral filter sigma in color space
        sigma_space: Bilateral filter sigma in coordinate space
        low_thresh: Low threshold for hysteresis
        high_thresh: High threshold for hysteresis
        use_fast: Use fast vectorized version (recommended)
    
    Returns:
        Ec: Coarse edge map
        Ef: Fine edge map
        edge_magnitude: Raw edge magnitude values
    """
    filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    luv_img = cv2.cvtColor(filtered, cv2.COLOR_BGR2Luv).astype(np.float32)
    L, u, v = cv2.split(luv_img)
    
    # Compute gradients for each channel using Sobel
    L_grad_x = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    L_grad_y = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    
    u_grad_x = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3)
    u_grad_y = cv2.Sobel(u, cv2.CV_64F, 0, 1, ksize=3)
    
    v_grad_x = cv2.Sobel(v, cv2.CV_64F, 1, 0, ksize=3)
    v_grad_y = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3)
    
    L_mag = np.sqrt(L_grad_x**2 + L_grad_y**2)
    u_mag = np.sqrt(u_grad_x**2 + u_grad_y**2)
    v_mag = np.sqrt(v_grad_x**2 + v_grad_y**2)
    
    edge_magnitude = np.sqrt(L_mag**2 + u_mag**2 + v_mag**2)
    edge_magnitude_norm = cv2.normalize(edge_magnitude, None, 0, 255, 
                                       cv2.NORM_MINMAX).astype(np.uint8)
    
    # hysteresis thresholding
    Ec = (edge_magnitude >= high_thresh).astype(np.uint8) * 255
    Ef_temp = (edge_magnitude >= low_thresh).astype(np.uint8)
    Ef = np.zeros_like(Ef_temp)
    kernel = np.ones((3, 3), np.uint8)
    strong_dilated = cv2.dilate(Ec, kernel, iterations=1)
    Ef = cv2.bitwise_and(Ef_temp * 255, strong_dilated)
    Ef = cv2.bitwise_or(Ef, Ec)

    return Ec, Ef, edge_magnitude_norm


def edge_clustering (Ec):
    pass


def paper_algorithm (frame : np.ndarray):
    '''
    The drum visual analysis as specified by the paper "Visual analysis for drum sequence transcription"
    https://ieeexplore.ieee.org/document/7098815/metrics#metrics
    '''
    
    ## THE METHOD:

    # Edge Detection
    Ec, Ef, edge_magnitude_norm = modified_edge_detection(img=frame)
    cv2.imshow("test", Ec)
    # Edge Clustering








def select_video_file():
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        )
    )
    
    return file_path



def process_video_frame_by_frame(video_path : str):
    cap = cv2.VideoCapture(video_path)
    
    # Loop through video frames
    while True:
        ret, frame = cap.read()

        # If ret is False -> video has ended
        if not ret:
            print("Finished processing all frames.")
            break

        # PROCESS THE FRAME: frame = NumPy array in BGR format
        paper_algorithm(frame=frame)
        
    cap.release()




if __name__ == "__main__":
    # Get the video
    video_path = select_video_file()

    # Run the algorithm on the video, frame by frame
    process_video_frame_by_frame(video_path=video_path)