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


def edge_clustering (Ec, curvature_radius=10, 
                               max_window=5, curv_percentile=80,
                               min_cluster_size=5):
    """
    Complete pipeline: Find connected components and segment by curvature.
    
    Args:
        edge_map: Binary edge map (Ec)
        curvature_radius: Radius for osculating circle fitting
        max_window: Window for local maxima detection
        curv_percentile: Percentile threshold for segmentation
        min_cluster_size: Minimum points to keep a cluster
    
    Returns:
        C: List of clusters {C0, C1, ..., Cn}
    """


    # Use OpenCV connected components
    num_labels, labels = cv2.connectedComponents(Ec, connectivity=8)
    
    clusters = []
    for label in range(1, num_labels):  # Skip background (0)
        points = np.argwhere(labels == label)
        if len(points) > 3:  
            cluster = [(pt[1], pt[0]) for pt in points]
            clusters.append(cluster)


    final_clusters = []

    def fit_circle_algebraic(points):
        """
        Fit a circle to points using algebraic (Pratt) method.
        Based on "Least-Squares Fitting of Circles and Ellipses" by Chernov.
        
        Args:
            points: List or array of (x, y) coordinates
        
        Returns:
            (cx, cy, radius) or None if fitting fails
        """
        points = np.array(points)
        if len(points) < 3:
            return None
        
        # Center the data
        x = points[:, 0]
        y = points[:, 1]
        x_m = np.mean(x)
        y_m = np.mean(y)
        
        u = x - x_m
        v = y - y_m
        
        # Build design matrix
        Suu = np.sum(u * u)
        Suv = np.sum(u * v)
        Svv = np.sum(v * v)
        Suuu = np.sum(u * u * u)
        Suvv = np.sum(u * v * v)
        Svvv = np.sum(v * v * v)
        Svuu = np.sum(v * u * u)
        
        # Solve linear system
        A = np.array([[Suu, Suv], [Suv, Svv]])
        b = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svvv + Svuu)])
        
        try:
            uc, vc = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        
        # Transform back to original coordinates
        cx = uc + x_m
        cy = vc + y_m
        
        # Calculate radius
        radius = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(points))
        
        return (cx, cy, radius)

    def estimate_curvature_at_point(cluster, point_idx, radius=10):
        """
        Estimate curvature at a point by fitting osculating circle.
        Curvature = 1/radius of osculating circle.
        
        Args:
            cluster: List of (x, y) points in the cluster
            point_idx: Index of point to estimate curvature at
            radius: Search radius for nearby points
        
        Returns:
            Curvature value (1/R) or 0 if estimation fails
        """
        cluster_array = np.array(cluster)
        point = cluster_array[point_idx]
        
        # Find points within radius
        distances = np.sqrt(np.sum((cluster_array - point)**2, axis=1))
        nearby_mask = distances <= radius
        
        # Need at least 3 points to fit circle
        if np.sum(nearby_mask) < 3:
            return 0.0
        
        nearby_points = cluster_array[nearby_mask]
        
        # Fit circle to nearby points
        circle = fit_circle_algebraic(nearby_points)
        
        if circle is None:
            return 0.0
        
        cx, cy, r = circle
        
        # Curvature is reciprocal of radius
        if r > 0:
            return 1.0 / r
        else:
            return 0.0
    def compute_cluster_curvatures(cluster, search_radius=10):
        """
        Compute curvature for all points in a cluster.
        
        Args:
            cluster: List of (x, y) points
            search_radius: Radius for osculating circle estimation
        
        Returns:
            Array of curvature values
        """
        curvatures = np.zeros(len(cluster))
        
        for i in range(len(cluster)):
            curvatures[i] = estimate_curvature_at_point(cluster, i, search_radius)
        
        return curvatures


    def find_local_maxima(values, window_size=5, threshold_percentile=75):
        """
        Find local maxima in curvature array.
        
        Args:
            values: Array of curvature values
            window_size: Window size for local maximum detection
            threshold_percentile: Only keep maxima above this percentile
        
        Returns:
            Indices of local maxima
        """
        if len(values) < window_size:
            return []
        
        maxima_indices = []
        half_window = window_size // 2
        
        # Find threshold value
        threshold = np.percentile(values, threshold_percentile)
        
        for i in range(half_window, len(values) - half_window):
            window = values[i - half_window:i + half_window + 1]
            
            # Check if current point is maximum in window and above threshold
            if values[i] == np.max(window) and values[i] > threshold:
                maxima_indices.append(i)
        
        return maxima_indices

    def segment_cluster_by_curvature(cluster, curvature_radius=10, 
                                  max_window=5, curv_percentile=80):
        """
        Segment a cluster at points of high curvature (local maxima).
        
        Args:
            cluster: List of (x, y) points
            curvature_radius: Radius for osculating circle
            max_window: Window size for maxima detection
            curv_percentile: Percentile threshold for curvature maxima
        
        Returns:
            List of sub-clusters
        """
        if len(cluster) < 10:  # Don't segment very small clusters
            return [cluster]
        
        # Compute curvatures
        curvatures = compute_cluster_curvatures(cluster, curvature_radius)
        
        # Find local maxima (segmentation points)
        split_indices = find_local_maxima(curvatures, max_window, curv_percentile)
        
        if len(split_indices) == 0:
            return [cluster]
        
        # Split cluster at maxima points
        sub_clusters = []
        start_idx = 0
        
        for split_idx in split_indices:
            if split_idx - start_idx > 3:  # Minimum cluster size
                sub_clusters.append(cluster[start_idx:split_idx])
            start_idx = split_idx
        
        # Add remaining points
        if len(cluster) - start_idx > 3:
            sub_clusters.append(cluster[start_idx:])
        
        return sub_clusters if len(sub_clusters) > 0 else [cluster]


    for i, cluster in enumerate(clusters):
        sub_clusters = segment_cluster_by_curvature(
            cluster, 
            curvature_radius, 
            max_window, 
            curv_percentile
        )
        
        # Filter by minimum size
        for sub_cluster in sub_clusters:
            if len(sub_cluster) >= min_cluster_size:
                final_clusters.append(sub_cluster)
        
    return final_clusters



def paper_algorithm (frame : np.ndarray):
    '''
    The drum visual analysis as specified by the paper "Visual analysis for drum sequence transcription"
    https://ieeexplore.ieee.org/document/7098815/metrics#metrics
    '''
    
    ## THE METHOD:

    # Edge Detection
    Ec, Ef, edge_magnitude_norm = modified_edge_detection(img=frame)
    cv2.imshow("test", Ec)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Edge Clustering
    final_clusters = edge_clustering(Ec)







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