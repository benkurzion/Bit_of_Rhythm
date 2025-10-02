import numpy as np
import cv2 as cv
import os
from numpy.linalg import norm, inv
from scipy.stats import multivariate_normal as mv_norm

# Global constants remain the same
init_weight = [0.7, 0.11, 0.1, 0.09]
init_u = np.zeros(3)
# initial Covariance matrix
init_sigma = 225*np.eye(3)
init_alpha = 0.05


class GMM():
    def __init__(self, alpha=init_alpha): # Removed data_dir and train_num from __init__
        self.alpha = alpha
        self.img_shape = None

        self.weight = None
        self.mu = None
        self.sigma = None
        self.K = 4 # Fixed K to 4, as used in original logic
        self.B = None

    def check(self, pixel, mu, sigma):
        '''
        Check whether a pixel match a Gaussian distribution. Matching means pixel is less than
        2.5 standard deviations away from a Gaussian distribution.
        '''
        x = np.asmatrix(np.reshape(pixel, (3, 1)))
        u = np.asmatrix(mu).T
        sigma = np.asmatrix(sigma)
        # calculate Mahalanobis distance
        # Use .I for inverse on a matrix object
        d = np.sqrt((x-u).T * sigma.I * (x-u))
        if d < 2.5:
            return True
        else:
            return False

    def _initialize_model(self, img_init):
        '''
        Initializes the GMM parameters using the first frame of the video.
        '''
        self.img_shape = img_init.shape
        H, W, C = self.img_shape
        
        # Initialize K components (K=4)
        K = self.K 

        # Initialize weights: H x W x K
        self.weight = np.array([[init_weight for j in range(W)] for i in range(H)])
        
        # Initialize means: H x W x K x 3
        # All means start with the pixel value of the first frame
        self.mu = np.array([[[np.array(img_init[i][j]).reshape(1,3) for k in range(K)] 
                             for j in range(W)] for i in range(H)])
                             
        # Initialize covariance: H x W x K x 3 x 3
        # All sigmas start with the high initial variance
        self.sigma = np.array([[[init_sigma for k in range(K)] for j in range(W)] 
                             for i in range(H)])

        # B is the number of background components (initially 1)
        self.B = np.ones(self.img_shape[0:2], dtype=np.int32)
        
        print(f"Model initialized for {H}x{W} image shape with K={K} components.")


    def update_with_frame(self, img):
        '''
        Updates the GMM for all pixels based on the current frame.
        '''
        H, W, C = img.shape
        K = self.K
        
        if self.mu is None:
            # Should not happen if _initialize_model is called correctly, 
            # but acts as a safeguard.
            self._initialize_model(img) 

        for i in range(H):
            for j in range(W):
                # Check whether match the existing K Gaussian distributions
                match = -1
                for k in range(K):
                    if self.check(img[i][j], self.mu[i][j][k], self.sigma[i][j][k]):
                        match = k
                        break
                
                # a match found
                if match != -1:
                    mu = self.mu[i][j][match]
                    sigma = self.sigma[i][j][match]
                    x = img[i][j].astype(np.float64) # Use float64 for stability
                    delta = x - mu
                    
                    # Calculate rho: alpha * N(x | mu, sigma)
                    # Flatten pixel value for mv_norm.pdf
                    rho = self.alpha * mv_norm.pdf(x.flatten(), mu.flatten(), sigma) 
                    
                    # Update weights (W_k = (1-alpha)W_k + alpha * M_k, where M_k=1 for match)
                    self.weight[i][j] = (1 - self.alpha) * self.weight[i][j]
                    self.weight[i][j][match] += self.alpha
                    
                    # Update mean: mu = mu + rho * delta
                    self.mu[i][j][match] = mu + rho * delta
                    
                    # Update covariance: sigma = sigma + rho * (delta*delta.T - sigma)
                    # Need to reshape delta for the outer product
                    delta_mat = delta.reshape(-1, 1) 
                    self.sigma[i][j][match] = sigma + rho * (np.matmul(delta_mat, delta_mat.T) - sigma)
                
                # if none of the K distributions match the current value
                if match == -1:
                    w_list = [self.weight[i][j][k] for k in range(K)]
                    id = w_list.index(min(w_list))
                    
                    # Replace the least probable distribution
                    self.mu[i][j][id] = np.array(img[i][j]).reshape(1, 3)
                    self.sigma[i][j][id] = np.array(init_sigma)
                    # The weight is not explicitly set to a low value here, 
                    # but will be reduced in the next frame's weight update.
                    
        self.reorder()

    def reorder(self, T=0.90):
        '''
        Reorder the estimated components and determine the number of background components B.
        '''
        H, W, _ = self.img_shape
        K = self.K
        
        for i in range(H):
            for j in range(W):
                k_weight = self.weight[i][j]
                
                # Calculate the norm of the standard deviation (sqrt of sigma is complex)
                # The original code uses norm(np.sqrt(sigma)). We'll use 
                # a measure related to standard deviation, e.g., the trace/determinant of sigma. 
                # For simplicity and to match the intent of the original code's ratio:
                k_norm = np.array([norm(self.sigma[i][j][k]) for k in range(K)])
                
                # Avoid division by zero/near-zero norms
                k_norm[k_norm < 1e-6] = 1e-6 
                
                ratio = k_weight / k_norm
                descending_order = np.argsort(-ratio) # Use -ratio for descending sort

                # Reorder the arrays
                self.weight[i][j] = self.weight[i][j][descending_order]
                self.mu[i][j] = self.mu[i][j][descending_order]
                self.sigma[i][j] = self.sigma[i][j][descending_order]

                # Determine B (number of background components)
                cum_weight = 0
                self.B[i][j] = 1 # Default minimum B=1
                for index in range(K):
                    cum_weight += self.weight[i][j][index]
                    if cum_weight > T:
                        self.B[i][j] = index + 1
                        break


    def infer(self, img):
        '''
        Infer whether a pixel is background or foreground.
        Background pixels are set to white (255, 255, 255).
        '''
        result = np.array(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                is_background = False
                # Check only the first B components
                for k in range(self.B[i][j]):
                    if self.check(img[i][j], self.mu[i][j][k], self.sigma[i][j][k]):
                        is_background = True
                        break
                
                if is_background:
                    result[i][j] = [255, 255, 255] # Set to white
                    
        return result