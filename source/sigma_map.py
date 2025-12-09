# sigma_map.py methods to make sigma maps to pass into the adaptive filter

import cv2
import numpy as np

# -----------------------------------------------------------------------
# computes a sigma map based on local mean intensity
    # the more extream the mean intensity, the higher the sigma
    # this is because extream intensities are more likely to be areas without edges
    # sigma in [min_sigma, max_sigma]
def compute_mean_sigma_map(img, k=7, min_sigma=0.5, max_sigma=3.0):
    # Compute local mean intensity using a box filter
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean = cv2.filter2D(img, -1, kernel)

    # recenter around 0.5
    dist = np.abs(local_mean - 0.5)

    # normalize to [0, 1]
    dist_norm = dist * 2

    # map to [min_sigma, max_sigma]
    return map_range(min_sigma, max_sigma, dist_norm)


# -----------------------------------------------------------------------
# computes a sigma map based on local variance
    # the greater the variance, the lower the sigma
    # this is because high variance areas are more likely to be edges
    # sigma in [min_sigma, max_sigma]
def compute_variance_sigma_map(img, k=7, min_sigma=0.5, max_sigma=3.0):
    # compute local variance Var=E[X^2]−(E[X])^2
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_mean_sq = cv2.filter2D(img * img, -1, kernel)
    variance = local_mean_sq - local_mean**2

    # normalize variance to [0,1]
    var_min, var_max = variance.min(), variance.max()
    var_norm = (variance - var_min) / (var_max - var_min + 1e-8)

    # invert: high variance → low sigma
    inv = 1.0 - var_norm

    # map to sigma range
    return map_range(min_sigma, max_sigma, inv)


# -----------------------------------------------------------------------
# compute adaptive sigma map based on local intensity
    # greater sigma in low or high intensity areas based on mode
    # honestly cant logic why this would work just feel like it might
    # sigma in [min_sigma, max_sigma]
def compute_intensity_sigma_map(img, mode="low", k=7, min_sigma=0.5, max_sigma=3.0):
    # compute local mean intensity using a box filter
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean = cv2.filter2D(img, -1, kernel)

    # normalize to [0,1]
    int_min, int_max = local_mean.min(), local_mean.max()
    int_norm = (local_mean - int_min) / (int_max - int_min + 1e-8)

    # invert: low intensity → high sigma
    if mode == "low":
        int_norm = 1.0 - int_norm
        
    # map to sigma range
    return map_range(min_sigma, max_sigma, int_norm)


# -----------------------------------------------------------------------
# compute adaptive sigma map based on Canny edge detection
    # edge areas get low sigma, non-edge areas get high sigma
    # when mode is blurred the edge map is blurred to create a smoother sigma map
    # sigma in [min_sigma, max_sigma]
def compute_canny_sigma_map(img, mode="clean", k=7, min_sigma=0.5, max_sigma=3.0):
    # Apply Canny edge detection -> outputs 8 bit image
    edges = cv2.Canny((img * 255).astype(np.uint8), 100, 200)

    if mode == "blurred":
        # Blur edges to create smoother sigma map
        edges = cv2.GaussianBlur(edges, (k, k), 0)

    # Normalize edges to [0,1]
    edges_norm = edges.astype(np.float32) / 255.0

    # Invert: edges (1) → low sigma, non-edges (0) → high sigma
    inv = 1.0 - edges_norm

    # map to sigma range
    return map_range(min_sigma, max_sigma, inv)





# util function 
def map_range(min_sigma, max_sigma, inv):
    sigma_map = min_sigma + (max_sigma - min_sigma) * inv
    return sigma_map.astype(np.float32)