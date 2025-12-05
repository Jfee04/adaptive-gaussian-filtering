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
    sigma_map = min_sigma + (max_sigma - min_sigma) * dist_norm
    return sigma_map.astype(np.float32)




# -----------------------------------------------------------------------
# computes a sigma map based on local variance
    # the more greater the variance, the lower the sigma
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
    sigma_map = min_sigma + (max_sigma - min_sigma) * inv
    return sigma_map.astype(np.float32)

