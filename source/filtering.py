#filtering.py  filter methods

import cv2
import numpy as np
from sigma_map import compute_mean_sigma_map, compute_variance_sigma_map, compute_intensity_sigma_map, compute_canny_sigma_map



#Normal Guassian (baseline)
def uniform_gaussian(img, k=7, sigma=0):
    return cv2.GaussianBlur(img,(k,k), sigma)


def adaptive_gaussian_vectorized(img, sigma_map, k=7, bins=8):
    sigmas = np.linspace(sigma_map.min(), sigma_map.max(), bins).astype(np.float32)

    # Precompute filtered images
    stack = np.stack([cv2.GaussianBlur(img, (k,k), s) for s in sigmas], axis=0)

    # Compute bin indices for each pixel
    idx = np.digitize(sigma_map, sigmas, right=True)
    idx = np.clip(idx, 0, bins - 1)

    # Fast pixel lookup: stack[bin, i, j]
    out = stack[idx, np.arange(img.shape[0])[:, None], np.arange(img.shape[1])]

    return out.astype(np.float32)





#call this to do the filtering. Binning is used to speed up the process
def variable_gaussian(img, mode="mean", k=7, bins=8):
    img = img.astype(np.float32)

    if mode == "mean":
        sigma_map = compute_mean_sigma_map(img, k)
    elif mode == "variance":
        sigma_map = compute_variance_sigma_map(img, k)
    elif mode == "intensity_low":
        sigma_map = compute_intensity_sigma_map(img, "low", k)
    elif mode == "intensity_high":
        sigma_map = compute_intensity_sigma_map(img, "high", k)
    elif mode == "canny_clean":
        sigma_map = compute_canny_sigma_map(img, "clean", k)
    elif mode == "canny_blurred":
        sigma_map = compute_canny_sigma_map(img, "blurred", k)
    else:
        raise ValueError("mode must be 'mean' or 'variance'")

    return adaptive_gaussian_vectorized(img, sigma_map, k, bins=bins)


