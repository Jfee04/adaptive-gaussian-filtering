import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage import data
import matplotlib.pyplot as plt



#helper functions
def compute_local_variance(img, ksize):
    """
    Computes local variance in a kxk neighborhood.
    """
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_mean_sq = cv2.filter2D(img**2, -1, kernel)
    variance = local_mean_sq - local_mean**2
    return variance

def average_intensity(img, i, j, ksize):
    half_ksize = ksize // 2
    patch = img[max(0, i - half_ksize):min(img.shape[0], i + half_ksize + 1),
                max(0, j - half_ksize):min(img.shape[1], j + half_ksize + 1)]
    return np.mean(patch)

def normalize(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def compute_sigma_map(img, k=7):
    # Compute local mean intensity using a box filter
    kernel = np.ones((k, k), np.float32) / (k * k)
    local_mean = cv2.filter2D(img, -1, kernel)

    # Sigma in [0.5, 3.0]
    sigma_map = 0.5 + 2.5 * local_mean
    return sigma_map.astype(np.float32)





#Normal Guassian (baseline)
def uniform_gaussian(img, sigma=1.5):
    return gaussian_filter(img, sigma=sigma)



#Intensity Variable Guassian 
# doesnt do a whole lot to make the image sharper 
def variable_gaussian(img, k=7, bins=8):
    img = img.astype(np.float32)

    # Step 1: compute sigma map
    sigma_map = compute_sigma_map(img, k)

    # Step 2: adaptive filtering with discretized sigma
    return adaptive_gaussian_vectorized(img, sigma_map, bins=bins)


def adaptive_gaussian_vectorized(img, sigma_map, bins=8):
    sigmas = np.linspace(sigma_map.min(), sigma_map.max(), bins).astype(np.float32)

    # Precompute filtered images
    stack = np.stack([gaussian_filter(img, sigma=s) for s in sigmas], axis=0)

    # Compute bin indices for each pixel
    idx = np.digitize(sigma_map, sigmas, right=True)
    idx = np.clip(idx, 0, bins - 1)

    # Fast pixel lookup: stack[bin, i, j]
    out = stack[idx, np.arange(img.shape[0])[:, None], np.arange(img.shape[1])]

    return out.astype(np.float32)




if __name__ == "__main__":
    candidates = [
        "source/cameraman.png",
        "source/cameraman.tif",
        "cameraman.png",
        "cameraman.tif",
    ]

    img = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    noisy = img + 0.1 * np.random.randn(*img.shape)
    noisy = np.clip(noisy, 0, 1)

    base = uniform_gaussian(noisy)

    intensity7 = variable_gaussian(noisy, k=7, bins=400)

    intensity15 = variable_gaussian(noisy, k=15, bins=400)





    combined = np.hstack([
        img,
        noisy,
        base,
        intensity7,
        intensity15
    ])

    cv2.imshow("All filters", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



