import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

# --------------------------------------------
# Utility Functions
# --------------------------------------------
def compute_local_variance(img, ksize):
    """
    Computes local variance in a kxk neighborhood.
    """
    kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    local_mean = cv2.filter2D(img, -1, kernel)
    local_mean_sq = cv2.filter2D(img**2, -1, kernel)
    variance = local_mean_sq - local_mean**2
    return variance

def normalize(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


# --------------------------------------------
# 1. Variance-Based Adaptive Gaussian Filtering
# --------------------------------------------
def adaptive_gaussian_variance(img):
    """
    Multi-scale variance-based sigma:
    - Small variance → large sigma (smooth)
    - Large variance → small sigma (preserve edges)
    """

    var5 = compute_local_variance(img, 5)
    var9 = compute_local_variance(img, 9)
    var21 = compute_local_variance(img, 21)

    # Combine multi-scale variance
    combined_var = normalize(var5 + var9 + var21)

    # Map variance to sigma range [0.5, 3.0]
    sigma_map = 3.0 - 2.5 * combined_var

    filtered = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered[i, j] = gaussian_filter(img, sigma=sigma_map[i, j])[i, j]

    return filtered


# --------------------------------------------
# 2. Discretized σ with Precomputed Kernels
# --------------------------------------------
def adaptive_gaussian_discrete(img, bins=8):
    sigmas = np.linspace(0.5, 3.0, bins)
    kernels = [gaussian_filter(img, s) for s in sigmas]

    # Use local variance to choose bins
    var = compute_local_variance(img, 9)
    var_norm = normalize(var)

    # Convert variance to discrete bin index
    bin_index = np.floor(var_norm * (bins - 1)).astype(int)

    filtered = np.zeros_like(img)
    for b in range(bins):
        filtered[bin_index == b] = kernels[b][bin_index == b]

    return filtered


# --------------------------------------------
# 3. Edge-Aware Gaussian Filtering
# --------------------------------------------
def adaptive_edge(img):
    edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
    edges = edges > 0

    smooth_region = gaussian_filter(img, sigma=3.0)
    edge_region = gaussian_filter(img, sigma=0.6)

    result = img.copy()
    result[edges] = edge_region[edges]
    result[~edges] = smooth_region[~edges]

    return result


# --------------------------------------------
# 4. Intensity-Dependent Spread (IDS)
# Lower intensity → higher sigma
# --------------------------------------------
def adaptive_intensity_dependent(img):
    norm = normalize(img)
    sigma_map = 0.5 + 2.5 * (1 - norm)

    filtered = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered[i, j] = gaussian_filter(img, sigma=sigma_map[i, j])[i, j]

    return filtered


# --------------------------------------------
# 5. Contrast-Dependent Spread (CDS)
# Higher local contrast → lower sigma
# --------------------------------------------
def adaptive_contrast(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = normalize(np.sqrt(gx**2 + gy**2))

    sigma_map = 3.0 - 2.5 * mag  # high contrast → small sigma

    filtered = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            filtered[i, j] = gaussian_filter(img, sigma=sigma_map[i, j])[i, j]

    return filtered


# --------------------------------------------
# Uniform Gaussian (baseline)
# --------------------------------------------
def uniform_gaussian(img, sigma=1.5):
    return gaussian_filter(img, sigma=sigma)


# --------------------------------------------
# Comparison Metrics
# --------------------------------------------
def compare_methods(original, noisy, methods):
    for name, result in methods.items():
        print(f"\n=== {name} ===")
        print(f"MSE:  {mse(original, result):.4f}")
        print(f"PSNR: {psnr(original, result):.4f} dB")


# --------------------------------------------
# Main Demo
# --------------------------------------------
if __name__ == "__main__":
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    noisy = img + 0.05 * np.random.randn(*img.shape)
    noisy = np.clip(noisy, 0, 1)

    print("Processing adaptive filters...")

    results = {
        "Uniform Gaussian": uniform_gaussian(noisy),
        "Variance-Based": adaptive_gaussian_variance(noisy),
        "Discrete Sigma": adaptive_gaussian_discrete(noisy),
        "Edge-Aware": adaptive_edge(noisy),
        "Intensity-Dependent Spread": adaptive_intensity_dependent(noisy),
        "Contrast-Dependent Spread": adaptive_contrast(noisy),
    }

    compare_methods(img, noisy, results)

    # Save outputs
    for name, out in results.items():
        cv2.imwrite(name.replace(" ", "_") + ".png", (out * 255).astype(np.uint8))

    print("Done! Filtered images saved to working directory.")
