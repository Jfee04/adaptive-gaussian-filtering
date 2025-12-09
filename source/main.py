#main.py main method to test filtering teqnieques and compare


import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from filtering import uniform_gaussian, variable_gaussian
from evaluation import evaluate_methods





if __name__ == "__main__":
    candidates = [
        "source/cameraman.png",
        "source/cameraman.tif",
        "cameraman.png",
        "cameraman.tif",
    ]

    img = cv2.imread("img/cameraman.tif", cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    noisy = img + 0.3 * np.random.randn(*img.shape)
    noisy = np.clip(noisy, 0, 1)



    # low = uniform_gaussian(noisy, sigma=0.5)
    # medium = uniform_gaussian(noisy, sigma=1.5)
    # high = uniform_gaussian(noisy, sigma=3.0)
    # extra = uniform_gaussian(noisy, sigma=5.0)

    # Compute filtered images
base = uniform_gaussian(noisy)

mean7 = variable_gaussian(noisy, "mean", k=7, bins=256)
mean15 = variable_gaussian(noisy, "mean", k=15, bins=256)

var7 = variable_gaussian(noisy, "variance", k=7, bins=256)
var15 = variable_gaussian(noisy, "variance", k=15, bins=256)

int7Low = variable_gaussian(noisy, "intensity_low", k=7, bins=256)
int7High = variable_gaussian(noisy, "intensity_high", k=7, bins=256)

cannyClean = variable_gaussian(noisy, "canny_clean", k=7, bins=256)
cannyBlurred = variable_gaussian(noisy, "canny_blurred", k=7, bins=256)


# Collect all results to display
results = {
    "Original": img,
    "Noisy": noisy,
    "Uniform Gaussian": base,
    "Mean Variable (k=7)": mean7,
    "Variance Variable (k=7)": var7,
    "Intensity Low (k=7)": int7Low,
    "Canny Clean (k=7)": cannyClean,
}

# Display using Matplotlib
n = len(results)
cols = 4
rows = int(np.ceil(n / cols))

plt.figure(figsize=(16, 4 * rows))

for i, (title, image) in enumerate(results.items(), 1):
    plt.subplot(rows, cols, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Evaluate methods (skip original/noisy)
methods_to_eval = {k: v for k, v in results.items() if k not in ["Original", "Noisy"]}
evaluate_methods(img, methods_to_eval, data_range=1.0)
