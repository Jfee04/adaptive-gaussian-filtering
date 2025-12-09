# expiroment.py script to run a series of experiments on different images
# with various filtering techniques and parameters, displaying
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filtering import uniform_gaussian, variable_gaussian
from evaluation import evaluate_methods

# -------------------------
# Helper function to add Gaussian noise
# -------------------------
def add_gaussian_noise(img, sigma=0.2):
    noisy = img + sigma * np.random.randn(*img.shape)
    return np.clip(noisy, 0, 1)

# -------------------------
# List of test images
# -------------------------
all_modes = [
    "uniform",
    "mean",
    "variance",
    "intensity_low",
    "intensity_high",
    "canny_clean",
    "canny_blurred",
]
ks = [5, 7, 9, 15, 21]

test_images = [
    {
        "name": "Cameraman",
        "path": "img/cameraman.tif",
        "noise_sigmas": [0.1, 0.2, 0.3],
        "modes": all_modes,
        "ks": ks,
    },
    {
        "name": "Lena",
        "path": "img/lena.png",
        "noise_sigmas": [0.1, 0.2, 0.3],
        "modes": all_modes,
        "ks": ks,
    },
    {
        "name": "Barbara",
        "path": "img/barbara.png",
        "noise_sigmas": [0.1, 0.2, 0.3],
        "modes": all_modes,
        "ks": ks,
    },
    {
        "name": "Checkerboard",
        "path": "img/checkerboard.png",
        "noise_sigmas": [0.1, 0.2, 0.3],
        "modes": all_modes,
        "ks": ks,
    },
    {
        "name": "Gradient",
        "path": "img/gradient.jpg",
        "noise_sigmas": [0.1, 0.2, 0.3],
        "modes": all_modes,
        "ks": ks,
    },
    {
        "name": "SimpleShapes",
        "path": "img/shapes.jpg",
        "noise_sigmas": [0.1, 0.2, 0.3],
        "modes": all_modes,
        "ks": ks,
    },
]


for img_info in test_images:

    # Load original image once
    img = cv2.imread(img_info["path"], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # Loop over all noise levels
    for sigma in img_info["noise_sigmas"]:
        
        noisy = add_gaussian_noise(img, sigma=sigma)

        # Prepare results dict for this noise level
        results = {
            "Original": img,
            "Noisy": noisy,
        }

        # Baseline uniform Gaussian
        base = uniform_gaussian(noisy, k=7, sigma=1.5)
        results["Uniform Gaussian"] = base

        # Adaptive / variable Gaussian filtering
        for mode in img_info["modes"]:
            for k in img_info["ks"]:

                key = f"{mode} k={k}"

                try:
                    filtered = variable_gaussian(noisy, mode=mode, k=k, bins=256)
                    results[key] = filtered
                except ValueError:
                    # Some modes may not apply to all images
                    continue

        # -------------------------
        # Visualization
        # -------------------------
        n_images = len(results)
        plt.figure(figsize=(4*n_images, 4))

        for i, (name, filtered) in enumerate(results.items()):
            plt.subplot(1, n_images, i + 1)
            plt.imshow(filtered, cmap="gray", vmin=0, vmax=1)
            plt.title(name, fontsize=10)
            plt.axis("off")

        plt.suptitle(f"{img_info['name']} (noise σ={sigma})")
        plt.tight_layout()

        out_path = f"output/results_{img_info['name'].replace(' ', '_')}_sigma{sigma}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        # -------------------------
        # Evaluation
        # -------------------------
        eval_set = {
            k: v for k, v in results.items()
            if k not in ["Original", "Noisy"]
        }

        eval_name = f"{img_info['name'].replace(' ', '_')}_sigma{sigma}"

        evaluate_methods(
            img,
            eval_set,
            data_range=1.0,
            out_dir="output",
            img_name=f"evaluation_{eval_name}"
        )
