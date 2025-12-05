#main.py main method to test filtering teqnieques and compare


import os
import cv2
import numpy as np

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

    base = uniform_gaussian(noisy)

    mean7 = variable_gaussian(noisy, "mean", k=7, bins=256)
    mean15 = variable_gaussian(noisy, "mean", k=15, bins=256)

    var7 = variable_gaussian(noisy, "variance", k=7, bins=256)
    var15 = variable_gaussian(noisy, "variance", k=15, bins=256)

    int7Low = variable_gaussian(noisy, "intensity_low", k=7, bins=256)
    int7High = variable_gaussian(noisy, "intensity_high", k=7, bins=256)

    cannyClean = variable_gaussian(noisy, "canny_clean", k=7, bins=256)
    cannyBlurred = variable_gaussian(noisy, "canny_blurred", k=7, bins=256)


    combined = np.hstack([
        img,
        noisy,
        base,
        mean7,
        var7,
        int7Low,
        cannyClean        
    ])


    cv2.imshow("All filters", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    results = {
        "Uniform Gaussian": base,
        "Mean Variable Gaussian": mean7,
        "Variance Variable Gaussian": var7,
        "Intensity Low Variable Gaussian": int7Low,
        "Canny Clean Variable Gaussian": cannyClean,
    }

    evaluate_methods(img, results, data_range=1.0)


