import os
import cv2
import numpy as np
from filtering import uniform_gaussian, variable_gaussian




if __name__ == "__main__":
    candidates = [
        "source/cameraman.png",
        "source/cameraman.tif",
        "cameraman.png",
        "cameraman.tif",
    ]

    img = cv2.imread("img/cameraman.tif", cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    noisy = img + 0.1 * np.random.randn(*img.shape)
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






    combined = np.hstack([
        img,
        noisy,
        base,
        mean7,
        mean15,
        var7,
        var15
    ])


    cv2.imshow("All filters", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



