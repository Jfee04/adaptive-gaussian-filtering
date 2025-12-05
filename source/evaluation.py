# evaluation.py
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# Compare multiple filtered images against the original image.
#`results` should be a dict: {method_name: filtered_image}.
# Prints MSE and PSNR for each method.
def evaluate_methods(original, results, data_range=1.0):

    print("\n=== Evaluation Metrics ===")
    for name, filtered in results.items():
        mse_val = mean_squared_error(original, filtered)
        psnr_val = peak_signal_noise_ratio(original, filtered, data_range=data_range)
        print(f"{name}: MSE = {mse_val:.6f}, PSNR = {psnr_val:.2f} dB")
