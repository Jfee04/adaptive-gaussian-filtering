# evaluation.py
import os
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

def evaluate_methods(original, results, data_range=1.0, out_dir="output", img_name="evaluation"):
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, img_name + ".txt")

    # Open file for writing
    with open(out_path, "w") as f:
        header = "=== Evaluation Metrics ===\n"
        print(header.strip())
        f.write(header)

        # Evaluate each method
        for name, filtered in results.items():
            mse_val = mean_squared_error(original, filtered)
            psnr_val = peak_signal_noise_ratio(original, filtered, data_range=data_range)

            line = f"{name}: MSE = {mse_val:.6f}, PSNR = {psnr_val:.2f} dB\n"

            # Print to console
            print(line.strip())

            # Write to file
            f.write(line)

    print(f"\nSaved evaluation results to: {out_path}")
