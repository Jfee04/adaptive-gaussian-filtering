import os
import re

OUTPUT_DIR = "output"

# Regex to extract "MSE = <float>" and "PSNR = <float>"
metric_pattern = re.compile(r"MSE\s*=\s*([0-9.]+),\s*PSNR\s*=\s*([0-9.]+)")

def extract_metrics(line):
    match = metric_pattern.search(line)
    if match:
        mse = float(match.group(1))
        psnr = float(match.group(2))
        return mse, psnr
    return None, None

def process_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return  # Not enough lines to process

    # Validate second line
    second = lines[1].strip()
    if not second.startswith("Uniform Gaussian:"):
        return  # Skip file entirely

    base_mse, base_psnr = extract_metrics(second)
    if base_mse is None:
        return  # malformed baseline line, skip

    updated_lines = []
    updated_lines.append(lines[0])  # Keep first line unchanged
    updated_lines.append(lines[1])  # Keep Uniform Gaussian line unchanged

    # Process all lines after second line
    for line in lines[2:]:
        mse, psnr = extract_metrics(line)

        # If metrics cannot be extracted, leave line unchanged
        if mse is None:
            updated_lines.append(line)
            continue

        # If the line does NOT beat the baseline: indent
        if not (mse < base_mse and psnr > base_psnr):
            updated_lines.append("        " + line)  # 8-space indent
        else:
            updated_lines.append(line)

    # Rewrite file
    with open(path, "w") as f:
        f.writelines(updated_lines)



for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".txt"):
        process_file(os.path.join(OUTPUT_DIR, filename))

