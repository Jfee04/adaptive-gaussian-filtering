import os
import re

OUTPUT_DIR = "output"
OUTFILE = "best_methods.txt"

metric_pattern = re.compile(r"MSE\s*=\s*([0-9.]+),\s*PSNR\s*=\s*([0-9.]+)")

def extract_metrics(line):
    match = metric_pattern.search(line)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def process_file(path, filename, global_results, per_file_results):
    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return

    # Baseline: Uniform Gaussian
    baseline_line = lines[1]
    if not baseline_line.startswith("Uniform Gaussian:"):
        return

    base_mse, base_psnr = extract_metrics(baseline_line)
    if base_mse is None:
        return

    file_entries = []

    # Process remaining lines
    for line in lines[2:]:
        mse, psnr = extract_metrics(line)
        if mse is None:
            continue

        delta_mse = mse - base_mse
        delta_psnr = psnr - base_psnr

        entry = {
            "file": filename,
            "line": line.strip(),
            "mse": mse,
            "psnr": psnr,
            "delta_mse": delta_mse,
            "delta_psnr": delta_psnr
        }

        global_results.append(entry)
        file_entries.append(entry)

    # Store per-file results
    per_file_results[filename] = file_entries


def sort_key(entry):
    # Sort by:
    #   1. Most negative ΔMSE
    #   2. Largest positive ΔPSNR
    return (entry["delta_mse"], -entry["delta_psnr"])



global_results = []
per_file_results = {}

# Process files
for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".txt"):
        full_path = os.path.join(OUTPUT_DIR, filename)
        process_file(full_path, filename, global_results, per_file_results)

# Sort global results
global_results.sort(key=sort_key)
top_10 = global_results[:10]

# Write output
with open(OUTFILE, "w") as f:

    # GLOBAL TOP 10
    f.write("=============== GLOBAL TOP 10 METHODS ===============\n")
    for r in top_10:
        f.write(
            f"[{r['file']}] "
            f"MSE={r['mse']:.6f}, PSNR={r['psnr']:.3f}, "
            f"ΔMSE={r['delta_mse']:.6f}, ΔPSNR={r['delta_psnr']:.3f}  "
            f"{r['line']}\n"
        )

    f.write("\n=============== TOP 2 PER FILE ===============\n")

    # PER-FILE TOP 2
    for filename, entries in per_file_results.items():
        if not entries:
            continue

        f.write(f"\n== {filename} ==\n")

        # Sort per file
        sorted_entries = sorted(entries, key=sort_key)
        top2 = sorted_entries[:2]

        for r in top2:
            f.write(
                f"[{r['file']}] "
                f"MSE={r['mse']:.6f}, PSNR={r['psnr']:.3f}, "
                f"ΔMSE={r['delta_mse']:.6f}, ΔPSNR={r['delta_psnr']:.3f}  "
                f"{r['line']}\n"
            )

print(f"best_methods.txt generated successfully.")
