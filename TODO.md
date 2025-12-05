# TODO

This is a running TODO list for the adaptive Gaussian filtering project.

---

## 1. Sigma Map Generation

- [ ] **Refine mean-based sigma map**
  - Ensure that areas near 0 or 1 in intensity get higher σ.
  - Check normalization and mapping to `[min_sigma, max_sigma]`.

- [ ] **Refine variance-based sigma map**
  - Ensure that high-variance areas (edges) get lower σ to preserve edges.
  - Consider using multi-scale variance (5×5, 9×9, 21×21).

- [x] **Edge-aware sigma map**
  - Use edge detection to assign lower σ to edge pixels.

- [x] **Intensity-dependent sigma map**
  - Increase σ in darker regions (Intensity-Dependent Spread model).

- [x] **Contrast-dependent sigma map**
  - Higher local contrast → lower σ, lower contrast → higher σ.

---

## 2. Adaptive Gaussian Filtering

- [x] **Vectorize per-pixel Gaussian filtering**
  - Remove nested loops using `adaptive_gaussian_vectorized`.
  - Ensure correct indexing when using discretized σ bins.

- [ ] **Discretized sigma filtering**
  - Precompute Gaussian-filtered images for bins.
  - Assign pixels to bins efficiently for fast filtering.

- [ ] **Test different kernel sizes**
  - Compare results for `k=7` and `k=15`.

- [ ] **Combine maps**
  - Optionally combine multiple sigma maps (e.g., variance + intensity).

---

## 3. Testing and Evaluation

- [~] **Compare adaptive methods with uniform Gaussian**
  - Metrics: MSE, PSNR, and visual comparison.
  - Include the following methods in comparison:
    - Uniform Gaussian
    - Mean-based adaptive Gaussian
    - Variance-based adaptive Gaussian
    - Edge-aware filtering
    - Intensity-dependent spread
    - Contrast-dependent spread

- [ ] **Visualize results**
  - Display side-by-side comparisons:
    - Original image
    - Noisy image
    - Uniform Gaussian
    - Intensity-based variable Gaussian
    - Variance-based variable Gaussian
    - Optional: edge-aware, contrast-dependent

---


## 4. Code Organization

- [x] **Separate map generation functions**
  - Moved `compute_mean_sigma_map`, `compute_variance_sigma_map`, etc., to `sigma_map.py`.

- [ ] **Document code**
  - Add explanations for each method.
  - Clarify why each sigma mapping approach works.

- [ ] **Add inline TODOs in code**
  - Use `# TODO:` comments for small, local improvements.

- [ ] **Extract repeat code to methods**
  - Find repeat code and make methods to replace them


---

## 5. Optional Extensions

- [ ] **Combine multiple adaptive approaches**
  - For example, use variance + intensity for sigma computation.
  
- [ ] **Support for color images**
  - Extend the methods to handle multi-channel RGB images.

- [ ] **Interactive visualization**
  - Allow switching between different filters dynamically for comparison.

---

[x] **Completed:**  
- Basic adaptive Gaussian filtering with mean and variance sigma maps.  
- Vectorized implementation using precomputed bins.  
- Main demo showing multiple filters side by side.
