# Engineering Changelog

## NSSC 2026 Data Analytics
### Mars HiRISE Unsupervised Anomaly Detection Challenge

This changelog documents the evolution of the pipeline across three iterations, each following a **symptom → diagnosis → fix → outcome** structure, as required by the Rules and Regulations.

---

## v1 — Baseline Autoencoder (MSE-only)

### Architecture

Convolutional autoencoder with:

- 4 convolutional encoding stages
- `AdaptiveAvgPool2d(4,4)` bottleneck
- 256-D latent space
- Upsampling decoder ending in a single unlearned bilinear interpolation from `64×64` to `227×227`

### Loss

MSE reconstruction loss only.

### Symptom

Reconstructions were severely blurred, capturing only coarse brightness gradients with no crater or terrain detail. Validation SSIM plateaued around **0.68**.

### Diagnosis

`AdaptiveAvgPool2d` averaged away high-frequency spatial detail before the latent projection. In addition, the decoder's final upsample was a single blind bilinear interpolation with no learned correction over a roughly **3.5× spatial gap**.

### Fix

- Replaced average pooling with an additional learned strided convolution in the encoder.
- Added a learned convolutional refinement block after the final decoder interpolation.
- This reduced the burden on the blind resize so that it only needed to correct a smaller residual gap.

### Outcome

Reconstructions preserved substantially more terrain structure. This revised architecture was adopted as the basis for **v2**.

**Best validation metrics:**

- MSE: **0.003207**
- SSIM: **0.667980**

---

## v2 — Improved Bottleneck + Combined MSE/SSIM Loss (Final Autoencoder)

### Architecture

Symmetric strided-convolution / transposed-convolution autoencoder with:

- Fully learned `8×8` spatial bottleneck
- 256-D latent space
- Learned refinement block after the final bilinear resize from `128×128` to `227×227`

### Loss

The final loss is:

\[
L = 10 \times \mathrm{MSE} + 1 \times (1-\mathrm{SSIM})
\]

The **10:1 weighting was not arbitrary**. A scale check on an untrained batch showed:

- Raw MSE ≈ **0.0269**
- `1 − SSIM` ≈ **0.400**
- Magnitude gap ≈ **14.87×**

A naive `0.6/0.4` weighting would therefore have allowed the SSIM term to dominate the gradient almost entirely. The 10:1 weighting brings the two terms to a more comparable gradient scale.

### Symptom

Fine details were still somewhat smoothed relative to the original images, although the reconstructions were substantially better than v1.

### Diagnosis

The remaining smoothing was attributed to the intrinsic difficulty of reconstructing fine, high-frequency Martian terrain texture from a **256-D bottleneck**, rather than to a clear architectural defect.

### Fix

Adopted **v2 as the final autoencoder** and generated 256-D latent vectors for the full approximately **10,422-image** dataset for downstream anomaly detection.

### Outcome

- Latent vectors were saved as `latents_v2.npy`.
- The 256-D latent representation was used as the input to the **Isolation Forest** scoring stage in Phase 2.
- The final autoencoder substantially improved preservation of terrain structure compared with v1.

---

## v3 — Anomaly-Detection Stage: Threshold Correction

### Architecture

No further autoencoder changes were made.

Anomaly scoring was performed entirely in the **frozen v2 latent space** using Isolation Forest.

### Loss

N/A — no additional autoencoder training was performed in this iteration.

### Symptom

An initial **median + 3×MAD** threshold flagged approximately **18–22%** of the dataset, directly contradicting the challenge's premise that anomalies are rare outliers.

### Diagnosis

A **Shapiro–Wilk normality test** (`p ≪ 0.05`) and a skewness diagnostic (≈ **1.4**) confirmed that the novelty-score distribution was strongly right-skewed and non-Gaussian.

This ruled out treating the score distribution as Gaussian for a `μ + kσ` threshold, even though such a threshold happened to produce a superficially plausible low flagged count.

A straight-line elbow heuristic also failed, converging inside the bulk of the distribution rather than beyond it.

A kernel density estimate (KDE) revealed an apparent **secondary mode** shortly after the main population mode, followed by a long, thinning tail.

### Fix

A MAD sensitivity sweep was run across a range of `k` values.

The final choice was:

**`k = 6.0 MAD`**

This threshold was selected because it lies **past the apparent secondary mode, in the thinning tail of the distribution**, making it a data-driven choice rather than an arbitrary or fixed-count cutoff.

The Isolation Forest's own contamination parameter was **not used to determine the final anomaly count**. The final anomaly count was determined by the statistical threshold.

### Outcome

The corrected threshold produced a defensible, low, data-driven anomaly rate.

The following outputs were regenerated using the corrected `k = 6.0` threshold:

- Novelty scores
- Metadata-linked anomaly locations
- Top-5 anomaly candidates for Phase 3

---

## Summary Table

| Version | Change Type | Key Metric / Result | Outcome |
|---|---|---|---|
| **v1** | Architecture + loss (baseline) | Validation SSIM ≈ **0.668** | Blurry, low-detail reconstructions |
| **v2** | Architecture (bottleneck) + loss reweighting | Validation MSE/SSIM improved over the baseline; final autoencoder | Final autoencoder; 256-D latents generated |
| **v3** | Anomaly-detection threshold | Initial flagged rate **18–22%** (rejected) → **k = 6.0 MAD** | Data-driven threshold; scores, metadata joins, and Top-5 candidates regenerated |

---

## Engineering Progression at a Glance

### v1

**Problem:** Excessive reconstruction blur and loss of fine terrain detail.

**Main cause:** Average pooling removed high-frequency information and the decoder relied on a blind final bilinear resize.

**Result:** Architecture redesigned.

### v2

**Problem:** Fine details remained somewhat smoothed.

**Main cause:** Remaining limitation was associated with compressing detailed Martian terrain into a 256-D representation rather than an obvious architecture failure.

**Result:** v2 was retained as the final autoencoder and its latent vectors were generated for the full dataset.

### v3

**Problem:** The first statistical threshold flagged far too many samples.

**Main cause:** Novelty scores were strongly right-skewed and non-Gaussian; the initial 3-MAD threshold and straight-line elbow fell inside the bulk distribution.

**Result:** KDE inspection and a MAD sensitivity sweep supported a final **6-MAD** threshold located in the thinning tail.

---

## Final Design Decision

The final pipeline therefore uses:

1. **v2 convolutional autoencoder**
2. **256-D latent representation**
3. **Combined loss:** `10 × MSE + 1 × (1 − SSIM)`
4. **Isolation Forest** operating on the frozen latent vectors
5. **Distribution-driven thresholding** using median + `6×MAD`
6. Regenerated anomaly scores, metadata joins, and Top-5 candidates using the corrected threshold