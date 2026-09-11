# Team Rivendell — Mars HiRISE Unsupervised Anomaly Detection

**Team Rivendell | NSCC/NSSC 2026 — Data Analytics**

Repository for Team Rivendell's submission to the **National Students' Space Challenge 2026 Data Analytics challenge**, focused on unsupervised anomaly detection in Mars HiRISE orbital imagery.

> **Pipeline:** HiRISE crop → convolutional autoencoder → 256-D latent space → Isolation Forest → robust threshold → anomaly candidates → reconstruction-error analysis → geological hypotheses

---

## Overview

The challenge provides approximately **10,422 grayscale HiRISE image crops**, each resized to **227 × 227 pixels**, without ground-truth anomaly labels or the locations/counts of the injected anomalies.

Our approach therefore uses a completely unsupervised pipeline:

```text
                    ┌──────────────────────┐
                    │  227×227 HiRISE Crop │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ V2 Convolutional     │
                    │ Autoencoder           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 256-D Latent Vector  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Isolation Forest     │
                    │ Novelty Scoring      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Robust MAD Threshold │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Anomaly Candidates   │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌──────────────────┐      ┌────────────────────┐
        │ Metadata /       │      │ V2 Decoder         │
        │ Location Analysis│      │ Reconstruction      │
        └──────────────────┘      └─────────┬──────────┘
                                            │
                                            ▼
                                  ┌────────────────────┐
                                  │ Pixel-wise Error   │
                                  │ Heatmaps            │
                                  └─────────┬──────────┘
                                            │
                                            ▼
                                  ┌────────────────────┐
                                  │ Geological          │
                                  │ Hypotheses          │
                                  └────────────────────┘
```

---

## Project Highlights

### Deep Latent Representation

We train a custom convolutional autoencoder **from scratch** using standard PyTorch layers. The pipeline does not rely on pretrained backbones, pretrained feature extractors, or transfer learning.

The final V2 architecture uses:

- Input: `1 × 227 × 227`
- Latent dimension: **256**
- Learned strided-convolution bottleneck
- Spatial bottleneck at **8 × 8**
- Transposed-convolution decoder
- Learned refinement after the final resize

### V1 → V2 Architecture Iteration

The project was developed through explicit engineering iterations.

| Version | Bottleneck | Decoder | Loss | Parameters |
|---|---|---|---|---:|
| V1 | `AdaptiveAvgPool2d(4,4)` | Blind bilinear resize | MSE | 3,497,153 |
| V2 | Learned `8×8` bottleneck | Bilinear resize + learned refinement | `10×MSE + 1×(1−SSIM)` | 10,099,521 |

V1 produced overly smooth reconstructions. We diagnosed the loss of high-frequency spatial information at the average-pooling bottleneck and the large blind resize in the decoder. V2 replaces this with learned downsampling and learned refinement.

### Combined MSE + SSIM Loss

The final training objective is:

\[
\mathcal{L}
=
10\,\mathrm{MSE}
+
1(1-\mathrm{SSIM})
\]

SSIM is implemented directly in the notebook using a Gaussian-window formulation.

The `10:1` weighting was selected after checking the initial scale of the loss terms:

```text
MSE       ≈ 0.0269
1 − SSIM  ≈ 0.400
```

Without scale matching, the SSIM component would dominate the objective.

### Latent-Space Visualization

The final 256-D representations are qualitatively explored using:

- **t-SNE**
- **UMAP**

These are used to inspect the structure of the learned representation and check for collapse or meaningful variation.

The projections are treated as **qualitative diagnostics**, not as proof of geological classes.

---

## Unsupervised Anomaly Detection

After the V2 autoencoder is finalized, a 256-dimensional latent vector is extracted for every image.

Isolation Forest is then applied in latent space:

```text
Image
  ↓
V2 Encoder
  ↓
256-D latent vector
  ↓
Isolation Forest
  ↓
Novelty Score
```

The sklearn decision-function convention is converted so that:

> **Higher Novelty Score = more anomalous**

This allows anomaly detection to operate on the learned terrain representation rather than directly on raw image pixels.

---

## Data-Driven Thresholding

A major part of the project is the choice of the final anomaly threshold.

We compare:

- Gaussian `μ + kσ`
- Robust `median + k × MAD`
- Elbow / curvature heuristic

The score distribution is additionally examined using:

- Shapiro–Wilk normality testing
- skewness
- sorted-score plots
- MAD sensitivity sweeps
- kernel density estimation (KDE)

An initial `median + 3×MAD` threshold flagged roughly **18–22%** of the dataset, which was inconsistent with the expected rarity of challenge outliers.

The final method uses:

\[
T = \mathrm{median}(\mathrm{score}) + 6\times\mathrm{MAD}(\mathrm{score})
\]

The `k = 6.0` threshold was selected because it lies beyond the apparent secondary mode and in the thinning upper tail of the observed novelty-score distribution.

This is described as a **data-driven robust threshold**, not as a universally valid statistical cutoff.

The Isolation Forest `contamination` value is retained as a fitting parameter, but the final reported anomaly count is determined by the selected threshold rather than by taking a fixed top-N count.

---

## Reconstruction Interpretability

The highest-scoring images that pass the anomaly threshold are sent through the same V2 decoder.

For each selected anomaly we examine:

1. Original crop
2. Reconstructed crop
3. Pixel-wise squared reconstruction error
4. Interior/background-excluded error

This gives a spatial view of the regions responsible for the unusual representation.

A high reconstruction error should **not** automatically be interpreted as proof of a geological anomaly. Possible causes include unusual terrain, sensor artifacts, domain shift, splicing, illumination differences, masking, and crop-boundary effects.

---

## Metadata and Location Analysis

Detected crops are linked back to their source-image metadata.

The analysis can therefore examine:

- latitude
- longitude
- sun angle
- season
- image resolution
- source image ID

This allows us to test whether detected anomalies are associated with acquisition conditions rather than assuming that every detection has a geological explanation.

An optional metadata-fusion experiment is also included in the notebook.

---

## Geological Interpretation

The final stage converts the strongest detections into **physically grounded hypotheses**.

These are deliberately written as hypotheses rather than confirmed classifications.

For example:

- Error concentrated along a sharp boundary may be consistent with an unfamiliar terrain feature, a splice, or a crop discontinuity.
- Diffuse error across much of a frame may be consistent with a sensor artifact or domain-shifted content.

The reconstruction maps are therefore used to support interpretation, while keeping the distinction between **machine-detected novelty** and **confirmed geological discovery** explicit.

---

## Repository Structure

The intended GitHub repository structure is:

```text
Team-Rivendell_NSCC-Data-Analytics-2026/
│
├── README.md
├── NSSC_2026_FINAL.ipynb
│
├── report/
│   ├── report.tex
│   ├── NSSC_2026_Report.pdf
│   └── CHANGELOG.pdf
│
├── results/
│   ├── novelty_scores.csv
│   ├── flagged_anomalies.csv
│   ├── latent_space_tsne.png
│   ├── latent_space_umap.png
│   └── reconstruction_errors/
│
├── figures/
│   └── report_figures/
│
└── .gitignore
```

The exact folder names can be adjusted to match the final repository contents.

### Files currently associated with the submission

| File | Purpose |
|---|---|
| `NSSC_2026_FINAL.ipynb` | Main end-to-end analysis notebook |
| `report/report.tex` | LaTeX source for the technical report |
| `report/NSSC_2026_Report.pdf` | Compiled report |
| `report/CHANGELOG.pdf` | Engineering iteration / design journal |
| `results/` | Generated numerical results and figures |

---

## Reproducing the Pipeline

### Environment

The notebook is intended primarily for **Google Colab**.

Recommended Python packages:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn scipy pillow umap-learn
```

### Dataset and Drive

The notebook expects the competition data and checkpoints to be accessible through Google Drive under:

```text
/content/drive/MyDrive/NSSC_HiRISE/
```

The pipeline uses:

```text
10,422 total images
9,901 training images
521 validation images
95/5 split
seed = 42
```

### Main generated model files

The finalized V2 checkpoint and latent representations are:

```text
checkpoints/cae_final_v2_mse_ssim.pt
checkpoints/latents_v2.npy
checkpoints/latent_filenames_v2.json
```

The anomaly-detection phase can be reproduced from the saved latent vectors without retraining the autoencoder.

### Execution Order

Run `NSSC_2026_FINAL.ipynb` from top to bottom:

```text
1. Dataset preparation
2. V1 baseline
3. V2 architecture
4. V2 training/evaluation
5. Latent extraction
6. t-SNE / UMAP
7. Isolation Forest
8. Threshold analysis
9. Metadata/location analysis
10. Metadata fusion (optional)
11. Top anomaly selection
12. Reconstruction + error heatmaps
13. Geological interpretation
14. Final Results
```

---

## Reproducibility Notes

The pipeline uses fixed random seeds where applicable, with the primary experimental seed set to:

```text
42
```

The latent filename index is saved alongside the latent vectors so that each vector can be matched back to its original crop.

When publishing final numbers, the notebook outputs, report, and repository figures should all correspond to the **same final successful run**.

---

## Limitations

This project is intentionally unsupervised and therefore has several limitations:

- No ground-truth anomaly labels are available for direct supervised validation.
- t-SNE and UMAP provide qualitative visualization rather than geological labels.
- Isolation Forest identifies unusual latent representations, not necessarily scientifically important anomalies.
- Reconstruction error is influenced by the learned representation and decoder.
- Masking, illumination, crop boundaries, and acquisition artifacts can influence anomaly scores.
- Geological explanations remain hypotheses until independently validated.

---

## Results

The final numerical results are generated by the notebook's **Final Results** section.

The final submission should synchronize these values across the notebook, GitHub repository, and LaTeX report:

- final V2 validation MSE
- final V2 validation SSIM
- latent dimension
- Isolation Forest settings
- final MAD threshold
- flagged count
- flagged percentage
- Top-5 filenames
- Top-5 Novelty Scores
- metadata/location observations
- reconstruction-error observations
- geological hypotheses

The repository deliberately avoids hard-coding unpublished final numbers until the final notebook run has been completed and checked.

---

## Team

### Team Rivendell

**National Students' Space Challenge 2026 — Data Analytics**

- **Sagnik Mitra**
- **Suvam Kar**

**Institution:** Institute of Radio Physics and Electronics, University of Calcutta

---

## Competition Context

This repository contains the team's technical implementation, experiment record, visualizations, and report for the **2026 Data Analytics challenge**.

The project follows the competition constraint of building the representation-learning pipeline without pretrained models or pretrained feature extractors.

---

## Citation / Reference

References and External Sources

The following sources support the established background methods, instrumentation, and software used in this repository:


```text
NASA — Mars Reconnaissance Orbiter Science Instruments / HiRISE.
https://science.nasa.gov/mission/mars-reconnaissance-orbiter/science-instruments/

NASA Planetary Data System — Mars Reconnaissance Orbiter HiRISE instrument context.
https://pds.nasa.gov/ds-view/pds/viewContext.jsp?identifier=urn%3Anasa%3Apds%3Acontext%3Ainstrument%3Ahirise.mro&version=1.2

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining.
https://doi.org/10.1109/ICDM.2008.17

Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600–612.
https://doi.org/10.1109/TIP.2003.819861

van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9, 2579–2605.
https://www.jmlr.org/papers/v9/vandermaaten08a.html

McInnes, L., Healy, J., Saul, N., & Großberger, L. (2018). UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software, 3(29), 861.
https://doi.org/10.21105/joss.00861

NIST/SEMATECH e-Handbook of Statistical Methods — Median Absolute Deviation.
https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm

PyTorch documentation — Conv2d and ConvTranspose2d.
https://docs.pytorch.org/docs/main/generated/torch.nn.Conv2d.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

scikit-learn documentation — IsolationForest.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
```


---

## Data and Model Artifacts

Competition-provided imagery and other restricted artifacts should only be redistributed when permitted by the competition organizers.

Large datasets, generated checkpoints, and other binary artifacts should generally be kept out of Git history unless they are explicitly required and permitted.

---



