# NSSC 2026 - HiRISE Unsupervised Anomaly Detection**

An end-to-end unsupervised anomaly detection pipeline for identifying unusual surface patterns in HiRISE image crops using convolutional autoencoders, latent-space novelty detection, statistical thresholding, and reconstruction-error analysis.

## Overview**

This project develops a fully unsupervised pipeline for detecting potentially unusual surface features in HiRISE imagery without using pretrained models or ground-truth anomaly labels.

The pipeline consists of four major stages:

1. Representation Learning

   - Train convolutional autoencoders on HiRISE image crops.

   - Compare a baseline V1 architecture with an improved V2 architecture.

2. Latent-Space Novelty Detection

   - Extract 256-dimensional latent representations from the final V2 model.

   - Apply Isolation Forest to assign novelty scores.

3. Statistical Anomaly Thresholding

   - Analyze the observed novelty-score distribution.

   - Compare multiple statistical thresholding approaches.

   - Select the final threshold using a robust median + MAD criterion.

4. Reconstruction Interpretability

   - Generate pixel-wise reconstruction-error maps.

   - Analyze the highest-novelty crops.

   - Form cautious geological hypotheses based on observed spatial patterns.

## Project Structure**

```text

NSSC-2026/

│

├── README.md

│

├── notebook/

│   └── NSSC_2026_Merged_fixed.ipynb

│

├── report/

│   └── Final_Report.pdf

│

├── changelog/

│   └── Engineering_Changelog.md

│

├── src/

│   ├── model_v1.py

│   ├── model_v2.py

│   ├── model_v3.py

│   └── other_required_code.py

│

├── results/

│   ├── figures/

│   │   ├── top5_reconstruction_analysis.png

│   │   ├── top5_clean_error_heatmaps.png

│   │   ├── umap_anomaly_overlay.png

│   │   ├── threshold_diagnostics.png

│   │   └── novelty_score_kde.png

│   │

│   └── tables/

│       ├── novelty_scores.csv

│       └── novelty_scores_with_metadata.csv

│

├── requirements.txt

│

└── .gitignore

Model Architecture

V1 - Baseline

V1 is the baseline convolutional autoencoder.

Key characteristics:

Grayscale HiRISE image input

256-dimensional latent representation

Adaptive average pooling bottleneck

Convolutional encoder

Transposed-convolution decoder

Final bilinear resizing stage

MSE-only reconstruction loss

Trainable parameters:

3,497,153

Best validation performance:

Validation MSE  : 0.003207

Validation SSIM : 0.667980

V2 - Final Autoencoder

V2 improves the bottleneck and reconstruction process of V1.

The main architectural changes are:

Replaces the adaptive-average-pooling bottleneck with a learned strided-convolution bottleneck.

Reduces the spatial representation to 8 × 8.

Projects the learned representation into a 256-dimensional latent vector.

Uses learned convolutional refinement after the final resize operation.

Uses a combined MSE + SSIM reconstruction objective.

Loss:

Loss = 10 × MSE + 1 × (1 − SSIM)

Trainable parameters:

10,099,521

Best validation performance:

Validation MSE  : 0.002164

Validation SSIM : 0.708526

The best validation metrics were obtained at Epoch 19 of the final 20-epoch training run.

Latent Representation

The final V2 encoder produces a 256-dimensional latent vector for each image crop.

For the complete dataset:

Number of image crops : 10,422

Latent dimensions     : 256

Latent matrix         : (10422, 256)

The latent vectors are used as the input to the unsupervised novelty detection stage.

Anomaly Detection

Isolation Forest

The 256-dimensional latent vectors are passed to an Isolation Forest.

Configuration:

n_estimators = 300

max_samples = auto

contamination = 0.05

The contamination parameter is used by Isolation Forest during model fitting. It is not used to determine the final anomaly count.

The Isolation Forest decision scores are sign-flipped so that:

Higher novelty score = More anomalous

The resulting scores are stored in:

results/tables/novelty_scores.csv

Statistical Thresholding

Instead of directly accepting the Isolation Forest contamination fraction as the final anomaly threshold, the observed novelty-score distribution is analyzed statistically.

Candidate approaches include:

Gaussian mean + standard deviation

Robust median + MAD

Elbow/curvature analysis of sorted scores

The final threshold is based on a robust median + MAD criterion:

Final Threshold = Median Score + 6 × MAD

This approach is intended to reduce sensitivity to skewed score distributions and extreme values.

The corresponding diagnostic figures are stored in:

results/figures/threshold_diagnostics.png

results/figures/novelty_score_kde.png

Top Anomalies

The highest-novelty crops from the final analysis are:

Rank    Image   Novelty Score

1   sample_01724.jpg    0.091290

2   sample_03876.jpg    0.084111

3   sample_00380.jpg    0.081736

4   sample_10097.jpg    0.081202

5   sample_07798.jpg    0.076856

These crops are subsequently analyzed using reconstruction-error heatmaps.

Reconstruction Interpretability

For high-novelty crops, the V2 decoder reconstructs the original image.

The pixel-wise reconstruction error is calculated as:

Error(x,y) = (Original(x,y) - Reconstruction(x,y))²

The resulting error maps highlight regions that the learned reconstruction model has difficulty representing.

Full Reconstruction Analysis

results/figures/top5_reconstruction_analysis.png

Shows:

Original crop

V2 reconstruction

Reconstruction-error heatmap

Interior-Only Error Analysis

results/figures/top5_clean_error_heatmaps.png

This analysis reduces the influence of invalid/background boundary regions and provides a cleaner view of the spatial distribution of reconstruction error.

Geological Interpretation

The reconstruction-error maps are used to formulate hypotheses rather than definitive geological classifications.

Examples of potentially relevant patterns include:

Elongated linear structures

Curved boundaries

Parallel bands

Localized high-error regions

Abrupt transitions between surface textures

Possible interpretations include:

Ridges

Troughs

Scarps

Dunes or grooves

Layered surface structures

Lineament-like features

Localized terrain-texture variations

Imaging geometry, illumination, crop boundaries, and reconstruction artifacts can also produce elevated error. Therefore, geological interpretations are treated as hypotheses requiring further investigation.

Metadata Analysis

Flagged crops are joined with source-image metadata using the image/source mapping.

The metadata-enriched results are stored in:

results/tables/novelty_scores_with_metadata.csv

Relevant metadata can include:

Source image ID

Latitude

Longitude

Sun angle

Season

Image resolution

This allows the detected anomalies to be examined for possible relationships with acquisition conditions and spatial location.

Visualizations

The project produces several diagnostic figures.

UMAP Latent-Space Visualization

umap_anomaly_overlay.png

Shows the distribution of latent representations and the locations of detected anomalies within the latent space.

Novelty Score Distribution

novelty_score_kde.png

Visualizes the observed novelty-score distribution.

Threshold Diagnostics

threshold_diagnostics.png

Provides statistical diagnostics used in selecting the final threshold.

Reconstruction Analysis

top5_reconstruction_analysis.png

Shows original images, reconstructions, and reconstruction-error maps for the highest-novelty candidates.

Clean Error Heatmaps

top5_clean_error_heatmaps.png

Shows interior-focused reconstruction-error maps after reducing boundary/background influence.

Reproducibility

The main experimental workflow is contained in:

notebook/NSSC_2026_Merged_fixed.ipynb

The source implementations are provided separately under:

src/

The main saved latent representation is:

latents_v2.npy

with the corresponding filename mapping:

latent_filenames_v2.json

The notebook records the training configuration, anomaly detection parameters, thresholding procedure, and generated outputs.

Requirements

Install the required Python packages using:

pip install -r requirements.txt

The project uses common scientific Python and deep-learning libraries including:

PyTorch

NumPy

Pandas

SciPy

scikit-learn

Matplotlib

UMAP

Pillow

Running the Project

The complete workflow can be executed from the Jupyter notebook:

notebook/NSSC_2026_Merged_fixed.ipynb

The general execution flow is:

HiRISE Images

      │

      ▼

Preprocessing

      │

      ▼

V1 Baseline Autoencoder

      │

      ▼

V2 Improved Autoencoder

      │

      ▼

256-D Latent Representations

      │

      ▼

Isolation Forest

      │

      ▼

Novelty Scores

      │

      ▼

Statistical Thresholding

      │

      ▼

Flagged Anomalies

      │

      ▼

V2 Reconstruction

      │

      ▼

Pixel-wise Error Maps

      │

      ▼

Geological Hypotheses

Results

The final V2 model improves reconstruction quality over the V1 baseline:

V1 → MSE 0.003207 | SSIM 0.667980

V2 → MSE 0.002164 | SSIM 0.708526

The final latent space contains:

10,422 samples × 256 dimensions

The anomaly detection stage identifies high-novelty candidates without requiring labelled anomalies or a pretrained classifier.

Limitations

The detected anomalies should not automatically be interpreted as confirmed geological discoveries.

Potential sources of false positives include:

Image boundaries

Illumination variations

Viewing geometry

Acquisition artifacts

Unusual but non-geological textures

Reconstruction-model limitations

The geological interpretations in this project are therefore presented as hypotheses based on the spatial structure of reconstruction errors.

Project Deliverables

The repository contains the major project deliverables:

Notebook: Complete experimental workflow

Report: Detailed methodology, results, figures, thresholds, and interpretations

Engineering Changelog: Evolution of the model architecture and design decisions

Source Code: V1, V2, V3 and supporting utilities

Results: Generated figures and numerical result tables

Authors

Suvam Kar , Sagnik Mitra

HiRISE Unsupervised Anomaly Detection Project
