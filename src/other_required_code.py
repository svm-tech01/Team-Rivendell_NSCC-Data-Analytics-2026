"""
NSSC 2026 - Other Required Code

Utility code shared by the notebook/project:
- HiRISE grayscale dataset
- SSIM implementation used by V2
- combined reconstruction loss
- robust threshold helpers
- reconstruction error helpers
- metadata alignment helper
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import math


IMG_SIZE = 227


class HiRISEDataset(Dataset):
    def __init__(self, image_paths, img_size=IMG_SIZE):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = Image.open(path).convert("L")

        if image.size != (self.img_size, self.img_size):
            image = image.resize(
                (self.img_size, self.img_size),
                Image.Resampling.BILINEAR,
            )

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        return image, os.path.basename(path)


def gaussian_window(window_size, sigma, channels=1, device="cpu"):
    coords = (
        torch.arange(window_size, dtype=torch.float32, device=device)
        - window_size // 2
    )
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.expand(
        channels, 1, window_size, window_size
    ).contiguous()


def ssim(
    img1,
    img2,
    window_size=11,
    sigma=1.5,
    C1=0.01 ** 2,
    C2=0.03 ** 2,
):
    """Compute mean SSIM for normalized grayscale tensors."""
    channels = img1.size(1)

    window = gaussian_window(
        window_size,
        sigma,
        channels,
        img1.device,
    )

    pad = window_size // 2

    mu1 = F.conv2d(
        img1, window, padding=pad, groups=channels
    )
    mu2 = F.conv2d(
        img2, window, padding=pad, groups=channels
    )

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=pad, groups=channels)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=pad, groups=channels)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=pad, groups=channels)
        - mu1_mu2
    )

    ssim_map = (
        (2 * mu1_mu2 + C1)
        * (2 * sigma12 + C2)
        / (
            (mu1_sq + mu2_sq + C1)
            * (sigma1_sq + sigma2_sq + C2)
        )
    )

    return ssim_map.mean()


def combined_loss(reconstruction, target, alpha=10.0, beta=1.0):
    """Final V2 loss: 10*MSE + 1*(1-SSIM)."""
    mse_value = F.mse_loss(reconstruction, target)
    ssim_value = ssim(reconstruction, target)

    loss = (
        alpha * mse_value
        + beta * (1.0 - ssim_value)
    )

    return loss, mse_value.item(), ssim_value.item()


def robust_threshold(scores, k=6.0):
    """Return median + k * normalized MAD."""
    scores = np.asarray(scores, dtype=float)

    median_score = float(np.median(scores))
    mad_score = float(
        stats_median_abs_deviation(scores)
    )

    threshold = median_score + k * mad_score
    return threshold, median_score, mad_score


def stats_median_abs_deviation(scores):
    """Small wrapper to keep scipy import localized."""
    from scipy import stats

    return stats.median_abs_deviation(
        scores,
        scale="normal",
    )


def reconstruction_errors(original, reconstruction, background_threshold=0.05):
    """
    Compute full-image and interior-only squared reconstruction errors.
    Background pixels at or below the threshold are excluded from the
    interior-only statistic.
    """
    original = np.asarray(original, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)

    error_map = (original - reconstruction) ** 2

    valid_mask = original > background_threshold
    valid_errors = error_map[valid_mask]

    return {
        "error_map": error_map,
        "mse": float(error_map.mean()),
        "interior_mse": float(valid_errors.mean()),
        "interior_max_error": float(valid_errors.max()),
    }


def align_metadata_to_filenames(
    merged_metadata,
    filenames,
):
    """
    Explicitly align metadata rows to latent filename ordering.
    """
    return (
        merged_metadata
        .set_index("filename")
        .loc[list(filenames)]
        .reset_index()
    )


def build_phase3_summary(top5, top5_results, crop_df, metadata_df):
    """
    Join Top-5 novelty results to crop/source metadata and reconstruction
    error metrics.
    """
    top5_metadata = (
        top5
        .merge(crop_df, on="filename", how="left")
        .merge(metadata_df, on="source_image_id", how="left")
    )

    summary = top5_metadata.copy()
    summary["full_mse"] = [
        r["mse"] for r in top5_results
    ]
    summary["interior_mse"] = [
        r["valid_mean_error"] for r in top5_results
    ]

    columns = [
        "filename",
        "source_image_id",
        "novelty_score",
        "full_mse",
        "interior_mse",
        "latitude",
        "longitude",
        "sun_angle",
        "season",
        "resolution",
    ]

    return summary[columns]
