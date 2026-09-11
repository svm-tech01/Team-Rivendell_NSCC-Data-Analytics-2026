"""
NSSC 2026 - Model V3 / Anomaly Detection Engine

The project's V3 stage is not a third autoencoder. It keeps the finalized V2
256-D latent representation and performs unsupervised novelty detection with
Isolation Forest followed by robust statistical thresholding.

Final threshold rule:
    threshold = median(score) + 6 * MAD(score)

Higher novelty_score means more anomalous.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats


class V3AnomalyDetector:
    def __init__(
        self,
        n_estimators=300,
        max_samples="auto",
        contamination=0.05,
        random_state=42,
        mad_multiplier=6.0,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.mad_multiplier = mad_multiplier

        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        self.median_score = None
        self.mad_score = None
        self.threshold = None

    def fit(self, Z, filenames=None):
        Z = np.asarray(Z)

        if Z.ndim != 2:
            raise ValueError("Z must be a 2-D latent matrix.")

        self.model.fit(Z)

        # sklearn uses lower values for more abnormal samples.
        # Sign flip makes higher values mean more novelty.
        novelty_score = -self.model.decision_function(Z)

        self.median_score = float(np.median(novelty_score))
        self.mad_score = float(
            stats.median_abs_deviation(
                novelty_score,
                scale="normal",
            )
        )

        self.threshold = (
            self.median_score
            + self.mad_multiplier * self.mad_score
        )

        if filenames is None:
            filenames = [f"sample_{i:05d}.jpg" for i in range(len(Z))]

        if len(filenames) != len(Z):
            raise ValueError("Number of filenames must match number of latent vectors.")

        result = pd.DataFrame({
            "filename": filenames,
            "novelty_score": novelty_score,
        })

        result["flagged"] = result["novelty_score"] > self.threshold
        result = result.sort_values(
            "novelty_score",
            ascending=False,
        ).reset_index(drop=True)

        self.results_ = result
        return result

    def top(self, n=5):
        if not hasattr(self, "results_"):
            raise RuntimeError("Call fit() before requesting top anomalies.")
        return self.results_.head(n).copy()

    def summary(self):
        if not hasattr(self, "results_"):
            raise RuntimeError("Call fit() before requesting a summary.")

        flagged = int(self.results_["flagged"].sum())
        total = len(self.results_)

        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "median_score": self.median_score,
            "mad_score": self.mad_score,
            "mad_multiplier": self.mad_multiplier,
            "threshold": self.threshold,
            "flagged_count": flagged,
            "flagged_percentage": 100.0 * flagged / total,
        }


if __name__ == "__main__":
    # Small smoke test only. Replace with the real V2 latent matrix in use.
    rng = np.random.default_rng(42)
    Z_demo = rng.normal(size=(100, 256))
    filenames_demo = [f"sample_{i:05d}.jpg" for i in range(100)]

    detector = V3AnomalyDetector()
    scores = detector.fit(Z_demo, filenames_demo)

    print(detector.summary())
    print(scores.head())
