"""A8: single global TRAIN-ONLY affine target normalisation (docs/A8_ABP_SCALE_SENSITIVITY_PREREGISTRATION.md §4).

y_norm = (y_mmHg - mu_train) / sigma_train with ONE scalar mu/sigma computed from every ABP sample value of the TRAIN subjects only.
Never per subject / per recording / per window; never from val or test; no clipping, no quantiles, no min-max. The inverse
y_mmHg = sigma * y_norm + mu is applied to every prediction before any clinical metric.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TargetNorm:
    mu: float
    sigma: float
    source: str = ""

    def forward(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mu) / self.sigma).astype(np.float32)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return (y * self.sigma + self.mu).astype(np.float32)

    @staticmethod
    def load(path: str | Path) -> "TargetNorm":
        d = json.loads(Path(path).read_text())
        return TargetNorm(float(d["mu_train"]), float(d["sigma_train"]), str(path))

    @staticmethod
    def identity() -> "TargetNorm":
        return TargetNorm(0.0, 1.0, "identity")

    @property
    def is_identity(self) -> bool:
        return self.mu == 0.0 and self.sigma == 1.0


def compute_train_stats(processed: Path, train_subjects: list[str]) -> dict:
    """Streaming exact mean/std over all target samples of the TRAIN subjects (float64 accumulators)."""
    n, s1, s2 = 0, 0.0, 0.0
    per_subject = {}
    for sub in sorted(train_subjects):
        y = np.load(Path(processed) / f"{sub}.npz")["y"].astype(np.float64)
        n += y.size
        s1 += y.sum()
        s2 += (y**2).sum()
        per_subject[sub] = {"n": int(y.size), "mean": float(y.mean()), "std": float(y.std())}
    mu = s1 / n
    var = s2 / n - mu**2
    return {"mu_train": float(mu), "sigma_train": float(np.sqrt(var)), "n_train_samples": int(n), "n_train_subjects": len(train_subjects), "per_subject": per_subject}


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
