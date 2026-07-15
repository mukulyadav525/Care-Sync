"""Adaptive (EWMA) baselines for residuals, conditioned on activity bucket."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class EWMAStats:
    """Exponentially-weighted mean/variance, optionally tracked per group key
    (e.g. per activity bucket) so "normal" is contextual (DESIGN.md section 8).

    Warm-up fix: we initialise variance to a large value (100, i.e. std≈10 bpm)
    and suppress z-scores for the first `n_warmup` observations per key.  This
    prevents the original var=1.0 initialisation from inflating z-scores during
    the cold-start period (e.g. ±5 bpm residuals gave z≈5 immediately, causing
    many false alerts in the first few minutes of a session).
    """

    def __init__(self, lam: float = 0.02, n_warmup: int = 50):
        self.lam = lam
        self.n_warmup = n_warmup
        self._mean: dict = defaultdict(float)
        # Large initial variance (std≈10) so early z-scores are conservative.
        self._var: dict = defaultdict(lambda: 100.0)
        self._count: dict = defaultdict(int)
        self._initialized: dict = defaultdict(bool)

    def update(self, value: float, key=0) -> tuple[float, float]:
        """Update stats for `key` with `value`. Returns (mean, std) post-update."""
        if not self._initialized[key]:
            self._mean[key] = value
            self._var[key] = 100.0  # std ≈ 10 bpm; conservative cold start
            self._initialized[key] = True
        else:
            delta = value - self._mean[key]
            self._mean[key] += self.lam * delta
            self._var[key] = (1 - self.lam) * (self._var[key] + self.lam * delta ** 2)
        self._count[key] += 1
        return self._mean[key], float(np.sqrt(self._var[key]))

    def zscore(self, value: float, key=0) -> float:
        """Return z-score.  Returns 0.0 during the warm-up window so that the
        cold-start period does not generate spurious alerts."""
        count_before = self._count[key]
        mean, std = self.update(value, key=key)
        if count_before < self.n_warmup:
            return 0.0
        std = max(std, 1e-6)
        return (value - mean) / std
