from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from mcmetrics.sigma.registry import BaseSigmaEstimator, register_sigma_estimator
from mcmetrics.sigma.spec import SigmaSpec


@register_sigma_estimator("diag_resid2")
@dataclass
class DiagResid2Sigma(BaseSigmaEstimator):
    """Diagonal Sigma estimator using squared residuals: Sigma_ii = e_i^2.

    Warning
    -------
    This estimator is *not* a standard econometric FGLS estimator by itself;
    it tends to overfit. It is provided as a simple pluggable example and for
    internal testing. Future FGLS estimators should generally impose structure
    (e.g., parametric variance functions, auxiliary regressions, etc.).
    """

    min_variance: float = 1e-12

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        resid: torch.Tensor,
        **kwargs,
    ) -> SigmaSpec:
        # resid expected shape: (R,n)
        s2 = resid * resid
        if self.min_variance > 0:
            s2 = torch.clamp(s2, min=float(self.min_variance))
        return SigmaSpec.diagonal(diag=s2, method=self.name, options={"min_variance": float(self.min_variance)})
