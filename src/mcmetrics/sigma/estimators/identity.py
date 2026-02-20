from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch

from mcmetrics.sigma.registry import BaseSigmaEstimator, register_sigma_estimator
from mcmetrics.sigma.spec import SigmaSpec


@register_sigma_estimator("identity")
@dataclass
class IdentitySigma(BaseSigmaEstimator):
    """Return Sigma = I (identity covariance).

    This is mainly useful as a baseline and for debugging.
    """

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> SigmaSpec:
        # Diagonal form is cheapest and sufficient for identity.
        n = int(X.shape[-2])
        diag = torch.ones(n, dtype=X.dtype, device=X.device)
        return SigmaSpec.diagonal(diag=diag, method=self.name)
