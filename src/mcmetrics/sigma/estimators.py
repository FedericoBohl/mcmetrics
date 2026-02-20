from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from mcmetrics.exceptions import ShapeError
from mcmetrics.linalg import solve_ls
from mcmetrics.sigma.spec import SigmaSpec


class SigmaEstimator(Protocol):
    """Protocol for Sigma estimators used in FGLS.

    Implementations must return a SigmaSpec.
    """

    def fit(self, X: torch.Tensor, y: torch.Tensor, resid: torch.Tensor) -> SigmaSpec:
        ...


def _white_design(X: torch.Tensor, *, has_const: bool) -> torch.Tensor:
    """Build the White auxiliary design matrix.

    X: (R,n,k)
    Returns Z: (R,n,p) with columns [1, x, vec(upper(x x'))]
    where x excludes the constant if has_const=True.
    """
    if X.ndim != 3:
        raise ShapeError(f"X must be (R,n,k). Got {tuple(X.shape)}")

    R, n, k = X.shape
    X0 = X[:, :, 1:] if has_const else X
    q = X0.shape[2]

    ones = torch.ones((R, n, 1), dtype=X.dtype, device=X.device)

    if q == 0:
        return ones

    # Quadratic and cross terms: upper triangle of x x'
    prod = X0.unsqueeze(-1) * X0.unsqueeze(-2)  # (R,n,q,q)
    idx = torch.triu_indices(q, q, device=X.device)
    quad = prod[:, :, idx[0], idx[1]]  # (R,n,q(q+1)/2)

    return torch.cat([ones, X0, quad], dim=2)


@dataclass
class WhiteDiagSigma:
    """White-style diagonal Sigma estimator.

    Fits an auxiliary regression of e^2 on the White design matrix Z and
    uses the fitted values as variance estimates.
    """

    has_const: bool = True
    min_var: float = 1e-12
    solve_method: str = "cholesky"

    def fit(self, X: torch.Tensor, y: torch.Tensor, resid: torch.Tensor) -> SigmaSpec:
        if resid.ndim != 2:
            raise ShapeError(f"resid must be (R,n). Got {tuple(resid.shape)}")
        Z = _white_design(X, has_const=self.has_const)
        e2 = resid * resid

        gamma, _, _ = solve_ls(Z, e2, solve_method=self.solve_method)
        e2_hat = (Z @ gamma.unsqueeze(-1)).squeeze(-1)
        e2_hat = torch.clamp(e2_hat, min=float(self.min_var))
        return SigmaSpec.diagonal(e2_hat)


@dataclass
class GreeneDummiesDiagSigma:
    """Greene-style diagonal Sigma estimator using group dummies.

    Given groups g(i) in {0,...,G-1}, estimate group variances as
      s_g^2 = mean(e_i^2 | g(i)=g)
    and set Sigma_ii = s_{g(i)}^2.
    """

    groups: torch.Tensor  # (n,)
    min_var: float = 1e-12

    def __post_init__(self) -> None:
        g = self.groups
        if not isinstance(g, torch.Tensor):
            g = torch.as_tensor(g)
        if g.ndim != 1:
            raise ShapeError(f"groups must be (n,). Got {tuple(g.shape)}")
        object.__setattr__(self, "groups", g)

    def fit(self, X: torch.Tensor, y: torch.Tensor, resid: torch.Tensor) -> SigmaSpec:
        if resid.ndim != 2:
            raise ShapeError(f"resid must be (R,n). Got {tuple(resid.shape)}")
        R, n = resid.shape
        g = self.groups.to(device=resid.device)
        if int(g.shape[0]) != n:
            raise ShapeError(f"groups length {int(g.shape[0])} must match n={n}")

        # Map to {0,...,G-1}
        uniq, inv = torch.unique(g, return_inverse=True)
        G = int(uniq.numel())

        # Build one-hot weights matrix W: (n,G)
        W = torch.nn.functional.one_hot(inv, num_classes=G).to(dtype=resid.dtype)
        counts = W.sum(dim=0).clamp_min(1.0)  # (G,)

        e2 = resid * resid  # (R,n)
        sum_e2 = e2 @ W  # (R,G)
        var_g = sum_e2 / counts.view(1, G)
        var_hat = var_g[:, inv]  # (R,n)
        var_hat = torch.clamp(var_hat, min=float(self.min_var))
        return SigmaSpec.diagonal(var_hat)
