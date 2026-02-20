from __future__ import annotations

import torch

from mcmetrics.exceptions import ShapeError


def _check_shapes(
    sigma_inv_X: torch.Tensor,
    resid_raw: torch.Tensor,
    bread_inv: torch.Tensor,
) -> tuple[int, int, int]:
    """
    sigma_inv_X: (R,n,k)  representing Sigma^{-1} X
    resid_raw : (R,n)    residuals in original space
    bread_inv : (R,k,k)  (X' Sigma^{-1} X)^{-1}
    """
    if sigma_inv_X.ndim != 3:
        raise ShapeError(f"sigma_inv_X must be (R,n,k). Got {tuple(sigma_inv_X.shape)}")
    if resid_raw.ndim != 2:
        raise ShapeError(f"resid_raw must be (R,n). Got {tuple(resid_raw.shape)}")
    if bread_inv.ndim != 3:
        raise ShapeError(f"bread_inv must be (R,k,k). Got {tuple(bread_inv.shape)}")

    R, n, k = sigma_inv_X.shape
    if resid_raw.shape != (R, n):
        raise ShapeError(f"Batch/obs dims mismatch: sigma_inv_X {tuple(sigma_inv_X.shape)}, resid_raw {tuple(resid_raw.shape)}")
    if bread_inv.shape != (R, k, k):
        raise ShapeError(f"bread_inv must be (R,k,k)={(R,k,k)}. Got {tuple(bread_inv.shape)}")
    return R, n, k


def vcov_gls_hc0(
    sigma_inv_X: torch.Tensor,
    resid_raw: torch.Tensor,
    bread_inv: torch.Tensor,
) -> torch.Tensor:
    """
    GLS-robust sandwich vcov (HC0):

        V = B [ (Sigma^{-1}X)' diag(e^2) (Sigma^{-1}X) ] B

    where B = (X' Sigma^{-1} X)^{-1} and e are residuals in original space.
    """
    _check_shapes(sigma_inv_X, resid_raw, bread_inv)
    e2 = resid_raw * resid_raw
    meat = torch.einsum("rni,rn,rnj->rij", sigma_inv_X, e2, sigma_inv_X)
    return bread_inv @ meat @ bread_inv


def vcov_gls_hc1(
    sigma_inv_X: torch.Tensor,
    resid_raw: torch.Tensor,
    bread_inv: torch.Tensor,
) -> torch.Tensor:
    """GLS-robust sandwich vcov (HC1): HC0 scaled by n/(n-k)."""
    R, n, k = _check_shapes(sigma_inv_X, resid_raw, bread_inv)
    df_resid = n - k
    if df_resid <= 0:
        raise ShapeError(f"Need n > k for HC1 scaling. Got n={n}, k={k}.")
    scale = float(n) / float(df_resid)
    return vcov_gls_hc0(sigma_inv_X, resid_raw, bread_inv) * scale
