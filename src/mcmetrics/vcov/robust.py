# src/mcmetrics/vcov/robust.py
from __future__ import annotations

import torch


def _check_shapes(X: torch.Tensor, resid: torch.Tensor) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be (R,n,k). Got {tuple(X.shape)}")
    if resid.ndim != 2:
        raise ValueError(f"resid must be (R,n). Got {tuple(resid.shape)}")
    if X.shape[0] != resid.shape[0] or X.shape[1] != resid.shape[1]:
        raise ValueError(f"Batch/obs dims mismatch: X {tuple(X.shape)}, resid {tuple(resid.shape)}")


def meat_white(X: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """
    White "meat" term: X' diag(e^2) X, batched over R.

    Inputs
    - X     : (R,n,k)
    - resid : (R,n)

    Output
    - meat  : (R,k,k)
    """
    _check_shapes(X, resid)
    e2 = resid * resid  # (R,n)
    # meat[r] = sum_i e2[r,i] * x[r,i,:] x[r,i,:]'
    return torch.einsum("rni,rn,rnj->rij", X, e2, X)


def vcov_hc0(X: torch.Tensor, resid: torch.Tensor, XtX_inv: torch.Tensor) -> torch.Tensor:
    """
    HC0 variance-covariance: (X'X)^(-1) [X' diag(e^2) X] (X'X)^(-1)

    Inputs
    - X       : (R,n,k)
    - resid   : (R,n)
    - XtX_inv : (R,k,k)  precomputed inverse of X'X

    Output
    - vcov    : (R,k,k)
    """
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ValueError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")

    M = meat_white(X, resid)              # (R,k,k)
    return XtX_inv @ M @ XtX_inv          # (R,k,k)


def vcov_hc1(X: torch.Tensor, resid: torch.Tensor, XtX_inv: torch.Tensor) -> torch.Tensor:
    """
    HC1 variance-covariance: HC0 * n/(n-k)

    Inputs
    - X       : (R,n,k)
    - resid   : (R,n)
    - XtX_inv : (R,k,k)

    Output
    - vcov    : (R,k,k)
    """
    _check_shapes(X, resid)
    R, n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Need n > k for HC1 scaling. Got n={n}, k={k}.")

    scale = float(n) / float(df_resid)
    return vcov_hc0(X, resid, XtX_inv) * scale