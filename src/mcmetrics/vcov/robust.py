# src/mcmetrics/vcov/robust.py
from __future__ import annotations

from typing import Literal

import torch


def _check_shapes(X: torch.Tensor, resid: torch.Tensor) -> None:
    if X.ndim != 3:
        raise ValueError(f"X must be (R,n,k). Got {tuple(X.shape)}")
    if resid.ndim != 2:
        raise ValueError(f"resid must be (R,n). Got {tuple(resid.shape)}")
    if X.shape[0] != resid.shape[0] or X.shape[1] != resid.shape[1]:
        raise ValueError(f"Batch/obs dims mismatch: X {tuple(X.shape)}, resid {tuple(resid.shape)}")


def meat_white(X: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """White "meat" term: X' diag(e^2) X, batched over R.

    Inputs
    - X     : (R,n,k)
    - resid : (R,n)

    Output
    - meat  : (R,k,k)
    """
    _check_shapes(X, resid)
    e2 = resid * resid  # (R,n)
    return torch.einsum("rni,rn,rnj->rij", X, e2, X)


def vcov_hc0(X: torch.Tensor, resid: torch.Tensor, XtX_inv: torch.Tensor) -> torch.Tensor:
    """HC0 variance-covariance: (X'X)^(-1) [X' diag(e^2) X] (X'X)^(-1)."""
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ValueError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")
    M = meat_white(X, resid)
    return XtX_inv @ M @ XtX_inv


def vcov_hc1(X: torch.Tensor, resid: torch.Tensor, XtX_inv: torch.Tensor) -> torch.Tensor:
    """HC1 variance-covariance: HC0 * n/(n-k)."""
    _check_shapes(X, resid)
    R, n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Need n > k for HC1 scaling. Got n={n}, k={k}.")
    scale = float(n) / float(df_resid)
    return vcov_hc0(X, resid, XtX_inv) * scale


def _coerce_clusters(
    clusters: torch.Tensor | list[int] | tuple[int, ...],
    *,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return (inv, G) where inv is (n,) int64 with values in {0,...,G-1}."""
    c = clusters if isinstance(clusters, torch.Tensor) else torch.as_tensor(clusters)
    c = c.to(device=device)
    if c.ndim != 1:
        raise ValueError(f"clusters must be 1D (n,). Got {tuple(c.shape)}")
    if c.shape[0] != n:
        raise ValueError(f"clusters length must be n={n}. Got {int(c.shape[0])}")

    _, inv = torch.unique(c, sorted=True, return_inverse=True)
    G = int(inv.max().item()) + 1 if inv.numel() > 0 else 0
    return inv.to(dtype=torch.int64), G


def vcov_cluster(
    X: torch.Tensor,
    resid: torch.Tensor,
    XtX_inv: torch.Tensor,
    clusters: torch.Tensor | list[int] | tuple[int, ...],
    *,
    correction: Literal["none", "CR1"] = "CR1",
) -> torch.Tensor:
    """One-way cluster-robust vcov (Arellano / Liang-Zeger), batched over R.

    Assumptions
    - clusters are common across replications (clusters is 1D length n).

    Implementation
    - S_rg = sum_{i in g} x_ri * e_ri   (k-vector)
    - meat_r = sum_g S_rg S_rg'
    - vcov_r = XtX_inv_r meat_r XtX_inv_r

    correction
    - "CR1": (G/(G-1)) * ((n-1)/(n-k))
    """
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ValueError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")

    R, n, k = X.shape
    inv, G = _coerce_clusters(clusters, n=n, device=X.device)
    if G <= 1:
        raise ValueError(f"Need at least 2 clusters. Got G={G}.")

    Xe = X * resid.unsqueeze(-1)  # (R,n,k)
    S = torch.zeros((R, G, k), device=X.device, dtype=X.dtype)
    idx = inv.view(1, n, 1).expand(R, n, k)
    S.scatter_add_(1, idx, Xe)
    meat = torch.einsum("rgk,rgl->rkl", S, S)  # (R,k,k)

    vc = XtX_inv @ meat @ XtX_inv

    if correction == "CR1":
        df_resid = n - k
        if df_resid <= 0:
            raise ValueError(f"Need n > k for CR1 correction. Got n={n}, k={k}.")
        scale = (float(G) / float(G - 1)) * (float(n - 1) / float(df_resid))
        vc = vc * float(scale)
    elif correction != "none":
        raise ValueError("correction must be one of {'none','CR1'}")

    return vc


def vcov_hac(
    X: torch.Tensor,
    resid: torch.Tensor,
    XtX_inv: torch.Tensor,
    *,
    max_lags: int = 1,
    kernel: Literal["bartlett"] = "bartlett",
) -> torch.Tensor:
    """HAC (Newey-West) vcov (batched over R) using Bartlett kernel."""
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ValueError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")
    if kernel != "bartlett":
        raise ValueError("Only kernel='bartlett' is currently supported")

    R, n, k = X.shape
    L = int(max(0, min(max_lags, n - 1)))

    Xe = X * resid.unsqueeze(-1)  # (R,n,k)
    meat = torch.einsum("rnk,rnl->rkl", Xe, Xe)  # Gamma_0

    for l in range(1, L + 1):
        w = 1.0 - float(l) / float(L + 1)
        A = Xe[:, l:, :]   # (R,n-l,k)
        B = Xe[:, :-l, :]  # (R,n-l,k)
        Gamma = torch.einsum("rnk,rnl->rkl", A, B)
        meat = meat + w * (Gamma + Gamma.transpose(-1, -2))

    return XtX_inv @ meat @ XtX_inv