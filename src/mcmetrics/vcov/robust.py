from __future__ import annotations

from typing import Literal

import torch

from mcmetrics.exceptions import NotSupportedError, ShapeError


def _check_shapes(X: torch.Tensor, resid: torch.Tensor) -> None:
    if X.ndim != 3:
        raise ShapeError(f"X must be (R,n,k). Got {tuple(X.shape)}")
    if resid.ndim != 2:
        raise ShapeError(f"resid must be (R,n). Got {tuple(resid.shape)}")
    if X.shape[0] != resid.shape[0] or X.shape[1] != resid.shape[1]:
        raise ShapeError(f"Batch/obs dims mismatch: X {tuple(X.shape)}, resid {tuple(resid.shape)}")


def meat_white(X: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    _check_shapes(X, resid)
    e2 = resid * resid
    return torch.einsum("rni,rn,rnj->rij", X, e2, X)


def vcov_hc0(X: torch.Tensor, resid: torch.Tensor, XtX_inv: torch.Tensor) -> torch.Tensor:
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ShapeError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")
    M = meat_white(X, resid)
    return XtX_inv @ M @ XtX_inv


def vcov_hc1(X: torch.Tensor, resid: torch.Tensor, XtX_inv: torch.Tensor) -> torch.Tensor:
    _check_shapes(X, resid)
    _, n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ShapeError(f"Need n > k for HC1 scaling. Got n={n}, k={k}.")
    scale = float(n) / float(df_resid)
    return vcov_hc0(X, resid, XtX_inv) * scale


def _coerce_clusters(
    clusters: torch.Tensor | list[int] | tuple[int, ...],
    *,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    c = clusters if isinstance(clusters, torch.Tensor) else torch.as_tensor(clusters)
    c = c.to(device=device)
    if c.ndim != 1:
        raise ShapeError(f"clusters must be 1D (n,). Got {tuple(c.shape)}")
    if c.shape[0] != n:
        raise ShapeError(f"clusters length must be n={n}. Got {int(c.shape[0])}")
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
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ShapeError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")

    R, n, k = X.shape
    inv, G = _coerce_clusters(clusters, n=n, device=X.device)
    if G <= 1:
        raise ShapeError(f"Need at least 2 clusters. Got G={G}.")

    Xe = X * resid.unsqueeze(-1)  # (R,n,k)
    S = torch.zeros((R, G, k), device=X.device, dtype=X.dtype)
    idx = inv.view(1, n, 1).expand(R, n, k)
    S.scatter_add_(1, idx, Xe)

    meat = torch.einsum("rgk,rgl->rkl", S, S)
    vc = XtX_inv @ meat @ XtX_inv

    if correction == "CR1":
        df_resid = n - k
        if df_resid <= 0:
            raise ShapeError(f"Need n > k for CR1 correction. Got n={n}, k={k}.")
        scale = (float(G) / float(G - 1)) * (float(n - 1) / float(df_resid))
        vc = vc * float(scale)
    elif correction != "none":
        raise NotSupportedError("correction must be one of {'none','CR1'}")

    return vc


def vcov_hac(
    X: torch.Tensor,
    resid: torch.Tensor,
    XtX_inv: torch.Tensor,
    *,
    max_lags: int = 1,
    kernel: Literal["bartlett"] = "bartlett",
) -> torch.Tensor:
    _check_shapes(X, resid)
    if XtX_inv.ndim != 3:
        raise ShapeError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")
    if kernel != "bartlett":
        raise NotSupportedError("Only kernel='bartlett' is currently supported")

    _, n, _ = X.shape
    L = int(max(0, min(max_lags, n - 1)))

    Xe = X * resid.unsqueeze(-1)
    meat = torch.einsum("rnk,rnl->rkl", Xe, Xe)

    for lag in range(1, L + 1):
        w = 1.0 - float(lag) / float(L + 1)
        A = Xe[:, lag:, :]
        B = Xe[:, :-lag, :]
        Gamma = torch.einsum("rnk,rnl->rkl", A, B)
        meat = meat + w * (Gamma + Gamma.transpose(-1, -2))

    return XtX_inv @ meat @ XtX_inv
