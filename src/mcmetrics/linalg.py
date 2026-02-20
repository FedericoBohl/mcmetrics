# src/mcmetrics/linalg.py
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch


SolveMethod = Literal["cholesky", "lstsq", "qr"]


def chol_spd(
    A: torch.Tensor,
    *,
    upper: bool = False,
    check_spd: bool = True,
    jitter: float = 0.0,
) -> torch.Tensor:
    """
    Batched Cholesky for SPD matrices using cholesky_ex.

    Parameters
    ----------
    A : (..., n, n)
    upper : if True returns upper-triangular factor U such that A = U^T U.
            if False returns lower-triangular factor L such that A = L L^T.
    check_spd : raise ValueError if factorization fails.
    jitter : if >0, adds jitter * I before factorization.

    Returns
    -------
    C : (..., n, n) triangular factor
    """
    if jitter and jitter > 0.0:
        n = A.shape[-1]
        eye = torch.eye(n, dtype=A.dtype, device=A.device)
        A = A + jitter * eye

    C, info = torch.linalg.cholesky_ex(A, upper=upper)
    if check_spd and torch.any(info != 0):
        raise ValueError("Matrix must be SPD; Cholesky factorization failed.")
    return C


def solve_triangular_left(
    C: torch.Tensor,
    B: torch.Tensor,
    *,
    upper: bool,
) -> torch.Tensor:
    """Solve C X = B for X with triangular C, batched."""
    return torch.linalg.solve_triangular(C, B, upper=upper, left=True)


def inv_via_solve(A: torch.Tensor) -> torch.Tensor:
    """Compute inverse via solve against identity. A: (..., k, k)."""
    k = A.shape[-1]
    I = torch.eye(k, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A, I)


def solve_ls(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    method: SolveMethod = "cholesky",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve least squares for each replication:

      beta = argmin ||y - X beta||_2

    Inputs
    ------
    X : (R,n,k)
    y : (R,n)

    Returns
    -------
    beta : (R,k)
    XtX_inv : (R,k,k)  (computed from normal equations for method='cholesky' or 'qr')
    """
    if X.ndim != 3 or y.ndim != 2:
        raise ValueError(f"Expected X (R,n,k) and y (R,n). Got X {tuple(X.shape)}, y {tuple(y.shape)}")
    if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
        raise ValueError("Batch/obs dims mismatch between X and y.")

    R, n, k = X.shape

    if method == "lstsq":
        # torch.linalg.lstsq supports batching in recent torch versions
        sol = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)  # (R,k)
        # XtX_inv is still needed for classic vcov; compute from normal equations
        XtX = torch.einsum("rnk,rnj->rkj", X, X)
        XtX_inv = inv_via_solve(XtX)
        return sol, XtX_inv

    if method == "qr":
        Q, Rm = torch.linalg.qr(X, mode="reduced")  # Q:(R,n,k), R:(R,k,k)
        Qt_y = torch.einsum("rnk,rn->rk", Q, y)
        beta = solve_triangular_left(Rm, Qt_y.unsqueeze(-1), upper=True).squeeze(-1)
        XtX_inv = inv_via_solve(torch.einsum("rnk,rnj->rkj", X, X))
        return beta, XtX_inv

    # default: normal equations + cholesky
    XtX = torch.einsum("rnk,rnj->rkj", X, X)          # (R,k,k)
    Xty = torch.einsum("rnk,rn->rk", X, y)            # (R,k)
    C = chol_spd(XtX, upper=False, check_spd=True, jitter=0.0)  # lower
    # Solve XtX beta = Xty using two triangular solves
    z = solve_triangular_left(C, Xty.unsqueeze(-1), upper=False).squeeze(-1)
    beta = solve_triangular_left(C.transpose(-1, -2), z.unsqueeze(-1), upper=True).squeeze(-1)

    XtX_inv = inv_via_solve(XtX)
    return beta, XtX_inv
