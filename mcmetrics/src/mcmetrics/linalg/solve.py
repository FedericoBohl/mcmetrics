from __future__ import annotations

from typing import Literal, Tuple

import torch

from mcmetrics.exceptions import ShapeError
from mcmetrics.linalg.chol import chol_inverse, chol_solve, safe_cholesky

SolveMethod = Literal["solve", "lstsq", "cholesky", "qr"]


def _batched_eye(k: int, R: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.eye(k, device=ref.device, dtype=ref.dtype).expand(R, k, k)


def solve_ls(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    solve_method: SolveMethod = "solve",
    chol_jitter: float = 0.0,
    chol_max_tries: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve least squares for each replication.

    X : (R,n,k)
    y : (R,n)
    returns (beta:(R,k), XtX:(R,k,k), XtX_inv:(R,k,k))
    """
    if X.ndim != 3:
        raise ShapeError(f"X must be (R,n,k). Got {tuple(X.shape)}")
    if y.ndim != 2:
        raise ShapeError(f"y must be (R,n). Got {tuple(y.shape)}")
    if X.shape[0] != y.shape[0] or X.shape[1] != y.shape[1]:
        raise ShapeError(f"Batch/obs dims mismatch: X {tuple(X.shape)}, y {tuple(y.shape)}")

    R, _, k = X.shape

    if solve_method == "qr":
        Q, Rm = torch.linalg.qr(X, mode="reduced")
        Qt_y = torch.einsum("rnk,rn->rk", Q, y)
        beta = torch.linalg.solve(Rm, Qt_y.unsqueeze(-1)).squeeze(-1)

        eye = _batched_eye(k, R, X)
        invR = torch.linalg.solve_triangular(Rm, eye, upper=True)
        XtX_inv = invR @ invR.transpose(-1, -2)
        XtX = Rm.transpose(-1, -2) @ Rm
        return beta, XtX, XtX_inv

    Xt = X.transpose(1, 2)
    XtX = Xt @ X
    Xty = Xt @ y.unsqueeze(-1)

    if solve_method == "solve":
        beta = torch.linalg.solve(XtX, Xty).squeeze(-1)
        XtX_inv = torch.linalg.solve(XtX, _batched_eye(k, R, X))
        return beta, XtX, XtX_inv

    if solve_method == "lstsq":
        beta = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)
        XtX_inv = torch.linalg.solve(XtX, _batched_eye(k, R, X))
        return beta, XtX, XtX_inv

    if solve_method == "cholesky":
        L = safe_cholesky(XtX, jitter=chol_jitter, max_tries=chol_max_tries)
        beta = chol_solve(Xty, L).squeeze(-1)
        XtX_inv = chol_inverse(L)
        return beta, XtX, XtX_inv

    raise ValueError(f"Unknown solve_method {solve_method!r}. Use 'solve','lstsq','cholesky','qr'.")
