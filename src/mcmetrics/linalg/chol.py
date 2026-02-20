from __future__ import annotations

from typing import Optional

import torch

from mcmetrics.exceptions import ShapeError, SingularMatrixError


def safe_cholesky(A: torch.Tensor, *, jitter: float = 0.0, max_tries: int = 1) -> torch.Tensor:
    """
    Batched Cholesky with optional diagonal jitter escalation.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ShapeError(f"A must be (...,k,k). Got {tuple(A.shape)}")

    max_tries = max(1, int(max_tries))

    A0 = 0.5 * (A + A.transpose(-1, -2))
    k = A0.shape[-1]
    eye = torch.eye(k, device=A0.device, dtype=A0.dtype)

    last_err: Optional[Exception] = None
    for t in range(max_tries):
        try:
            Aj = A0 + (jitter * (10.0**t)) * eye if jitter > 0.0 else A0
            return torch.linalg.cholesky(Aj)
        except Exception as e:
            last_err = e

    assert last_err is not None
    raise SingularMatrixError(str(last_err))


def chol_solve(B: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    return torch.cholesky_solve(B, L)


def chol_inverse(L: torch.Tensor) -> torch.Tensor:
    return torch.cholesky_inverse(L)
