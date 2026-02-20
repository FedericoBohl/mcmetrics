from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from mcmetrics.exceptions import InvalidWeightsError, NotSPDMatrixError, ShapeError
from mcmetrics.typing import as_torch
from mcmetrics.weights import as_batched_weights
from mcmetrics.sigma.spec import SigmaSpec


@dataclass(frozen=True)
class WhiteningInfo:
    """Whitening context returned by `whiten_system`.

    This object contains all objects needed to:
      - whiten vectors (residuals)
      - compute Sigma^{-1}X without forming Sigma^{-1}
      - compute logdet(Sigma) (for diagnostics)
    """

    diag_branch: bool
    n: int
    R: int

    # Diagonal branch
    sqrt_w: Optional[torch.Tensor] = None   # (R,n) = 1/sqrt(Sigma_ii)
    w_prec: Optional[torch.Tensor] = None   # (R,n) = 1/Sigma_ii
    Sigma_diag: Optional[torch.Tensor] = None  # (R,n)

    # Full branch
    chol: Optional[torch.Tensor] = None     # (n,n) or (R,n,n)
    chol_upper: bool = False

    # Diagnostics
    logdet: Optional[torch.Tensor] = None   # (R,)


def _coerce_full_sigma(Sigma: torch.Tensor, *, R: int, n: int) -> torch.Tensor:
    if Sigma.ndim == 2:
        if Sigma.shape != (n, n):
            raise ShapeError(f"Sigma must be (n,n)={(n,n)} or (R,n,n)={(R,n,n)}. Got {tuple(Sigma.shape)}")
        return Sigma
    if Sigma.ndim == 3:
        if Sigma.shape != (R, n, n):
            raise ShapeError(f"Sigma must be (R,n,n)={(R,n,n)}. Got {tuple(Sigma.shape)}")
        return Sigma
    raise ShapeError(f"Sigma must be 2D or 3D. Got {Sigma.ndim}D")


def _coerce_chol(chol: torch.Tensor, *, R: int, n: int) -> torch.Tensor:
    if chol.ndim == 2:
        if chol.shape != (n, n):
            raise ShapeError(f"chol_Sigma must be (n,n)={(n,n)} or (R,n,n)={(R,n,n)}. Got {tuple(chol.shape)}")
        return chol
    if chol.ndim == 3:
        if chol.shape != (R, n, n):
            raise ShapeError(f"chol_Sigma must be (R,n,n)={(R,n,n)}. Got {tuple(chol.shape)}")
        return chol
    raise ShapeError(f"chol_Sigma must be 2D or 3D. Got {chol.ndim}D")


def _validate_chol_spd(chol: torch.Tensor) -> None:
    d = torch.diagonal(chol, dim1=-2, dim2=-1)
    if not torch.isfinite(d).all() or (d <= 0).any():
        raise NotSPDMatrixError("chol_Sigma must have strictly positive diagonal entries (SPD requirement).")


def cholesky_spd(
    Sigma: torch.Tensor,
    *,
    upper: bool,
    check_spd: bool,
    jitter: float,
    max_tries: int,
) -> torch.Tensor:
    """Batched Cholesky with optional jitter escalation and explicit SPD error."""
    if Sigma.ndim < 2 or Sigma.shape[-1] != Sigma.shape[-2]:
        raise ShapeError(f"Sigma must be (...,n,n). Got {tuple(Sigma.shape)}")

    S = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    n = S.shape[-1]
    eye = torch.eye(n, device=S.device, dtype=S.dtype)

    max_tries = max(1, int(max_tries))

    for t in range(max_tries):
        Sj = S
        if jitter > 0.0:
            Sj = S + (float(jitter) * (10.0 ** t)) * eye

        C, info = torch.linalg.cholesky_ex(Sj, upper=upper)

        # We require factorization success in all cases.
        if torch.any(info != 0):
            continue

        return C

    raise NotSPDMatrixError("Sigma must be symmetric positive definite (SPD). Cholesky factorization failed.")


def _whiten_full_Xy(
    X: torch.Tensor,
    y: torch.Tensor,
    chol: torch.Tensor,
    *,
    chol_upper: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-whiten (X,y) given Cholesky factor of Sigma."""
    if not chol_upper:
        # Sigma = L L' -> whiten with L^{-1}
        Xs = torch.linalg.solve_triangular(chol, X, upper=False, left=True)
        ys = torch.linalg.solve_triangular(chol, y.unsqueeze(-1), upper=False, left=True).squeeze(-1)
        return Xs, ys

    # Sigma = U' U -> whiten with U^{-T}: solve U' z = rhs
    UT = chol.transpose(-1, -2)  # lower
    Xs = torch.linalg.solve_triangular(UT, X, upper=False, left=True)
    ys = torch.linalg.solve_triangular(UT, y.unsqueeze(-1), upper=False, left=True).squeeze(-1)
    return Xs, ys


def whiten_vec(v: torch.Tensor, info: WhiteningInfo) -> torch.Tensor:
    """Whiten a vector batch using WhiteningInfo."""
    if info.diag_branch:
        if info.sqrt_w is None:
            raise RuntimeError("Missing sqrt_w in WhiteningInfo for diagonal whitening.")
        return v * info.sqrt_w

    if info.chol is None:
        raise RuntimeError("Missing chol in WhiteningInfo for full whitening.")

    chol = info.chol
    if not info.chol_upper:
        return torch.linalg.solve_triangular(chol, v.unsqueeze(-1), upper=False, left=True).squeeze(-1)

    UT = chol.transpose(-1, -2)
    return torch.linalg.solve_triangular(UT, v.unsqueeze(-1), upper=False, left=True).squeeze(-1)


def sigma_inv_times_X(X: torch.Tensor, X_star: torch.Tensor, info: WhiteningInfo) -> torch.Tensor:
    """Compute Sigma^{-1}X without forming Sigma^{-1}."""
    if info.diag_branch:
        if info.w_prec is None:
            raise RuntimeError("Missing w_prec in WhiteningInfo for diagonal Sigma^{-1}X.")
        return X * info.w_prec.unsqueeze(-1)

    if info.chol is None:
        raise RuntimeError("Missing chol in WhiteningInfo for full Sigma^{-1}X.")
    chol = info.chol

    # - lower chol: Sigma=L L' and X_star=L^{-1}X -> Sigma^{-1}X = L^{-T} X_star
    # - upper chol: Sigma=U' U and X_star=U^{-T}X -> Sigma^{-1}X = U^{-1} X_star
    if not info.chol_upper:
        LT = chol.transpose(-1, -2)
        return torch.linalg.solve_triangular(LT, X_star, upper=True, left=True)

    return torch.linalg.solve_triangular(chol, X_star, upper=True, left=True)


def whiten_system(
    X: torch.Tensor,
    y: torch.Tensor,
    spec: SigmaSpec,
    *,
    check_spd: bool,
    chol_upper: bool,
    jitter: float,
    chol_max_tries: int,
) -> tuple[torch.Tensor, torch.Tensor, WhiteningInfo]:
    """Whiten (X,y) based on SigmaSpec.

    Returns
    -------
    X_star, y_star, info
      - X_star, y_star are the whitened system used for OLS.
      - info carries objects needed for inference and diagnostics.
    """
    R, n, _ = X.shape

    if spec.kind == "diag":
        # Prefer inv_sqrt if provided
        if spec.inv_sqrt is not None:
            inv_s = as_torch(spec.inv_sqrt, dtype=X.dtype, device=X.device)
            if inv_s.ndim == 0:
                inv_s = inv_s.view(1, 1).expand(R, n)
            elif inv_s.ndim == 1:
                if int(inv_s.shape[0]) != n:
                    raise ShapeError(f"inv_sqrt has shape {tuple(inv_s.shape)} but n={n}")
                inv_s = inv_s.view(1, n).expand(R, n)
            elif inv_s.ndim == 2:
                if tuple(inv_s.shape) != (R, n):
                    raise ShapeError(f"inv_sqrt must be (R,n)={(R,n)}. Got {tuple(inv_s.shape)}")
            else:
                raise ShapeError(f"inv_sqrt must be scalar, (n,), or (R,n). Got {tuple(inv_s.shape)}")

            if check_spd:
                if not torch.isfinite(inv_s).all():
                    raise InvalidWeightsError("inv_sqrt contains inf/nan")
                if (inv_s <= 0).any():
                    raise InvalidWeightsError("inv_sqrt must be strictly positive")

            sqrt_w = inv_s
            w_prec = inv_s * inv_s
            Sigma_diag = 1.0 / w_prec
        else:
            if spec.diag is None:
                raise ShapeError("Diagonal SigmaSpec requires 'diag' variances or 'inv_sqrt'.")
            w_prec, sqrt_w = as_batched_weights(
                spec.diag,
                R=R,
                n=n,
                mode="variance",
                dtype=X.dtype,
                device=X.device,
                check=check_spd,
            )
            Sigma_diag = 1.0 / w_prec

        X_star = X * sqrt_w.unsqueeze(-1)
        y_star = y * sqrt_w
        logdet = torch.sum(torch.log(Sigma_diag), dim=1)

        info = WhiteningInfo(
            diag_branch=True,
            n=n,
            R=R,
            sqrt_w=sqrt_w,
            w_prec=w_prec,
            Sigma_diag=Sigma_diag,
            chol=None,
            chol_upper=False,
            logdet=logdet,
        )
        return X_star, y_star, info

    # Full SPD matrix or its Cholesky
    if spec.kind == "chol":
        if spec.chol is None:
            raise ShapeError("Cholesky SigmaSpec requires 'chol'.")
        chol = _coerce_chol(as_torch(spec.chol, dtype=X.dtype, device=X.device), R=R, n=n)
        if check_spd:
            _validate_chol_spd(chol)
        chol_upper_eff = bool(spec.chol_upper)
    elif spec.kind == "full":
        if spec.full is None:
            raise ShapeError("Full SigmaSpec requires 'full'.")
        Sigma_full = _coerce_full_sigma(as_torch(spec.full, dtype=X.dtype, device=X.device), R=R, n=n)
        chol_upper_eff = bool(chol_upper)
        chol = cholesky_spd(
            Sigma_full,
            upper=chol_upper_eff,
            check_spd=check_spd,
            jitter=float(jitter),
            max_tries=int(chol_max_tries),
        )
    else:
        raise ShapeError(f"Unknown SigmaSpec.kind='{spec.kind}'.")

    X_star, y_star = _whiten_full_Xy(X, y, chol, chol_upper=chol_upper_eff)

    d = torch.diagonal(chol, dim1=-2, dim2=-1)
    if d.ndim == 1:
        logdet = (2.0 * torch.sum(torch.log(d))) * torch.ones(R, device=X.device, dtype=X.dtype)
    else:
        logdet = 2.0 * torch.sum(torch.log(d), dim=1)

    info = WhiteningInfo(
        diag_branch=False,
        n=n,
        R=R,
        sqrt_w=None,
        w_prec=None,
        Sigma_diag=None,
        chol=chol,
        chol_upper=chol_upper_eff,
        logdet=logdet,
    )
    return X_star, y_star, info
