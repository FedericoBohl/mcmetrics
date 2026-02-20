# src/mcmetrics/models/gls.py

from __future__ import annotations

from typing import Optional, Literal

import dataclasses
import math

import torch

from mcmetrics.typing import as_batched_xy, as_torch
from mcmetrics.models.ols import OLS

__all__ = ["GLS"]

# Robust vcov options implemented at GLS layer (sandwich in original space)
_VCOV_GLS_ROBUST = {"hc0", "hc1"}


# =============================================================================
# Helpers: keep compatibility with frozen dataclass results (OLSResults)
# =============================================================================

def _dataclass_fieldnames(obj) -> set[str]:
    if dataclasses.is_dataclass(obj):
        return {f.name for f in dataclasses.fields(obj)}
    return set()


def _has_attr(obj, name: str) -> bool:
    try:
        getattr(obj, name)
        return True
    except Exception:
        return False


def _safe_attach_attr(obj, name: str, value) -> bool:
    """
    Attach attribute bypassing frozen dataclass restriction.
    Returns True if it worked, False otherwise (e.g., slots without __dict__).
    """
    try:
        object.__setattr__(obj, name, value)
        return True
    except Exception:
        return False


def _safe_replace(res, **kwargs):
    """
    Update a (possibly frozen) dataclass result object.
    - Only replaces existing dataclass fields.
    - Preserves dynamically-attached `metadata` dict across dataclasses.replace().
    """
    if not dataclasses.is_dataclass(res):
        # non-dataclass fallback
        for k, v in kwargs.items():
            try:
                setattr(res, k, v)
            except Exception:
                pass
        return res

    names = _dataclass_fieldnames(res)
    payload = {k: v for k, v in kwargs.items() if k in names}
    if not payload:
        return res

    # Preserve dynamic metadata if metadata is not a declared field
    md_dyn = None
    if "metadata" not in names and _has_attr(res, "metadata"):
        md = getattr(res, "metadata", None)
        if isinstance(md, dict):
            md_dyn = md

    res2 = dataclasses.replace(res, **payload)

    if md_dyn is not None:
        _safe_attach_attr(res2, "metadata", md_dyn)

    return res2


def _ensure_metadata(res) -> tuple[object, Optional[dict]]:
    """
    Guarantee that `res.metadata` exists and is a dict.

    Strategy:
    1) If res.metadata already exists and is dict, use it.
    2) If 'metadata' is a declared dataclass field, set it via dataclasses.replace.
    3) Otherwise, attach a dynamic attribute 'metadata' via object.__setattr__.

    Returns (res, metadata_dict_or_None). If metadata can't be attached (e.g., slots), returns None.
    """
    md = getattr(res, "metadata", None)
    if isinstance(md, dict):
        return res, md

    if dataclasses.is_dataclass(res) and "metadata" in _dataclass_fieldnames(res):
        res = _safe_replace(res, metadata={})
        md = getattr(res, "metadata", None)
        return res, (md if isinstance(md, dict) else None)

    # Dynamic attach (works for normal dataclasses without slots)
    ok = _safe_attach_attr(res, "metadata", {})
    if not ok:
        return res, None
    return res, getattr(res, "metadata", None)


def _put_in_metadata(res, key: str, value) -> object:
    res, md = _ensure_metadata(res)
    if md is not None:
        md[key] = value
    return res


# =============================================================================
# Sigma parsing / whitening
# =============================================================================

def _as_batched_sigma_full(Sigma: torch.Tensor, R: int, n: int) -> torch.Tensor:
    """
    Accept Sigma as (n,n) or (R,n,n). Return tensor suitable for broadcasting
    against (R,n,*) right-hand sides.
    """
    if Sigma.ndim == 2:
        if Sigma.shape != (n, n):
            raise ValueError(
                f"Sigma must have shape (n,n)={(n,n)} or (R,n,n)={(R,n,n)}. Got {tuple(Sigma.shape)}."
            )
        return Sigma
    if Sigma.ndim == 3:
        if Sigma.shape != (R, n, n):
            raise ValueError(f"Sigma must have shape (R,n,n)={(R,n,n)}. Got {tuple(Sigma.shape)}.")
        return Sigma
    raise ValueError(f"Sigma (full) must be 2D or 3D. Got {Sigma.ndim}D.")


def _as_batched_sigma_diag(v: torch.Tensor, R: int, n: int) -> torch.Tensor:
    """
    Accept diagonal Sigma representation as:
      - (n,) or (R,n)
    Return tensor suitable for broadcasting against (R,n,*) arrays.
    """
    if v.ndim == 1:
        if v.shape[0] != n:
            raise ValueError(f"Sigma diagonal must have length n={n}. Got {tuple(v.shape)}.")
        return v
    if v.ndim == 2:
        if v.shape != (R, n):
            raise ValueError(f"Sigma diagonal must have shape (R,n)={(R,n)}. Got {tuple(v.shape)}.")
        return v
    raise ValueError(f"Sigma diagonal must be 1D or 2D. Got {v.ndim}D.")


def _chol_spd(
    Sigma: torch.Tensor,
    *,
    upper: bool,
    check_spd: bool,
    jitter: float,
) -> torch.Tensor:
    """
    Compute triangular factor from SPD matrix Sigma.

    Convention:
      - if upper=False: returns L such that Sigma = L @ L.T
      - if upper=True : returns U such that Sigma = U.T @ U
    """
    if jitter and jitter > 0.0:
        n = Sigma.shape[-1]
        eye = torch.eye(n, dtype=Sigma.dtype, device=Sigma.device)
        Sigma = Sigma + jitter * eye

    C, info = torch.linalg.cholesky_ex(Sigma, upper=upper)
    if check_spd and torch.any(info != 0):
        raise ValueError("Sigma must be symmetric positive definite (SPD). Cholesky factorization failed.")
    return C


def _validate_chol(C: torch.Tensor) -> None:
    """
    Lightweight validation when user supplies chol_Sigma:
    require strictly positive diagonal entries.
    """
    d = torch.diagonal(C, dim1=-2, dim2=-1)
    if torch.any(d <= 0):
        raise ValueError("chol_Sigma must have strictly positive diagonal entries (SPD requirement).")


def _diag_inv_sqrt_from_sigma(Sigma_diag: torch.Tensor, *, check_spd: bool) -> torch.Tensor:
    if check_spd and torch.any(Sigma_diag <= 0):
        raise ValueError("Sigma diagonal entries must be strictly positive (SPD requirement).")
    return torch.rsqrt(Sigma_diag)


def _diag_whiten_Xy(X: torch.Tensor, y: torch.Tensor, inv_sqrt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # inv_sqrt: (n,) or (R,n)
    X_star = X * inv_sqrt.unsqueeze(-1)
    y_star = y * inv_sqrt
    return X_star, y_star


def _diag_whiten_vec(v: torch.Tensor, inv_sqrt: torch.Tensor) -> torch.Tensor:
    return v * inv_sqrt


def _full_whiten_Xy(
    X: torch.Tensor,
    y: torch.Tensor,
    C: torch.Tensor,
    *,
    chol_upper: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Whitening using triangular solves.

    If chol_upper=False: C is L (lower), Sigma = L L^T, whiten via L^{-1}.
    If chol_upper=True : C is U (upper), Sigma = U^T U, whiten via U^{-T}.
    """
    if not chol_upper:
        X_star = torch.linalg.solve_triangular(C, X, upper=False, left=True)
        y_star = torch.linalg.solve_triangular(C, y.unsqueeze(-1), upper=False, left=True).squeeze(-1)
        return X_star, y_star

    # chol_upper=True -> whiten with U^{-T}: solve U^T * z = rhs
    UT = C.transpose(-1, -2)  # lower
    X_star = torch.linalg.solve_triangular(UT, X, upper=False, left=True)
    y_star = torch.linalg.solve_triangular(UT, y.unsqueeze(-1), upper=False, left=True).squeeze(-1)
    return X_star, y_star


def _full_whiten_vec(v: torch.Tensor, C: torch.Tensor, *, chol_upper: bool) -> torch.Tensor:
    if not chol_upper:
        return torch.linalg.solve_triangular(C, v.unsqueeze(-1), upper=False, left=True).squeeze(-1)

    UT = C.transpose(-1, -2)  # lower
    return torch.linalg.solve_triangular(UT, v.unsqueeze(-1), upper=False, left=True).squeeze(-1)


def _sigma_inv_times_X(X_star: torch.Tensor, C: torch.Tensor, *, chol_upper: bool) -> torch.Tensor:
    """
    Compute Sigma^{-1} X without forming Sigma^{-1}.

    If chol_upper=False: Sigma = L L^T, X_star = L^{-1} X, so Sigma^{-1}X = L^{-T} X_star.
    If chol_upper=True : Sigma = U^T U, X_star = U^{-T} X, so Sigma^{-1}X = U^{-1} X_star.
    """
    if not chol_upper:
        LT = C.transpose(-1, -2)  # upper
        return torch.linalg.solve_triangular(LT, X_star, upper=True, left=True)
    return torch.linalg.solve_triangular(C, X_star, upper=True, left=True)


def _batched_inv(A: torch.Tensor) -> torch.Tensor:
    """
    Return A^{-1} using solve against identity (stable).
    A: (R,k,k) or (k,k)
    """
    k = A.shape[-1]
    I = torch.eye(k, dtype=A.dtype, device=A.device)
    return torch.linalg.solve(A, I)


def _gls_vcov_sandwich_hc(
    X: torch.Tensor,
    resid_raw: torch.Tensor,
    *,
    sigma_inv_X: torch.Tensor,
    kind: Literal["HC0", "HC1"],
) -> torch.Tensor:
    """
    GLS-robust (sandwich) variance estimator:

      V = (X' S^{-1} X)^{-1} [ X' S^{-1} diag(e^2) S^{-1} X ] (X' S^{-1} X)^{-1}

    where S is the user-provided Sigma used in the GLS estimator, and e is residual in original space.
    """
    XtSiX = torch.einsum("rnk,rnj->rkj", X, sigma_inv_X)
    bread = _batched_inv(XtSiX)

    e2 = resid_raw ** 2  # (R,n)
    meat = torch.einsum("rn,rnk,rnj->rkj", e2, sigma_inv_X, sigma_inv_X)

    if kind.upper() == "HC1":
        R, n, k = X.shape
        if n <= k:
            raise ValueError("HC1 scaling requires n > k.")
        meat = meat * (float(n) / float(n - k))

    return bread @ meat @ bread


# =============================================================================
# GLS main
# =============================================================================

def GLS(
    X,
    y,
    *,
    Sigma=None,
    chol_Sigma=None,
    # Precomputed diagonal whitener: inv_sqrt(Sigma_diag) with shape (n,) or (R,n).
    inv_sqrt_Sigma: Optional[torch.Tensor] = None,
    # If ambiguous, user can force diagonal vs full.
    Sigma_is_diagonal: Optional[bool] = None,
    # Cholesky convention for full Sigma
    chol_upper: bool = False,
    check_spd: bool = True,
    jitter: float = 0.0,
    validate_chol: bool = True,
    # Inference
    vcov: str = "classic",
    solve_method: str = "cholesky",
    # Storage
    store_fitted: bool = False,
    store_resid: bool = False,
    store_y: bool = True,
    store_diagnostics: bool = False,
    beta_true=None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    """
    GLS with known Sigma via pre-whitening.

    Sigma representations
    ---------------------
    (1) Diagonal (heteroskedasticity only):
        - Sigma: (n,) or (R,n) with diagonal entries (variances) s_i^2
        - or inv_sqrt_Sigma: (n,) or (R,n) with 1/sqrt(s_i^2)

        Whitening:
          X* = inv_sqrt ⊙ X
          y* = inv_sqrt ⊙ y

        Note: Diagonal GLS is equivalent to WLS with weights w_i = 1/s_i^2.

    (2) Full (general SPD):
        - Sigma: (n,n) or (R,n,n)
        - or chol_Sigma: (n,n) or (R,n,n)

        Whitening uses triangular solves based on Cholesky factor.

    Inference
    ---------
    - vcov="classic": keep the OLS vcov computed on (X*, y*) (assumes correct Sigma).
    - vcov in {"HC0","HC1"}: GLS-robust sandwich in ORIGINAL space.

    Storage
    -------
    - store_resid stores whitened residual in `res.resid` if possible, and raw residual in:
        * `res.resid_raw` if possible, else res.metadata["resid_raw"].
    - store_diagnostics stores diagnostics dict in res.metadata["diagnostics"].
    """

    X_t, y_t, _ = as_batched_xy(X, y, dtype=dtype, device=device)
    R, n, k = X_t.shape

    if (Sigma is None) and (chol_Sigma is None) and (inv_sqrt_Sigma is None):
        raise ValueError("Provide Sigma, chol_Sigma, or inv_sqrt_Sigma (diagonal whitener).")

    # Decide diagonal vs full branch
    if Sigma_is_diagonal is not None:
        diag_branch = bool(Sigma_is_diagonal)
    else:
        if inv_sqrt_Sigma is not None:
            diag_branch = True
        elif chol_Sigma is not None:
            diag_branch = False
        else:
            Sigma0 = as_torch(Sigma, dtype=X_t.dtype, device=X_t.device)
            if Sigma0.ndim == 1:
                diag_branch = True
            elif Sigma0.ndim == 2:
                # (n,n) => full; otherwise interpret as (R,n) diagonal representation
                diag_branch = (Sigma0.shape != (n, n))
            elif Sigma0.ndim == 3:
                diag_branch = False
            else:
                raise ValueError(f"Sigma must be 1D/2D/3D. Got {Sigma0.ndim}D.")

    diag_inv_sqrt = None
    C = None

    # Build whiteners and transform
    if diag_branch:
        if inv_sqrt_Sigma is not None:
            inv_s = as_torch(inv_sqrt_Sigma, dtype=X_t.dtype, device=X_t.device)
            inv_s = _as_batched_sigma_diag(inv_s, R=R, n=n)
        else:
            Sigma_diag = as_torch(Sigma, dtype=X_t.dtype, device=X_t.device)
            Sigma_diag = _as_batched_sigma_diag(Sigma_diag, R=R, n=n)
            inv_s = _diag_inv_sqrt_from_sigma(Sigma_diag, check_spd=check_spd)

        diag_inv_sqrt = inv_s
        X_star, y_star = _diag_whiten_Xy(X_t, y_t, inv_s)

    else:
        if chol_Sigma is not None:
            C0 = as_torch(chol_Sigma, dtype=X_t.dtype, device=X_t.device)
            C0 = _as_batched_sigma_full(C0, R=R, n=n)
            if validate_chol:
                _validate_chol(C0)
            C = C0
        else:
            Sigma_full = as_torch(Sigma, dtype=X_t.dtype, device=X_t.device)
            Sigma_full = _as_batched_sigma_full(Sigma_full, R=R, n=n)
            C = _chol_spd(Sigma_full, upper=chol_upper, check_spd=check_spd, jitter=jitter)

        X_star, y_star = _full_whiten_Xy(X_t, y_t, C, chol_upper=chol_upper)

    # Run OLS on transformed data (keep it memory-light)
    # Always compute classic vcov on (X*,y*). Robust GLS vcov (HC0/HC1) overwrites below.
    res = OLS(
        X_star,
        y_star,
        vcov="classic",
        solve_method=solve_method,
        store_fitted=False,
        store_resid=False,
        store_y=False,
        beta_true=beta_true,
        dtype=X_t.dtype,
        device=X_t.device,
    )

    # Ensure metadata exists early so examples can always do res.metadata.get(...)
    res, _ = _ensure_metadata(res)

    # Original-space fitted and residual
    beta = res.params  # (R,k)
    yhat = torch.einsum("rnk,rk->rn", X_t, beta)
    resid_raw = y_t - yhat  # (R,n)

    # Whitening of residual (for diagnostics / store_resid)
    if diag_branch:
        resid_white = _diag_whiten_vec(resid_raw, diag_inv_sqrt)
    else:
        resid_white = _full_whiten_vec(resid_raw, C, chol_upper=chol_upper)

    # Inference override if robust requested
    vcov_req = (vcov or "classic").lower()
    if vcov_req in _VCOV_GLS_ROBUST:
        if diag_branch:
            # Sigma^{-1}X = X * inv_s^2
            sigma_inv_X = X_t * (diag_inv_sqrt ** 2).unsqueeze(-1)
        else:
            # Sigma^{-1}X without forming Sigma^{-1}
            sigma_inv_X = _sigma_inv_times_X(X_star, C, chol_upper=chol_upper)

        kind = "HC1" if vcov_req == "hc1" else "HC0"
        new_vcov = _gls_vcov_sandwich_hc(X_t, resid_raw, sigma_inv_X=sigma_inv_X, kind=kind)
        res = _safe_replace(res, vcov=new_vcov)

        # re-attach metadata if replace dropped dynamic attrs
        res, _ = _ensure_metadata(res)
        res = _put_in_metadata(res, "vcov", kind)
        res = _put_in_metadata(res, "vcov_mode", "gls_sandwich")
    else:
        res = _put_in_metadata(res, "vcov", "classic")
        res = _put_in_metadata(res, "vcov_mode", "whitened_ols")

    # Attach GLS metadata
    res = _put_in_metadata(
        res,
        "gls",
        {
            "diag_branch": bool(diag_branch),
            "chol_upper": bool(chol_upper),
            "Sigma_provided": Sigma is not None,
            "chol_Sigma_provided": chol_Sigma is not None,
            "inv_sqrt_Sigma_provided": inv_sqrt_Sigma is not None,
            "jitter": float(jitter),
        },
    )

    # Storage: prefer dataclass fields if they exist; otherwise store in metadata.
    fields = _dataclass_fieldnames(res)

    if store_y:
        if "y" in fields:
            res = _safe_replace(res, y=y_t)
            res, _ = _ensure_metadata(res)
        else:
            res = _put_in_metadata(res, "y", y_t)

    if store_fitted:
        if "fitted" in fields:
            res = _safe_replace(res, fitted=yhat)
            res, _ = _ensure_metadata(res)
        else:
            res = _put_in_metadata(res, "fitted", yhat)

    if store_resid:
        if "resid" in fields:
            res = _safe_replace(res, resid=resid_white)
            res, _ = _ensure_metadata(res)
        else:
            res = _put_in_metadata(res, "resid", resid_white)

        if "resid_raw" in fields:
            res = _safe_replace(res, resid_raw=resid_raw)
            res, _ = _ensure_metadata(res)
        else:
            res = _put_in_metadata(res, "resid_raw", resid_raw)

    # Diagnostics: store in metadata["diagnostics"]
    if store_diagnostics:
        objective = torch.sum(resid_white ** 2, dim=1)  # (R,)

        # logdet(Sigma)
        if diag_branch:
            if inv_sqrt_Sigma is not None:
                Sigma_diag_eff = 1.0 / (diag_inv_sqrt ** 2)
            else:
                Sigma_diag_eff = as_torch(Sigma, dtype=X_t.dtype, device=X_t.device)
                Sigma_diag_eff = _as_batched_sigma_diag(Sigma_diag_eff, R=R, n=n)

            if Sigma_diag_eff.ndim == 1:
                logdet = torch.sum(torch.log(Sigma_diag_eff)) * torch.ones(R, dtype=X_t.dtype, device=X_t.device)
            else:
                logdet = torch.sum(torch.log(Sigma_diag_eff), dim=1)
        else:
            d = torch.diagonal(C, dim1=-2, dim2=-1)
            if d.ndim == 1:
                logdet = 2.0 * torch.sum(torch.log(d)) * torch.ones(R, dtype=X_t.dtype, device=X_t.device)
            else:
                logdet = 2.0 * torch.sum(torch.log(d), dim=1)

        loglik = -0.5 * (float(n) * math.log(2.0 * math.pi) + logdet + objective)

        res = _put_in_metadata(
            res,
            "diagnostics",
            {
                "objective": objective,
                "logdet_Sigma": logdet,
                "loglik_gaussian": loglik,
            },
        )

    return res