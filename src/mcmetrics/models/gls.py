from __future__ import annotations

from typing import Literal, Optional, Union

import math

import torch

from mcmetrics.exceptions import ShapeError
from mcmetrics.linalg import solve_ls
from mcmetrics.results import OLSResults
from mcmetrics.sigma import SigmaSpec, sigma_from_inputs
from mcmetrics.sigma.whiten import sigma_inv_times_X, whiten_system, whiten_vec
from mcmetrics.typing import as_batched_xy
from mcmetrics.vcov.classic import vcov_classic

SolveMethod = Literal["solve", "lstsq", "cholesky", "qr"]
VcovType = Literal["classic", "HC0", "HC1"]


def GLS(
    X,
    y,
    *,
    # Preferred: pass a SigmaSpec directly
    sigma_spec: Optional[SigmaSpec] = None,
    # Backward-compatible inputs
    Sigma=None,
    chol_Sigma=None,
    inv_sqrt_Sigma=None,
    Sigma_is_diagonal: Optional[bool] = None,
    chol_upper: bool = False,
    # Numerical controls
    check_spd: bool = True,
    chol_jitter: float = 0.0,
    chol_max_tries: int = 1,
    # Estimation controls
    has_const: bool = True,
    solve_method: SolveMethod = "solve",
    vcov: VcovType = "classic",
    use_t: bool | None = None,
    beta_true: Optional[torch.Tensor] = None,
    # Storage
    store_y: bool = True,
    store_fitted: bool = False,
    store_resid: bool = False,
    store_diagnostics: bool = False,
    # Device
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> OLSResults:
    """GLS with known Sigma, implemented via pre-whitening.

    Shapes
    ------
    - X: (R,n,k) or (n,k)
    - y: (R,n) or (n,)
    - Sigma:
        * diagonal (variances): (n,) or (R,n)
        * full SPD matrix:      (n,n) or (R,n,n)

    Notes
    -----
    - For diagonal Sigma, GLS is equivalent to WLS with weights = 1 / Sigma_diag.
    - For full Sigma, the Cholesky factor is computed with safe_cholesky.
    - `resid` (if stored) is the working (whitened) residual; `resid_raw` is y - Xb.
    """

    X_t, y_t, param_names = as_batched_xy(X, y, dtype=dtype, device=device)
    R, n, k = X_t.shape

    df_resid = n - k
    if df_resid <= 0:
        raise ShapeError(f"Need n > k for GLS. Got n={n}, k={k} (df_resid={df_resid}).")

    if sigma_spec is None:
        sigma_spec = sigma_from_inputs(
            Sigma=Sigma,
            chol_Sigma=chol_Sigma,
            inv_sqrt_Sigma=inv_sqrt_Sigma,
            Sigma_is_diagonal=Sigma_is_diagonal,
            dtype=X_t.dtype,
            device=X_t.device,
            chol_upper=chol_upper,
        )

    # Pre-whiten
    X_star, y_star, state = whiten_system(
        X_t,
        y_t,
        sigma_spec,
        check_spd=check_spd,
        chol_jitter=chol_jitter,
        chol_max_tries=chol_max_tries,
    )

    # Solve OLS on transformed system
    beta, _, XtX_inv = solve_ls(
        X_star,
        y_star,
        solve_method=solve_method,
        chol_jitter=chol_jitter,
        chol_max_tries=chol_max_tries,
    )

    # Fitted and residuals in original space
    fitted = (X_t @ beta.unsqueeze(-1)).squeeze(-1)
    resid_raw = y_t - fitted
    resid_work = whiten_vec(resid_raw, state)

    ssr = (resid_work * resid_work).sum(dim=1)
    sigma2 = ssr / float(df_resid)

    vcov_key = vcov.upper() if vcov != "classic" else "CLASSIC"
    if vcov_key == "CLASSIC":
        cov_type = "nonrobust"
        vcov_mat = vcov_classic(XtX_inv, sigma2)
        default_use_t = True
    elif vcov_key == "HC0":
        cov_type = "HC0"
        sigma_inv_X = sigma_inv_times_X(X_star, state)
        e2 = resid_raw * resid_raw
        meat = torch.einsum("rn,rnk,rnj->rkj", e2, sigma_inv_X, sigma_inv_X)
        vcov_mat = XtX_inv @ meat @ XtX_inv
        default_use_t = False
    elif vcov_key == "HC1":
        cov_type = "HC1"
        sigma_inv_X = sigma_inv_times_X(X_star, state)
        e2 = resid_raw * resid_raw
        meat = torch.einsum("rn,rnk,rnj->rkj", e2, sigma_inv_X, sigma_inv_X)
        vcov_mat = XtX_inv @ meat @ XtX_inv
        vcov_mat = vcov_mat * (float(n) / float(df_resid))
        default_use_t = False
    else:
        raise ShapeError("Unknown vcov. Use 'classic','HC0','HC1'.")

    if use_t is None:
        use_t = default_use_t

    df_model = (k - 1) if has_const else k
    if df_model <= 0:
        df_model = k

    extras: dict[str, object] = {
        "solve_method": solve_method,
        "vcov": vcov,
        "sigma_kind": sigma_spec.kind,
        "chol_upper": bool(getattr(sigma_spec, "chol_upper", False)),
        "chol_jitter": float(chol_jitter),
        "chol_max_tries": int(chol_max_tries),
    }

    metadata: dict[str, object] = {
        "sigma": {
            "kind": sigma_spec.kind,
            "chol_upper": bool(getattr(sigma_spec, "chol_upper", False)),
        },
        "whitening": {
            "kind": state.kind,
            "chol_upper": bool(state.chol_upper),
        },
    }

    diagnostics: dict[str, torch.Tensor] = {}
    if store_diagnostics:
        logdet = state.logdet
        if logdet is None:
            logdet = torch.zeros(R, dtype=X_t.dtype, device=X_t.device)
        ll = -0.5 * (float(n) * math.log(2.0 * math.pi) + logdet + ssr)
        diagnostics = {
            "objective": ssr,
            "logdet_Sigma": logdet,
            "loglik_gaussian": ll,
        }

    return OLSResults(
        params=beta,
        vcov=vcov_mat,
        sigma2=sigma2,
        ssr=ssr,
        _nobs=n,
        df_resid=df_resid,
        df_model=df_model,
        has_const=has_const,
        fitted=fitted if store_fitted else None,
        resid=resid_work if store_resid else None,
        y=y_t if store_y else None,
        resid_raw=resid_raw if store_resid else None,
        metadata=metadata,
        diagnostics=diagnostics,
        cov_type=cov_type,
        backend="torch",
        use_t=bool(use_t),
        beta_true=beta_true,
        param_names=param_names,
        model_name="GLS",
        method_name="Generalized Least Squares",
        extras=extras,
    )
