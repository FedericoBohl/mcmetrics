from __future__ import annotations

from typing import Literal, Optional, Union

import math

import torch

from mcmetrics.exceptions import ShapeError
from mcmetrics.linalg import solve_ls
from mcmetrics.results import OLSResults
from mcmetrics.typing import as_batched_xy, as_torch
from mcmetrics.vcov.classic import vcov_classic
from mcmetrics.vcov.gls import vcov_gls_hc0, vcov_gls_hc1
from mcmetrics.sigma.spec import SigmaSpec, coerce_sigma_spec
from mcmetrics.sigma.whiten import WhiteningInfo, sigma_inv_times_X, whiten_system, whiten_vec

SolveMethod = Literal["solve", "lstsq", "cholesky", "qr"]
VcovType = Literal["classic", "HC0", "HC1"]

__all__ = ["GLS"]


def GLS(
    X,
    y,
    *,
    sigma_spec: Optional[SigmaSpec] = None,
    Sigma=None,
    chol_Sigma=None,
    inv_sqrt_Sigma: Optional[torch.Tensor] = None,
    Sigma_is_diagonal: Optional[bool] = None,
    has_const: bool = True,
    check_spd: bool = True,
    chol_upper: bool = False,
    jitter: float = 0.0,
    chol_max_tries: int = 1,
    solve_method: SolveMethod = "solve",
    vcov: VcovType = "classic",
    use_t: bool | None = None,
    beta_true: Optional[torch.Tensor] = None,
    store_y: bool = True,
    store_fitted: bool = False,
    store_resid: bool = False,
    store_diagnostics: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> OLSResults:
    """
    Generalized Least Squares with known Sigma.

    Supported Sigma representations
    -------------------------------
    1) Diagonal Sigma (heteroskedasticity only)
        - Sigma: scalar, (n,), or (R,n) representing variances
        - inv_sqrt_Sigma: scalar, (n,), or (R,n) representing 1/sqrt(variances)

       This path is equivalent to WLS with precision weights w_i = 1/Sigma_ii.

    2) Full Sigma (general SPD)
        - Sigma: (n,n) or (R,n,n)
        - chol_Sigma: (n,n) or (R,n,n)

    Inference
    ---------
    - vcov='classic': (X' Sigma^{-1} X)^{-1} * sigma2, where sigma2 = (e' Sigma^{-1} e) / (n-k)
    - vcov in {'HC0','HC1'}: GLS-robust sandwich using original residuals.
    """

    X, y, param_names = as_batched_xy(X, y, dtype=dtype, device=device)
    R, n, k = X.shape

    df_resid = n - k
    if df_resid <= 0:
        raise ShapeError(f"Need n > k for GLS. Got n={n}, k={k} (df_resid={df_resid}).")

    # SigmaSpec coercion (shared with future FGLS)
    if sigma_spec is None:
        sigma_spec = coerce_sigma_spec(
            Sigma,
            chol_Sigma=chol_Sigma,
            inv_sqrt_Sigma=inv_sqrt_Sigma,
            Sigma_is_diagonal=Sigma_is_diagonal,
            chol_upper=chol_upper,
            method="known",
        )

    X_star, y_star, info = whiten_system(
        X,
        y,
        sigma_spec,
        check_spd=check_spd,
        chol_upper=chol_upper,
        jitter=float(jitter),
        chol_max_tries=int(chol_max_tries),
    )

    diagnostics: dict[str, object] = {}
    metadata: dict[str, object] = {
        "model": "GLS",
        "sigma_spec": sigma_spec.to_dict(),
        "diag_branch": bool(info.diag_branch),
        "chol_upper": bool(info.chol_upper),
        "check_spd": bool(check_spd),
        "jitter": float(jitter),
        "chol_max_tries": int(chol_max_tries),
    }

    # ---------------------------------------------------------------------
    # Estimation on whitened system
    # ---------------------------------------------------------------------
    beta, XtX, XtX_inv = solve_ls(X_star, y_star, solve_method=solve_method)
    fitted = (X @ beta.unsqueeze(-1)).squeeze(-1)
    resid_raw = y - fitted

    # GLS objective SSR in whitening metric: e' Sigma^{-1} e = ||e*||^2
    resid_w = whiten_vec(resid_raw, info)

    ssr = (resid_w * resid_w).sum(dim=1)
    sigma2 = ssr / float(df_resid)

    # GLS analogue of TSS / R^2 computed in whitened space
    if has_const:
        ybar = y_star.mean(dim=1, keepdim=True)
        tss_star = ((y_star - ybar) ** 2).sum(dim=1)
    else:
        tss_star = (y_star * y_star).sum(dim=1)
    rsq_star = 1.0 - ssr / tss_star
    if has_const:
        adj_rsq_star = 1.0 - (1.0 - rsq_star) * (float(n - 1) / float(df_resid))
    else:
        adj_rsq_star = 1.0 - (1.0 - rsq_star) * (float(n) / float(df_resid))

    df_model = (k - 1) if has_const else k
    if df_model <= 0:
        df_model = k

    # ---------------------------------------------------------------------
    # Inference (vcov)
    # ---------------------------------------------------------------------
    vcov_key = vcov.upper() if vcov != "classic" else "CLASSIC"
    if vcov_key == "CLASSIC":
        cov_type = "nonrobust"
        vcov_mat = vcov_classic(XtX_inv, sigma2)
        default_use_t = True
        metadata["vcov_mode"] = "whitened_ols"
    elif vcov_key == "HC0":
        cov_type = "HC0"
        sigma_inv_X = sigma_inv_times_X(X, X_star, info)
        vcov_mat = vcov_gls_hc0(sigma_inv_X, resid_raw, XtX_inv)
        default_use_t = False
        metadata["vcov_mode"] = "gls_sandwich"
    elif vcov_key == "HC1":
        cov_type = "HC1"
        sigma_inv_X = sigma_inv_times_X(X, X_star, info)
        vcov_mat = vcov_gls_hc1(sigma_inv_X, resid_raw, XtX_inv)
        default_use_t = False
        metadata["vcov_mode"] = "gls_sandwich"
    else:
        raise ShapeError("Unknown vcov for GLS. Use 'classic','HC0','HC1'.")

    if use_t is None:
        use_t = default_use_t

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    if store_diagnostics:
        objective = ssr
        logdet = info.logdet
        if logdet is None:
            # Should not happen; whitening always computes logdet.
            logdet = torch.zeros(R, device=X.device, dtype=X.dtype)
        loglik = -0.5 * (float(n) * math.log(2.0 * math.pi) + logdet + objective)
        diagnostics = {
            "objective": objective,
            "logdet_Sigma": logdet,
            "loglik_gaussian": loglik,
        }
    else:
        diagnostics = {}

    extras: dict[str, object] = {
        "solve_method": solve_method,
        "vcov": vcov,
        "tss_weighted": tss_star,
        "rsquared_weighted": rsq_star,
        "adj_rsquared_weighted": adj_rsq_star,
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
        resid=resid_raw if store_resid else None,
        y=y if store_y else None,
        metadata=metadata,
        diagnostics=diagnostics,
        model_name="GLS",
        method_name="Generalized Least Squares",
        cov_type=cov_type,
        backend="torch",
        param_names=param_names,
        use_t=bool(use_t),
        beta_true=beta_true,
        extras=extras,
    )
