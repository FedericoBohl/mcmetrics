# src/mcmetrics/models/ols.py
from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from mcmetrics.typing import as_batched_xy
from mcmetrics.results import OLSResults
from mcmetrics.vcov.robust import vcov_hc0, vcov_hc1

SolveMethod = Literal["solve", "lstsq"]
VcovType = Literal["classic", "HC0", "HC1"]


def OLS(
    X,
    y,
    *,
    has_const: bool = True,
    solve_method: SolveMethod = "solve",
    vcov: VcovType = "classic",
    use_t: bool | None = None,
    beta_true: Optional[torch.Tensor] = None,
    store_y: bool = True,
    store_fitted: bool = False,
    store_resid: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> OLSResults:
    """
    Batched OLS for Monte Carlo replications with memory-light defaults.

    Inputs
    - X: (R,n,k) or (n,k) or pandas.DataFrame (single sample)
    - y: (R,n) or (n,) or pandas.Series / 1-col DataFrame (single sample)
    """
    X, y, param_names = as_batched_xy(X, y, dtype=dtype, device=device)

    R, n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Need n > k for OLS. Got n={n}, k={k} (df_resid={df_resid}).")

    Xt = X.transpose(1, 2)                 # (R,k,n)
    XtX = Xt @ X                           # (R,k,k)
    Xty = Xt @ y.unsqueeze(-1)             # (R,k,1)

    if solve_method == "solve":
        beta = torch.linalg.solve(XtX, Xty).squeeze(-1)  # (R,k)
    elif solve_method == "lstsq":
        beta = torch.linalg.lstsq(XtX, Xty).solution.squeeze(-1)
    else:
        raise ValueError(f"Unknown solve_method {solve_method}")

    fitted = (X @ beta.unsqueeze(-1)).squeeze(-1)        # (R,n)
    resid = y - fitted                                   # (R,n)

    ssr = (resid ** 2).sum(dim=1)                         # (R,)
    sigma2 = ssr / float(df_resid)                        # (R,)

    I = torch.eye(k, device=X.device, dtype=X.dtype).expand(R, k, k)
    XtX_inv = torch.linalg.solve(XtX, I)                  # (R,k,k)

    df_model = (k - 1) if has_const else k
    if df_model <= 0:
        df_model = k

    vcov_key = vcov.upper() if vcov != "classic" else "CLASSIC"
    if vcov_key == "CLASSIC":
        cov_type = "nonrobust"
        vcov_mat = XtX_inv * sigma2.view(R, 1, 1)
        default_use_t = True
    elif vcov_key == "HC0":
        cov_type = "HC0"
        vcov_mat = vcov_hc0(X, resid, XtX_inv)
        default_use_t = False
    elif vcov_key == "HC1":
        cov_type = "HC1"
        vcov_mat = vcov_hc1(X, resid, XtX_inv)
        default_use_t = False
    else:
        raise ValueError(f"Unknown vcov='{vcov}'. Use 'classic', 'HC0', or 'HC1'.")

    if use_t is None:
        use_t = default_use_t

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
        resid=resid if store_resid else None,
        y=y if store_y else None,
        cov_type=cov_type,
        backend="torch",
        use_t=bool(use_t),
        beta_true=beta_true,
        param_names=param_names,  # <- names from DataFrame when single sample
    )