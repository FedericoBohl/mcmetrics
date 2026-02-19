# src/mcmetrics/models/ols.py
from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from mcmetrics.results import OLSResults
from mcmetrics.typing import as_batched_xy
from mcmetrics.vcov.robust import vcov_cluster, vcov_hac, vcov_hc0, vcov_hc1

SolveMethod = Literal["solve", "lstsq", "cholesky", "qr"]
VcovType = Literal["classic", "HC0", "HC1", "cluster", "HAC"]


def _solve_ls(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    solve_method: SolveMethod,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve least squares for each replication.

    Inputs
    - X: (R,n,k)
    - y: (R,n)

    Returns
    - beta: (R,k)
    - XtX:  (R,k,k)
    - XtX_inv: (R,k,k)
    """
    R, n, k = X.shape

    if solve_method == "qr":
        # X = Q R, beta = R^{-1} Q' y
        Q, Rm = torch.linalg.qr(X, mode="reduced")  # Q:(R,n,k), Rm:(R,k,k)
        Qt_y = torch.einsum("rnk,rn->rk", Q, y)  # (R,k)
        beta = torch.linalg.solve(Rm, Qt_y.unsqueeze(-1)).squeeze(-1)  # (R,k)
        XtX = Rm.transpose(-1, -2) @ Rm
        I = torch.eye(k, device=X.device, dtype=X.dtype).expand(R, k, k)
        XtX_inv = torch.linalg.solve(XtX, I)
        return beta, XtX, XtX_inv

    Xt = X.transpose(1, 2)  # (R,k,n)
    XtX = Xt @ X            # (R,k,k)
    Xty = Xt @ y.unsqueeze(-1)  # (R,k,1)

    if solve_method == "solve":
        beta = torch.linalg.solve(XtX, Xty).squeeze(-1)
    elif solve_method == "lstsq":
        beta = torch.linalg.lstsq(XtX, Xty).solution.squeeze(-1)
    elif solve_method == "cholesky":
        L = torch.linalg.cholesky(XtX)
        beta = torch.cholesky_solve(Xty, L).squeeze(-1)
    else:
        raise ValueError(f"Unknown solve_method {solve_method}")

    I = torch.eye(k, device=X.device, dtype=X.dtype).expand(R, k, k)
    XtX_inv = torch.linalg.solve(XtX, I)
    return beta, XtX, XtX_inv


def OLS(
    X,
    y,
    *,
    has_const: bool = True,
    solve_method: SolveMethod = "solve",
    vcov: VcovType = "classic",
    clusters: Optional[torch.Tensor] = None,
    cluster_correction: Literal["none", "CR1"] = "CR1",
    hac_max_lags: int = 1,
    hac_kernel: Literal["bartlett"] = "bartlett",
    use_t: bool | None = None,
    beta_true: Optional[torch.Tensor] = None,
    store_y: bool = True,
    store_fitted: bool = False,
    store_resid: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> OLSResults:
    """Batched OLS for Monte Carlo replications with memory-light defaults.

    Inputs
    - X: (R,n,k) or (n,k) or pandas.DataFrame (single sample)
    - y: (R,n) or (n,) or pandas.Series / 1-col DataFrame (single sample)

    vcov
    - classic : homoskedastic
    - HC0/HC1 : White robust
    - cluster : one-way cluster robust (requires clusters)
    - HAC     : Newey-West (requires ordered observations)
    """
    X, y, param_names = as_batched_xy(X, y, dtype=dtype, device=device)

    R, n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Need n > k for OLS. Got n={n}, k={k} (df_resid={df_resid}).")

    beta, XtX, XtX_inv = _solve_ls(X, y, solve_method=solve_method)

    fitted = (X @ beta.unsqueeze(-1)).squeeze(-1)  # (R,n)
    resid = y - fitted                             # (R,n)

    ssr = (resid * resid).sum(dim=1)               # (R,)
    sigma2 = ssr / float(df_resid)                 # (R,)

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

    elif vcov_key == "CLUSTER":
        if clusters is None:
            raise ValueError("vcov='cluster' requires clusters=(n,) labels")
        cov_type = "cluster"
        vcov_mat = vcov_cluster(X, resid, XtX_inv, clusters, correction=cluster_correction)
        default_use_t = False

    elif vcov_key == "HAC":
        cov_type = "HAC"
        vcov_mat = vcov_hac(X, resid, XtX_inv, max_lags=hac_max_lags, kernel=hac_kernel)
        default_use_t = False

    else:
        raise ValueError("Unknown vcov. Use 'classic','HC0','HC1','cluster','HAC'.")

    if use_t is None:
        use_t = default_use_t

    extras = {
        "solve_method": solve_method,
        "vcov": vcov,
        "cluster_correction": cluster_correction if vcov_key == "CLUSTER" else None,
        "hac_max_lags": int(hac_max_lags) if vcov_key == "HAC" else None,
        "hac_kernel": hac_kernel if vcov_key == "HAC" else None,
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
        resid=resid if store_resid else None,
        y=y if store_y else None,
        cov_type=cov_type,
        backend="torch",
        use_t=bool(use_t),
        beta_true=beta_true,
        param_names=param_names,
        model_name="OLS",
        method_name="Least Squares",
        extras=extras,
    )