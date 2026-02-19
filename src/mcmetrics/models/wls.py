# src/mcmetrics/models/wls.py
from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from mcmetrics.results import OLSResults
from mcmetrics.typing import as_batched_xy
from mcmetrics.vcov.robust import vcov_cluster, vcov_hac, vcov_hc0, vcov_hc1
from mcmetrics.weights import WeightsMode, as_batched_weights

SolveMethod = Literal["solve", "lstsq", "cholesky", "qr"]
VcovType = Literal["classic", "HC0", "HC1", "cluster", "HAC"]


def _solve_ls(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    solve_method: SolveMethod,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve LS for each replication.

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
        Q, Rm = torch.linalg.qr(X, mode="reduced")
        Qt_y = torch.einsum("rnk,rn->rk", Q, y)
        beta = torch.linalg.solve(Rm, Qt_y.unsqueeze(-1)).squeeze(-1)
        XtX = Rm.transpose(-1, -2) @ Rm
        I = torch.eye(k, device=X.device, dtype=X.dtype).expand(R, k, k)
        XtX_inv = torch.linalg.solve(XtX, I)
        return beta, XtX, XtX_inv

    Xt = X.transpose(1, 2)
    XtX = Xt @ X
    Xty = Xt @ y.unsqueeze(-1)

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


def WLS(
    X,
    y,
    weights,
    *,
    weights_mode: WeightsMode = "precision",
    has_const: bool = True,
    check_weights: bool = True,
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
    """Batched WLS for Monte Carlo replications.

    The model is y = X beta + u, with weights proportional to precision:
      w_i \\propto 1/Var(u_i | X)

    Implementation uses the transformed system:
      sqrt(w_i) y_i = sqrt(w_i) x_i' beta + e_i

    Inputs
    - X: (R,n,k) or (n,k)
    - y: (R,n) or (n,)
    - weights: scalar, (n,), or (R,n)

    vcov
    - classic : homoskedastic in transformed equation
    - HC0/HC1 : White robust on transformed equation
    - cluster : cluster robust on transformed equation (requires clusters)
    - HAC     : Newey-West on transformed equation
    """
    X, y, param_names = as_batched_xy(X, y, dtype=dtype, device=device)

    R, n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError(f"Need n > k for WLS. Got n={n}, k={k} (df_resid={df_resid}).")

    w, sqrt_w = as_batched_weights(
        weights,
        R=R,
        n=n,
        mode=weights_mode,
        dtype=X.dtype,
        device=X.device,
        check=check_weights,
    )

    Xw = X * sqrt_w.unsqueeze(-1)  # (R,n,k)
    yw = y * sqrt_w               # (R,n)

    beta, XtX, XtX_inv = _solve_ls(Xw, yw, solve_method=solve_method)

    fitted = (X @ beta.unsqueeze(-1)).squeeze(-1)  # (R,n) fitted on original scale
    resid = y - fitted                             # (R,n)

    wrss = (w * resid * resid).sum(dim=1)          # (R,)
    sigma2 = wrss / float(df_resid)                # (R,) (transformed equation)

    df_model = (k - 1) if has_const else k
    if df_model <= 0:
        df_model = k

    resid_w = resid * sqrt_w  # (R,n)

    vcov_key = vcov.upper() if vcov != "classic" else "CLASSIC"

    if vcov_key == "CLASSIC":
        cov_type = "nonrobust"
        vcov_mat = XtX_inv * sigma2.view(R, 1, 1)
        default_use_t = True

    elif vcov_key == "HC0":
        cov_type = "HC0"
        vcov_mat = vcov_hc0(Xw, resid_w, XtX_inv)
        default_use_t = False

    elif vcov_key == "HC1":
        cov_type = "HC1"
        vcov_mat = vcov_hc1(Xw, resid_w, XtX_inv)
        default_use_t = False

    elif vcov_key == "CLUSTER":
        if clusters is None:
            raise ValueError("vcov='cluster' requires clusters=(n,) labels")
        cov_type = "cluster"
        vcov_mat = vcov_cluster(Xw, resid_w, XtX_inv, clusters, correction=cluster_correction)
        default_use_t = False

    elif vcov_key == "HAC":
        cov_type = "HAC"
        vcov_mat = vcov_hac(Xw, resid_w, XtX_inv, max_lags=hac_max_lags, kernel=hac_kernel)
        default_use_t = False

    else:
        raise ValueError("Unknown vcov. Use 'classic','HC0','HC1','cluster','HAC'.")

    if use_t is None:
        use_t = default_use_t

    # Weight diagnostics
    wmin = float(w.min().detach().cpu().item())
    wmax = float(w.max().detach().cpu().item())
    wmean = float(w.mean().detach().cpu().item())

    extras: dict[str, object] = {
        "solve_method": solve_method,
        "vcov": vcov,
        "weights_mode": weights_mode,
        "weights_min": wmin,
        "weights_max": wmax,
        "weights_mean": wmean,
        "weights_ratio": (wmax / wmin) if wmin > 0 else float("inf"),
        "cluster_correction": cluster_correction if vcov_key == "CLUSTER" else None,
        "hac_max_lags": int(hac_max_lags) if vcov_key == "HAC" else None,
        "hac_kernel": hac_kernel if vcov_key == "HAC" else None,
    }

    # Weighted R^2 (if y is stored)
    if store_y:
        ybar_w = (w * y).sum(dim=1) / w.sum(dim=1)
        tss_w = (w * (y - ybar_w.view(R, 1)) ** 2).sum(dim=1)
        r2_w = 1.0 - wrss / tss_w
        extras["tss_weighted"] = tss_w
        extras["rsquared_weighted"] = r2_w
        if has_const:
            extras["adj_rsquared_weighted"] = 1.0 - (1.0 - r2_w) * (float(n - 1) / float(df_resid))
        else:
            extras["adj_rsquared_weighted"] = 1.0 - (1.0 - r2_w) * (float(n) / float(df_resid))

    return OLSResults(
        params=beta,
        vcov=vcov_mat,
        sigma2=sigma2,
        ssr=wrss,  # for WLS, ssr is *weighted* SSR
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
        model_name="WLS",
        method_name="Weighted Least Squares",
        extras=extras,
    )