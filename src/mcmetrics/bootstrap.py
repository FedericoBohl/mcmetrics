from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from mcmetrics.typing import as_batched_xy
from mcmetrics.vcov.robust import vcov_hc0, vcov_hc1
from mcmetrics.weights import WeightsMode, as_batched_weights

VcovType = Literal["classic", "HC0", "HC1"]


def _solve_xtx(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute beta and XtX_inv for each replication.

    X: (R,n,k), y: (R,n)
    Returns beta: (R,k), XtX_inv: (R,k,k)
    """
    R, _, k = X.shape
    Xt = X.transpose(1, 2)
    XtX = Xt @ X
    Xty = Xt @ y.unsqueeze(-1)
    beta = torch.linalg.solve(XtX, Xty).squeeze(-1)
    eye = torch.eye(k, device=X.device, dtype=X.dtype).expand(R, k, k)
    XtX_inv = torch.linalg.solve(XtX, eye)
    return beta, XtX_inv


def wild_bootstrap_pvalue_beta0(
    X,
    y,
    *,
    beta0: float,
    j: int,
    weights=None,
    weights_mode: WeightsMode = "precision",
    vcov: VcovType = "HC1",
    B: int = 999,
    seed: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Wild bootstrap p-values for H0: beta_j = beta0, batched over replications.

    NOTE: this utility is intentionally minimal and not yet part of the public API.
    """
    X, y, _ = as_batched_xy(X, y, dtype=dtype, device=device)
    R, n, k = X.shape
    if not (0 <= j < k):
        raise ValueError(f"j must be in [0,{k-1}]. Got {j}.")

    if weights is not None:
        w, sqrt_w = as_batched_weights(
            weights,
            R=R,
            n=n,
            mode=weights_mode,
            dtype=X.dtype,
            device=X.device,
            check=True,
        )
        Xw = X * sqrt_w.unsqueeze(-1)
        yw = y * sqrt_w
        beta_hat, XtX_inv = _solve_xtx(Xw, yw)
        resid = yw - (Xw @ beta_hat.unsqueeze(-1)).squeeze(-1)
        if vcov == "HC0":
            V = vcov_hc0(Xw, resid, XtX_inv)
        else:
            V = vcov_hc1(Xw, resid, XtX_inv)
    else:
        beta_hat, XtX_inv = _solve_xtx(X, y)
        resid = y - (X @ beta_hat.unsqueeze(-1)).squeeze(-1)
        if vcov == "HC0":
            V = vcov_hc0(X, resid, XtX_inv)
        else:
            V = vcov_hc1(X, resid, XtX_inv)

    se = torch.sqrt(torch.diagonal(V, dim1=1, dim2=2)[:, j])
    t_obs = (beta_hat[:, j] - float(beta0)) / se

    g = torch.Generator(device=X.device)
    if seed is not None:
        g.manual_seed(int(seed))

    # Rademacher weights
    v = torch.randint(0, 2, (B, R, n), generator=g, device=X.device, dtype=torch.int64)
    v = 2 * v - 1  # {-1, +1}

    # Bootstrap residuals: e* v
    e_star = resid.unsqueeze(0) * v.to(resid.dtype)

    # y* under H0: replace coefficient j with beta0 in fitted values
    beta0_vec = beta_hat.clone()
    beta0_vec[:, j] = float(beta0)
    y0 = (X @ beta0_vec.unsqueeze(-1)).squeeze(-1)
    y_star = y0.unsqueeze(0) + e_star

    # Estimate bootstrap stats (loop over B; acceptable: B is not the hot path of your library)
    # (Later we can batch this, but it's fine as a dev utility.)
    t_star = torch.empty((B, R), device=X.device, dtype=X.dtype)
    for b in range(B):
        bb, XtX_inv_b = _solve_xtx(X, y_star[b])
        resid_b = y_star[b] - (X @ bb.unsqueeze(-1)).squeeze(-1)
        Vb = vcov_hc1(X, resid_b, XtX_inv_b) if vcov == "HC1" else vcov_hc0(X, resid_b, XtX_inv_b)
        se_b = torch.sqrt(torch.diagonal(Vb, dim1=1, dim2=2)[:, j])
        t_star[b] = (bb[:, j] - float(beta0)) / se_b

    pvals = (t_star.abs() >= t_obs.abs().unsqueeze(0)).to(torch.float64).mean(dim=0)
    return pvals.to(X.dtype)
