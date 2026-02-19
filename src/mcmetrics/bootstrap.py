# src/mcmetrics/bootstrap.py
from __future__ import annotations

from typing import Literal, Optional, Union

import torch

from mcmetrics.typing import as_batched_xy
from mcmetrics.weights import WeightsMode, as_batched_weights
from mcmetrics.vcov.robust import vcov_hc0, vcov_hc1

VcovType = Literal["classic", "HC0", "HC1"]


def _solve_xtx(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute beta and XtX_inv for each replication.

    Inputs
    - X: (R,n,k)
    - y: (R,n)

    Returns
    - beta: (R,k)
    - XtX_inv: (R,k,k)
    """
    R, n, k = X.shape
    Xt = X.transpose(1, 2)
    XtX = Xt @ X
    Xty = Xt @ y.unsqueeze(-1)
    beta = torch.linalg.solve(XtX, Xty).squeeze(-1)
    I = torch.eye(k, device=X.device, dtype=X.dtype).expand(R, k, k)
    XtX_inv = torch.linalg.solve(XtX, I)
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
    chunk_R: int = 256,
    chunk_B: int = 128,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Wild bootstrap p-values for H0: beta_j = beta0, batched over replications.

    Implements a wild bootstrap t-test for a single coefficient.

    If weights is provided, runs the bootstrap on the transformed WLS equation
    sqrt(w) y = sqrt(w) X beta + e.

    Returns
    - pvals: (R,) tensor
    """
    X, y, _ = as_batched_xy(X, y, dtype=dtype, device=device)
    Rtot, n, k = X.shape
    if not (0 <= j < k):
        raise ValueError(f"j must be in [0,{k-1}]. Got {j}.")
    if B <= 0:
        raise ValueError("B must be >= 1")

    # Optional WLS transform
    if weights is not None:
        w, sqrt_w = as_batched_weights(
            weights,
            R=Rtot,
            n=n,
            mode=weights_mode,
            dtype=X.dtype,
            device=X.device,
            check=True,
        )
        Xt = X * sqrt_w.unsqueeze(-1)
        yt = y * sqrt_w
    else:
        Xt = X
        yt = y

    gen = None
    if seed is not None:
        gen = torch.Generator(device=Xt.device)
        gen.manual_seed(int(seed))

    pvals = torch.empty((Rtot,), device=Xt.device, dtype=Xt.dtype)

    for r0 in range(0, Rtot, int(chunk_R)):
        r1 = min(Rtot, r0 + int(chunk_R))
        Xb = Xt[r0:r1]  # (Rc,n,k)
        yb = yt[r0:r1]  # (Rc,n)
        Rc = Xb.shape[0]

        beta_hat, XtX_inv = _solve_xtx(Xb, yb)  # (Rc,k), (Rc,k,k)
        fitted = (Xb @ beta_hat.unsqueeze(-1)).squeeze(-1)
        resid = yb - fitted

        if vcov == "classic":
            ssr = (resid * resid).sum(dim=1)
            sigma2 = ssr / float(n - k)
            vc = XtX_inv * sigma2.view(Rc, 1, 1)
        elif vcov == "HC0":
            vc = vcov_hc0(Xb, resid, XtX_inv)
        elif vcov == "HC1":
            vc = vcov_hc1(Xb, resid, XtX_inv)
        else:
            raise ValueError("vcov must be one of {'classic','HC0','HC1'}")

        se = torch.sqrt(torch.diagonal(vc, dim1=-2, dim2=-1))  # (Rc,k)
        t_obs = (beta_hat[:, j] - float(beta0)) / se[:, j]     # (Rc,)
        t_abs = torch.abs(t_obs)

        # Restricted fit under H0: beta_j fixed at beta0
        xj = Xb[:, :, j]  # (Rc,n)
        if k == 1:
            fitted0 = xj * float(beta0)
        else:
            Xrest = torch.cat([Xb[:, :, :j], Xb[:, :, j + 1 :]], dim=2)  # (Rc,n,k-1)
            y0 = yb - xj * float(beta0)
            beta_rest, _ = _solve_xtx(Xrest, y0)
            fitted0 = xj * float(beta0) + (Xrest @ beta_rest.unsqueeze(-1)).squeeze(-1)

        resid0 = yb - fitted0  # (Rc,n)

        count = torch.zeros((Rc,), device=Xb.device, dtype=torch.int64)
        done = 0
        while done < B:
            Bb = int(min(chunk_B, B - done))
            done += Bb

            u = torch.randint(0, 2, (Bb, Rc, n), device=Xb.device, generator=gen)
            v = u.to(Xb.dtype) * 2.0 - 1.0  # +/-1

            y_star = fitted0.unsqueeze(0) + resid0.unsqueeze(0) * v  # (Bb,Rc,n)

            Xty_star = torch.einsum("rnk,brn->brk", Xb, y_star)
            beta_star = torch.einsum("rkl,brl->brk", XtX_inv, Xty_star)  # (Bb,Rc,k)

            resid_star = y_star - torch.einsum("rnk,brk->brn", Xb, beta_star)

            if vcov == "classic":
                ssr_star = (resid_star * resid_star).sum(dim=2)  # (Bb,Rc)
                sigma2_star = ssr_star / float(n - k)
                vc_star = XtX_inv.unsqueeze(0) * sigma2_star.view(Bb, Rc, 1, 1)
            elif vcov == "HC0":
                e2 = resid_star * resid_star
                meat = torch.einsum("rni,brn,rnj->brij", Xb, e2, Xb)
                vc_star = XtX_inv.unsqueeze(0) @ meat @ XtX_inv.unsqueeze(0)
            else:  # HC1
                e2 = resid_star * resid_star
                meat = torch.einsum("rni,brn,rnj->brij", Xb, e2, Xb)
                vc_star = XtX_inv.unsqueeze(0) @ meat @ XtX_inv.unsqueeze(0)
                vc_star = vc_star * (float(n) / float(n - k))

            se_star = torch.sqrt(torch.diagonal(vc_star, dim1=-2, dim2=-1))  # (Bb,Rc,k)
            t_star = (beta_star[:, :, j] - float(beta0)) / se_star[:, :, j]   # (Bb,Rc)

            count += (torch.abs(t_star) >= t_abs.view(1, Rc)).sum(dim=0).to(torch.int64)

        p_block = (count.to(Xb.dtype) + 1.0) / float(B + 1)
        pvals[r0:r1] = p_block

    return pvals