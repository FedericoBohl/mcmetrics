"""
WLS example (batched Monte Carlo, no Python loops in the hot path)

Goal
- Show how to use mcmetrics.WLS on data shaped like (R, n, k).
- Use weights that are constant across replications (shape (n,)), so the user sees the simplest weighted workflow.

Design
- Fixed regressors across replications (common in Monte Carlo).
- Heteroskedastic errors with known variance: Var(e_i | X) = v_i, where v_i depends on x1.
- Correct WLS weights: w_i = 1 / v_i.

Run
- Classic (nonrobust) vcov: efficient if weights are correctly specified.
- HC1 robust vcov: sandwich inference on the transformed (weighted) problem.
"""
from __future__ import annotations

import torch

from mcmetrics import WLS


def main() -> None:
    # ----------------------------
    # Settings
    # ----------------------------
    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    R = 2_000  # replications
    n = 250  # observations
    k = 3  # intercept + 2 regressors

    beta_true = torch.tensor([1.0, 0.8, -0.5], device=device, dtype=dtype)

    # ----------------------------
    # Fixed X across replications
    # ----------------------------
    x1 = torch.linspace(-2.0, 2.0, n, device=device, dtype=dtype)
    x2 = torch.randn(n, device=device, dtype=dtype)

    X1 = x1.expand(R, n)
    X2 = x2.expand(R, n)

    X = torch.stack(
        [
            torch.ones((R, n), device=device, dtype=dtype),
            X1,
            X2,
        ],
        dim=2,
    )  # (R,n,k)

    # ----------------------------
    # Known heteroskedasticity + correct weights
    # ----------------------------
    v = 0.5 + 1.5 * (x1**2)  # (n,) must be strictly positive
    w = 1.0 / v  # (n,) broadcast to (R,n) inside WLS

    z = torch.randn((R, n), device=device, dtype=dtype)
    eps = torch.sqrt(v).expand(R, n) * z
    y = (X @ beta_true.view(k, 1)).squeeze(-1) + eps

    # ----------------------------
    # WLS estimation
    # ----------------------------
    res_classic = WLS(
        X,
        y,
        weights=w,
        has_const=True,
        vcov="classic",
        beta_true=beta_true,
        store_y=False,
        store_fitted=False,
        store_resid=False,
        dtype=dtype,
        device=device,
    )

    res_hc1 = WLS(
        X,
        y,
        weights=w,
        has_const=True,
        vcov="HC1",
        beta_true=beta_true,
        store_y=False,
        store_fitted=False,
        store_resid=False,
        dtype=dtype,
        device=device,
    )

    def stderr_from_vcov(vcov: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.diagonal(vcov, dim1=1, dim2=2))

    b = res_classic.params
    se_classic = stderr_from_vcov(res_classic.vcov)
    se_hc1 = stderr_from_vcov(res_hc1.vcov)

    print("\n=== WLS (classic vcov) ===")
    print("beta_true:", beta_true.detach().cpu().numpy())
    print("mean(beta_hat):", b.mean(dim=0).detach().cpu().numpy())
    print("sd(beta_hat): ", b.std(dim=0, unbiased=True).detach().cpu().numpy())
    print("mean(stderr): ", se_classic.mean(dim=0).detach().cpu().numpy())

    print("\n=== WLS (HC1 robust vcov) ===")
    print("mean(stderr): ", se_hc1.mean(dim=0).detach().cpu().numpy())

    r0 = 0
    print("\n=== Replication r=0 ===")
    print("beta_hat:", b[r0].detach().cpu().numpy())
    print("stderr (classic):", se_classic[r0].detach().cpu().numpy())
    print("stderr (HC1): ", se_hc1[r0].detach().cpu().numpy())


if __name__ == "__main__":
    main()
