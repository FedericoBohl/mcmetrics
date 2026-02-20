"""
Plotting example (LaTeX-like style).

This script:
- Generates Monte Carlo replications
- Fits OLS
- Produces useful plots:
  * parameter histogram for beta1
  * residual series and residual histogram for replication r=0
  * Normal Q-Q plot of residuals for replication r=0
  * residual ACF for replication r=0

Notes:
- If residuals are not stored, we pass X and y to plot() so it can compute them on-the-fly.
"""

import torch

from mcmetrics import OLS


def main() -> None:
    torch.set_default_dtype(torch.float64)

    R = 2000
    n = 100

    x1 = torch.randn(R, n)
    x2 = torch.randn(R, n)
    X = torch.stack([torch.ones(R, n), x1, x2], dim=2)

    beta_true = torch.tensor([1.0, 0.8, -0.2])
    eps = 0.8 * torch.randn(R, n)
    y = torch.einsum("rnk,k->rn", X, beta_true) + eps

    reg = OLS(
        X, y,
        vcov="classic",
        beta_true=beta_true,
        store_y=False,
        store_resid=False,    # do NOT store residuals
        store_fitted=False,
    )

    # -----------------------------
    # Parameter histogram (beta1)
    # -----------------------------
    reg.plot(kind="params", j=1, show=True, latex=True, use_tex=False)

    # -----------------------------
    # Residual diagnostics for a chosen replication
    # We pass X,y because resid is not stored.
    # -----------------------------
    r = 0
    reg.plot(kind="resid", r=r, X=X, y=y, show=True, latex=True)
    reg.plot(kind="resid_hist", r=r, X=X, y=y, show=True, latex=True)
    reg.plot(kind="qq_resid", r=r, X=X, y=y, show=True, latex=True)
    reg.plot(kind="acf_resid", r=r, X=X, y=y, nlags=20, confint=True, show=True, latex=True)


if __name__ == "__main__":
    main()