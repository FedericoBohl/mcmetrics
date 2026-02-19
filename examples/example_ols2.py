"""
More advanced example: Monte Carlo replications + hypothesis testing.

This script:
- Generates R replications
- Fits OLS with robust vcov (HC1)
- Reports:
  * rejection rates for H0: beta = beta_null (componentwise)
  * a joint Wald test: H0: beta1 = 0 AND beta2 = 0
"""

import torch

from mcmetrics import OLS


def main() -> None:
    torch.set_default_dtype(torch.float64)

    # -----------------------------
    # Monte Carlo design
    # -----------------------------
    R = 5000
    n = 80

    x1 = torch.randn(R, n)
    x2 = torch.randn(R, n)

    X = torch.stack([torch.ones(R, n), x1, x2], dim=2)  # (R,n,k=3)

    beta_true = torch.tensor([1.0, 0.8, -0.2])

    # Heteroskedastic errors (example): var depends on |x1|
    u = torch.randn(R, n)
    sigma = 0.5 + 0.8 * torch.abs(x1)
    eps = sigma * u

    y = torch.einsum("rnk,k->rn", X, beta_true) + eps  # (R,n)

    # -----------------------------
    # Fit OLS with robust vcov
    # -----------------------------
    reg = OLS(
        X, y,
        vcov="HC1",
        beta_true=beta_true,
        store_y=False,         # no need for R^2 here
        store_resid=False,     # memory-light
        store_fitted=False,    # memory-light
    )

    print(reg.summary(param_names=["const", "x1", "x2"]))

    # -----------------------------
    # (A) Componentwise tests: H0: beta = beta_null
    # -----------------------------
    alpha = 0.05

    # Example null: test beta2 = 0 (keep other components at their true values)
    beta_null = torch.tensor([1.0, 0.8, 0.0])

    rej_rates = reg.rejection_rate_beta0(beta_null, alpha=alpha)
    print("\nRejection rates for H0: beta = beta_null (componentwise)")
    for name, rr in zip(["const", "x1", "x2"], rej_rates.tolist()):
        print(f"  {name:>5s}: {rr:0.4f}")

    # -----------------------------
    # (B) Joint Wald test: H0: beta1=0 AND beta2=0
    # -----------------------------
    Rm = torch.tensor([
        [0.0, 1.0, 0.0],  # selects beta1
        [0.0, 0.0, 1.0],  # selects beta2
    ])
    q = torch.tensor([0.0, 0.0])

    w = reg.wald_test(Rm, q, alpha=alpha, use_f=False)  # robust -> Chi^2 by default
    joint_rej = w["rej"].to(torch.float64).mean().item()

    print(f"\nJoint Wald test H0: beta1=0 and beta2=0")
    print(f"  Rejection rate @ {alpha:0.2f}: {joint_rej:0.4f}")


if __name__ == "__main__":
    main()