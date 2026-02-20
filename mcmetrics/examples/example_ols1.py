"""
Basic OLS example (Monte Carlo aware, but we run a single sample).

This script:
- Creates a simple regression dataset
- Fits OLS
- Prints a compact Monte Carlo-style summary (here R=1)
"""

import torch

from mcmetrics import OLS


def main() -> None:
    torch.set_default_dtype(torch.float64)

    # -----------------------------
    # Generate a single dataset
    # -----------------------------
    n = 200
    x1 = torch.randn(n)
    x2 = torch.randn(n)

    # Include constant manually
    X = torch.stack([torch.ones(n), x1, x2], dim=1)  # (n,k=3)

    beta_true = torch.tensor([1.0, 0.8, -0.2])
    eps = 0.5 * torch.randn(n)
    y = (X @ beta_true) + eps  # (n,)

    # -----------------------------
    # Fit (single sample => internally becomes R=1)
    # -----------------------------
    reg = OLS(
        X, y,
        vcov="classic",
        beta_true=beta_true,
        store_y=True,          # allow R^2/F-stat in the summary
        store_resid=False,     # memory-light
        store_fitted=False,    # memory-light
    )

    print(reg.summary(param_names=["const", "x1", "x2"]))


if __name__ == "__main__":
    main()