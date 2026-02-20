# Example 3: Diagonal Sigma with precomputed inv_sqrt_Sigma (Monte Carlo-friendly)
import torch
from mcmetrics import GLS

torch.manual_seed(2)

R, n, k = 100, 300, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([0.3, -0.1, 0.7]).expand(R, k)

Sigma_diag = torch.exp(torch.linspace(-1.0, 1.0, n))  # (n,) variances > 0
inv_sqrt = torch.rsqrt(Sigma_diag)                    # 1/sqrt(variance)

u = torch.randn(R, n) * torch.sqrt(Sigma_diag)
y = torch.einsum("rnk,rk->rn", X, beta_true) + u

# Pass the whitener directly to avoid repeated sqrt/rsqrt costs
res = GLS(
    X, y,
    inv_sqrt_Sigma=inv_sqrt,
    Sigma_is_diagonal=True,   # explicit: interpret as diagonal whitener
    vcov="classic",
    store_y=True,
)

print(res.summary())