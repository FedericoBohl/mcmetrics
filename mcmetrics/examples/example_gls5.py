# Example 5: Full Sigma with precomputed Cholesky (avoid recomputation)
import torch
from mcmetrics import GLS

def make_spd(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A @ A.T + 0.5 * torch.eye(n)

torch.manual_seed(4)

R, n, k = 25, 80, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([0.2, 0.5, -0.7]).expand(R, k)

Sigma = make_spd(n, seed=21)
chol = torch.linalg.cholesky(Sigma)

z = torch.randn(R, n)
u = (chol @ z.unsqueeze(-1)).squeeze(-1)
y = torch.einsum("rnk,rk->rn", X, beta_true) + u

res = GLS(
    X, y,
    chol_Sigma=chol,          # re-use factor
    vcov="classic",
    store_y=True,
)

print(res.summary())