# Example 4: Full (non-diagonal) Sigma, broadcasted across replications (n,n) -> (R,n,n)
import torch
from mcmetrics import GLS

def make_spd(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A @ A.T + 0.5 * torch.eye(n)

torch.manual_seed(3)

R, n, k = 40, 120, 2
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.2, -0.4]).expand(R, k)

Sigma = make_spd(n, seed=9)
C = torch.linalg.cholesky(Sigma)

z = torch.randn(R, n)
u = (C @ z.unsqueeze(-1)).squeeze(-1)
y = torch.einsum("rnk,rk->rn", X, beta_true) + u

res = GLS(
    X, y,
    Sigma=Sigma,
    vcov="classic",
    store_y=True,
    store_resid=True,
    store_diagnostics=True,   # after patch, goes into res.metadata["diagnostics"]
)

print(res.summary())

diag = res.metadata.get("diagnostics", {})
if "logdet_Sigma" in diag:
    print("logdet(Sigma):", diag["logdet_Sigma"].item() if diag["logdet_Sigma"].ndim == 0 else diag["logdet_Sigma"])
if "objective" in diag:
    print("mean objective:", diag["objective"].mean().item())