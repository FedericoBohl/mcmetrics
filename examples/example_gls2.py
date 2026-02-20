# Example 2: Diagonal Sigma (heteroskedasticity) — fast path (no n×n matrices)
import torch
from mcmetrics import GLS

torch.manual_seed(1)

R, n, k = 200, 500, 4
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.0, 0.5, -0.2, 0.1]).expand(R, k)

idx = torch.linspace(0.2, 2.0, n)
Sigma_diag = (0.5 + idx) ** 2  # variances (n,)

u = torch.randn(R, n) * torch.sqrt(Sigma_diag)
y = torch.einsum("rnk,rk->rn", X, beta_true) + u

res = GLS(
    X, y,
    Sigma=Sigma_diag,
    vcov="classic",
    store_y=True,
    store_fitted=True,
    store_resid=True,        # after patch, this should populate whitened resid
)

print(res.summary())

# Robust way to get whitened residuals (works even if res.resid is None)
beta = res.params
yhat = torch.einsum("rnk,rk->rn", X, beta)
resid_raw = y - yhat
inv_sqrt = torch.rsqrt(Sigma_diag)            # (n,)
resid_white = resid_raw * inv_sqrt            # (R,n)

print("whitened resid std:", resid_white.std().item())