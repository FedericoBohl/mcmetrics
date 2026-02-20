# Example 1: GLS with identity covariance (GLS == OLS)
import torch
from mcmetrics import OLS, GLS

torch.manual_seed(0)

R, n, k = 50, 200, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.0, -0.5, 0.8]).expand(R, k)
u = torch.randn(R, n)
y = torch.einsum("rnk,rk->rn", X, beta_true) + u

Sigma_I = torch.eye(n)

res_gls = GLS(X, y, Sigma=Sigma_I, vcov="classic", store_y=True)
res_ols = OLS(X, y, vcov="classic", store_y=True)

print(res_gls.summary())
print("max |beta_gls - beta_ols|:", (res_gls.params - res_ols.params).abs().max().item())