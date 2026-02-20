# Example 6: Robust GLS vcov (HC0 / HC1) when Sigma may be misspecified
import torch
from mcmetrics import GLS

torch.manual_seed(5)

R, n, k = 150, 250, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.0, 0.0, 0.5]).expand(R, k)

true_s2 = 0.2 + torch.linspace(0.5, 2.0, n) ** 2
u = torch.randn(R, n) * torch.sqrt(true_s2)
y = torch.einsum("rnk,rk->rn", X, beta_true) + u

model_s2 = 0.5 + torch.linspace(0.2, 1.5, n) ** 2

res_classic = GLS(X, y, Sigma=model_s2, vcov="classic", store_y=True)
res_hc0     = GLS(X, y, Sigma=model_s2, vcov="HC0",     store_y=True)
res_hc1     = GLS(X, y, Sigma=model_s2, vcov="HC1",     store_y=True)

print("=== classic ===")
print(res_classic.summary())
print("=== HC0 ===")
print(res_hc0.summary())
print("=== HC1 ===")
print(res_hc1.summary())