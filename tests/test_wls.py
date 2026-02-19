# tests/test_wls.py
import torch

from mcmetrics import OLS, WLS


def test_wls_scalar_weight_matches_ols_params_and_vcov():
    torch.manual_seed(0)
    R, n, k = 7, 40, 3
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    res_ols = OLS(X, y, vcov="classic")
    res_wls = WLS(X, y, weights=2.0, vcov="classic")

    assert torch.allclose(res_ols.params, res_wls.params, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res_ols.vcov, res_wls.vcov, atol=1e-6, rtol=1e-6)


def test_wls_broadcast_weights_n_vs_Rn():
    torch.manual_seed(1)
    R, n, k = 5, 25, 4
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    w_n = torch.linspace(0.5, 2.0, n)              # (n,)
    w_Rn = w_n.view(1, n).expand(R, n).clone()     # (R,n)

    res1 = WLS(X, y, weights=w_n, vcov="classic")
    res2 = WLS(X, y, weights=w_Rn, vcov="classic")

    assert torch.allclose(res1.params, res2.params, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res1.vcov, res2.vcov, atol=1e-6, rtol=1e-6)