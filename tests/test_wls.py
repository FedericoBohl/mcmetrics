import torch

from mcmetrics import OLS
from mcmetrics.models.wls import WLS


def _make_xy(R: int, n: int, k: int, dtype, seed=1234):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn((R, n, k), generator=g, dtype=dtype)
    X[:, :, 0] = 1.0
    beta = torch.linspace(0.2, 0.2 * k, k, dtype=dtype)
    y = (X @ beta.view(1, k, 1)).squeeze(-1) + 0.1 * torch.randn((R, n), generator=g, dtype=dtype)
    return X, beta, y


def test_wls_equals_ols_when_weights_one(torch_dtype):
    R, n, k = 4, 35, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)

    res_ols = OLS(X, y, vcov="classic")
    res_wls = WLS(X, y, weights=1.0, vcov="classic")

    assert torch.max(torch.abs(res_ols.params - res_wls.params)).item() < 1e-10
    assert torch.max(torch.abs(res_ols.ssr - res_wls.ssr)).item() < 1e-10
    assert torch.max(torch.abs(res_ols.sigma2 - res_wls.sigma2)).item() < 1e-10
    assert torch.max(torch.abs(res_ols.vcov - res_wls.vcov)).item() < 1e-10


def test_wls_weights_broadcast_equivalence(torch_dtype):
    R, n, k = 3, 40, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)

    w_scalar = 2.0
    w_n = torch.full((n,), 2.0, dtype=torch_dtype)
    w_Rn = torch.full((R, n), 2.0, dtype=torch_dtype)

    res0 = WLS(X, y, weights=w_scalar, vcov="classic")
    res1 = WLS(X, y, weights=w_n, vcov="classic")
    res2 = WLS(X, y, weights=w_Rn, vcov="classic")

    assert torch.max(torch.abs(res0.params - res1.params)).item() < 1e-10
    assert torch.max(torch.abs(res0.params - res2.params)).item() < 1e-10


def test_wls_rejects_nonpositive_weights(torch_dtype):
    R, n, k = 2, 30, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)
    w = torch.ones(n, dtype=torch_dtype)
    w[0] = 0.0

    try:
        WLS(X, y, weights=w, check_weights=True)
    except Exception as e:
        assert "positive" in str(e).lower()
