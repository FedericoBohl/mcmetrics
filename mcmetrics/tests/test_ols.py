import pytest
import torch

from mcmetrics import OLS

def test_ols_shapes_and_consistency():
    torch.manual_seed(0)
    R, n, k = 10, 50, 3
    X = torch.randn(R, n, k)
    beta = torch.tensor([1.0, 0.8, -0.2])
    y = (X @ beta).squeeze(-1) + 0.1 * torch.randn(R, n)

    res = OLS(X, y, vcov="classic", store_resid=True, store_y=True)
    assert res.params.shape == (R, k)
    assert res.vcov.shape == (R, k, k)
    assert res.resid is not None and res.resid.shape == (R, n)
    assert res.y is not None and res.y.shape == (R, n)

    # sanity: ssr matches resid
    ssr2 = (res.resid ** 2).sum(dim=1)
    assert torch.max(torch.abs(ssr2 - res.ssr)) < 1e-6


def test_ols_accepts_single_sample_and_returns_batched(torch_dtype):
    R, n, k = 4, 25, 3
    Xb = torch.randn((R, n, k), dtype=torch_dtype)
    Xb[:, :, 0] = 1.0
    yb = torch.randn((R, n), dtype=torch_dtype)

    X = Xb[0]  # (n,k)
    y = yb[0]  # (n,)
    res = OLS(X, y)

    assert res.params.shape == (1, k)
    assert res.vcov.shape == (1, k, k)
    assert res.sigma2.shape == (1,)
    assert res.ssr.shape == (1,)
    assert res.nobs == n
    assert res.R == 1
    assert res.k == k

    assert res.y is not None
    assert res.y.shape == (1, n)
    assert res.fitted is None
    assert res.resid is None


def test_ols_batched_shapes_and_df(torch_dtype):
    R, n, k = 5, 30, 4
    X = torch.randn((R, n, k), dtype=torch_dtype)
    X[:, :, 0] = 1.0
    y = torch.randn((R, n), dtype=torch_dtype)

    res = OLS(X, y, has_const=True)
    assert res.params.shape == (R, k)
    assert res.vcov.shape == (R, k, k)
    assert res.sigma2.shape == (R,)
    assert res.ssr.shape == (R,)
    assert res.df_resid == n - k
    assert res.df_model == (k - 1)


def test_ols_matches_direct_solve(torch_dtype):
    R, n, k = 3, 40, 3
    g = torch.Generator().manual_seed(7)

    X = torch.randn((R, n, k), generator=g, dtype=torch_dtype)
    X[:, :, 0] = 1.0
    beta_true = torch.tensor([0.5, -0.25, 0.1], dtype=torch_dtype)

    noise = 0.05 * torch.randn((R, n), generator=g, dtype=torch_dtype)
    y = (X @ beta_true.view(1, k, 1)).squeeze(-1) + noise

    res = OLS(X, y, solve_method="solve")

    Xt = X.transpose(1, 2)
    XtX = Xt @ X
    Xty = Xt @ y.unsqueeze(-1)
    beta_direct = torch.linalg.solve(XtX, Xty).squeeze(-1)

    assert torch.max(torch.abs(res.params - beta_direct)).item() < 1e-10


def test_ols_solve_method_lstsq_close_to_solve(torch_dtype):
    R, n, k = 3, 50, 4
    g = torch.Generator().manual_seed(11)

    X = torch.randn((R, n, k), generator=g, dtype=torch_dtype)
    X[:, :, 0] = 1.0
    y = torch.randn((R, n), generator=g, dtype=torch_dtype)

    res_solve = OLS(X, y, solve_method="solve")
    res_lstsq = OLS(X, y, solve_method="lstsq")

    # float32 has ~1e-7 precision; allow a looser tolerance than float64
    if torch_dtype == torch.float32:
        assert torch.allclose(res_solve.params, res_lstsq.params, rtol=1e-5, atol=1e-6)
    else:
        assert torch.allclose(res_solve.params, res_lstsq.params, rtol=1e-10, atol=1e-10)


def test_ols_vcov_defaults_use_t(torch_dtype):
    R, n, k = 2, 30, 3
    X = torch.randn((R, n, k), dtype=torch_dtype)
    X[:, :, 0] = 1.0
    y = torch.randn((R, n), dtype=torch_dtype)

    res_classic = OLS(X, y, vcov="classic")
    assert res_classic.cov_type == "nonrobust"
    assert res_classic.use_t is True

    res_hc0 = OLS(X, y, vcov="HC0")
    assert res_hc0.cov_type == "HC0"
    assert res_hc0.use_t is False

    res_hc1 = OLS(X, y, vcov="HC1")
    assert res_hc1.cov_type == "HC1"
    assert res_hc1.use_t is False


def test_ols_raises_when_n_leq_k(torch_dtype):
    R, n, k = 2, 3, 3
    X = torch.randn((R, n, k), dtype=torch_dtype)
    y = torch.randn((R, n), dtype=torch_dtype)
    with pytest.raises(ValueError):
        OLS(X, y)
