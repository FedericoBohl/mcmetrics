\
import torch
import pytest

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
    # Same numerical weights provided as scalar / (n,) / (R,n) should yield identical results.
    R, n, k = 3, 30, 4
    X, _, y = _make_xy(R, n, k, torch_dtype)

    w_scalar = 2.0
    w_vec = torch.full((n,), 2.0, dtype=torch_dtype)
    w_mat = torch.full((R, n), 2.0, dtype=torch_dtype)

    res_s = WLS(X, y, weights=w_scalar, vcov="classic")
    res_v = WLS(X, y, weights=w_vec, vcov="classic")
    res_m = WLS(X, y, weights=w_mat, vcov="classic")

    assert torch.max(torch.abs(res_s.params - res_v.params)).item() < 1e-10
    assert torch.max(torch.abs(res_s.params - res_m.params)).item() < 1e-10
    assert torch.max(torch.abs(res_s.sigma2 - res_v.sigma2)).item() < 1e-10
    assert torch.max(torch.abs(res_s.sigma2 - res_m.sigma2)).item() < 1e-10


def test_wls_weights_mode_equivalence(torch_dtype):
    R, n, k = 2, 25, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)

    # Construct a non-trivial precision vector w>0 and its corresponding variance v=1/w.
    w = 1.0 + torch.linspace(0.2, 1.0, n, dtype=torch_dtype)
    v = 1.0 / w
    sqrt_w = torch.sqrt(w)
    sqrt_v = torch.sqrt(v)

    res_prec = WLS(X, y, weights=w, weights_mode="precision", vcov="classic")
    res_var = WLS(X, y, weights=v, weights_mode="variance", vcov="classic")
    res_sqrt_prec = WLS(X, y, weights=sqrt_w, weights_mode="sqrt_precision", vcov="classic")
    res_sqrt_var = WLS(X, y, weights=sqrt_v, weights_mode="sqrt_variance", vcov="classic")

    assert torch.max(torch.abs(res_prec.params - res_var.params)).item() < 1e-10
    assert torch.max(torch.abs(res_prec.params - res_sqrt_prec.params)).item() < 1e-10
    assert torch.max(torch.abs(res_prec.params - res_sqrt_var.params)).item() < 1e-10

    assert torch.max(torch.abs(res_prec.sigma2 - res_var.sigma2)).item() < 1e-10
    assert torch.max(torch.abs(res_prec.sigma2 - res_sqrt_prec.sigma2)).item() < 1e-10
    assert torch.max(torch.abs(res_prec.sigma2 - res_sqrt_var.sigma2)).item() < 1e-10


def test_wls_vcov_defaults_use_t(torch_dtype):
    R, n, k = 2, 30, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)
    w = 1.0 + torch.linspace(0.1, 0.3, n, dtype=torch_dtype)

    res_classic = WLS(X, y, weights=w, vcov="classic")
    assert res_classic.cov_type == "nonrobust"
    assert res_classic.use_t is True

    res_hc0 = WLS(X, y, weights=w, vcov="HC0")
    assert res_hc0.cov_type == "HC0"
    assert res_hc0.use_t is False

    res_hc1 = WLS(X, y, weights=w, vcov="HC1")
    assert res_hc1.cov_type == "HC1"
    assert res_hc1.use_t is False


def test_wls_cluster_requires_clusters(torch_dtype):
    R, n, k = 2, 20, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)
    w = 1.0

    with pytest.raises(ValueError):
        WLS(X, y, weights=w, vcov="cluster", clusters=None)


def test_wls_cluster_runs(torch_dtype):
    R, n, k = 2, 24, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)
    w = 1.0 + torch.linspace(0.1, 0.3, n, dtype=torch_dtype)

    # Two clusters (0/1), common across replications
    clusters = torch.tensor([0] * (n // 2) + [1] * (n - n // 2))

    res = WLS(X, y, weights=w, vcov="cluster", clusters=clusters, cluster_correction="CR1")

    assert res.vcov.shape == (R, k, k)
    assert res.cov_type == "cluster"
    assert res.use_t is False


def test_wls_hac_runs(torch_dtype):
    R, n, k = 2, 30, 3
    X, _, y = _make_xy(R, n, k, torch_dtype)
    w = 1.0 + torch.linspace(0.1, 0.3, n, dtype=torch_dtype)

    res = WLS(X, y, weights=w, vcov="HAC", hac_max_lags=1, hac_kernel="bartlett")

    assert res.vcov.shape == (R, k, k)
    assert res.cov_type == "HAC"
    assert res.use_t is False


def test_wls_raises_when_n_leq_k(torch_dtype):
    R, n, k = 2, 3, 3
    X = torch.randn((R, n, k), dtype=torch_dtype)
    y = torch.randn((R, n), dtype=torch_dtype)
    with pytest.raises(ValueError):
        WLS(X, y, weights=1.0)
