import pytest
import torch

from mcmetrics.models.ols import OLS
from mcmetrics.models.wls import WLS
from mcmetrics.models.gls import GLS


def _make_spd(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A @ A.T + 0.5 * torch.eye(n)


def test_gls_equals_ols_when_sigma_identity_full():
    torch.manual_seed(123)
    R, n, k = 5, 40, 3
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    Sigma = torch.eye(n)
    res_gls = GLS(X, y, Sigma=Sigma, vcov="classic")
    res_ols = OLS(X, y, vcov="classic")

    assert torch.allclose(res_gls.params, res_ols.params, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res_gls.vcov, res_ols.vcov, atol=1e-6, rtol=1e-6)


def test_gls_diag_equals_wls_precision():
    torch.manual_seed(7)
    R, n, k = 4, 30, 2
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    # Sigma diagonal (variances)
    Sigma_diag = torch.linspace(0.5, 2.0, n)

    res_gls = GLS(X, y, Sigma=Sigma_diag, vcov="classic")
    # WLS with weights = precision = 1/Sigma_diag
    w = 1.0 / Sigma_diag
    res_wls = WLS(X, y, w=w, vcov="classic")

    assert torch.allclose(res_gls.params, res_wls.params, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res_gls.vcov, res_wls.vcov, atol=1e-6, rtol=1e-6)


def test_gls_diag_matches_full_diag_dense():
    torch.manual_seed(9)
    R, n, k = 3, 12, 3
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    Sigma_diag = torch.linspace(0.8, 1.6, n)
    Sigma_dense = torch.diag(Sigma_diag)

    res_diag = GLS(X, y, Sigma=Sigma_diag, vcov="classic")
    res_full = GLS(X, y, Sigma=Sigma_dense, vcov="classic")

    assert torch.allclose(res_diag.params, res_full.params, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res_diag.vcov, res_full.vcov, atol=1e-6, rtol=1e-6)


def test_inv_sqrt_sigma_reuses_whitener_and_matches_diag_path():
    torch.manual_seed(11)
    R, n, k = 2, 20, 2
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    Sigma_diag = torch.linspace(0.6, 1.8, n)
    inv_s = torch.rsqrt(Sigma_diag)

    res1 = GLS(X, y, Sigma=Sigma_diag, vcov="classic")
    res2 = GLS(X, y, inv_sqrt_Sigma=inv_s, Sigma_is_diagonal=True, vcov="classic")

    assert torch.allclose(res1.params, res2.params, atol=1e-6, rtol=1e-6)
    assert torch.allclose(res1.vcov, res2.vcov, atol=1e-6, rtol=1e-6)


def test_error_when_sigma_diag_not_positive():
    torch.manual_seed(13)
    R, n, k = 2, 10, 2
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    Sigma_bad = torch.ones(n)
    Sigma_bad[3] = 0.0
    with pytest.raises(ValueError):
        GLS(X, y, Sigma=Sigma_bad, Sigma_is_diagonal=True)


def test_error_when_sigma_full_not_spd():
    torch.manual_seed(17)
    R, n, k = 2, 10, 2
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    diag = torch.ones(n)
    diag[2] = -1.0
    Sigma_bad = torch.diag(diag)
    with pytest.raises(ValueError, match="SPD"):
        GLS(X, y, Sigma=Sigma_bad, vcov="classic", check_spd=True)


def test_gls_robust_hc0_reduces_to_ols_hc0_when_sigma_identity():
    torch.manual_seed(19)
    R, n, k = 3, 25, 3
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)

    Sigma = torch.eye(n)

    res_gls = GLS(X, y, Sigma=Sigma, vcov="HC0")
    res_ols = OLS(X, y, vcov="HC0")

    assert torch.allclose(res_gls.vcov, res_ols.vcov, atol=1e-6, rtol=1e-6)


def test_store_diagnostics_outputs_expected_keys():
    torch.manual_seed(23)
    R, n, k = 2, 15, 2
    X = torch.randn(R, n, k)
    y = torch.randn(R, n)
    Sigma_diag = torch.linspace(0.7, 1.3, n)

    res = GLS(X, y, Sigma=Sigma_diag, store_resid=True, store_diagnostics=True, Sigma_is_diagonal=True)
    assert hasattr(res, "diagnostics")
    assert "objective" in res.diagnostics
    assert "logdet_Sigma" in res.diagnostics
    assert "loglik_gaussian" in res.diagnostics
