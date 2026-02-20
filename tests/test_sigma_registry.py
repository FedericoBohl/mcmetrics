import pytest

from mcmetrics.sigma import get_sigma_estimator, list_sigma_estimators


def test_sigma_registry_has_builtin_estimators():
    methods = list_sigma_estimators()
    assert "identity" in methods
    assert "diag_resid2" in methods


def test_get_sigma_estimator_returns_class():
    cls = get_sigma_estimator("identity")
    est = cls()
    assert hasattr(est, "fit")


def test_get_sigma_estimator_unknown_raises():
    with pytest.raises(Exception):
        get_sigma_estimator("does_not_exist")
