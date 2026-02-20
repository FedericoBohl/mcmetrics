from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

from mcmetrics.exceptions import NotSupportedError


class BaseSigmaEstimator:
    """Base class for Sigma estimators used by FGLS.

    Estimators should implement `fit(...)` and return a SigmaSpec.
    """

    name: str = "base"

    def fit(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


_REGISTRY: dict[str, Type[BaseSigmaEstimator]] = {}


def register_sigma_estimator(name: str) -> Callable[[Type[BaseSigmaEstimator]], Type[BaseSigmaEstimator]]:
    """Decorator to register a Sigma estimator class."""

    def deco(cls: Type[BaseSigmaEstimator]) -> Type[BaseSigmaEstimator]:
        key = str(name).strip().lower()
        cls.name = key
        _REGISTRY[key] = cls
        return cls

    return deco


def get_sigma_estimator(name: str) -> Type[BaseSigmaEstimator]:
    """Return the estimator class registered under `name`."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise NotSupportedError(
            f"Unknown Sigma estimation method '{name}'. "
            f"Available: {', '.join(sorted(_REGISTRY.keys())) or '(none)'}"
        )
    return _REGISTRY[key]


def list_sigma_estimators() -> list[str]:
    """List available Sigma estimation methods."""
    return sorted(_REGISTRY.keys())


# Register built-in estimators (minimal set; expands with FGLS work)
from mcmetrics.sigma.estimators.identity import IdentitySigma  # noqa: E402,F401
from mcmetrics.sigma.estimators.diag_resid2 import DiagResid2Sigma  # noqa: E402,F401
