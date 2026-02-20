from __future__ import annotations

from typing import Callable

import torch

from mcmetrics.exceptions import NotSupportedError
from mcmetrics.sigma.estimators import GreeneDummiesDiagSigma, SigmaEstimator, WhiteDiagSigma

_REGISTRY: dict[str, Callable[..., SigmaEstimator]] = {}


def register_sigma_estimator(name: str, factory: Callable[..., SigmaEstimator]) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Estimator name must be non-empty")
    _REGISTRY[key] = factory


def list_sigma_estimators() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_sigma_estimator(name: str) -> Callable[..., SigmaEstimator]:
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise NotSupportedError(
            f"Unknown sigma estimator {name!r}. Available: {', '.join(list_sigma_estimators())}"
        )
    return _REGISTRY[key]


# -----------------------------------------------------------------------------
# Built-ins
# -----------------------------------------------------------------------------


def _white_factory(
    *,
    has_const: bool = True,
    min_var: float = 1e-12,
    solve_method: str = "cholesky",
) -> SigmaEstimator:
    return WhiteDiagSigma(has_const=has_const, min_var=min_var, solve_method=solve_method)


def _greene_factory(groups: torch.Tensor, *, min_var: float = 1e-12) -> SigmaEstimator:
    return GreeneDummiesDiagSigma(groups=groups, min_var=min_var)


register_sigma_estimator("white", _white_factory)
register_sigma_estimator("greene_dummies", _greene_factory)
