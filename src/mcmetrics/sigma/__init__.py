from __future__ import annotations

"""Sigma estimation and representation.

This subpackage is designed to support FGLS workflows where the user chooses
how to estimate the covariance matrix Sigma. The core concept is SigmaSpec,
which represents Sigma in a structured way (diagonal/full/cholesky) and can be
used consistently by GLS and (future) FGLS.
"""

from mcmetrics.sigma.spec import SigmaSpec
from mcmetrics.sigma.registry import (
    BaseSigmaEstimator,
    get_sigma_estimator,
    list_sigma_estimators,
    register_sigma_estimator,
)

__all__ = [
    "SigmaSpec",
    "BaseSigmaEstimator",
    "register_sigma_estimator",
    "get_sigma_estimator",
    "list_sigma_estimators",
]
