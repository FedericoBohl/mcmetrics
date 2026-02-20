from __future__ import annotations

"""Built-in Sigma estimators.

These are intentionally minimal. The project will expand this module with
additional estimators (White-type, parametric forms, etc.) as FGLS work proceeds.
"""

__all__ = [
    "IdentitySigma",
    "DiagResid2Sigma",
]

from mcmetrics.sigma.estimators.identity import IdentitySigma
from mcmetrics.sigma.estimators.diag_resid2 import DiagResid2Sigma
