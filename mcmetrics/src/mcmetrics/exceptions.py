from __future__ import annotations


class MCmetricsError(Exception):
    """Base exception for mcmetrics."""


class ShapeError(MCmetricsError, ValueError):
    """Invalid shape or dimension mismatch."""


class InvalidWeightsError(MCmetricsError, ValueError):
    """Invalid weights: non-positive, NaN/inf, or incompatible shapes."""


class SingularMatrixError(MCmetricsError, RuntimeError):
    """Matrix is singular or numerically non-invertible."""


class NotSupportedError(MCmetricsError, NotImplementedError):
    """Feature is not supported."""
