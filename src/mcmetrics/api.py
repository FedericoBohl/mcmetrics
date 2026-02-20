# src/mcmetrics/api.py
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from mcmetrics.models.ols import OLS
from mcmetrics.models.wls import WLS

__all__ = ["OLS", "WLS", "__version__"]

try:
    __version__ = version("mcmetrics")
except PackageNotFoundError:  # editable/local
    __version__ = "0.0.0"