from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import torch

from mcmetrics.exceptions import ShapeError
from mcmetrics.typing import as_torch


SigmaKind = Literal["diag", "full", "chol"]


@dataclass(frozen=True)
class SigmaSpec:
    """Structured representation of a covariance matrix Sigma.

    Rationale
    ---------
    Users (and estimators) can represent Sigma in multiple ways:

      - diagonal variances (heteroskedasticity): Sigma = diag(s2)
      - full SPD matrix: Sigma
      - Cholesky factor of Sigma: chol

    SigmaSpec acts as a normalized container so GLS and future FGLS can share
    the same whitening pipeline.

    Notes
    -----
    - For kind='diag', `diag` must contain variances (Sigma_ii), not std dev.
    - For kind='chol', `chol` must satisfy Sigma = L L' (if chol_upper=False)
      or Sigma = U' U (if chol_upper=True).
    """

    kind: SigmaKind

    # diagonal representation (variances)
    diag: Optional[Any] = None
    inv_sqrt: Optional[Any] = None  # optional: 1/sqrt(diag)

    # full representation
    full: Optional[Any] = None

    # factor representation
    chol: Optional[Any] = None
    chol_upper: bool = False

    # bookkeeping for FGLS (method name + options)
    method: str = "known"
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Lightweight dict for metadata."""
        return {
            "kind": self.kind,
            "chol_upper": bool(self.chol_upper),
            "method": self.method,
            "options": dict(self.options) if isinstance(self.options, dict) else {},
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @staticmethod
    def diagonal(
        diag: Any = None,
        *,
        inv_sqrt: Any | None = None,
        method: str = "known",
        options: Optional[dict[str, Any]] = None,
    ) -> "SigmaSpec":
        return SigmaSpec(
            kind="diag",
            diag=diag,
            inv_sqrt=inv_sqrt,
            method=method,
            options={} if options is None else dict(options),
        )

    @staticmethod
    def full_matrix(
        Sigma: Any,
        *,
        method: str = "known",
        options: Optional[dict[str, Any]] = None,
    ) -> "SigmaSpec":
        return SigmaSpec(
            kind="full",
            full=Sigma,
            method=method,
            options={} if options is None else dict(options),
        )

    @staticmethod
    def cholesky(
        chol: Any,
        *,
        chol_upper: bool = False,
        method: str = "known",
        options: Optional[dict[str, Any]] = None,
    ) -> "SigmaSpec":
        return SigmaSpec(
            kind="chol",
            chol=chol,
            chol_upper=bool(chol_upper),
            method=method,
            options={} if options is None else dict(options),
        )


def coerce_sigma_spec(
    Sigma: Any,
    *,
    chol_Sigma: Any | None = None,
    inv_sqrt_Sigma: Any | None = None,
    Sigma_is_diagonal: bool | None = None,
    chol_upper: bool = False,
    method: str = "known",
) -> SigmaSpec:
    """Coerce user inputs to a SigmaSpec.

    This keeps backward compatibility with GLS(X,y,Sigma=..., chol_Sigma=..., ...).
    """

    if isinstance(Sigma, SigmaSpec):
        return Sigma

    if chol_Sigma is not None:
        return SigmaSpec.cholesky(chol_Sigma, chol_upper=chol_upper, method=method)

    if inv_sqrt_Sigma is not None:
        return SigmaSpec.diagonal(diag=None, inv_sqrt=inv_sqrt_Sigma, method=method)

    if Sigma is None:
        raise ShapeError("Provide Sigma (diagonal or full) or chol_Sigma / inv_sqrt_Sigma.")

    # Infer diagonal vs full if needed
    if Sigma_is_diagonal is not None:
        if Sigma_is_diagonal:
            return SigmaSpec.diagonal(diag=Sigma, method=method)
        return SigmaSpec.full_matrix(Sigma, method=method)

    # Heuristic inference by shape
    S0 = as_torch(Sigma)
    if S0.ndim in {0, 1}:
        return SigmaSpec.diagonal(diag=Sigma, method=method)
    if S0.ndim == 2:
        # If square, interpret as full; otherwise interpret as diagonal (R,n)
        if S0.shape[0] == S0.shape[1]:
            return SigmaSpec.full_matrix(Sigma, method=method)
        return SigmaSpec.diagonal(diag=Sigma, method=method)
    if S0.ndim == 3:
        return SigmaSpec.full_matrix(Sigma, method=method)

    raise ShapeError(f"Sigma must be scalar/1D/2D/3D. Got {S0.ndim}D.")
