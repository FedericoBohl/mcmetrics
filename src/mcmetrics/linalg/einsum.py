from __future__ import annotations

from typing import Any
import torch


def einsum(equation: str, *operands: Any) -> torch.Tensor:
    """
    Thin wrapper around torch.einsum.

    Rationale: keeps a single hook point if you later want to:
    - add shape checks
    - swap backend (JAX) in a controlled way
    """
    return torch.einsum(equation, *operands)
