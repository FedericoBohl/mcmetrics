from __future__ import annotations

from typing import Any

import torch


def einsum(equation: str, *operands: Any) -> torch.Tensor:
    """Thin wrapper around torch.einsum."""
    return torch.einsum(equation, *operands)
