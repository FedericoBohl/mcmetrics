from __future__ import annotations

from typing import Literal, Optional, Union
import warnings
import torch

from mcmetrics.exceptions import InvalidWeightsError, ShapeError

WeightsMode = Literal["precision", "variance", "sqrt_precision", "sqrt_variance"]


def as_batched_weights(
    weights,
    *,
    R: int,
    n: int,
    mode: WeightsMode = "precision",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    check: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Coerce weights to (R,n) precision weights and return (w, sqrt_w).

    Accepted inputs:
      - scalar
      - (n,)
      - (R,n)

    mode:
      - "precision"      : w = 1/Var(u_i)
      - "variance"       : v = Var(u_i) (converted to w=1/v)
      - "sqrt_precision" : s = sqrt(w)
      - "sqrt_variance"  : s = sqrt(v) (converted to w=1/s^2)
    """
    if isinstance(weights, torch.Tensor):
        w = weights
    else:
        w = torch.as_tensor(weights)

    if dtype is not None:
        w = w.to(dtype=dtype)
    if device is not None:
        w = w.to(device=device)

    # Shape to (R,n)
    if w.ndim == 0:
        w = w.view(1, 1).expand(R, n)
    elif w.ndim == 1:
        if int(w.shape[0]) != n:
            raise ShapeError(f"weights has shape {tuple(w.shape)} but n={n}")
        w = w.view(1, n).expand(R, n)
    elif w.ndim == 2:
        if tuple(w.shape) != (R, n):
            raise ShapeError(f"weights must be (R,n)={(R,n)}. Got {tuple(w.shape)}")
    else:
        raise ShapeError(f"weights must be scalar, (n,), or (R,n). Got {tuple(w.shape)}")

    # Convert to precision weights
    if mode == "precision":
        w_prec = w
    elif mode == "variance":
        w_prec = 1.0 / w
    elif mode == "sqrt_precision":
        w_prec = w * w
    elif mode == "sqrt_variance":
        w_prec = 1.0 / (w * w)
    else:
        raise ShapeError("mode must be one of {'precision','variance','sqrt_precision','sqrt_variance'}")

    if check:
        if not torch.isfinite(w_prec).all():
            raise InvalidWeightsError("weights contain inf/nan after coercion")
        if (w_prec <= 0).any():
            raise InvalidWeightsError("weights must be strictly positive")

        wmin = float(w_prec.min().detach().cpu().item())
        wmax = float(w_prec.max().detach().cpu().item())
        if wmin <= 0:
            raise InvalidWeightsError("weights must be strictly positive")

        ratio = wmax / wmin
        if ratio > 1e8:
            warnings.warn(
                f"Very large weight ratio max/min = {ratio:.2e}. This can cause numerical issues.",
                RuntimeWarning,
                stacklevel=2,
            )

    sqrt_w = torch.sqrt(w_prec)
    return w_prec, sqrt_w
