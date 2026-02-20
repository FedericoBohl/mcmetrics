from __future__ import annotations

import torch

from mcmetrics.exceptions import ShapeError


def vcov_classic(XtX_inv: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    Classic (homoskedastic) vcov.

    XtX_inv : (R,k,k)
    sigma2  : (R,) or scalar
    """
    if XtX_inv.ndim != 3 or XtX_inv.shape[-1] != XtX_inv.shape[-2]:
        raise ShapeError(f"XtX_inv must be (R,k,k). Got {tuple(XtX_inv.shape)}")

    if not isinstance(sigma2, torch.Tensor):
        sigma2 = torch.as_tensor(sigma2, device=XtX_inv.device, dtype=XtX_inv.dtype)
    else:
        sigma2 = sigma2.to(device=XtX_inv.device, dtype=XtX_inv.dtype)

    if sigma2.ndim == 0:
        return XtX_inv * sigma2
    if sigma2.ndim != 1:
        raise ShapeError(f"sigma2 must be (R,) or scalar. Got {tuple(sigma2.shape)}")

    R = XtX_inv.shape[0]
    if sigma2.shape[0] != R:
        raise ShapeError(f"sigma2 must have length R={R}. Got {int(sigma2.shape[0])}")

    return XtX_inv * sigma2.view(R, 1, 1)
