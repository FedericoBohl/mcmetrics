# src/mcmetrics/typing.py
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


ArrayLike = Union[torch.Tensor, np.ndarray, Sequence[Any]]


def _is_pandas_df(x: Any) -> bool:
    try:
        import pandas as pd  # type: ignore
        return isinstance(x, pd.DataFrame)
    except Exception:
        return False


def _is_pandas_series(x: Any) -> bool:
    try:
        import pandas as pd  # type: ignore
        return isinstance(x, pd.Series)
    except Exception:
        return False


def _require_pandas() -> None:
    try:
        import pandas as _  # noqa: F401
    except Exception as e:
        raise ImportError("pandas is required to pass DataFrame/Series inputs. Install with: pip install pandas") from e


def as_torch(
    x: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Convert common array-likes to torch.Tensor.

    Supports:
    - torch.Tensor
    - numpy.ndarray
    - Python lists/tuples (nested)
    - pandas.DataFrame / pandas.Series (if pandas installed)
    """
    if _is_pandas_df(x) or _is_pandas_series(x):
        _require_pandas()
        x = x.to_numpy()  # type: ignore[attr-defined]

    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t

    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def as_batched_xy(
    X: Any,
    y: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[list[str]]]:
    """
    Standardize inputs to:
      X: (R,n,k)
      y: (R,n)

    Accept:
      X: (n,k) or (R,n,k)
      y: (n,) or (R,n)

    Extra feature:
      If X is a pandas.DataFrame and it is a single-sample (2D) input,
      return param_names = list(X.columns). Otherwise param_names=None.
    """
    param_names: Optional[list[str]] = None

    if _is_pandas_df(X):
        _require_pandas()
        if hasattr(X, "columns"):
            param_names = [str(c) for c in X.columns]  # type: ignore[attr-defined]
        X = X.to_numpy()  # type: ignore[attr-defined]

    if _is_pandas_series(y):
        _require_pandas()
        y = y.to_numpy()  # type: ignore[attr-defined]
    elif _is_pandas_df(y):
        _require_pandas()
        # allow a single-column DataFrame as y
        y_np = y.to_numpy()  # type: ignore[attr-defined]
        if y_np.ndim != 2 or y_np.shape[1] != 1:
            raise ValueError("If y is a DataFrame, it must have exactly one column.")
        y = y_np[:, 0]

    Xt = as_torch(X, dtype=dtype, device=device)
    yt = as_torch(y, dtype=dtype, device=device)

    if Xt.ndim == 2:
        Xt = Xt.unsqueeze(0)
    if yt.ndim == 1:
        yt = yt.unsqueeze(0)

    if Xt.ndim != 3:
        raise ValueError(f"X must be (R,n,k) or (n,k). Got {tuple(Xt.shape)}")
    if yt.ndim != 2:
        raise ValueError(f"y must be (R,n) or (n,). Got {tuple(yt.shape)}")
    if Xt.shape[0] != yt.shape[0] or Xt.shape[1] != yt.shape[1]:
        raise ValueError(f"Batch/obs dims mismatch: X {tuple(Xt.shape)}, y {tuple(yt.shape)}")

    # If user passed batched X (3D) we do not trust names even if they came from a DF
    if Xt.shape[0] != 1:
        param_names = None

    return Xt, yt, param_names