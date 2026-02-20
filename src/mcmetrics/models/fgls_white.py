from __future__ import annotations

from mcmetrics.models.fgls import FGLS


def FGLS_white(*args, **kwargs):
    """Convenience wrapper: FGLS(method='white')."""
    kwargs = dict(kwargs)
    kwargs.setdefault("method", "white")
    return FGLS(*args, **kwargs)
