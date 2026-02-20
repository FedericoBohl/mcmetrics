# src/mcmetrics/plotting.py
from __future__ import annotations


def apply_latex_style(*, use_tex: bool = False) -> None:
    """
    Apply a LaTeX-like matplotlib style.

    - use_tex=False uses mathtext + serif fonts (no LaTeX installation required).
    - use_tex=True enables full LaTeX rendering (requires a LaTeX installation).
    """
    import matplotlib as mpl

    rc = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 175,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "text.usetex": bool(use_tex),
    }

    mpl.rcParams.update(rc)