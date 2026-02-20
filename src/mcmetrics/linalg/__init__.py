\
"""
mcmetrics.linalg

Low-level linear algebra utilities for batched econometrics.

Conventions
-----------
- Replication axis: R
- Observation axis: n
- Parameter axis: k

Shapes
------
- X: (R, n, k)
- y: (R, n)
"""
from .solve import SolveMethod, solve_ls
from .chol import safe_cholesky, chol_inverse, chol_solve
from .einsum import einsum

__all__ = [
    "SolveMethod",
    "solve_ls",
    "safe_cholesky",
    "chol_inverse",
    "chol_solve",
    "einsum",
]
