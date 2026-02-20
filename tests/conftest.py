import torch
import pytest

@pytest.fixture(scope="session")
def torch_dtype():
    # Use float64 in tests for numerical stability.
    return torch.float64

def make_design(R: int, n: int, k: int, *, seed: int = 123, dtype=torch.float64):
    """
    Deterministic-ish design generator with full column rank (almost surely).
    Returns:
      X : (R,n,k) with a constant in column 0
      beta_true : (k,)
      y : (R,n)
    """
    g = torch.Generator().manual_seed(seed)
    # Column 0: constant
    X = torch.empty((R, n, k), dtype=dtype)
    X[:, :, 0] = 1.0
    if k > 1:
        X[:, :, 1:] = torch.randn((R, n, k - 1), generator=g, dtype=dtype)

    beta_true = torch.arange(1, k + 1, dtype=dtype)
    beta_true = beta_true / beta_true.abs().sum()  # keep magnitudes modest

    # Noise per replication
    eps = 0.1 * torch.randn((R, n), generator=g, dtype=dtype)
    y = (X @ beta_true.view(1, k, 1)).squeeze(-1) + eps
    return X, beta_true, y
