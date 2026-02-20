# mcmetrics

`mcmetrics` is a small library for **fast batched econometrics** designed for Monte Carlo simulations.

The core idea is to run many replications at once on tensors shaped like `(R, n, k)` (for `X`) and `(R, n)` (for `y`) **without Python loops in the hot path**.

## What it does

- Batched linear regression on tensors: estimate many replications in parallel.
- A clean API focused on Monte Carlo workflows (bias/RMSE/coverage diagnostics live in the results object).

## Backend

The current implementation uses **PyTorch** (CPU/GPU) as the computational backend.

A NumPy/JAX backend may be added later, but it is not the focus of the current codebase.

## Features

Implemented:
- **OLS** (batched) on `(R, n, k)` / `(R, n)` (and single-sample inputs `(n, k)` / `(n,)`)
- **WLS** (batched) with weights as:
  - scalar (constant across all replications and observations)
  - `(n,)` (common across replications)
  - `(R, n)` (per replication)
- Variance-covariance estimators:
  - `"classic"` (homoskedastic)
  - `"HC0"`, `"HC1"` (White robust)

Optional / advanced (if enabled in your version):
- `"cluster"` (one-way cluster robust) and `"HAC"` (Newey–West) vcov options
- Wild bootstrap utilities under `mcmetrics.bootstrap` (not exported at top-level)

Roadmap:
- **GLS / FGLS** (future)

## Installation (development)

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

pip install -U pip
pip install -e .
```

## Quickstart (OLS)

```python
import torch
from mcmetrics import OLS

torch.manual_seed(0)

R, n, k = 1000, 200, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.0, 0.8, -0.5])
y = (X @ beta_true.view(k, 1)).squeeze(-1) + torch.randn(R, n)

res = OLS(X, y, vcov="HC1", store_y=False)
print("mean(beta_hat):", res.params.mean(dim=0))
```

## Quickstart (WLS)

### 1) Constant weight (scalar)

Passing a scalar weight applies the same weight to all replications and observations.

```python
import torch
from mcmetrics import WLS

torch.manual_seed(0)

R, n, k = 2000, 250, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.0, 0.8, -0.5])
y = (X @ beta_true.view(k, 1)).squeeze(-1) + torch.randn(R, n)

res = WLS(X, y, weights=1.0, vcov="HC1", store_y=False)
print("mean(beta_hat):", res.params.mean(dim=0))
```

### 2) Common weights across replications `(n,)`

A typical Monte Carlo setup uses fixed regressors across replications and known heteroskedasticity by observation:

```python
import torch
from mcmetrics import WLS

torch.manual_seed(123)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

R, n, k = 2000, 250, 3
beta_true = torch.tensor([1.0, 0.8, -0.5], device=device, dtype=dtype)

x1 = torch.linspace(-2.0, 2.0, n, device=device, dtype=dtype)
x2 = torch.randn(n, device=device, dtype=dtype)

X = torch.stack(
    [torch.ones((R, n), device=device, dtype=dtype),
     x1.expand(R, n),
     x2.expand(R, n)],
    dim=2,
)

# Var(e_i | X) = v_i depends on x1; correct precision weights: w_i = 1/v_i
v = 0.5 + 1.5 * (x1 ** 2)
w = 1.0 / v                      # (n,)

eps = torch.sqrt(v).expand(R, n) * torch.randn((R, n), device=device, dtype=dtype)
y = (X @ beta_true.view(k, 1)).squeeze(-1) + eps

res = WLS(X, y, weights=w, vcov="classic", store_y=False)
print("mean(beta_hat):", res.params.mean(dim=0))
```

## Advanced: Wild bootstrap (optional)

`mcmetrics` focuses on fast batched estimation and standard analytic vcov estimators. In some settings you may want
finite-sample inference based on resampling. A common robust option under heteroskedasticity is the **wild bootstrap**.

Bootstrap utilities live in `mcmetrics.bootstrap` and are intentionally **not exported** in `mcmetrics.__init__`
to keep the core API clean.

Example (bootstrap p-values for a single coefficient test):

```python
import torch
from mcmetrics import WLS
from mcmetrics.bootstrap import wild_bootstrap_pvalue_beta0

torch.manual_seed(0)

R, n, k = 500, 200, 3
X = torch.randn(R, n, k)
beta_true = torch.tensor([1.0, 0.8, -0.5])
y = (X @ beta_true.view(k, 1)).squeeze(-1) + torch.randn(R, n)

res = WLS(X, y, weights=1.0, vcov="HC1", store_y=False)

# Test H0: beta_1 = 0.8
p_boot = wild_bootstrap_pvalue_beta0(
    X, y,
    beta0=0.8,
    j=1,
    weights=1.0,      # optional; if provided, bootstrap runs on the WLS-transformed equation
    vcov="HC1",
    B=999,
    seed=123,
)

print("Mean bootstrap p-value:", p_boot.mean().item())
```

Notes:
- Bootstrap methods are more computationally expensive than analytic vcov estimators because they re-solve the model many times.
- In Monte Carlo work, bootstrap p-values can be a robustness check when you care about finite-sample size/power.

## Contributing

This is an early-stage project. If you want to contribute:
- open an issue with a minimal reproducible example
- include tensor shapes `(R,n,k)` / `(R,n)` and expected behavior