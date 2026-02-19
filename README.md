# mcmetrics

`mcmetrics` is a small library for **fast batched econometrics** designed for Monte Carlo simulations:
OLS / WLS / GLS / FGLS on tensors shaped like `(R, n, k)` without Python loops in the hot path.

Backends:
- **NumPy** (default)
- **JAX** (optional) for `jit`-compiled runs
- **PyTorch** (optional) for GPU-friendly workflows

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