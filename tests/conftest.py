import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(params=[torch.float32, torch.float64])
def torch_dtype(request):
    """Run numeric tests under both float32 and float64."""
    return request.param
