import pytest
import torch


@pytest.fixture(scope="session", params=[torch.float32, torch.float64])
def torch_dtype(request):
    return request.param
