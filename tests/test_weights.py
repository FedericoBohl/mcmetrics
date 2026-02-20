import torch
import pytest

from mcmetrics.weights import as_batched_weights


def test_as_batched_weights_shapes_and_modes(torch_dtype):
    R, n = 3, 10

    # Scalar -> (R,n)
    w0, s0 = as_batched_weights(2.0, R=R, n=n, mode="precision", dtype=torch_dtype)
    assert w0.shape == (R, n)
    assert s0.shape == (R, n)
    assert torch.allclose(w0, torch.full((R, n), 2.0, dtype=torch_dtype))

    # (n,) -> (R,n)
    w1, _ = as_batched_weights(torch.arange(1, n + 1, dtype=torch_dtype), R=R, n=n, mode="precision")
    assert w1.shape == (R, n)

    # Mode conversions
    v = torch.linspace(0.5, 2.0, n, dtype=torch_dtype)  # variance
    w_from_v, _ = as_batched_weights(v, R=R, n=n, mode="variance")
    assert torch.allclose(w_from_v[0], 1.0 / v)

    sqrt_w = torch.sqrt(1.0 / v)
    w_from_sqrt_w, _ = as_batched_weights(sqrt_w, R=R, n=n, mode="sqrt_precision")
    assert torch.allclose(w_from_sqrt_w[0], 1.0 / v)

    sqrt_v = torch.sqrt(v)
    w_from_sqrt_v, _ = as_batched_weights(sqrt_v, R=R, n=n, mode="sqrt_variance")
    assert torch.allclose(w_from_sqrt_v[0], 1.0 / v)


def test_as_batched_weights_rejects_bad_shapes(torch_dtype):
    R, n = 2, 5
    with pytest.raises(ValueError):
        as_batched_weights(torch.ones(n + 1, dtype=torch_dtype), R=R, n=n)

    with pytest.raises(ValueError):
        as_batched_weights(torch.ones((R, n + 1), dtype=torch_dtype), R=R, n=n)

    with pytest.raises(ValueError):
        as_batched_weights(torch.ones((R, n, 1), dtype=torch_dtype), R=R, n=n)


def test_as_batched_weights_rejects_nonpositive(torch_dtype):
    R, n = 2, 6
    with pytest.raises(ValueError):
        as_batched_weights(torch.zeros(n, dtype=torch_dtype), R=R, n=n)

    with pytest.raises(ValueError):
        as_batched_weights(-1.0, R=R, n=n)
