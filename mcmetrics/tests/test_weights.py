import torch

from mcmetrics.weights import as_batched_weights


def test_as_batched_weights_shapes(torch_dtype):
    R, n = 3, 10

    w0, _ = as_batched_weights(2.0, R=R, n=n, mode="precision")
    assert w0.shape == (R, n)

    w1, _ = as_batched_weights(torch.arange(1, n + 1, dtype=torch_dtype), R=R, n=n, mode="precision")
    assert w1.shape == (R, n)

    w2, _ = as_batched_weights(torch.ones((R, n), dtype=torch_dtype), R=R, n=n, mode="precision")
    assert w2.shape == (R, n)
