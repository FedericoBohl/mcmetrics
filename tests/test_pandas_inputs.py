\
import pytest

pd = pytest.importorskip("pandas")

import torch
from mcmetrics import OLS


def test_param_names_from_dataframe():
    n, k = 12, 3
    X = pd.DataFrame({"const": [1.0] * n, "x1": list(range(n)), "x2": [0.5] * n})
    y = pd.Series([1.0] * n)

    res = OLS(X, y)

    assert res.param_names == ["const", "x1", "x2"]
    assert res.params.shape == (1, k)
