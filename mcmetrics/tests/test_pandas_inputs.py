import pytest

from mcmetrics import OLS

pd = pytest.importorskip("pandas")


def test_ols_accepts_pandas_dataframe_and_series():
    n = 40
    X = pd.DataFrame({"const": [1.0] * n, "x": list(range(n))})
    y = pd.Series([0.1 * i for i in range(n)])

    res = OLS(X, y)
    assert res.params.shape[0] == 1
    assert res.k == 2
