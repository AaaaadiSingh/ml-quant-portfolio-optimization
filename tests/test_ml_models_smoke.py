from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.ml_models import predict_1d, train_rf, train_ridge_linreg


def test_ml_shape_and_no_nan():
    n = 300
    p = 77
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.normal(size=n))
    for factory, n_fit in [
        (train_ridge_linreg, n),
        (train_rf, 200),
    ]:
        m = factory(X.iloc[:n_fit], y.iloc[:n_fit])
        preds = predict_1d(m, X.iloc[200:])
        assert preds.shape == (100,), f"shape mismatch for {factory.__name__}: {preds.shape}"
        assert np.isfinite(preds).all(), f"non-finite preds from {factory.__name__}"
