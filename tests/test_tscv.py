import numpy as np
# import pandas as pd
import mlstack.tscv as tscv


def test_embargo_split():
    """Test if TS embargo CV class and method return the same results
    """
    X = np.random.rand(1000, 2)
    # y = np.random.rand(1000, 1)

    emb_split = tscv.TimeSeriesEmbargoSplit(embargo_size=5, n_splits=5)
    emb_cv = emb_split.split(X)

    func_cv = tscv.time_series_embargo_split(X, embargo_size=5, n_splits=5)

    for u, v in zip(emb_cv, func_cv):
        # ensure they return all the same
        print(f'u last 10 = {u[0][-10:]}')
        print(f'v last 10 = {v[0][-10:]}')
        assert(np.array_equal(u[0], v[0]))
