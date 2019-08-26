import numpy as np
from scipy.stats import kstest

import mlstack.sklearn as mls


def test_log_uniform_draw():
    a = 1e-3
    b = 1e3
    size = 1000
    ns = mls.log_uniform_draw(size, a=a, b=b)

    # KS test null hypothesis is the data come from the specified distribution
    # therefore, failing to reject the null means confirms that the null
    # is true.
    ks = kstest(
        rvs=np.log(ns), cdf="uniform", args=(np.log(a), np.log(b / a)), N=size
    )

    # assert that the null cannot be rejected, i.e. data is uniform.
    print(f"K-S test p-value: {ks.pvalue:.5e}")
    # assert failing to reject null
    assert ks.pvalue > 0.05
