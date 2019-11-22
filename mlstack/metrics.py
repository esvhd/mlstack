import numpy as np


def ccc(x, y, ddof=1):
    """Computes concordance correlation coefficient
    """
    covar = np.cov(x, y, ddof=ddof)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=ddof), np.var(y, ddof=ddof)

    rho = 2 * covar / (var_x + var_y + np.pow(mean_x - mean_y, 2))
    return rho
