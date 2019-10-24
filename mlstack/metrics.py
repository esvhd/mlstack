import numpy as np
from scipy.stats import pearsonr


def ccc(x, y, ddof: int = 1):
    """Cordance Correlation Coefficient (CCC). Penalises correlation which deviate
    away from the 45-degree line of agreement.

    Parameters
    ----------
    x : [type]
        [description]
    y : [type]
        [description]
    ddof : int, optional
        CCC is not immune to baised vs un-biased measures of variance.
        Lin's paper used 1/N, whereas some others used 1/(N - 1)

    Returns
    -------
    float
        [description]
    """
    assert x.ndim == 1
    assert y.ndim == 1

    rho, _ = pearsonr(x, y)

    var_x = np.var(x, ddof=ddof)
    var_y = np.var(y, ddof=ddof)

    top = 2 * rho * np.sqrt(var_x) * np.sqrt(var_y)
    bottom = var_x + var_y + np.pow(np.mean(x) - np.mean(y), 2)

    return top / bottom
