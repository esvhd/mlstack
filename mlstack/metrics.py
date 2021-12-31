import numpy as np
from scipy.stats import pearsonr, spearmanr


def ccc2(x, y, ddof=1):
    """Computes concordance correlation coefficient
    """
    rho, _ = pearsonr(x, y)

    var_x = np.var(x, ddof=ddof)
    var_y = np.var(y, ddof=ddof)

    top = 2 * rho * np.sqrt(var_x) * np.sqrt(var_y)
    bottom = var_x + var_y + np.power(np.mean(x) - np.mean(y), 2)

    return top / bottom


def ccc(x, y, ddof: int = 1):
    """Cordance Correlation Coefficient (CCC). Penalises correlation which
    deviate away from the 45-degree line of agreement.

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
    if x.ndim > 1:
        x = x.squeeze()
    if y.ndim > 1:
        y = y.squeeze()
    assert x.ndim == 1, f"Expecting 1-D array but got {x.shape}."
    assert y.ndim == 1, f"Expecting 1-D array but got {y.shape}."

    # returns 2x2 covariance matrix
    covar = np.cov(x, y, ddof=ddof)

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=ddof), np.var(y, ddof=ddof)

    rho = 2 * covar[0, 1] / (var_x + var_y + np.power(mean_x - mean_y, 2))

    return rho


def spearman(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = y_true.squeeze()
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    assert y_true.ndim == 1, "spearman score expects y_true.ndim == 1"
    assert y_pred.ndim == 1, "spearman score expects y_pred.ndim == 1"

    p, _ = spearmanr(y_true, y_pred)
    return p


def rmspe(y_true, y_pred) -> float:
    """Root Mean Square Percentage Error

    Parameters
    ----------
    y_true : [type]
        True labels
    y_pred : [type]
        Predicted lables

    Returns
    -------
    float
        [description]
    """
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
