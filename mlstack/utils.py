import numpy as np
import matplotlib.pyplot as plt


def diag_plot(y_true, y_pred, figsize=(5, 5), **scatter_kws):
    """Diagonal plot

    Parameters
    ----------
    y_true : [type]
        [description]
    y_pred : [type]
        [description]
    figsize : tuple, optional
        figure size, by default (5, 5)

    Returns
    -------
    [type]
        [description]
    """
    # prepare axes limits
    x_min = min(np.min(y_true), np.min(y_pred))
    x_max = max(np.max(y_true), np.max(y_pred))

    # create some buffer
    x_min -= np.abs(x_min) * 0.05
    x_max += np.abs(x_max) * 0.05
    # create 45-dgree diagonal line
    diag = np.linspace(x_min, x_max, 100)

    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(x=y_true, y=y_pred, **scatter_kws)
    ax.plot(diag, diag, ls="--", c="g", lw=1.0, alpha=0.5)
    ax.set_ylim((x_min, x_max))
    ax.set_xlim((x_min, x_max))
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    return ax
