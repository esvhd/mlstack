import numpy as np


def sequential_x_y_split(data, train_len, target_len, ban_len, debug=False):
    '''
    Splits a n-dimensional time series into X/Y datasets.

    Given a 3-D dataset, similar to a batched training data set.

    Parameters
    ----------
    data : TYPE
        3-D time series data, shape of (N, D, L) where:
        N: number of days / blocks
        D: no. of features / dimensions
        L: sequence length for all dimensions
    train_len : int
        training sequence length
    target_len : int
        target sequence length
    ban_len : int
        A buffer zone between two consecutive samples. This is introduced to
        try to break auto-correlation between neighbouring sample sets.
        Autocorrelation is a prominent feature of financial time series,
        especially price series.
    debug : bool, optional
        Default False.

    Returns
    -------
    Tuple, (X, y).
    X.sahpe: (N, K, D, train_len)
    y.shape: (N, K, D, target_len)

    Where K is the number of samples per day / block, e.g. for the same day,
    there may be K samples of (D, train_len) time series.

    '''
    assert(data.ndim == 3), 'Expects data.ndims == 3, (N, dims, samples)'
    N, D, L = data.shape
    # N: number of days
    # D: no. of features / dimensions
    # L: sequence length for all dimensions

    step = train_len + target_len + ban_len
    # work out how many samples there will be
    samples_per_day = L // step

    X = np.zeros((N, samples_per_day, D, train_len))
    y = np.zeros((N, samples_per_day, D, target_len))

    if debug:
        print('data.shape: ', data.shape)
        print('step size: ', step)
        print('Expected non-overlapping samples per day: ', samples_per_day)
        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)

    for i in range(samples_per_day):
        train_start = i * step
        train_end = train_start + train_len
        # target_start == train_end
        target_end = train_end + target_len

        if debug:
            print('Iter: %d, train_start: %d, train_end: %d, target_end: %d'
                  % (i, train_start, train_end, target_end))

        X[:, i, :] = data[:, :, train_start:train_end]
        y[:, i, :] = data[:, :, train_end:target_end]

    return (X, y)
