import numpy as np
import warnings


def sequential_xy_split(data, train_len, target_len, ban_len, debug=False):
    '''
    Splits a n-dimensional time series into X/Y datasets.

    Given a 3-D dataset, similar to a batched training data set. Each X and Y
    forms a block of sequential series, i.e. X is followed by Y without gap.

    An gap can be inserted between two (X, Y) pairs. This means that the length
    of a full X, Y, ban block is X_len + Y_len + ben_len.

    If the last chunk of the time series isn't as long as X_len + Y_len,
    a warning is raised and the last chuck is discarded.

    Parameters
    ----------
    data : numpy.ndarray
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

    # make sure residual is at least train_len + target_len
    tail_size = L % step
    if tail_size != 0 and tail_size < train_len + target_len:
        warnings.warn('Time series tail size is %d, expecting at least %d. '
                      'Last chunk is discarded.'
                      % (tail_size, train_len + target_len))
        # no need for this as samples_per_day is an integer division, therefore
        # would automatically stop before the tail chunk
        # data = data[:, :, :-tail_size]

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


def sequential_xy_split_2d(data, train_len, target_len, ban_len, debug=False):
    '''
    Create X, Y splits for sequential data from 2D input.

    Parameters
    ----------
    data : TYPE
        2D data
    train_len : TYPE

    target_len : TYPE

    ban_len : TYPE

    debug : bool, optional

    '''
    assert(data.ndim == 2), 'Expects 2D data but got %d.' % data.ndim

    D, L = data.shape
    # D: data dimension
    # L: data length

    step = train_len + target_len + ban_len
    # work out how many samples there will be
    num_samples = L // step

    # make sure residual is at least train_len + target_len
    tail_size = L % step
    if tail_size != 0 and tail_size < train_len + target_len:
        warnings.warn('Time series tail size is %d, expecting at least %d. '
                      'Last chunk is discarded.'
                      % (tail_size, train_len + target_len))
        # no need for this as num_samples is an integer division, therefore
        # would automatically stop before the tail chunk
        # data = data[:, :-tail_size]

    X = np.zeros((num_samples, D, train_len))
    y = np.zeros((num_samples, D, target_len))

    if debug:
        print('data.shape: ', data.shape)
        print('step size: ', step)
        print('Expected non-overlapping samples per day: ', num_samples)
        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)

    for i in range(num_samples):
        train_start = i * step
        train_end = train_start + train_len
        # target_start == train_end
        target_end = train_end + target_len

        if debug:
            print('Iter: %d, train_start: %d, train_end: %d, target_end: %d'
                  % (i, train_start, train_end, target_end))

        X[i] = data[:, train_start:train_end]
        y[i] = data[:, train_end:target_end]

    return (X, y)
