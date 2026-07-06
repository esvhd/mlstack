"""
Time series CV tools
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit


def wide_to_long(
    df: pd.DataFrame,
    id_var: str = "index",
    var_name: str = "name",
    value_name: str = "value",
) -> pd.DataFrame:
    """Convert wide format dataframe to long format. Result dataframe is sorted
    by var_name which contain original df's column names and by original df's
    index as field names.

    Parameters
    ----------
    df : pd.DataFrame
        input wide data frame
    id_var : str, optional
        field name, by default "index", i.e. using input df's index
    var_name : str, optional
        variable name, using column names in the input df, by default "name"
    value_name : str, optional
        Value column name in the output long format df, by default "value"

    Returns
    -------
    pd.DataFrame
        long format data frame with NaNs dropped.
    """
    # reset index so date becomes a column. a named index (e.g. a
    # DatetimeIndex named "date") becomes a column under its own name,
    # not "index" - resolve the id column accordingly.
    df = df.reset_index()
    if id_var not in df.columns:
        id_var = df.columns[0]
    df = (
        df.melt(id_vars=id_var, var_name=var_name, value_name=value_name)
        # remove NaN not aligned in wide format, e.g. due to holidays
        .dropna(axis=0)
    )
    df.sort_values([var_name, id_var], inplace=True)
    return df.reset_index(drop=True)


def time_series_embargo_split(
    X_wide, embargo_size: int, n_splits: int, max_train_size=None
):
    """Generate time series CV with embargo, i.e. a gap exists between train
    and test data.

    Parameters
    ----------
    X_wide : TYPE
        wide format time series
    embargo_size : int
        embargo length
    n_splits : int
        no. of cv folds
    max_train_size : None, optional
        see sklear.TimeSeriesSplit() docs

    Yields
    ------
    tuple of indices of train and test data
    """
    assert embargo_size >= 0, "embargo_size must be non negative."

    # split with time series split
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
    base_cv = tscv.split(X_wide)

    # NOTE: this function is a generator (it contains yield), so an early
    # `return base_cv` would silently yield nothing. Always iterate.
    for train_idx, test_idx in base_cv:
        if embargo_size > 0:
            assert len(train_idx) > embargo_size
            if len(train_idx) < 2 * embargo_size:
                warnings.warn(
                    f"Training size {len(train_idx)} < 2x "
                    + f"Embargo size {embargo_size}"
                )
            # chop off the last few samples in training
            train_idx = train_idx[:-embargo_size]
        yield train_idx, test_idx


def long_format_ts_embargo_split(
    long_df: pd.DataFrame,
    time_col: str,
    embargo_size: int,
    n_splits: int,
    sort_time_col=True,
    debug=False,
):
    """Generate time series embargo split indices for a long format data frame.
    A column representing time must exist.

    Parameters
    ----------
    long_df : pd.DataFrame
        long format time series
    time_col : str
        column representing time
    embargo_size : int
        embargo length
    n_splits : int
        no. of cv folds.
    sort_time_col : bool, optional
        whether to sort by time_col, default True. Turn off if data is already
        sorted.
    debug : bool, optional


    Yields
    ------
    tuple of positional train and test indices into long_df (row numbers
    after sorting when sort_time_col=True, otherwise row numbers of the
    input frame).
    """
    # convert from long to wide format, use X to hold dates
    assert time_col in long_df.columns, f"{time_col} does NOT exist in frame."

    # avoid mutating the caller's dataframe
    long_df = long_df.copy()

    # make sure data is sorted in time order
    if sort_time_col:
        long_df.sort_values(time_col, ascending=True, inplace=True)
        long_df.reset_index(drop=True, inplace=True)

    # np.asarray: pandas .unique() may return a DatetimeArray, which has
    # no .squeeze() and does not index like an ndarray
    X = np.asarray(long_df[time_col].unique()).reshape((-1, 1))

    cv = time_series_embargo_split(X, embargo_size, n_splits=n_splits)

    # map dates onto original frame
    for train, test in cv:
        train_time, test_time = X[train].squeeze(), X[test].squeeze()

        if debug:
            print(f"train size = {len(train)}, test size = {len(test)}")
            print(f"train time tail = {train_time[-5:]}")
            print(f"test time head = {test_time[:5]}")

        train_mask = long_df[time_col].isin(train_time)
        test_mask = long_df[time_col].isin(test_time)

        # identify train / test blocks with date masks. Return POSITIONS
        # (not index labels) so downstream iloc-style CV consumers work
        # with any input index.
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)

        yield (train_idx, test_idx)


class TimeSeriesEmbargoSplit:
    def __init__(self, embargo_size: int, n_splits: int = 3):
        """Time Series Embargo CV Split class, works with sklearn.

        Parameters
        ----------
        embargo_size : int
            embargo size, cut off from the tail of training data.
        n_splits : int, optional
            number of time series splits, by default 3
        """
        assert embargo_size > -1, "Embargo size must be non-negative."

        self.embargo_size = embargo_size
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        # Start with base time series split, then take off the last few
        # from training indices (aka embargo size)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        base_cv = tscv.split(X, y, groups=groups)

        # NOTE: generator function - an early `return base_cv` would
        # silently yield nothing. Always iterate.
        for train_idx, test_idx in base_cv:
            if self.embargo_size > 0:
                assert len(train_idx) > self.embargo_size
                if len(train_idx) < 2 * self.embargo_size:
                    warnings.warn(
                        f"Training size {len(train_idx)} < 2x "
                        + f"Embargo size {self.embargo_size}"
                    )
                train_idx = train_idx[: -self.embargo_size]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        # required by sklearn's CV interface
        return self.n_splits
