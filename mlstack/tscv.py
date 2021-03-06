"""
Time series CV tools
"""
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
    df = (
        df
        # reset index so date becomes a column
        .reset_index().melt(
            id_vars=id_var, var_name=var_name, value_name="value"
        )
        # remove NaN not aligned in wide format, e.g. due to holidays
        .dropna(axis=0)
    )
    df.sort_values([var_name, id_var], inplace=True)
    # reindex
    df.index = [x for x in range(len(df))]
    return df


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
    assert embargo_size >= 0, "embargo_size must be non negetive."

    # split with time series split
    tscv = TimeSeriesSplit(n_splits, max_train_size=max_train_size)
    base_cv = tscv.split(X_wide)

    if embargo_size < 1:
        return base_cv

    # for each pair, chop off the last few samples in training
    for train_idx, test_idx in base_cv:
        assert len(train_idx) > embargo_size
        if len(train_idx) < 2 * embargo_size:
            warnings.warn(
                f"Training size {len(train_idx)} < 2x "
                + f"Embargo size {embargo_size}"
            )
        if embargo_size > 0:
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
    tuple of train and test indices
    """
    # convert from long to wide format, use X to hold dates
    assert time_col in long_df.columns, f"{time_col} does NOT exist in frame."

    # make sure data is sorted in time order
    if sort_time_col:
        long_df.sort_values(time_col, ascending=True, inplace=True)
        long_df.index = [x for x in range(len(long_df))]

    # X is np.ndarray
    X = long_df[time_col].unique().reshape((-1, 1))

    # pdb.set_trace()

    # 1st way
    # tscv = TimeSeriesEmbargoSplit(embargo_size=10, n_splits=n_splits)
    # cv = tscv.split(X)

    # 2nd way, one more iter loop so may be a bit slower.
    cv = time_series_embargo_split(X, embargo_size, n_splits=n_splits)

    # map dates onto original frame
    for idx in cv:
        train, test = idx
        train_time, test_time = X[train].squeeze(), X[test].squeeze()

        if debug:
            print(f"train size = {len(train)}, test size = {len(test)}")
            print(f"train time tail = {train_time[-5:]}")
            print(f"test time head = {test_time[:5]}")

        # if i don't do .values below, the reuslts are all False.
        # this is because numpy and pandas use different datetime types...
        # train_mask = (x in train_time for x in long_df[time_col].values)
        # test_mask = (x in test_time for x in long_df[time_col].values)
        train_mask = long_df[time_col].isin(train_time)
        test_mask = long_df[time_col].isin(test_time)

        # identify train / test blocks with date masks, return their indices
        train_idx = long_df.loc[train_mask].index
        test_idx = long_df.loc[test_mask].index

        # pdb.set_trace()

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

        if self.embargo_size < 1:
            return base_cv

        for train_idx, test_idx in base_cv:
            assert len(train_idx) > self.embargo_size
            if len(train_idx) < 2 * self.embargo_size:
                warnings.warn(
                    f"Training size {len(train_idx)} < 2x "
                    + f"Embargo size {self.embargo_size}"
                )
            if self.embargo_size > 0:
                train_idx = train_idx[: -self.embargo_size]
            yield train_idx, test_idx
