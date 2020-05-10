import time
import numpy as np
import pandas as pd
import joblib
import tqdm
import lightgbm as lgb

from typing import List, Optional, Dict, Any, AnyStr, Union, Tuple
from scipy.stats import rv_continuous
from scipy.stats import pearsonr, spearmanr

import sklearn.preprocessing as skp
from sklearn.linear_model import Ridge
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import brier_score_loss, log_loss, make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, roc_auc_score, r2_score

from sklearn.model_selection import (
    RandomizedSearchCV,
    BaseCrossValidator,
    train_test_split,
)
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from .metrics import ccc, spearman
from .utils import diag_plot


mcc_scorer = make_scorer(mcc)

brier_scorer = make_scorer(
    brier_score_loss, greater_is_better=False, needs_proba=True
)

nll_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

mae_scorer = make_scorer(
    mean_absolute_error, greater_is_better=False, needs_proba=False
)

mse_scorer = make_scorer(
    mean_squared_error, greater_is_better=False, needs_proba=False
)

ccc_scorer = make_scorer(ccc, greater_is_better=True, needs_proba=False)

spearman_scorer = make_scorer(
    spearman, greater_is_better=True, needs_proba=False
)


def score_reg(
    y_truth,
    y_pred,
    debug=False,
    plot=False,
    nobs: int = None,
    p: int = None,
    verbose: bool = True,
) -> Dict:
    """Run all commonly used regression metrics.

    Parameters
    ----------
    y_truth : [type]
        [description]
    y_pred : [type]
        [description]
    debug : bool, optional
        [description], by default True
    plot : bool, optional
        [description], by default False
    nobs : int, optional
        sample size, by default None. If both nobs and p are given the adjusted
        R2 can be computed.
    p : int, optional
        no. of features, by default None.

    Returns
    -------
    Dict
        [description]
    """

    if y_truth.ndim > 1:
        y_truth = y_truth.squeeze()
    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()

    if debug:
        print(f"Shapes: {y_truth.shape} vs {y_pred.shape}")

    r2 = r2_score(y_truth, y_pred)
    if nobs is not None and p is not None and nobs > 0 and p > 1:
        # compute adjusted R-squared
        r2 = 1 - (1 - r2) * (nobs - 1) / (nobs - p - 1)
        if verbose:
            print(f"Adjusted R^2 score: {r2:.3f}")
    elif verbose:
        print(f"R^2 score: {r2:.3f}")

    corr, pval = pearsonr(y_truth, y_pred)

    con_corr = ccc(y_truth, y_pred, ddof=1)

    rho, pval2 = spearmanr(y_truth, y_pred)

    mae = mean_absolute_error(y_truth, y_pred)

    mse = mean_squared_error(y_truth, y_pred)

    if verbose:
        # print(f"Pearson correlation {corr[0]:.5f}, p-value = {pval[0]:.5e}")
        print(f"Pearson correlation {corr:.5f}, p-value = {pval:.5e}")
        # print(f"Pearson correlation {corr}, p-value = {pval}")
        print(f"Spearman correlation {rho:.5f}, p-value = {pval:.5e}")
        print(f"Concordance correlation {con_corr:.5f}")
        print(f"MAE: {mae:.5e}")
        print(f"MSE: {mse:.5e}")

    out = {
        "r2": r2,
        "pearsonr": corr,
        "pearsonr_pval": pval,
        "ccc": con_corr,
        "spearmanr": rho,
        "spearmanr_pval": pval2,
        "mae": mae,
        "mse": mse,
    }

    if plot:
        # produce diagnal plot, choose max of 500 samples
        N = np.min([500, len(y_truth)])
        idx = np.random.choice(range(0, len(y_truth)), size=N, replace=False)
        print(f"Randomly choose {len(idx)} samples to plot...")
        if isinstance(y_truth, np.ndarray):
            x_sam = y_truth[idx]
        else:
            # assume pandas
            x_sam = y_truth.iloc[idx]
        if isinstance(y_pred, np.ndarray):
            y_sam = y_pred[idx]
        else:
            # assume pandas
            y_sam = y_pred.iloc[idx]
        ax = diag_plot(x_sam, y_sam, alpha=0.2, marker="x")
        ax.set_ylabel("y_pred")
        ax.set_xlabel("y_truth")

    return out


def get_regression_scorers() -> Dict:
    scorers = {
        "mae": mae_scorer,
        "mse": mse_scorer,
        "r2": make_scorer(r2_score, greater_is_better=True, needs_proba=False),
        "ccc": ccc_scorer,
        "spearman": spearman_scorer,
    }
    return scorers


def reg_baseline(
    model,
    X,
    y,
    X_test=None,
    y_test=None,
    normalize: bool = True,
    plot: bool = True,
    permute_imp: bool = False,
    verbose: bool = True,
) -> Tuple:
    """Baseline evaluation scores.

    Parameters
    ----------
    model : [type]
        [description]
    X : [type]
        [description]
    y : [type]
        [description]
    X_test : [type], optional
        [description], by default None
    y_test : [type], optional
        [description], by default None
    normalize : bool, optional
        [description], by default True
    plot : bool, optional
        [description], by default True

    Returns
    -------
    Tuple
        Tuple of two: (model, scores_dict)
        If permute_imp set to True, returns (model, scores_dict, imp_scores)
    """
    if normalize:
        scaler = skp.StandardScaler()
        X = scaler.fit_transform(X)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    model.fit(X, y)
    if X_test is not None and y_test is not None:
        # use test set for eval
        if verbose:
            print("Test set eval:")
        y_pred = model.predict(X_test)
    else:
        if verbose:
            print("Training set eval:")
        y_pred = model.predict(X)
        # use training set as test set
        X_test = X
        y_test = y

    scores = score_reg(y_test, y_pred, plot=plot, verbose=verbose)
    if permute_imp:
        imps, _ = permute_imp_sk(model, X_test, y_test)
        return model, scores, imps
    else:
        return model, scores


def reg_baseline_ridge(
    alpha,
    X,
    y,
    X_test=None,
    y_test=None,
    fit_intercept: bool = True,
    normalize: bool = True,
    plot: bool = True,
    permute_imp: bool = False,
    verbose: bool = True,
) -> Tuple:
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize)
    out = reg_baseline(
        model,
        X,
        y,
        X_test,
        y_test,
        normalize,
        plot,
        permute_imp=permute_imp,
        verbose=verbose,
    )
    return out


def reg_baseline_cv(
    X_train,
    y_train,
    model=None,
    cv=None,
    params_dict=None,
    X_test=None,
    y_test=None,
    refit="spearman",
    plot=True,
    **srch_kws,
):
    if model is None:
        print("Using default pipeline: standard scaler + ridge")
        model = make_pipeline(skp.StandardScaler(), Ridge())
        params_dict = {"ridge__alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0]}

    scoring = get_regression_scorers()

    # TODO: add case for no randomized cv
    if params_dict is not None:
        srch = RandomizedSearchCV(
            model, params_dict, cv=cv, scoring=scoring, refit=refit, **srch_kws
        )
        srch.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
        srch = model
    y_in = srch.predict(X_train)
    print("Training set performance:")
    score_reg(y_train, y_in, plot=False)

    if X_test is not None and y_test is not None:
        # compute test scores
        print("Test set performance:")
        y_pred = srch.predict(X_test)
        score_reg(y_test, y_pred, plot=(plot and len(y_test) < 2000))

    return srch


def prediction_sign(x):
    """Used to convert prediction to class labels based on the sign

    Parameters
    ----------
    x : [type]
        prediction values

    Returns
    -------
    [type]
        Integer class labels
    """
    return np.sign(x).astype("int")


def sign_binary_class_score(score_func, y_true, y_pred, **kwargs):
    class_truth = prediction_sign(y_true)
    class_pred = prediction_sign(y_pred)

    # clean class labels
    class_truth[class_truth == 0] = np.random.choice([1, -1])
    class_pred[class_pred == 0] = np.random.choice([1, -1])

    return score_func(class_truth, class_pred, **kwargs)


def sign_mcc_score(y_true, y_pred, **kwargs):
    return sign_binary_class_score(mcc, y_true, y_pred, **kwargs)


def sign_f1_score(y_true, y_pred, **kwargs):
    return sign_binary_class_score(f1_score, y_true, y_pred, **kwargs)


def sign_roc_auc_score(y_true, y_pred, **kwargs):
    return sign_binary_class_score(roc_auc_score, y_true, y_pred, **kwargs)


sign_mcc_scorer = make_scorer(
    sign_mcc_score, greater_is_better=True, needs_proba=False
)

sign_f1_scorer = make_scorer(
    sign_f1_score, greater_is_better=True, needs_proba=False
)

sign_roc_auc_scorer = make_scorer(
    sign_roc_auc_score, greater_is_better=True, needs_proba=False
)


def perf_time(func):
    """Decorator for showing function call time spent.

    Parameters
    ----------
    func : [type]
        function to be called

    Returns
    -------
    [type]
        wrapper function
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        x = func(*args, **kwargs)
        spent = time.perf_counter() - start
        print(f"Time taken (s): {spent:.3e}")
        return x

    return wrapper


@perf_time
def param_search_lgb(
    params: dict,
    n_iter: int,
    X_train,
    y_train,
    cv=None,
    learner_n_jobs: int = -1,
    search_n_jobs: int = 1,
    X_test=None,
    y_test=None,
    device="cpu",
    cv_filename="cv_results.h5",
    params_filename="params.txt",
    **kwargs,
) -> dict:
    """Assissted param search for lightgbm classifier.

    Holds max_depth iterates through num_leaves, then performace random search
    for each (max_depth, num_leaves) combo.

    This is useful because the relationship num_leaves < 2**max_depth should
    hold. If blindly using a random search, the pair selected may violate
    this condition.

    The method writs cv results into a HDF5 file, indexed by num_leave_n
    keys, where n is the parameter used for num_leaves.

    It also writes the best params into a text file.

    Parameters
    ----------
    params : dict
        Random search parameter dict. Must have {'max_depth', 'num_leaves'}.
    n_iter : int
        number of searches
    X_train : TYPE

    y_train : TYPE

    cv : None, optional
        cross valiation indices if given.
    learner_n_jobs : int, optional
        default -1, use all cpus for learner fitting
    search_n_jobs : int, optional
        default 1, use only 1 cpu for search. this is because by default the
        learner is allowed to use all CPUs already.
    X_test : None, optional
        test/valid data
    y_test : None, optional
        test/valid targets
    device : str, optional
        default 'cpu', can be either of {'cpu', 'gpu'}
    cv_filename : str, optional
        file to store cv results
    params_filename : str, optional
        file to store best params
    **kwargs
        kwargs passed to sklearn.RandomizedSearchCV

    Returns
    -------
    dict
    Keys:
    -----
    best_param:
        best parameters
    best_score:
        best achieved score
    best_learner:
        fitted best model
    """
    # some params
    max_depth = params.pop("max_depth")
    num_leaves = params.pop("num_leaves")
    assert max_depth is not None
    assert len(max_depth) == 1
    assert num_leaves is not None
    assert len(num_leaves) > 0

    max_depth = max_depth[0]
    out = dict()

    best_score = None
    best_params = None
    best_learner = None
    start = time.process_time()
    for n in num_leaves:
        print(f"max_depth = {max_depth}, num_leaves = {n}...")
        learner = lgb.LGBMClassifier(
            max_depth=max_depth,
            num_leaves=n,
            # boosting_type='gbdt',
            # objective='xentropy',
            # eval_metric='binary_logloss',
            # early_stopping_rounds=100,
            # verbose_eval=200,
            device=device,
            # verbosity=lgb_verbosity,
            n_jobs=learner_n_jobs,
        )

        rs = RandomizedSearchCV(
            learner,
            params,
            cv=cv,
            n_jobs=search_n_jobs,
            n_iter=n_iter,
            return_train_score=False,
            **kwargs,
        )
        # model_selection._search.format_results() sometimes has an bug and
        # returns nothing, causing ValueError when unpacking output.
        rs.fit(X_train, y_train, verbose=-1)

        if best_score is None:
            best_score = rs.best_score_
            best_learner = rs.best_estimator_

            best_params = rs.best_params_.copy()
            best_params["max_depth"] = max_depth
            best_params["num_leaves"] = n
        elif best_score < rs.best_score_:
            best_score = rs.best_score_
            best_learner = rs.best_estimator_

            best_params = rs.best_params_.copy()
            best_params["max_depth"] = max_depth
            best_params["num_leaves"] = n

        key = f"max_depth_{max_depth}_num_leaves_{n}"

        # store this search object for later use
        out[key] = rs

        # save cv scores
        if cv is not None:
            pd.DataFrame(rs.cv_results_).to_hdf(cv_filename, key=key, mode="a")

        # predict test set.
        if X_test is not None and y_test is not None:
            assert len(X_test) == len(y_test)
            y_pred = rs.predict(X_test)
            train_loss = rs.score(X_train, y_train)
            test_loss = rs.score(X_test, y_test)
            test_mcc = mcc(y_test, y_pred)

            msg = (
                f"{key}, Dev score: {train_loss:.3f}, Test score: "
                + f"{test_loss:.3f}, Test MCC: {test_mcc:.3f}\n"
            )
            print(msg)

            print(f"{key}, Save TSCV Best params = {best_params}")
            with open(params_filename, "a") as fp:
                fp.write(msg)
                fp.write(str(best_params))
                fp.write("\n")

    time_taken = time.process_time() - start
    print("Time taken (s): ", time_taken)

    # write final best param
    with open(params_filename, "a") as fp:
        # convert to python types for json writes
        # best_params = {k: np.asscalar(v) for k, v in best_params.items()}
        # fp.write(json.dumps(best_params))
        fp.write("Final result:\n")
        fp.write(str(best_params))
        fp.write("\n\n")

    out["best_params"] = best_params
    out["best_score"] = best_score
    out["best_learner"] = best_learner

    return out


def save_lgb_model(estimator):
    # write lightgbm booster to file
    estimator.booster_.save_model("lgb_booster.txt")
    # gbm = lgb.Booster(model_file='lightgbm.txt')

    # write sklearn estimator
    joblib.dump(estimator, "lgb_sklearn.pkl")
    # model = joblib.load('lightgbm.pkl')
    print("Model saved to file.")


def load_lgb_booster(filepath) -> lgb.Booster:
    gbm = lgb.Booster(model_file=filepath)
    return gbm


def load_sklearn_estimator(filepath):
    model = joblib.load(filepath)
    return model


def nll_metric(model, X, y):
    # TODO: validate shape, check log_loss docs here. Make class labels
    # match predicted proba
    y_pred = model.predict_proba(X)
    return log_loss(y, y_pred)


@perf_time
def permutation_importances(
    model, X_train: pd.DataFrame, y_train, metric=nll_metric
) -> pd.Series:
    """Permuation importance metrics. Measures the difference of metric
    between baseline and randomly permuted rows for a given feature.

    The direction of the metric, improving or declining, shows the importance.

    Parameters
    ----------
    model : TYPE
        sklearn estimator
    X_train : pd.DataFrame

    y_train : TYPE

    metric : TYPE, optional
        evaluation metric. default is NLL for classification.

    Returns
    -------
    TYPE
    """
    baseline = metric(model, X_train, y_train)
    imp = []

    for col in tqdm.tqdm(X_train.columns):
        print(f"Computing metrics for {col}...")
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(model, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    imp = pd.Series(np.array(imp), index=X_train.columns)

    return imp


def compute_permute_imp(estimator, X, y, out_file: str) -> pd.Series:
    """Helper function for permutation importance.

    Parameters
    ----------
    estimator : TYPE

    X : TYPE

    y : TYPE

    out_file : str


    Returns
    -------
    TYPE
    """
    print("Computing permutation importance values...")
    X = X.copy()
    # y = y.copy()
    permute_imp = permutation_importances(estimator, X, y)

    # sig file
    if out_file is not None:
        permute_imp.to_hdf(out_file, key="permute")
        # write a status file to indicate computation completed
        # with open('status.txt', 'a') as fp:
        #     fp.write('permutation imp done.')

    return permute_imp


@perf_time
def random_search_train(
    learner,
    params_dict: dict,
    X_train,
    y_train,
    cv_method: Union[int, BaseCrossValidator],
    X_test=None,
    y_test=None,
    n_splits=5,
    scoring={"mae": mae_scorer, "mse": mse_scorer},
    refit="mse",
    n_iter: int = 10,
    **cv_kws,
) -> RandomizedSearchCV:
    """Convenient method for running random search with sklearn models.

    Parameters
    ----------
    learner : [type]
        [description]
    params_dict : dict
        hyperparameters dict
    cv_method : [type]
        callable method to create cv ojbect
    X : [type]
        features / X
    y : [type]
        targets / y
    scoring : dict, optional
        scoring methods, by default {'mae': mae_scorer, 'mse': mse_scorer}
    refit : optional
        when there are more then one scoring function, we need to specify
        how the best learner if refitted, i.e. which socring metric to use
        to choose the best, by default 'mse'.
    n_iter : int, optional
        number of searches, by default 10
    cv_kws : optional
        kwargs passed to cv_method call

    Returns
    -------
    [type]
        [description]
    """
    if isinstance(cv_method, int):
        # otherwise, cv is no. of folds
        folds = cv_method
    else:
        # use provided cv split method to create CV folds
        cv = cv_method(n_splits=n_splits, **cv_kws)
        folds = cv.split(X_train, y_train)

    if isinstance(scoring, dict):
        print(f"Scoring: {scoring.keys()}, refit: {refit}")
    # run random search
    rs = RandomizedSearchCV(
        learner,
        params_dict,
        cv=folds,
        n_jobs=-1,
        scoring=scoring,
        n_iter=n_iter,
        refit=refit,
    )
    rs.fit(X_train, y_train)

    print(f"Train best score: {rs.best_score_:.4e}")
    if X_test is not None and y_test is not None:
        test_score = rs.score(X_test, y_test)
        print(f"Test score: {test_score:.4e}")

    return rs


def score(
    learner,
    X,
    y,
    scorer_dict: dict = {
        "mae": mae_scorer,
        "mse": mse_scorer,
        "mcc": sign_mcc_scorer,
        "f1_score": sign_f1_scorer,
        "roc_auc": sign_roc_auc_scorer,
    },
) -> pd.Series:
    """Function to generate various scores / metrics.

    Returns
    -------
    pd.Series
        [description]
    """
    # y_pred = learner.predict(X)
    scores = {
        name: scorer(learner, X, y) for name, scorer in scorer_dict.items()
    }
    return pd.Series(scores)


def pipeline_train(
    estimator: BaseEstimator,
    data: pd.DataFrame,
    params_dict: Dict[AnyStr, Any],
    target_var: str,
    is_classification: bool,
    cat_cols: Optional[List[str]],
    one_hot=False,
    normalize=True,
    impute_strategy: str = None,
    impute_indicator: bool = False,
    cv=None,
    cv_splits=5,
    test_size: Union[int, float] = 0.2,
    split_shuffle=False,
    search_iter: int = 10,
    verbose=False,
    **cv_kws,
):
    """Build pipeline with basic preprocessing pipeline and random search cv.
    Steps:
    1. Categorify data features
    2. impute data if needed (pipeline)
    3. normalise if needed (pipeline)
    4. add model (pipeline)
    5. random search

    Parameters
    ----------
    estimator : BaseEstimator
        prediction model
    data : pd.DataFrame
        data examples, with both features and labels.
    params_dict : Dict[AnyStr, Any]
        Tuning parameter dict for the model, parameter keys should have
        prefix 'model__'.
    target_var : str
        target variable name, i.e. column name for the labels / prediction
        targets.
    is_classification : bool
        Whether the task is classification or not. If true, MCC and NLL is
        chosen as the scoring methods, and Negative Log-Likelihood would be
        the method for refit after random search.
        If set to False, MAE and MSE are the scoring method and MSE is used
        for refit.
    cat_cols : Optional[List[str]]
        Categorical feature names. Those columns not named here, except the
        target variable, are assumed to be continuous variables.
    one_hot : bool, optional
        Whether to use one hot encoding for categorial features,
        by default False, which works for tree models. Otherwise, set to True
        for models that cannot handle ordinal features.
    normalize : bool, optional
        Whether to normalise in pipeline, by default True
    impute_strategy : str, optional
        Imputation strategy, see sklearn.impute.SimpleImputer for details,
        by default None
    impute_indicator : bool, optional
        Whether to add indicator for rows imputed, by default False
    cv : [type], optional
        CV method class, by default None
    cv_splits : int, optional
        CV splits, by default 5
    test_size : Union[int, float], optional
        ratio or number of test data size, by default 0.2, i.e. 20%
    split_shuffle : bool, optional
        Whether data will be shuffled when splitting into train and test data,
        by default False
    search_iter : int, optional
        # of random search rounds, by default 10
    verbose : bool, optional
        [description], by default False

    Returns
    -------
    sklearn.model_selection.RandomSearchCV
        [description]
    """
    # Transform categorical variables
    # Assume categorical variables are either ordinal or one hot
    # ordinal used for tree methods, whereas one hot is used for others
    # such as regression
    if cat_cols is not None:
        cont_mask = [
            x not in cat_cols and x != target_var for x in data.columns
        ]
        cont_cols = data.columns.values[cont_mask].tolist()

        if one_hot:
            cat_encoder = skp.OneHotEncoder(sparse=False)
            if verbose:
                print("Use One-Hot encoding for categorical variables.")

            # encode category
            cat_data = cat_encoder.fit_transform(data[cat_cols])
            # categories are stored as list of ndarrays
            cat_cols = np.concatenate(cat_encoder.categories_).tolist()
            cat_data = pd.DataFrame(
                cat_data, index=data.index, columns=cat_cols
            )

            # concat with continuous variables
            h, _ = data.shape
            expected_w = len(cont_cols) + len(cat_cols) + 1
            data = pd.concat([data[cont_cols + [target_var]], cat_data], axis=1)
            assert data.shape == (h, expected_w)
        else:
            cat_encoder = skp.OrdinalEncoder()
            h, w = data.shape
            cat_data = cat_encoder.fit_transform(data[cat_cols])
            cat_data = pd.DataFrame(
                cat_data, index=data.index, columns=cat_cols
            )
            # join with continuous variables
            data = pd.concat([data[cont_cols + [target_var]], cat_data], axis=1)
            assert data.shape == (h, w + 1)
            if verbose:
                print("Use Ordinal encoding for categorical variables.")
        if verbose:
            print("Converted category columns.")
    else:
        # all columns are continuous, skip target column
        cont_mask = [x != target_var for x in data.columns]
        cont_cols = data.columns.values[cont_mask].tolist()

    # all feature columns
    if cat_cols is not None:
        feature_cols = cat_cols + cont_cols
    else:
        feature_cols = cont_cols

    if verbose:
        if cat_cols is not None:
            print(f"Categorical features # {len(cat_cols)}: {cat_cols[:5]}")
        print(f"Continuous features # {len(cont_cols)}: {cont_cols[:5]}")
        print(f"Total # of features: {len(feature_cols)}")

    # convert columna names to indices, cannot use column names with
    # column transformer which returns nd.arrays, in a pipeline
    X = data[feature_cols]
    cont_cols_loc = [X.columns.get_loc(x) for x in cont_cols]
    # cat_cols_loc = [X.columns.get_loc(x) for x in cat_cols]

    # prediction pipeline including preprocessing steps within a CV fold
    model_steps = []
    # build processing pipe, need separate column transformers for each step
    # column transfomers with multiple steps would concat the output from
    # all steps into a new feature array
    if impute_strategy is not None:
        imputer = SimpleImputer(
            strategy=impute_strategy, add_indicator=impute_indicator
        )
        impute_trans = ColumnTransformer(
            [("impute", imputer, cont_cols_loc)],
            remainder="passthrough",
            sparse_threshold=0,
        )
        model_steps.append(("impute", impute_trans))
        if verbose:
            print(f"Added imputer with strategy: {impute_strategy}.")

        if impute_indicator:
            # TODO: categorify missing indicator columns
            print("TODO: categorify missing indicator columns")
            pass

    if normalize:
        # somehow this step sees a sparse matrix and cannot standardise
        # with mean...
        cont_encoder = ColumnTransformer(
            [("normalise", skp.StandardScaler(), cont_cols_loc)],
            remainder="passthrough",
        )
        model_steps.append(("normalise", cont_encoder))
        if verbose:
            print("Added normalisation for continuous variables.")

    model_steps.append(("model", estimator))
    pipe = Pipeline(model_steps, verbose=verbose)
    if verbose:
        print(f"Pipeline = {pipe}")

    # training set is used in CV / params search
    if is_classification:
        scoring = {"mcc": mcc_scorer, "nll": nll_scorer}
        refit_choice = "nll"

        # encode target as categories
        target_encoder = skp.OrdinalEncoder()
        y = target_encoder.fit_transform(
            data[target_var].values.reshape((-1, 1))
        )
        if verbose:
            print(f"Encode target var as categorical: {y.dtype}")
    else:
        scoring = {"mae": mae_scorer, "mse": mse_scorer}
        refit_choice = "mse"
        y = data[target_var].values.reshape((-1, 1))

    # split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=split_shuffle
    )
    if verbose:
        print(f"Scoring methods = {scoring.keys()}, refit: {refit_choice}")
        print(f"Train size:, x = {x_train.shape}, y = {y_train.shape}")
        print(f"Test size:, x = {x_test.shape}, y = {y_test.shape}")

    # # build random search
    rs = random_search_train(
        pipe,
        params_dict,
        x_train,
        y_train.ravel(),
        cv_method=cv,
        X_test=x_test,
        y_test=y_test.ravel(),
        n_splits=cv_splits,
        scoring=scoring,
        refit=refit_choice,
        n_iter=search_iter,
        **cv_kws,
    )

    return rs


class LogUniformDist(rv_continuous):
    def __init___(self, **kwargs):
        super().__init__(self, **kwargs)

    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform_draw(n: int, a: float = 1e-3, b: float = 1e3):
    """Draw from log uniform distrubtion. This is useful for hyperparameter
    searches where sampling uniformly between a and b isn't efficient.
    This is more effecitive when the parameters' scale matters more, e.g.
    1, 10, 100, 1000, ... are more effective than uniform values between
    1 and 1000.

    Parameters
    ----------
    n : int
        no. of draws
    a : float, optional
        lower bound, by default 1e-3
    b : float, optional
        upper bound, by default 1e3

    Returns
    -------
    ndarray
        draws
    """
    dist = LogUniformDist(a=a, b=b, name="log_uniform")
    draws = dist.rvs(size=n)
    return draws


def train_test_split_ts(
    data: pd.DataFrame, ban_zone: int, shuffle=False, **kwargs
):
    """Split dataset into train and test set, insert a ban zone if needed.

    Parameters
    ----------
    data : pd.DataFrame
        [description]
    ban_zone : int
        Non-negative number, data is taken away from both the tail of train
        and head of test data.

    Returns
    -------
    [type]
        [description]
    """
    train, test = train_test_split(data, shuffle=shuffle, **kwargs)
    if ban_zone != 0:
        ban_zone = abs(ban_zone)
        # fmt: off
        assert ban_zone < len(train), f"ban_zone {ban_zone} must be less than length of training set {len(train)}."
        assert ban_zone < len(test), f"ban_zone {ban_zone} must be less than length of test set {len(test)}."
        # fmt: on
        train = train.iloc[:-ban_zone]
        test = test.iloc[ban_zone:]
    return train, test


def permute_imp_sk(
    estimator,
    X,
    y,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    n_jobs=-1,
    rand_seed=None,
) -> Tuple[pd.DataFrame, Dict]:
    """Compute permutation importance with sklearn's API.

    Parameters
    ----------
    estimator : [type]
        [description]
    X : [type]
        [description]
    y : [type]
        [description]
    scoring : str, optional
        [description], by default "neg_mean_absolute_error"
    n_repeats : int, optional
        [description], by default 10
    n_jobs : int, optional
        [description], by default -1
    rand_seed : [type], optional
        [description], by default None

    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        [description]
    """
    imp_dict = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=rand_seed,
    )

    mean = imp_dict.get("importances_mean")
    std = imp_dict.get("importances_std")

    df = pd.DataFrame.from_dict({"score_mean": mean, "score_std": std})
    df.sort_values("score_mean", inplace=True)
    if isinstance(X, pd.DataFrame):
        df.index = X.columns
    return df, imp_dict
