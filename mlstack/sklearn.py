import time
import numpy as np
import pandas as pd
import joblib
import tqdm
import lightgbm as lgb

from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import brier_score_loss, log_loss, make_scorer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, BaseCrossValidator


mcc_scorer = make_scorer(mcc)

brier_scorer = make_scorer(brier_score_loss,
                           greater_is_better=False, needs_proba=True)

nll_scorer = make_scorer(log_loss, greater_is_better=False,
                         needs_proba=True)

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False,
                         needs_proba=False)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False,
                         needs_proba=False)


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
        print(f'Time taken (s): {spent:.3e}')
        return x
    return wrapper


@perf_time
def param_search_lgb(params: dict, n_iter: int,
                     X_train, y_train,
                     cv=None,
                     learner_n_jobs: int = -1,
                     search_n_jobs: int = 1,
                     X_test=None, y_test=None,
                     device='cpu',
                     cv_filename='cv_results.h5',
                     params_filename='params.txt',
                     **kwargs) -> dict:
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
    max_depth = params.pop('max_depth')
    num_leaves = params.pop('num_leaves')
    assert(max_depth is not None)
    assert(len(max_depth) == 1)
    assert(num_leaves is not None)
    assert(len(num_leaves) > 0)

    max_depth = max_depth[0]
    out = dict()

    best_score = None
    best_params = None
    best_learner = None
    start = time.process_time()
    for n in num_leaves:
        print(f'max_depth = {max_depth}, num_leaves = {n}...')
        learner = lgb.LGBMClassifier(max_depth=max_depth,
                                     num_leaves=n,
                                     # boosting_type='gbdt',
                                     # objective='xentropy',
                                     # eval_metric='binary_logloss',
                                     # early_stopping_rounds=100,
                                     # verbose_eval=200,
                                     device=device,
                                     # verbosity=lgb_verbosity,
                                     n_jobs=learner_n_jobs)

        rs = RandomizedSearchCV(learner, params,
                                cv=cv,
                                n_jobs=search_n_jobs,
                                n_iter=n_iter,
                                return_train_score=False,
                                **kwargs)
        # model_selection._search.format_results() sometimes has an bug and
        # returns nothing, causing ValueError when unpacking output.
        rs.fit(X_train, y_train, verbose=-1)

        if best_score is None:
            best_score = rs.best_score_
            best_learner = rs.best_estimator_

            best_params = rs.best_params_.copy()
            best_params['max_depth'] = max_depth
            best_params['num_leaves'] = n
        elif best_score < rs.best_score_:
            best_score = rs.best_score_
            best_learner = rs.best_estimator_

            best_params = rs.best_params_.copy()
            best_params['max_depth'] = max_depth
            best_params['num_leaves'] = n

        key = f'max_depth_{max_depth}_num_leaves_{n}'

        # store this search object for later use
        out[key] = rs

        # save cv scores
        if cv is not None:
            pd.DataFrame(rs.cv_results_).to_hdf(cv_filename, key=key, mode='a')

        # predict test set.
        if X_test is not None and y_test is not None:
            assert(len(X_test) == len(y_test))
            y_pred = rs.predict(X_test)
            train_loss = rs.score(X_train, y_train)
            test_loss = rs.score(X_test, y_test)
            test_mcc = mcc(y_test, y_pred)

            msg = (f'{key}, Dev score: {train_loss:.3f}, Test score: ' +
                   f'{test_loss:.3f}, Test MCC: {test_mcc:.3f}\n')
            print(msg)

            print(f'{key}, Save TSCV Best params = {best_params}')
            with open(params_filename, 'a') as fp:
                fp.write(msg)
                fp.write(str(best_params))
                fp.write('\n')

    time_taken = time.process_time() - start
    print('Time taken (s): ', time_taken)

    # write final best param
    with open(params_filename, 'a') as fp:
        # convert to python types for json writes
        # best_params = {k: np.asscalar(v) for k, v in best_params.items()}
        # fp.write(json.dumps(best_params))
        fp.write('Final result:\n')
        fp.write(str(best_params))
        fp.write('\n\n')

    out['best_params'] = best_params
    out['best_score'] = best_score
    out['best_learner'] = best_learner

    return out


def save_lgb_model(estimator):
    # write lightgbm booster to file
    estimator.booster_.save_model('lgb_booster.txt')
    # gbm = lgb.Booster(model_file='lightgbm.txt')

    # write sklearn estimator
    joblib.dump(estimator, 'lgb_sklearn.pkl')
    # model = joblib.load('lightgbm.pkl')
    print('Model saved to file.')


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
def permutation_importances(model,
                            X_train: pd.DataFrame,
                            y_train,
                            metric=nll_metric):
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
        print(f'Computing metrics for {col}...')
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(model, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    imp = pd.Series(np.array(imp), index=X_train.columns)

    return imp


def compute_permute_imp(estimator, X, y, out_file: str):
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
    print('Computing permutation importance values...')
    X = X.copy()
    # y = y.copy()
    permute_imp = permutation_importances(estimator, X, y)

    # sig file
    if out_file is not None:
        permute_imp.to_hdf(out_file, key='permute')
        # write a status file to indicate computation completed
        # with open('status.txt', 'a') as fp:
        #     fp.write('permutation imp done.')

    return permute_imp


@perf_time
def random_search_train(learner, params_dict: dict,
                        cv_method: BaseCrossValidator,
                        X_train, y_train,
                        X_test=None, y_test=None,
                        n_splits=5,
                        scoring={'mae': mae_scorer, 'mse': mse_scorer},
                        refit='mse',
                        n_iter: int = 10):
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

    Returns
    -------
    [type]
        [description]
    """
    # use provided cv split method to create CV folds
    cv = cv_method(n_splits=n_splits)
    folds = cv.split(X_train)

    # run random search
    rs = RandomizedSearchCV(learner, params_dict, cv=folds, n_jobs=-1,
                            scoring=scoring,
                            n_iter=n_iter,
                            refit=refit)
    rs.fit(X_train, y_train)

    print(f'Train best score: {rs.best_score_:.4e}')
    if X_test is not None and y_test is not None:
        test_score = rs.score(X_test, y_test)
        print(f'Test score: {test_score:.4e}')

    return rs
