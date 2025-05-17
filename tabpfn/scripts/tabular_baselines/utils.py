from __future__ import annotations

import logging
import time
from typing import Literal
import warnings
import uuid

import numpy as np
from scipy.linalg._decomp_update import LinAlgError

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from functools import partial, update_wrapper

import hyperopt
import pandas as pd
from pandas.api.types import CategoricalDtype

from hyperopt import Trials, fmin, rand, space_eval
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ..tabular_evaluation_utils import DatasetEvaluation
from ..tabular_metrics import get_scoring_direction
from tabpfn.scripts.tabular_metrics.classification import auc_metric_ovo
from tabpfn.datasets import DistributionShiftDataset
from tabpfn.utils import target_is_multiclass

# Keep it that way, since % autoreload in jupyter notebook destroys the from A import B structure
# This destroys calls to isinstance(var , B). But isinstance(var, A.B) still works.
# https://github.com/ipython/ipython/issues/12399
from .custom_folds import DistributionShiftSplit

CV = 3
MULTITHREAD = 8  # Number of threads baselines are able to use at most


def get_random_seed(random_state, y, mod=True):
    """Generates a random seed based on the random_state and the sum
    of the labels. This is to ensure that the seed is different for
    different splits of the same dataset.
    """
    try:
        seed = random_state + int(y[:].sum())
    except:
        seed = random_state

    # This is done in order to not change previous results.
    # In some parts of the code the mod operation was applied,
    # while in others it was not.
    # np.random.RandomState would support seeds in [0, 2**32 - 1]
    if mod:
        seed = seed % 10000

    # Only fix after the fact to avoid breaking random seeds that worked previously
    if seed and seed < 0:
        return get_random_seed(random_state, abs(y), mod=mod)

    return seed


def eval_f(
    params,
    clf_,
    x,
    y,
    task_type,
    metric_used,
    preprocess_kwargs={},
    use_metric_as_scorer=False,
    verbose=False,
):
    if use_metric_as_scorer:
        scoring = metric_used
    else:
        # Wrap scorer to catch NaN in y_score separately from other NaN in data.

        if metric_used.__name__ == auc_metric_ovo.__name__:
            partial_scorer = partial(metric_used, labels=np.unique(y))
            update_wrapper(partial_scorer, metric_used)

        scoring = make_scorer(
            partial_scorer,
            needs_proba=target_is_multiclass(task_type),
            greater_is_better=get_scoring_direction(metric_used) == 1,
        )  # get_scoring_string(metric_used, usage="sklearn_cv")

    if verbose:
        print(
            f"Starting cross-validation with parameters {params} and scoring {scoring}.",
        )

    if (
        "train_dist_shift_domain" in preprocess_kwargs
        and np.unique(preprocess_kwargs["train_dist_shift_domain"]).shape[0] >= 2
    ):
        cv = DistributionShiftSplit(
            domain_indicators=preprocess_kwargs["train_dist_shift_domain"],
            gap=0,
            max_train_size=None,
            n_splits=CV,
            test_size=None,
        )
    else:
        cv = CV

    try:
        scores = cross_val_score(
            clf_(**params),
            x,
            y,
            n_jobs=1,  # The algorithms use all cores/GPUs available. Large values for n_jobs result in very inefficient fitting.
            cv=cv,
            scoring=scoring,
            error_score="raise",  # to avoid crashing a fold resulting in a better mean performance.
        )
    except ValueError as e:
        if str(e) == "Encountered NaN in y_score.":
            print(f"Skipping params due to NaN in predictions: {params}")
            return np.nan
        else:
            print(f"Encountered exception during cross-validation: {e!s}")
            print(f"The error occurred while using these parameters: {params}")
            raise e
    except Exception as e:
        # Print error to stdout to also be able to find the relationship between
        # tried parameters and the error itself
        print(f"Encountered exception during cross-validation: {e!s}")
        print(f"The error occurred while using these parameters: {params}")
        raise e

    mean_score = np.nanmean(scores)

    if verbose:
        print(
            f"Cross-validation with parameters {params} yielded a score of {mean_score}. Complete score list: {scores}",
        )

    return mean_score * -1


def _cost_fn(
    params,
    clf_,
    x,
    y,
    task_type,
    metric_used,
    preprocess_kwargs,
    use_metric_as_scorer,
    verbose,
):
    """Top-level wrapper to support pickling of the cost function for hyperopt."""
    return eval_f(
        params,
        clf_,
        x,
        y,
        task_type,
        metric_used,
        preprocess_kwargs=preprocess_kwargs,
        use_metric_as_scorer=use_metric_as_scorer,
        verbose=verbose,
    )


def _hyperopt_random_state(*, rstate):
    return (
        np.random.RandomState(rstate)
        if hyperopt.__version__ in ("0.2.4", "0.2.5")
        else np.random.default_rng(rstate)
    )


def _init_ray() -> None | object:
    """Try to init ray to use for HPO."""
    try:
        import os
        import platform

        import psutil
        import ray
        import torch
    except ImportError:
        print(
            "Ray or its dependencies is not installed. Not using robust timeout and fitting for hyperopt.",
        )
        return None, None, None

    num_cpus = (
        psutil.cpu_count(logical=True)
        if os.name == "nt" or platform.system() == "Darwin"
        else len(os.sched_getaffinity(0))
    )
    num_gpus = torch.cuda.device_count()
    ray_mem_in_b = int(int(os.environ.get("RAY_MEM_IN_GB", default=8)) * (1024.0**3))

    print(f"Found {num_cpus} CPUs and {num_gpus} GPUs.")

    if not ray.is_initialized():
        ray_args = dict(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            address="local",
            _memory=ray_mem_in_b,
            object_store_memory=ray_mem_in_b,
            include_dashboard=False,
            logging_level=logging.INFO,
            log_to_driver=True,
        )
        if os.environ.get("TABPFN_CLUSTER_SETUP") == "5a5ba760":
            ray_args["_temp_dir"] = os.environ.get("TMPDIR")
        elif os.environ.get("RAY_INIT_TMPDIR", False):
            os_env = os.environ.get("RAY_INIT_TMPDIR")
            tmp_dir_base_path = "/tmp_ray" if os_env == "True" else os_env

            ts = str(uuid.uuid4())[:8]

            ray_dir = f"{tmp_dir_base_path}/hpo_{ts}"

            if not os.path.isabs(ray_dir):
                ray_dir = os.path.abspath(ray_dir)

            ray_args["_temp_dir"] = ray_dir
        ray.init(**ray_args)

    return ray, num_cpus, num_gpus


def _get_eval_func_for_hpo(
    *,
    clf_,
    x,
    y,
    task_type,
    metric_used,
    preprocess_kwargs,
    use_metric_as_scorer,
    verbose,
    start_time,
    max_time,
    method_name,
):
    """Get the eval function for hyperopt.

    If ray is installed, we use a robust function that is able to handle a lot of errors. Otherwise, fallback to the normal eval_f function.
    """
    _ray, num_cpus, num_gpus = _init_ray()

    if _ray is None:
        return lambda params: eval_f(
            params,
            clf_,
            x,
            y,
            task_type=task_type,
            metric_used=metric_used,
            preprocess_kwargs=preprocess_kwargs,
            use_metric_as_scorer=use_metric_as_scorer,
            verbose=verbose,
        )

    # Build function
    def fn(*args, **kwargs):
        rest_time = int(max_time - (time.time() - start_time))

        timeout_reached = False

        if rest_time > 1:
            try:
                remote_fn = _ray.remote(max_calls=1, max_retries=1)(_cost_fn)
                ref = remote_fn.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
                    *args,
                    clf_=clf_,
                    x=x,
                    y=y,
                    task_type=task_type,
                    metric_used=metric_used,
                    preprocess_kwargs=preprocess_kwargs,
                    use_metric_as_scorer=use_metric_as_scorer,
                    verbose=verbose,
                    **kwargs,
                )
                finished, unfinished = _ray.wait(
                    [ref],
                    num_returns=1,
                    # time left with a small overhead of 5s to avoid re-queuing the task before hyperopt times out.
                    timeout=rest_time,
                )
                if finished:
                    res = _ray.get(finished[0])
                else:
                    timeout_reached = True
            except _ray.exceptions.WorkerCrashedError as ex:
                print(f"WORKER CRASHED: {ex}")
                print(
                    "CONFIG FAILED DUE TO UNKNOWN REASON (LIKELY KILLED DUE TO OOM FROM OS).",
                )
                timeout_reached = False
                res = {
                    "status": hyperopt.STATUS_FAIL,
                    "failure": "MemOut",
                }
            except Exception as e:
                # -> fail the trial instead of crashing the whole process if we know the edge case problem

                if method_name == "catboost" and str(e).endswith(
                    "Too few sampling units (subsample=0.8, bootstrap_type=MVS): please increase sampling rate or disable sampling",
                ):
                    # Specific workaround for https://github.com/catboost/catboost/issues/2555
                    warnings.warn(
                        f"Expected CatBoost crash happened with {e!s}. Failing trial instead of HPO.",
                        stacklevel=2,
                    )
                    timeout_reached = False
                    res = {
                        "status": hyperopt.STATUS_FAIL,
                        "failure": "CatBoostBug",
                    }
                elif isinstance(e, ValueError) and str(e).endswith(
                    "Number of classes in y_true not equal to the number of columns in 'y_score'",
                ):
                    # Specific workaround for cross-validation from sklearn not working correctly.
                    warnings.warn(
                        f"Sklearn cross-validation failed with {e!s}. Failing trial instead of HPO.",
                        stacklevel=2,
                    )
                    timeout_reached = False
                    res = {
                        "status": hyperopt.STATUS_FAIL,
                        "failure": "SklearnCVBug",
                    }
                elif (
                    method_name == "linear_quantile"
                    and isinstance(e, TypeError)
                    and str(e).endswith(
                        "params = solution[:n_params] - solution[n_params : 2 * n_params]\nTypeError: 'NoneType' object is not subscriptable",
                    )
                ):
                    # Specific workaround for Linear programming for QuantileRegressor crashing due Numerical difficulties encountered resulting
                    # from the alpha value. Fail the trial instead of HPO.
                    warnings.warn(
                        f"Linear_quantile model failed with {e!s}. Failing trial instead of HPO.",
                        stacklevel=2,
                    )
                    timeout_reached = False
                    res = {
                        "status": hyperopt.STATUS_FAIL,
                        "failure": "LinearProgQuantileRegressorBug",
                    }
                elif method_name == "xgb":
                    from xgboost.core import XGBoostError

                    if isinstance(e, XGBoostError) and (
                        "adaptive.cc:131: Check failed: h_row_set.empty()" in str(e)
                    ):
                        # workaround for edge case quantile regression bug in xgboost
                        warnings.warn(
                            f"XGB model failed with {e!s}. Failing trial instead of HPO.",
                            stacklevel=2,
                        )
                        timeout_reached = False
                        res = {
                            "status": hyperopt.STATUS_FAIL,
                            "failure": "XGBQuantileRegressorBug",
                        }
                    elif (
                        isinstance(e, ValueError)
                        and "mean_pinball_loss" in str(e)
                        and str(e).endswith(
                            "ValueError: Input contains infinity or a value too large for dtype('float32')."
                        )
                    ):
                        # workaround for edge case where extreme parameters make predictions inf in xgboost
                        warnings.warn(
                            f"XGB model failed with {e!s}. Failing trial instead of HPO.",
                            stacklevel=2,
                        )
                        timeout_reached = False
                        res = {
                            "status": hyperopt.STATUS_FAIL,
                            "failure": "XGBQuantileRegressorBug",
                        }
                    else:
                        raise e
                else:
                    raise e
        else:
            timeout_reached = True

        if timeout_reached:
            print("TERMINATING DUE TO TIME-OUT.")
            res = {
                "status": hyperopt.STATUS_FAIL,
                "failure": "TimeOut",
            }
            time.sleep(5)  # wait for hyperopt

        return res

    return fn


def _run_hpo(
    clf_,
    x,
    y,
    max_time: int,
    random_state,
    task_type,
    metric_used,
    preprocess_kwargs,
    use_metric_as_scorer,
    verbose,
    key,
    param_grid,
):
    # Init args
    best = {}
    failed = False
    trials = Trials()
    rstate = get_random_seed(random_state, y)
    rstate_gen = _hyperopt_random_state(rstate=rstate)

    start_time = time.time()

    print("Start HPO Search.")
    print(f"Total time for hpo: {int(max_time-(time.time()-start_time))}")

    fn = _get_eval_func_for_hpo(
        clf_=clf_,
        x=x,
        y=y,
        task_type=task_type,
        metric_used=metric_used,
        preprocess_kwargs=preprocess_kwargs,
        use_metric_as_scorer=use_metric_as_scorer,
        verbose=verbose,
        start_time=start_time,
        max_time=max_time,
        method_name=key,
    )

    print("Fit Default")
    default = fn({})
    if isinstance(default, dict):
        print(
            f"Default config ran into {default['failure']}. Failed to train anything!",
        )
        failed = True
    elif np.isnan(default):
        # In case no CV fold was found that satisfies the split criteria e.g.
        # the classes in train and validation are equal, default returns
        # a nan result. This would lead to fmin() failing with the error
        # hyperopt.exceptions.AllTrialsFailed
        # Reproducibility: analcatdata_marketing fails consistently on
        # split_number 3 as in the overall train portion only 1 sample
        # belongs to class 0, so no split exists that will satisfy our
        # conditions.
        # In that case, we just skip the hpo and use the default params.
        print(
            f"HPO on method {key} failed as no CV split satisfying the conditions was found. Using default params.",
        )
    else:
        try:
            print("Start Hyperopt")

            hpo_best = fmin(
                fn=fn,
                space=param_grid,
                algo=rand.suggest,
                rstate=rstate_gen,
                trials=trials,
                timeout=max(int(max_time - (time.time() - start_time)), 1),
                verbose=True,
                show_progressbar=False,  # The seed is deterministic but varies for each dataset and each split of it
                max_evals=10000,
            )
            valid_trial_losses = [
                t["result"]["loss"]
                for t in trials.trials
                if t["result"]["status"] == hyperopt.STATUS_OK
            ]
            print(valid_trial_losses)
            if valid_trial_losses:
                best_score = np.nanmin(valid_trial_losses)

                # Only use the parameters in case they are better than the default.
                if best_score < default:
                    best = space_eval(param_grid, hpo_best)

            if verbose:
                print("<=========================== HPO ===========================>")
                print(
                    f"Number of hpo configurations evaluated: {len(valid_trial_losses)}",
                )
                print(f"Best parameters: {best}")
                print("<===========================================================>")
        except hyperopt.exceptions.AllTrialsFailed:
            print("HPO failed. Using default params.")

    return best, failed


def _fallback_predictions(
    metric_used, x, y, test_x, task_type, additional_args, method_name
):
    if task_type in {"multiclass", "dist_shift_multiclass"}:
        from sklearn.dummy import DummyClassifier

        pred = DummyClassifier(strategy="most_frequent").fit(x, y).predict_proba(test_x)
        return DatasetEvaluation(
            y=None,
            pred=pred,
            additional_args=additional_args,
            algorithm_name=method_name,
        )

    if task_type == "regression":
        # regression
        from sklearn.dummy import DummyRegressor

        pred = DummyRegressor(strategy="mean").fit(x, y).predict(test_x)
        return DatasetEvaluation(
            y=None,
            pred=pred,
            additional_args=additional_args,
            algorithm_name=method_name,
        )

    raise ValueError(f"Unknown task type: {task_type}")


def eval_complete_f(
    x,
    y,
    test_xs,
    task_type,
    preprocess_kwargs,
    key,
    param_grid,
    clf_,
    metric_used,
    max_time,
    no_tune: dict | None,  # if None, do HPO. If dict, use dict as default parameters
    random_state,
    use_metric_as_scorer=False,
    verbose: bool = True,
    method_name=None,
) -> dict[str, DatasetEvaluation]:
    if no_tune is None:
        best, failed = _run_hpo(
            clf_=clf_,
            x=x,
            y=y,
            max_time=max_time,
            random_state=random_state,
            task_type=task_type,
            metric_used=metric_used,
            preprocess_kwargs=preprocess_kwargs,
            use_metric_as_scorer=use_metric_as_scorer,
            verbose=verbose,
            key=key,
            param_grid=param_grid,
        )
    else:
        best, failed = no_tune.copy(), False

    start = time.time()
    if not failed:
        clf = clf_(**best)
        try:
            clf.fit(x, y)
        except (
            LinAlgError
        ) as e:  # This can happen for linear models if the data is too ill-conditioned
            print(f"Encountered LinAlgError during fit: {e!s}")
            print(f"The error occurred while using these parameters: {best}")
            failed = True
        except ValueError as e:
            print(f"Encountered ValueError during fit: {e!s}")
            print(f"The error occurred while using these parameters: {best}")
            failed = True
        except AssertionError:
            print("Encountered AssertionError during fit.")
            print(f"The error occurred while using these parameters: {best}")
            failed = True

    fit_time = time.time() - start
    additional_args = {"best_config": best, "failed": failed, "fit_time": fit_time}

    if failed:
        print("Failed to train model, returning default predictions!")
        return {
            test_portion: _fallback_predictions(
                metric_used=metric_used,
                x=x,
                y=y,
                test_x=test_x,
                task_type=task_type,
                additional_args=additional_args,
                method_name=method_name,
            )
            for test_portion, test_x in test_xs.items()
        }

    evaluations = {}
    for test_portion, test_x in test_xs.items():
        start = time.time()

        if target_is_multiclass(task_type):
            pred = clf.predict_proba(test_x)
        else:
            pred = clf.predict(test_x)

        inference_time = time.time() - start
        additional_args["inference_time"] = inference_time

        evaluations[test_portion] = DatasetEvaluation(
            y=None,
            pred=pred,
            additional_args=additional_args,
            algorithm_name=method_name,
        )

    return evaluations


class NumpyToPandasEncoder(BaseEstimator, TransformerMixin):
    # TODO: Might replace with TabularDataset.to_pandas() at some point. This however acts on the dataset level, not on the
    # level of X, y.
    def __init__(self, attribute_names, cat_features):
        self.attribute_names = attribute_names

        assert len(attribute_names) > 0, "The attribute names must not be empty."
        assert (
            not cat_features or np.min(cat_features) >= 0
        ), "The minimum index of the categorical features is out of bounds."
        assert not cat_features or np.max(cat_features) < len(
            attribute_names
        ), "The maximum index of the categorical features is out of bounds."

        cat_feature_indices = cat_features
        self.cat_features = [attribute_names[i] for i in cat_feature_indices]

        self.categories_ = {}

        self.fitted = False

    def set_output(
        self, *, transform: Literal["default", "pandas"] | None = None
    ) -> BaseEstimator:
        print(
            "This custom transformer always takes a numpy array as input and returns a pandas dataframe as output."
        )

        return self

    def fit(self, X, y=None):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        assert (
            len(self.attribute_names) == X.shape[1]
        ), "The attribute names do not match the shape of X."

        X_df = pd.DataFrame(X, columns=self.attribute_names)

        # Identify categories in the training data
        for c in self.cat_features:
            self.categories_[c] = X_df[c].astype("category").cat.categories.tolist()

        self.fitted = True

        return self

    def transform(self, X, y=None):
        assert self.fitted == True, "Must call fit first."

        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        assert (
            len(self.attribute_names) == X.shape[1]
        ), "The attribute names do not match the shape of X."

        # We only support numeric dtypes for now due to loads of edge cases with other dtypes.
        assert np.issubdtype(X.dtype, np.number), "The input array must be numerical."

        X_df = pd.DataFrame(X, columns=self.attribute_names)

        # By default Pandas dataframes don't infer the correct data type for the columns.
        X_df = X_df.apply(pd.to_numeric, errors="coerce")

        for c in self.cat_features:
            current_categories = X_df[c].astype("category").cat.categories.tolist()
            existing_categories_set = set(self.categories_[c])

            # We want to keep new categories in transform for now.
            # This is done for consistency of the categorical codes of multiple transforms.
            new_categories = [
                cat for cat in current_categories if cat not in existing_categories_set
            ]

            if new_categories:
                self.categories_[c] += new_categories

            cat_dtype = CategoricalDtype(categories=self.categories_[c], ordered=False)

            X_df[c] = X_df[c].astype(cat_dtype)

        return X_df


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_strategy="mean", categorical_strategy="most_frequent"):
        self.numeric_prefix = "numeric"
        self.categorical_prefix = "categorical"

        self.numeric_strategy = numeric_strategy

        self.categorical_strategy = categorical_strategy

        self.column_transformer = None  # Initialized in fit

        self.fitted = False

    def set_output(
        self, *, transform: Literal["default", "pandas"] | None = None
    ) -> BaseEstimator:
        print(
            "This custom transformer always takes a pandas array as input and returns a pandas dataframe as output."
        )

        return self

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

        # Identify categorical columns
        # Use a dict as dicts are ordered since Python 3.7
        categorical_cols = {
            name: None for name in X.select_dtypes(include=["category"]).columns
        }

        # Create a list of transformers. The way the ColumnTransformer is built, the relative order of the columns is kept.
        column_transformers = []
        for col in X.columns:
            if col in categorical_cols:
                column_transformers.append(
                    (
                        f"{self.categorical_prefix}_{col}",
                        SimpleImputer(
                            missing_values=np.nan,
                            strategy=self.categorical_strategy,
                            keep_empty_features=True,
                        ),
                        [col],
                    )
                )
            else:
                column_transformers.append(
                    (
                        f"{self.numeric_prefix}_{col}",
                        SimpleImputer(
                            missing_values=np.nan,
                            strategy=self.numeric_strategy,
                            keep_empty_features=True,
                        ),
                        [col],
                    )
                )

        column_transformer = ColumnTransformer(column_transformers)
        column_transformer.set_output(transform="pandas")

        column_transformer.fit(X)

        self.column_transformer = column_transformer

        self.fitted = True

        return self

    def transform(self, X, y=None):
        assert self.fitted, "Must call fit first."
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

        # Identify categorical columns
        categorical_cols = {
            name: None for name in X.select_dtypes(include=["category"]).columns
        }

        # Save the categorical codes, works as most_frequent does not introduce new categories.
        assert (
            self.categorical_strategy == "most_frequent"
        ), "Only most frequent strategy is currently supported."

        categories = {}
        for idx, col in enumerate(categorical_cols):
            categories[col] = CategoricalDtype(
                categories=X[col].cat.categories, ordered=False
            )

        X = self.column_transformer.transform(X)

        # Remove the prefixes from the column names
        X.columns = [c.split("__", 1)[1] for c in X.columns]

        # Reapply the correct dtypes
        for idx, col in enumerate(categorical_cols):
            X[col] = X[col].astype(categories[col])

        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, remove_high_cardinality=False, onehot_drop=None):
        self.remove_high_cardinality = remove_high_cardinality
        self.high_cardinality_features = {}  # Track high cardinality features
        self.high_cardinality_limit = (
            20  # Limit above which a feature is considered to have high cardinality
        )

        self.onehot_drop = onehot_drop

        self.one_hot_prefix = "onehot"
        self.passthrough_prefix = "passthrough"

        self.fitted = False

        self.column_transformer = None  # Initialized in fit

    def set_output(
        self, *, transform: Literal["default", "pandas"] | None = None
    ) -> BaseEstimator:
        print(
            "This custom transformer always takes a pandas array as input and returns a pandas dataframe as output."
        )

        return self

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

        # Identify categorical columns
        # Use a dict as dicts are ordered since Python 3.7
        categorical_cols = {
            name: None for name in X.select_dtypes(include=["category"]).columns
        }

        # Remove high cardinality features
        if self.remove_high_cardinality:
            keys_to_remove = []
            for c in categorical_cols:
                if X[c].nunique() > self.high_cardinality_limit:
                    # NOTE: Keep them as category dtype for successive steps in the pipeline
                    # self.high_cardinality_features[c] = X[c].cat.categories.dtype
                    keys_to_remove.append(c)

            # Need to remove keys in a separate loop as we cannot modify the dict while iterating over it
            for key in keys_to_remove:
                del categorical_cols[key]

        # Create a list of transformers. For categorical columns use OneHotEncoder. For others, just passthrough.
        # The way the ColumnTransformer is built, the relative order of the columns is kept.
        column_transformers = []
        for idx, col in enumerate(X.columns):
            if col in categorical_cols:
                column_transformers.append(
                    (
                        f"{self.one_hot_prefix}_{col}",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            drop=self.onehot_drop,
                        ),
                        [col],
                    )
                )
            else:
                column_transformers.append(
                    (f"{self.passthrough_prefix}_{col}", "passthrough", [col])
                )

        column_transformer = ColumnTransformer(column_transformers)
        column_transformer.set_output(transform="pandas")

        column_transformer.fit(X)

        self.column_transformer = column_transformer

        self.fitted = True

        return self

    def transform(self, X, y=None):
        assert self.fitted, "Must call fit first."
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

        # NOTE: Keep them as category dtype for successive steps in the pipeline
        # Revert high cardinality features to their original data types
        # for c, dtype in self.high_cardinality_features.items():
        #    assert c in X, f"Column {c} not found in X."
        #
        #    X[c] = X[c].astype(dtype)

        X = self.column_transformer.transform(X)

        # Make sure the categorical codes are consistent across multiple transforms
        cat_dtype = CategoricalDtype(categories=[0, 1], ordered=False)

        # Reapply the correct dtypes
        for c in X.columns:
            if c.startswith(self.one_hot_prefix):
                X[c] = X[c].astype(cat_dtype)

        # Remove the prefixes from the column names
        X.columns = [c.split("__", 1)[1] for c in X.columns]

        return X


class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    This custom implementation of the MinMaxScaler was needed, as we need to track categorical features.

    Besides the numerical change of categorical features by scaling them, which is not an issue, their dtype is also
    changed from category to float. This is not desired, as we need to track the categorical features throughout the
    pipeline.
    """

    def __init__(self, onehot_used):
        self.categorical_cols = []
        self.scaler = None  # Initialized in fit

        self.onehot_used = onehot_used

        # TODO: Weird nan bug here
        assert self.onehot_used, "One-hot encoding is not supported in this version."

        self.scaler_prefix = "scaled"
        self.passthrough_prefix = "passthrough"

        self.fitted = False

        self.column_transformer = None  # Initialized in fit

    def set_output(
        self, *, transform: Literal["default", "pandas"] | None = None
    ) -> BaseEstimator:
        print(
            "This custom transformer always takes a pandas array as input and returns a pandas dataframe as output."
        )

        return self

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

        # Identify categorical columns
        # Use a dict as dicts are ordered since Python 3.7
        categorical_cols = {
            name: None for name in X.select_dtypes(include=["category"]).columns
        }

        # Create a list of transformers. For categorical columns use OneHotEncoder. For others, just passthrough.
        # The way the ColumnTransformer is built, the relative order of the columns is kept.
        column_transformers = []
        for idx, col in enumerate(X.columns):
            # Skip MinMaxScaler for categorical columns if one-hot encoding is used.
            if self.onehot_used and col in categorical_cols:
                column_transformers.append(
                    (f"{self.passthrough_prefix}_{col}", "passthrough", [col])
                )
            else:
                column_transformers.append(
                    (
                        f"{self.scaler_prefix}_{col}",
                        MinMaxScaler(),
                        [col],
                    )
                )

        column_transformer = ColumnTransformer(column_transformers)
        column_transformer.set_output(transform="pandas")

        column_transformer.fit(X)

        self.column_transformer = column_transformer

        self.fitted = True

        return self

    def transform(self, X, y=None):
        assert self.fitted, "Must call fit first."
        assert isinstance(X, pd.DataFrame), "X must be a pandas dataframe."

        # Identify categorical columns
        # Use a dict as dicts are ordered since Python 3.7
        categorical_cols = {
            name: None for name in X.select_dtypes(include=["category"]).columns
        }

        # Save the categorical codes
        categories = {}
        # Calculate the transformation for each category by the MinMaxScaler.
        # Currently done to preserve backwards compatibility.

        if not self.onehot_used and len(categorical_cols) > 0:
            # TODO: Implement this correctly.
            raise NotImplementedError("There are edge cases that lead this to fail.")

            # for idx, c in enumerate(categorical_cols):
            #     scaler = self.column_transformer.transformers_[idx][1]
            #     assert (
            #         scaler.clip == False
            #     ), "The MinMaxScaler must not clip the data for this to work."
            #
            #     categories_transformed = X[c].cat.categories.to_numpy()
            #     categories_transformed *= scaler.scale_[0]
            #     categories_transformed += scaler.min_[0]
            #
            #     categories[c] = CategoricalDtype(
            #         categories=categories_transformed, ordered=False
            #     )

        X = self.column_transformer.transform(X)

        # Remove the prefixes from the column names
        X.columns = [c.split("__", 1)[1] for c in X.columns]

        # Restore the category dtype
        if not self.onehot_used:
            for c in categorical_cols:
                X[c] = X[c].astype(categories[c])

        return X


def preprocess_and_impute(
    train_ds,
    test_ds,
    impute,
    one_hot,
    standardize,
    is_classification=True,
    onehot_drop=None,
    dist_shift_append_domain=False,
):
    import warnings

    # ignore all warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        preprocess_kwargs = {}

        # Append the domain as a feature if needed
        # Needs to be done before the pipeline is fitted as this alters the attribute names
        preprocess_dist_shift(
            train_ds, test_ds, preprocess_kwargs, dist_shift_append_domain
        )

        attribute_names = train_ds.attribute_names
        cat_features = train_ds.categorical_feats

        pipeline = fit_preprocessing_pipeline(
            train_ds.x.cpu().numpy(),
            impute,
            one_hot,
            standardize,
            attribute_names,
            cat_features=cat_features,
            onehot_drop=onehot_drop,
        )

        if is_classification:
            y = train_ds.y.long().cpu().numpy()
        else:
            y = train_ds.y.cpu().numpy()

        train_ds_x = pipeline.transform(train_ds.x.cpu().numpy())
        test_ds_xs = {
            test_portion: pipeline.transform(ds.x.cpu().numpy())
            for test_portion, ds in test_ds.items()
        }

        attribute_names = train_ds_x.columns.to_list()
        category_columns = train_ds_x.select_dtypes(include=["category"]).columns
        category_column_indices = [
            train_ds_x.columns.get_loc(col) for col in category_columns
        ]

    return (
        train_ds_x,
        y,
        test_ds_xs,
        attribute_names,
        category_column_indices,
        preprocess_kwargs,
    )


def preprocess_dist_shift(train_ds, test_ds, preprocess_kwargs, append_domain=True):
    if not isinstance(train_ds, DistributionShiftDataset):
        return

    preprocess_kwargs["train_dist_shift_domain"] = train_ds.dist_shift_domain

    if append_domain:
        train_ds.append_domain()
        for test_portion, ds in test_ds.items():
            ds.append_domain()


def fit_preprocessing_pipeline(
    x_train,
    impute,
    one_hot,
    standardize,
    attribute_names,
    cat_features=[],
    onehot_drop=None,
):
    steps = []

    numpy_to_pandas = NumpyToPandasEncoder(
        attribute_names=attribute_names, cat_features=cat_features
    )
    steps.append(("numpy_to_pandas", numpy_to_pandas))

    if impute:
        steps.append(("imputer", CustomImputer()))

    if one_hot:
        steps.append(
            (
                "one_hot_encoder",
                CustomOneHotEncoder(
                    remove_high_cardinality=True, onehot_drop=onehot_drop
                ),
            )
        )

    if standardize:
        steps.append(("scaler", CustomMinMaxScaler(onehot_used=one_hot)))

    pipeline = Pipeline(steps)

    # fit the pipeline
    pipeline.fit(x_train)
    pipeline.set_output(transform="pandas")

    return pipeline
