import warnings

import numpy as np
from hyperopt import hp
import math

from ...tabular_metrics import get_scoring_string

from tabpfn.utils import target_is_multiclass

from ..utils import (
    MULTITHREAD,
    eval_complete_f,
    get_random_seed,
    preprocess_and_impute,
)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Catboost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf

param_grid_hyperopt = {
    "learning_rate": hp.loguniform(
        "learning_rate", math.log(math.pow(math.e, -5)), math.log(1)
    ),
    "random_strength": hp.randint("random_strength", 1, 20),
    "l2_leaf_reg": hp.loguniform("l2_leaf_reg", math.log(1), math.log(10)),
    "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1),
    "leaf_estimation_iterations": hp.randint("leaf_estimation_iterations", 1, 20),
    "iterations": hp.randint(
        "iterations", 100, 4000
    ),  # This is smaller than in paper, 4000 leads to ram overusage
}


def catboost_metric(
    train_ds,
    test_ds,
    metric_used,
    max_time=300,
    no_tune=None,
    gpu_id=None,
    random_state=0,
    device: str = "cpu",
    pipeline_kwargs=None,
    **kwargs,
):
    from catboost import CatBoostClassifier, CatBoostRegressor

    task_type = train_ds.task_type

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    (
        x,
        y,
        test_xs,
        attribute_names,
        categorical_feats,
        preprocess_kwargs,
    ) = preprocess_and_impute(
        train_ds=train_ds,
        test_ds=test_ds,
        impute=False,
        one_hot=False,  # Must be False as otherwise cat_features != categorical_feats anymore.
        standardize=False,
        is_classification=target_is_multiclass(task_type),
        **pipeline_kwargs,
    )

    # TODO: Might put into preprocessing pipeline as well.
    def nan_to_num(x, column_indices, num=-1):
        x.iloc[:, column_indices] = np.nan_to_num(x.iloc[:, column_indices], -1)
        return x

    # TODO: Might put into preprocessing pipeline as well.
    def cat_to_str(x, column_indices):
        for idx in column_indices:
            col_name = x.columns[idx]
            x[col_name] = x[col_name].astype(str)
        return x

    if categorical_feats:
        # Nans in categorical features must be encoded as separate class
        x = nan_to_num(x, categorical_feats)
        test_xs = {
            test_portion: nan_to_num(test_x, categorical_feats)
            for test_portion, test_x in test_xs.items()
        }

        # Categorical columns in catboost have to be integers or strings
        x = cat_to_str(x, categorical_feats)
        test_xs = {
            test_portion: cat_to_str(test_x, categorical_feats)
            for test_portion, test_x in test_xs.items()
        }

    gpu_params = {}
    if device != "cpu":
        import torch

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise ValueError("No GPUs found. Please set device='cpu' or use a GPU.")
        gpu_params = {
            "task_type": "GPU",
            "devices": "0" if gpu_id is None else str(gpu_id),
        }

    def clf_(**params):
        if target_is_multiclass(task_type):
            return CatBoostClassifier(
                loss_function=get_scoring_string(metric_used, usage="catboost"),
                thread_count=MULTITHREAD,
                used_ram_limit="4gb",
                random_seed=get_random_seed(random_state, y, mod=False),
                logging_level="Silent",
                cat_features=train_ds.categorical_feats,
                **gpu_params,
                **params,
            )
        else:
            return CatBoostRegressor(
                loss_function=get_scoring_string(metric_used, usage="catboost"),
                thread_count=MULTITHREAD,
                used_ram_limit="4gb",
                random_seed=get_random_seed(random_state, y, mod=False),
                logging_level="Silent",
                cat_features=train_ds.categorical_feats,
                **gpu_params,
                **params,
            )

    return eval_complete_f(
        x,
        y,
        test_xs,
        task_type,
        preprocess_kwargs,
        "catboost",
        param_grid_hyperopt,
        clf_,
        metric_used,
        max_time,
        no_tune,
        random_state,
        method_name="catboost",
    )
