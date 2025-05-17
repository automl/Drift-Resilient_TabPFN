import warnings
import math
import torch

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from ...tabular_metrics import get_scoring_string

from tabpfn.utils import target_is_multiclass

from ..utils import (
    preprocess_and_impute,
    eval_complete_f,
    MULTITHREAD,
)

from hyperopt import hp

# XGBoost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
param_grid_hyperopt = {
    "learning_rate": hp.loguniform("learning_rate", -7, math.log(1)),
    "max_depth": hp.randint("max_depth", 1, 10),
    "subsample": hp.uniform("subsample", 0.2, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.2, 1),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0.2, 1),
    "min_child_weight": hp.loguniform("min_child_weight", -16, 5),
    "alpha": hp.loguniform("alpha", -16, 2),
    "lambda": hp.loguniform("lambda", -16, 2),
    "gamma": hp.loguniform("gamma", -16, 2),
    "n_estimators": hp.randint(
        "n_estimators", 100, 4000
    ),  # This is smaller than in paper
}


def xgb_metric(
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
    import xgboost as xgb

    # XGB Documentation:
    # XGB handles categorical data appropriately without using One Hot Encoding, categorical features are experimental
    # XGB handles missing values appropriately without imputation

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
        one_hot=False,
        standardize=False,
        is_classification=target_is_multiclass(task_type),
        **pipeline_kwargs,
    )

    enable_categorical = True

    def category_to_int(x):
        for c in x.columns:
            if x[c].dtype == "category":
                x[c] = pd.to_numeric(x[c], errors="coerce")

        return x

    if not enable_categorical:
        x = category_to_int(x)
        test_xs = {
            test_portion: category_to_int(test_x)
            for test_portion, test_x in test_xs.items()
        }

    gpu_params = {}
    if device != "cpu":
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise ValueError("No GPUs found. Please set device='cpu' or use a GPU.")
        gpu_params = {
            "tree_method": "hist",
            "device": f"cuda:{0 if gpu_id is None else gpu_id}",
        }

    def clf_(**params):
        if target_is_multiclass(task_type):
            return xgb.XGBClassifier(
                use_label_encoder=False,
                nthread=MULTITHREAD,
                num_class=len(np.unique(y)),
                enable_categorical=enable_categorical,
                **params,
                **gpu_params,
                objective=get_scoring_string(metric_used, usage="xgb"),
            )
        else:
            return xgb.XGBRegressor(
                use_label_encoder=False,
                nthread=MULTITHREAD,
                enable_categorical=enable_categorical,
                **params,
                **gpu_params,
                objective=get_scoring_string(metric_used, usage="xgb"),
            )

    return eval_complete_f(
        x,
        y,
        test_xs,
        task_type,
        preprocess_kwargs,
        "xgb",
        param_grid_hyperopt,
        clf_,
        metric_used,
        max_time,
        no_tune,
        random_state,
        method_name="xgb",
    )
