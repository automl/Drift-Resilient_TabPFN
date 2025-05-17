import math
import warnings

import numpy as np
from hyperopt import hp

from ...tabular_baselines.utils import (
    eval_complete_f,
    preprocess_and_impute,
)
from ...tabular_metrics import get_scoring_string

from tabpfn.utils import target_is_multiclass

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


param_grid_hyperopt = {
    "num_leaves": hp.randint("num_leaves", 5, 50),
    "max_depth": hp.randint("max_depth", 3, 20),
    "learning_rate": hp.loguniform("learning_rate", -3, math.log(1.0)),
    "n_estimators": hp.randint("n_estimators", 50, 2000)
    # , 'feature_fraction': 0.8,
    # , 'subsample': 0.2
    ,
    "min_child_weight": hp.choice(
        "min_child_weight", [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    ),
    "subsample": hp.uniform("subsample", 0.2, 0.8),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.2, 0.8),
    "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
    "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 5, 10, 20, 50, 100]),
}  # 'normalize': [False],


def lightgbm_metric(
    train_ds,
    test_ds,
    metric_used,
    max_time=300,
    no_tune=None,
    random_state=0,
    pipeline_kwargs=None,
    **kwargs,
):
    from lightgbm import LGBMClassifier, LGBMRegressor

    task_type = train_ds.task_type

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    (
        x,
        y,
        test_xs,
        attribute_names,
        cat_features,
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

    def clf_(**params):
        objective = get_scoring_string(
            metric_used, usage="lightgbm", multiclass=len(np.unique(y)) > 2
        )
        if target_is_multiclass(task_type):
            return LGBMClassifier(
                use_missing=True,
                objective=objective,
                **params,
            )
        return LGBMRegressor(
            use_missing=True,
            objective=objective,
            **params,
        )

    return eval_complete_f(
        x,
        y,
        test_xs,
        task_type,
        preprocess_kwargs,
        "lightgbm",
        param_grid_hyperopt,
        clf_,
        metric_used,
        max_time,
        no_tune,
        random_state,
    )
