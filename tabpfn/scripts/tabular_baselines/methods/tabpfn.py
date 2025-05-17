from __future__ import annotations
import warnings

import numpy as np
from ...tabular_baselines.utils import get_random_seed
from ...tabular_evaluation_utils import DatasetEvaluation
from ...estimator import TabPFNModelPathsConfig
from tabpfn.utils import print_once


def _get_task_type(task_type, *, metric_used):
    assert task_type in [
        "regression",
        "multiclass",
        "quantile_regression",
        "dist_shift_multiclass",
    ], f"Metric is {task_type}"

    # TODO: Already overwritten to regression in dataset init, right? So this could be removed, right?
    if task_type == "quantile_regression":
        task_type = "regression"

    return task_type


def _warnings(*, max_time):
    if max_time is not None:
        print_once(
            "Maximum time is not enforced on the transformer metric, it is only used too keep the same interface as the other metrics.",
        )


def _get_model(*, input_model, model_type, paths_config, task_type, device, seed):
    from tabpfn.best_models import get_best_tabpfn

    match input_model:
        case None:
            match paths_config:
                case None:
                    print("Using default paths_config.")
                    return get_best_tabpfn(
                        task_type, device=device, model_type=model_type, seed=seed
                    )
                case _:
                    print(f"Using custom paths_config: {paths_config}")
                    return get_best_tabpfn(
                        task_type,
                        device=device,
                        model_type=model_type,
                        seed=seed,
                        paths_config=paths_config,
                    )
        case _:
            return input_model


def _determine_old_tabpfn(*, input_model) -> bool:
    old_tabpfn = not hasattr(input_model, "set_categorical_features")
    if old_tabpfn:
        warnings.warn(
            "The old TabPFN model is not informed about categorical features, as it does not support `set_categorical_features`.",
            stacklevel=2,
        )
    return old_tabpfn


def _fit(*, input_model, task_type, train_ds, old_tabpfn):
    if not old_tabpfn:
        input_model.set_categorical_features(train_ds.categorical_feats)

    if task_type == "dist_shift_multiclass":
        assert (
            train_ds.concatenated_domain == False
        ), "The domain information has already been concatenated."

        input_model.fit(
            train_ds.x,
            train_ds.y,
            additional_x={"dist_shift_domain": train_ds.dist_shift_domain},
        )
    elif old_tabpfn:
        input_model.fit(train_ds.x, train_ds.y, overwrite_warning=True)
    else:
        input_model.fit(train_ds.x, train_ds.y)


def _predict(*, input_model, task_type, test_ds, quantiles, full_predict):
    additional_args = {}
    pred_full = {}

    if task_type == "multiclass":
        # TabPFNClassifier
        pred = input_model.predict_proba(test_ds.x)
    elif task_type == "regression":
        # TabPFNRegressor
        if hasattr(input_model, "predict_full") and (
            quantiles is not None or full_predict
        ):
            pred_full = input_model.predict_full(test_ds.x)
            if quantiles is not None:
                pred = np.stack([pred_full[f"quantile_{q:.2f}"] for q in quantiles], 1)
            else:
                pred = pred_full[input_model.get_optimization_mode()]
        else:
            pred = input_model.predict(test_ds.x)
    elif task_type == "dist_shift_multiclass":  # TabPFNDistShiftClassifier
        pred = input_model.predict_proba(
            test_ds.x, additional_x={"dist_shift_domain": test_ds.dist_shift_domain}
        )
    else:
        raise NotImplementedError(f"Metric is {task_type}")

    if full_predict:
        additional_args["pred_full"] = pred_full
    return pred, additional_args


def transformer_metric(
    train_ds,
    test_ds,
    metric_used,
    max_time=None,
    random_state=None,
    device="cpu",
    classifier=None,  # keep name classifier for backwards compatibility
    model_type: str = "single",  # ["single_light", "single_fast"]
    # Allows to hardcode the used model paths
    paths_config: None | TabPFNModelPathsConfig = None,
    full_predict: bool = False,
    pipeline_kwargs=None,
    **kwargs,  # unused but required for backwards compatibility
):
    task_type = _get_task_type(task_type=train_ds.task_type, metric_used=metric_used)

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    _warnings(max_time=max_time)
    seed = get_random_seed(random_state, train_ds.y, mod=False)
    model = _get_model(
        input_model=classifier,
        model_type=model_type,
        paths_config=paths_config,
        task_type=task_type,
        device=device,
        seed=seed,
    )
    old_tabpfn = _determine_old_tabpfn(input_model=model)

    if task_type == "dist_shift_multiclass":
        dist_shift_append_domain = pipeline_kwargs["dist_shift_append_domain"]
        model.set_append_domain(append_domain=dist_shift_append_domain)

    _fit(
        input_model=model, task_type=task_type, train_ds=train_ds, old_tabpfn=old_tabpfn
    )

    evaluations = {}
    for test_portion, ds in test_ds.items():
        pred, additional_args = _predict(
            input_model=model,
            task_type=task_type,
            test_ds=ds,
            quantiles=getattr(metric_used, "quantiles", None),
            full_predict=full_predict,
        )

        evaluations[test_portion] = DatasetEvaluation(
            y=None, pred=pred, additional_args=additional_args, algorithm_name="TabPFN"
        )

    return evaluations
