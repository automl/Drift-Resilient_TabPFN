from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional
from copy import deepcopy

import sklearn
from torch import nn

from tabpfn.local_settings import model_string_config
from tabpfn.scripts.estimator import (
    PreprocessorConfig,
    TabPFNBaseModel,
    TabPFNClassificationConfig,
    TabPFNDistShiftClassificationConfig,
    TabPFNConfig,
    TabPFNModelPathsConfig,
    get_tabpfn,
)

# Best model paths, descending order of performance


def get_model_strings(model_string_config):
    if model_string_config == "LOCAL":
        # Support for local debugging.
        from tabpfn.local_settings import local_model_path

        local_dir = Path(local_model_path).resolve()

        model_strings = {
            "multiclass": [
                {
                    "path": str(local_dir / "model_classification.cpkt"),
                    "wandb_id": "-1",
                },
            ],
            "dist_shift_multiclass": [
                {
                    "path": str(local_dir / "tabpfn_dist_model_1.cpkt"),
                    "wandb_id": "-1",
                },
                {
                    "path": str(local_dir / "tabpfn_dist_model_2.cpkt"),
                    "wandb_id": "-1",
                },
                {
                    "path": str(local_dir / "tabpfn_dist_model_3.cpkt"),
                    "wandb_id": "-1",
                },
            ],
        }

        if not any(
            Path(model_info[0]["path"]).exists()
            for model_info in model_strings.values()
        ):
            raise FileNotFoundError(
                f"Could not find any models locally at path: {local_dir}."
            )
    else:
        raise ValueError(f"Unknown model_string_config {model_string_config=}")

    return model_strings


model_strings = get_model_strings(model_string_config)


### BEST CONFIGS ###

best_tabpfn_configs = {
    "multiclass": {
        "single_fast": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            task_type_config=TabPFNClassificationConfig(),
            N_ensemble_configurations=1,
            save_peak_memory="True",
            optimize_metric="roc",
        ),
        "single": TabPFNConfig(
            model_name="tabpfn_single_v_2_1",
            task_type="multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["multiclass"]]
            ),
            task_type_config=TabPFNClassificationConfig(),
            N_ensemble_configurations=32,
            # save_peak_memory="auto", # TODO: Re-enable when memory tracking is adapted to the specific model
            optimize_metric="roc",
        ),
    },
    "dist_shift_multiclass": {
        "single_fast": TabPFNConfig(
            model_name="tabpfn_single_fast_v_2_1",
            task_type="dist_shift_multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["dist_shift_multiclass"]],
                task_type="dist_shift_multiclass",
            ),
            task_type_config=TabPFNDistShiftClassificationConfig(),
            N_ensemble_configurations=1,
            save_peak_memory="True",
            optimize_metric="roc",
        ),
        "single": TabPFNConfig(
            model_name="tabpfn_single_v_2_1",
            task_type="dist_shift_multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["dist_shift_multiclass"]],
                task_type="dist_shift_multiclass",
            ),
            task_type_config=TabPFNDistShiftClassificationConfig(),
            N_ensemble_configurations=32,
            # save_peak_memory="auto", # TODO: Re-enable when memory tracking is adapted to the specific model
            optimize_metric="roc",
        ),
        "best_dist": TabPFNConfig(
            model_name="tabpfn_best_dist_v_2_1",
            task_type="dist_shift_multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["dist_shift_multiclass"]],
                task_type="dist_shift_multiclass",
            ),
            preprocess_transforms=(
                PreprocessorConfig(
                    name="robust",
                    append_original=True,
                    categorical_name="numeric",
                    subsample_features=-1,
                    global_transformer_name="None",
                ),
            ),
            softmax_temperature=-0.10536051565782628,
            average_logits=False,
            task_type_config=TabPFNDistShiftClassificationConfig(
                multiclass_decoder="shuffle"
            ),
            use_poly_features=False,
            add_fingerprint_features=True,
            remove_outliers=7.0,
            N_ensemble_configurations=32,
            save_peak_memory="True",
            batch_size_inference=1,
            optimize_metric="roc",
            subsample_samples=0.99,
            feature_shift_decoder="shuffle",
            fp16_inference=True,
        ),
        "best_base": TabPFNConfig(
            model_name="tabpfn_best_base_v_2_1",
            task_type="dist_shift_multiclass",
            model_type="single",
            paths_config=TabPFNModelPathsConfig(
                paths=[m["path"] for m in model_strings["dist_shift_multiclass"]],
                task_type="dist_shift_multiclass",
            ),
            preprocess_transforms=(
                PreprocessorConfig(
                    name="safepower",
                    append_original=True,
                    categorical_name="onehot",
                    subsample_features=0.9,
                    global_transformer_name="svd",
                ),
            ),
            softmax_temperature=-0.2876820724517809,
            average_logits=False,
            task_type_config=TabPFNDistShiftClassificationConfig(
                multiclass_decoder="shuffle"
            ),
            use_poly_features=False,
            add_fingerprint_features=True,
            remove_outliers=9.0,
            N_ensemble_configurations=32,
            save_peak_memory="True",
            batch_size_inference=1,
            optimize_metric="roc",
            subsample_samples=0.99,
            feature_shift_decoder="shuffle",
            fp16_inference=True,
        ),
    },
}

best_tabpfn_configs["multiclass"]["best"] = best_tabpfn_configs["multiclass"]["single"]


def get_best_tabpfn_config(
    task_type: str,
    model_type: str = "single_fast",
    debug: bool = False,
    model: Optional[nn.Module] = None,
    paths_config=None,
    c: Optional[Dict] = None,
    return_list_of_config_per_model_string: bool = False,
) -> TabPFNConfig | list[TabPFNConfig]:
    config = copy.deepcopy(best_tabpfn_configs[task_type][model_type])

    # In debug mode sets small parameters for the best models
    if debug:
        config.N_ensemble_configurations = min(2, config.N_ensemble_configurations)

    assert not (model and paths_config), "Only one of model and paths_config can be set"
    if model:
        config.model = model
        config.c = c
        config.paths_config = None

    if paths_config:
        config.paths_config = paths_config
        config.model = None
        config.c = None

    if return_list_of_config_per_model_string:
        assert (
            config.paths_config is not None
        ), "paths_config must be set to return list of configs per model string!"

        config_per_model_string = []
        for model_string in config.paths_config.model_strings:
            tmp_config = copy.deepcopy(config)
            tmp_config.paths_config = TabPFNModelPathsConfig(
                paths=[model_string], task_type=task_type
            )
            config_per_model_string.append(tmp_config)

        return config_per_model_string

    return config


### GETTING BEST MODELS ###


def _infer_config_overwrite(config, inference_config_overwrite, verbose=False):
    """
    This function looks for the keys in inference_config_overwrite and overwrites the corresponding value in either config, config.model_type_config, config.paths_config or config.task_type_config.
    """
    sub_configs = [
        config.model_type_config,
        config.paths_config,
        config.task_type_config,
    ]
    for key in inference_config_overwrite.keys():
        if key in config.__dict__:
            config.__dict__[key] = inference_config_overwrite[key]
        else:
            found = False
            for sub_config in sub_configs:
                if sub_config and key in sub_config.__dict__:
                    sub_config.__dict__[key] = inference_config_overwrite[key]
                    found = True
                    if verbose:
                        print(f"updated {key=} in subconfig")
                    break

            if not found:
                raise ValueError(f"Unknown config key {key}")
    if verbose:
        print("building tabpfn model with config", config)


def get_best_tabpfn(
    task_type: str,
    model_type: str = "single_fast",
    paths_config: TabPFNModelPathsConfig = None,
    model=None,
    c=None,
    debug: bool = False,  # If True, uses small parameters for the best models. Useful for debugging and testing.
    inference_config_overwrite: dict = None,
    **kwargs,
) -> TabPFNBaseModel:
    inference_config_overwrite = deepcopy(inference_config_overwrite)
    if (
        inference_config_overwrite is not None
        and "paths_config" in inference_config_overwrite
    ):
        assert (
            paths_config is None
        ), "paths_config can't be set if it's in inference_config_overwrite"
        paths_config = inference_config_overwrite.pop("paths_config")
    config = get_best_tabpfn_config(
        task_type, model_type, debug=debug, model=model, c=c, paths_config=paths_config
    )
    if inference_config_overwrite is not None:
        _infer_config_overwrite(config, inference_config_overwrite)

    return get_tabpfn(config, **kwargs)


def get_all_best_tabpfns(
    task_type: str, debug: bool = False, **kwargs
) -> Dict[str, TabPFNBaseModel]:
    configs = {
        model_type: get_best_tabpfn_config(task_type, model_type, debug=debug)
        for model_type in best_tabpfn_configs[task_type].keys()
    }

    return {k: get_tabpfn(config, **kwargs) for (k, config) in configs.items()}
