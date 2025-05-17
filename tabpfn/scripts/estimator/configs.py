from __future__ import annotations

import dataclasses
from functools import lru_cache

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict, Literal
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
import math
import torch

from tabpfn.local_settings import wandb_entity, get_wandb_project


def get_params_from_config(c):
    return (
        {}  # here you add things that you want to use from the config to do inference in transformer_predict
    )


@dataclass(eq=True, frozen=True)
class PreprocessorConfig:
    """
    Configuration for data preprocessors.

    Attributes:
        name (Literal): Name of the preprocessor.
        categorical_name (Literal): Name of the categorical encoding method. Valid options are "none", "numeric",
                                "onehot", "ordinal", "ordinal_shuffled". Default is "none".
        append_original (bool): Whether to append the original features to the transformed features. Default is False.
        subsample_features (float): Fraction of features to subsample. -1 means no subsampling. Default is -1.
        global_transformer_name (str): Name of the global transformer to use. Default is None.
    """

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # different quantile transformations with few quantiles up to a lot
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (inside the transformer we anyways do a standardization)
    ]
    categorical_name: Literal[
        "none", "numeric", "onehot", "ordinal", "ordinal_shuffled"
    ] = "none"
    # categorical_name meanings:
    # "none": categorical features are pretty much treated as ordinal, just not resorted
    # "numeric": categorical features are treated as numeric, that means they are also power transformed for example
    # "onehot": categorical features are onehot encoded
    # "ordinal": categorical features are sorted and encoded as integers from 0 to n_categories - 1
    # "ordinal_shuffled": categorical features are encoded as integers from 0 to n_categories - 1 in a random order
    append_original: bool = False
    subsample_features: Optional[float] = -1
    global_transformer_name: Optional[str] = None
    # if True, the transformed features (e.g. power transformed) are appended to the original features

    def __str__(self):
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                "_subsample_feats_" + str(self.subsample_features)
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )

    def can_be_cached(self):
        return not self.subsample_features > 0

    def to_dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class EnsembleConfiguration:
    """
    Configuration for an ensemble member.

    Attributes:
        class_shift_configuration (torch.Tensor | None): Permutation to apply to classes. Only used for classification.
        feature_shift_configuration (int | None): Random seed for feature shuffling.
        preprocess_transform_configuration (PreprocessorConfig): Preprocessor configuration to use.
        styles_configuration (int | None): Styles configuration to use.
        subsample_samples_configuration (int | None): Indices of samples to use for this ensemble member.
    """

    class_shift_configuration: torch.Tensor | None = None
    feature_shift_configuration: int | None = None
    preprocess_transform_configuration: PreprocessorConfig = PreprocessorConfig("none")
    styles_configuration: int | None = None
    subsample_samples_configuration: int | None = None


@dataclass
class TabPFNConfig:
    """
    Configuration for TabPFN models.

    Check TabPFNBaseEstimator for more information on attributes.
    """

    task_type: str
    model_type: Literal[
        "best",
        "single",
        "single_fast",
    ]
    paths_config: TabPFNModelPathsConfig
    task_type_config: TabPFNClassificationConfig | TabPFNRegressionConfig | TabPFNDistShiftClassificationConfig | None = (
        None
    )
    model_type_config: None = None

    model_name: str = "tabpfn"  # This name will be tracked on wandb

    preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
        PreprocessorConfig("safepower", categorical_name="numeric"),
        PreprocessorConfig("power", categorical_name="numeric"),
    )
    feature_shift_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate", "auto_rotate"
    ] = "shuffle"  # local_shuffle breaks, because no local configs are generated with high feature number
    normalize_with_test: bool = False
    fp16_inference: bool = True
    N_ensemble_configurations: int | None = 1
    average_logits: bool = True
    transformer_predict_kwargs: Optional[Dict] = field(default_factory=dict)
    save_peak_memory: Literal["True", "False", "auto"] = "True"
    batch_size_inference: int = None
    add_fingerprint_features: bool = False
    max_poly_features: int = 50
    use_poly_features: bool = True
    softmax_temperature: float = math.log(0.9)
    subsample_samples: float = -1
    remove_outliers: float = -1

    optimize_metric: Optional[str | None] = None
    c: Optional[Dict] = field(default_factory=dict)
    model: Optional = None

    def to_kwargs(self):
        kwargs = dataclasses.asdict(self)
        del kwargs["task_type"]
        del kwargs["model_type"]

        if self.task_type_config is not None:
            kwargs.update(dataclasses.asdict(self.task_type_config))

        if (
            kwargs.get("paths_config", None) is not None
            and kwargs.get("model", None) is not None
        ):
            raise ValueError(
                "Either paths_config or model must be specified, not both."
            )
        elif kwargs.get("paths_config", None) is not None:
            assert (
                len(kwargs["paths_config"]["model_strings"]) == 1
            ), "Only one model can be used as config for our TabPFNBaseEstimator models."
            kwargs["model_string"] = kwargs["paths_config"]["model_strings"][0]
        elif kwargs.get("model", None) is not None:
            kwargs["model_string"] = "tabpfn"
        else:
            raise ValueError(
                f"Either paths_c"
                f"onfig or model must be specified paths_config {kwargs.get('paths_config', None)} model {kwargs.get('model', None)}"
            )

        del kwargs["paths_config"]
        del kwargs["task_type_config"]
        del kwargs["model_type_config"]
        del kwargs["model_name"]

        return kwargs

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            + ", ".join(f"{k}: {repr(v)[:100]}" for k, v in asdict(self).items())
            + ")"
        )


@dataclass
class TabPFNModelPathsConfig:
    """
    paths: list of model paths or WANDB_IDs, e.g. ["/path/to/model.pth", "WANDB_ID:1a2b3c4d"]

    The model_strings attribute is automatically generated from the paths attribute, and contains the actual paths to the models on our cluster.
    If a WANDB_ID is given, the model_string is automatically fetched from WANDB.
    If a path is given, the model_string is the path itself.
    """

    paths: list[str]

    model_strings: list[str] = dataclasses.field(init=False)

    task_type: str = "multiclass"

    def __post_init__(self):
        # Initialize Model paths
        self.model_strings = []

        for path in self.paths:
            self.model_strings.append(path)

    def to_dict(self):
        return dataclasses.asdict(self)


### TASK TYPE CONFIGS ###


@dataclass
class TabPFNClassificationConfig:
    multiclass_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate"
    ] = "shuffle"


@dataclass
class TabPFNRegressionConfig:
    regression_y_preprocess_transforms: Tuple[str, ...] = (
        None,
        "power",
    )
    cancel_nan_borders: bool = False
    super_bar_dist_averaging: bool = False


@dataclass
class TabPFNDistShiftClassificationConfig:
    multiclass_decoder: Literal[
        "shuffle", "none", "local_shuffle", "rotate"
    ] = "shuffle"
