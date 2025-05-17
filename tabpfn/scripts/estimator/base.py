from __future__ import annotations

import itertools
import math
import os
import random
import traceback
import typing as tp
import warnings
from copy import copy, deepcopy
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from packaging import version
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted

from tabpfn import utils
from tabpfn.scripts.estimator import preprocessing
from .configs import (
    EnsembleConfiguration,
    PreprocessorConfig,
    TabPFNConfig,
    get_params_from_config,
)
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from ..model_builder import load_model
from tabpfn.utils import (
    NOP,
    normalize_data,
    print_once,
    timing_clear,
    timing_collect,
    timing_end,
    timing_start,
)
from .scoring_utils import score_regression, score_classification

SAVE_PEAK_MEM_FACTOR = 8

# a flag needed to allow displaying the user a warning if we need to change the model to work on CPU
# to be removed when pytorch fixes the issue
did_change_model = False


class TabPFNBaseModel(BaseEstimator):
    """
    Base class for TabPFN models.

    This class provides common functionality for all TabPFN models, including input validation,
    data preprocessing, ensemble configuration generation, and prediction.

    Attributes:
        model_processed_ (Any): The processed model object.
        c_processed_ (Dict): The processed model configuration dictionary.
        is_classification_ (bool): Whether this is a classification model.
        num_classes_ (int): The number of classes for classification models.
        _rnd (np.random.Generator): The random number generator to use.
    """

    def _more_tags(self):
        return {"allow_nan": True}

    models_in_memory = {}
    semisupervised_indicator = np.nan

    maximum_free_memory_in_gb: Optional[float] = None
    computational_budget = 4.5e12
    processor_speed: float = 1.0
    sklearn_compatible_precision: bool = False
    inference_mode: bool = True
    predict_function_for_shap = "predict"

    categorical_features: List[int] = []

    cached_preprocessors_ = None
    cached_models_ = None
    cached_shuffles_ = None

    def __init__(
        self,
        # Tunable parameters
        model: Optional[Any] = None,
        c: Optional[Dict] = None,
        model_string: str = "",
        N_ensemble_configurations: int = 10,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        feature_shift_decoder: Literal[
            "shuffle", "none", "local_shuffle", "rotate", "auto_rotate"
        ] = "shuffle",  # local_shuffle breaks, because no local configs are generated with high feature number
        average_logits: bool = False,
        optimize_metric: Optional[str] = None,
        transformer_predict_kwargs: Optional[Dict] = None,
        model_name: str = "tabpfn",  # This name will be tracked on wandb
        softmax_temperature: Optional[float] = math.log(0.8),
        use_poly_features=False,
        max_poly_features=None,
        remove_outliers=0.0,
        add_fingerprint_features=False,
        subsample_samples: Optional[float] = -1,
        transductive=False,
        normalize_with_test: bool = False,
        # The following parameters are not tunable, but specify the execution mode
        fit_at_predict_time: bool = True,
        device: tp.Literal["cuda", "cpu", "auto"] = "auto",
        seed: Optional[int] = 0,
        show_progress: bool = True,
        batch_size_inference: int = None,
        fp16_inference: bool = True,
        save_peak_memory: Literal["True", "False", "auto"] = "True",
    ) -> None:
        """
        You need to specify a model either by setting the `model_string` or by setting `model` and `c`,
        where the latter is the config.

        Parameters:
            model_string: The model string is the path to the model
            preprocess_transforms: A tuple of strings, specifying the preprocessing steps to use.
                You can use the following strings as elements '(none|power|quantile|robust)[_all][_and_none]', where the first
                part specifies the preprocessing step and the second part specifies the features to apply it to and
                finally '_and_none' specifies that the original features should be added back to the features in plain.
                Finally, you can combine all strings without `_all` with `_onehot` to apply one-hot encoding to the categorical
                features specified with `self.fit(..., categorical_features=...)`.
            feature_shift_decoder: ["False", "True", "auto"] Whether to shift features for each ensemble configuration
            model: The model, if you want to specify it directly, this is used in combination with c
            c: The config, if you want to specify it directly, this is used in combination with model
            seed: The default seed to use for the model. If None, a random seed is used and each fit and predict call
                will yield different predictions otherwise they will be deterministic based on the seed.
            device: The device to use for inference, "auto" means that it will use cuda if available, otherwise cpu
            fp16_inference: Whether to use fp16 for inference on GPU, does not affect CPU inference.
            N_ensemble_configurations: The number of ensemble configurations to use, the most important setting
            batch_size_inference: The batch size to use for inference, this does not affect the results, just the
                memory usage and speed. A higher batch size is faster but uses more memory. Setting the batch size to None
                means that the batch size is automatically determined based on the memory usage and the maximum free memory
                specified with `maximum_free_memory_in_gb`.
            normalize_with_test: If True, the test set is used to normalize the data, otherwise the training set is used only.
            average_logits: Whether to average logits or probabilities for ensemble members
            save_peak_memory: Whether to save the peak memory usage of the model, can enable up to 8 times larger datasets to fit into memory.
                "True", means always enabled, "False", means always disabled, "auto" means that it will be set based on the memory usage
            use_poly_features: Whether to use polynomial features as the last preprocessing step
            max_poly_features: Maximum number of polynomial features to use, None means unlimited
            add_fingerprint_features: If True, will add one feature of random values, that will be added to
                the input features. This helps discern duplicated samples in the transformer model.
            fit_at_predict_time: Whether to train the model lazily, i.e. only when it is needed for inference in predict[_proba]
            subsample_samples: If not None, will use a random subset of the samples for training in each ensemble configuration
            remove_outliers: If not 0.0, will remove outliers from the input features, where values with a standard deviation
                larger than remove_outliers will be removed.
        """
        self.device = device
        self.model = model
        self.c = c
        self.N_ensemble_configurations = N_ensemble_configurations
        self.model_string = model_string
        self.batch_size_inference = batch_size_inference
        self.fp16_inference = fp16_inference

        self.feature_shift_decoder = feature_shift_decoder
        self.seed = seed
        self.softmax_temperature = softmax_temperature
        self.normalize_with_test = normalize_with_test
        self.average_logits = average_logits
        self.optimize_metric = optimize_metric
        self.transformer_predict_kwargs = transformer_predict_kwargs
        self.preprocess_transforms = preprocess_transforms
        self.show_progress = show_progress
        self.model_name = model_name

        self.save_peak_memory = save_peak_memory

        self.subsample_samples = subsample_samples
        self.use_poly_features = use_poly_features
        self.max_poly_features = max_poly_features
        self.transductive = transductive
        self.remove_outliers = remove_outliers

        self.add_fingerprint_features = add_fingerprint_features
        self.fit_at_predict_time = fit_at_predict_time

    def optimizes_balanced_metric(self):
        if self.optimize_metric is None:
            return False
        return self.optimize_metric in ("balanced_acc", "auroc_ovo")

    def set_categorical_features(self, categorical_features):
        self.categorical_features = categorical_features

    def _init_rnd(self):
        seed = self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
        # utils.print_once("using seed, ", seed, "with self.device", self.device)
        self._rnd = np.random.default_rng(seed)
        if hasattr(self, "model_processed_"):
            self.model_processed_.seed = self.seed
            self.model_processed_._init_rnd()

    def init_model_and_get_model_config(self) -> None:
        """
        Initialize the model and its associated configuration.

        It loads the model and its configuration into memory. If the device is CPU, it also resets the attention bias
        in PyTorch's MultiheadAttention module if needed.
        """
        self.device_ = self.device
        if self.device_ == "auto":
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_.startswith("cpu") and torch.cuda.is_available():
            warnings.warn(
                "You are using a CPU device, but CUDA is available. This will likely lead to inferior performance."
                "Fix this by setting device='cuda'."
            )
        self.normalize_std_only_ = False
        # Model file specification (Model name, Epoch)
        if self.model is None:
            model_key = (self.model_string, self.device_)
            if model_key in TabPFNBaseModel.models_in_memory:
                (
                    self.model_processed_,
                    self.c_processed_,
                ) = TabPFNBaseModel.models_in_memory[model_key]
            else:
                self.model_processed_, self.c_processed_ = load_model(
                    self.model_string, self.device_, verbose=False
                )
                self.model_processed_ = self.model_processed_[2]
                TabPFNBaseModel.models_in_memory[model_key] = (
                    self.model_processed_,
                    self.c_processed_,
                )
        else:
            assert (
                self.c is not None and len(self.c.keys()) > 0
            ), "If you specify the model you need to set the config, c"
            self.model_processed_ = self.model
            self.c_processed_ = self.c
        # style, temperature = self.load_result_minimal(style_file, i, e)

        # Set variables
        self.is_classification_ = not hasattr(
            self.model_processed_.criterion, "borders"
        )

        if not self.is_classification_:
            self.num_classes_ = None  # this defines the end of the indexes used in softmax, we just want all and [:None] is the whole range.

        if self.device_.startswith("cpu"):
            global did_change_model
            did_change_model = False

            def reset_attention_bias(module):
                global did_change_model
                if isinstance(module, torch.nn.MultiheadAttention):
                    if module.in_proj_bias is None:
                        module.in_proj_bias = torch.nn.Parameter(
                            torch.zeros(3 * module.kdim)
                        )
                        did_change_model = True
                    if module.out_proj.bias is None:
                        module.out_proj.bias = torch.nn.Parameter(
                            torch.zeros(1 * module.kdim)
                        )
                        did_change_model = True

            self.model_processed_.apply(reset_attention_bias)
            if did_change_model:
                print_once(
                    "Changed model to be compatible with CPU, this is needed for the current version of "
                    "PyTorch, see issue: https://github.com/pytorch/pytorch/issues/97128. "
                    "The model will be slower if reused on GPU."
                )

        if hasattr(self.model_processed_, "to_device_for_forward"):
            self.model_processed_.to_device_for_forward(self.device_)
        else:
            self.model_processed_.to(self.device_)
        self.model_processed_.eval()

        self.max_num_features_ = self.c_processed_.get("num_features")
        if self.max_num_features_ is None:
            self.max_num_features_ = self.c_processed_.get(
                "max_num_features_in_training"
            )
            if self.max_num_features_ is None:
                print("falling back to unlimited features")
                self.max_num_features_ = -1
        if "PerFeat" in type(self.model_processed_).__name__:
            self.max_num_features_ = -1  # Unlimited features

        self.differentiable_hps_as_style_ = self.c_processed_[
            "differentiable_hps_as_style"
        ]
        if self.c_processed_.get("features_per_group", self.max_num_features_) == 1:
            # print_once(
            #     "The feature shuffling is turned off, as your model only has one feature per group. It thus is permutation invariant."
            # )
            self.feature_shift_decoder = "none"

        style = None  # Currently we do not support style, code left for later usage
        self.num_styles_, self.style_ = self.init_style(
            style, differentiable_hps_as_style=self.differentiable_hps_as_style_
        )

        self._poly_ignore_nan_features = True
        self._poly_degree = 2
        self.semisupervised_enabled_ = self.c_processed_.get(
            "semisupervised_enabled", False
        )
        self.adapt_encoder()

        self.preprocess_transforms = [
            PreprocessorConfig(**p) if type(p) is dict else p
            for p in self.preprocess_transforms
        ]

        self._save_peak_memory = False

        self._init_rnd()

    def _adapt_encoder(self, encoder):
        norm_layer = [
            e for e in encoder if "InputNormalizationEncoderStep" in str(e.__class__)
        ][0]
        norm_layer.remove_outliers = self.remove_outliers and self.remove_outliers > 0.0
        if norm_layer.remove_outliers:
            norm_layer.remove_outliers_sigma = self.remove_outliers
        norm_layer.seed = self.seed
        norm_layer.reset_seed()

    def adapt_encoder(self):
        # TODO: EnsembleTabPFN and other models, need to adapt this function
        if hasattr(self.model_processed_, "encoder"):
            self._adapt_encoder(self.model_processed_.encoder)

    def get_save_peak_memory(
        self, X, safety_factor: float = 5.0, **overwrite_params
    ) -> bool:
        if self.save_peak_memory == "True":
            return True
        elif self.save_peak_memory == "False":
            return False
        elif self.save_peak_memory == "auto":
            return (
                self.estimate_memory_usage(
                    X, "gb", save_peak_mem_factor=False, **overwrite_params
                )
                * safety_factor
                > self.get_max_free_memory_in_gb()
            )
        else:
            raise ValueError(
                f"Unknown value for save_peak_memory {self.save_peak_memory}"
            )

    def get_batch_size_inference(self, X, safety_factor=4.0, **overwrite_params) -> int:
        if self.batch_size_inference is None:
            if self.device_.startswith("cpu"):
                return 1  # No batching on CPU
            capacity = self.get_max_free_memory_in_gb()
            usage = self.estimate_memory_usage(
                X, "gb", batch_size_inference=1, **overwrite_params
            )
            estimated_max_size = math.floor(capacity / usage / safety_factor)
            estimated_max_size = max(1, estimated_max_size)

            return estimated_max_size
        else:
            return self.batch_size_inference

    def is_initialized(self):
        return hasattr(self, "model_processed_")

    @staticmethod
    def check_training_data(
        clf: TabPFNBaseModel, X: np.ndarray, y: np.ndarray
    ) -> Tuple:
        """
        Validates the input training data X and y.

        Parameters:
            clf (TabPFNBaseModel): The TabPFN model object.
            X (ndarray): The input feature matrix of shape (n_samples, n_features).
            y (ndarray): The target vector of shape (n_samples,).

        Returns:
            Tuple[ndarray, ndarray]: The validated X and y.

        Raises:
            ValueError: If the number of features in X exceeds the maximum allowed for the model.
                        If X and y have inconsistent lengths.
        """
        if clf.max_num_features_ > -1 and X.shape[1] > clf.max_num_features_:
            raise ValueError(
                "The number of features for this classifier is restricted to ",
                clf.max_num_features_,
            )

        X = check_array(
            X, accept_sparse="csr", dtype=np.float32, force_all_finite=False
        )
        y = check_array(y, ensure_2d=False, dtype=np.float32, force_all_finite=False)

        check_consistent_length(X, y)

        return X, y

    def _pre_train_hook(self):
        """
        Pre-train hook to be called before the actual training in fit().
        :return: None
        """
        pass

    def _fit(self):
        """
        Fits the model to the training data.

        This method should be implemented by subclasses to specify the actual training logic.
        """
        raise NotImplementedError

    def fit(self, X, y, additional_x=None, additional_y=None) -> TabPFNBaseModel:
        """
        Fits the model to the input data `X` and `y`.

        The actual training logic is delegated to the `_fit` method, which should be implemented by subclasses.

        Parameters:
            X (Union[ndarray, torch.Tensor]): The input feature matrix of shape (n_samples, n_features).
            y (Union[ndarray, torch.Tensor]): The target labels of shape (n_samples,).
            additional_x (Optional[Dict[str, torch.Tensor]]): Additional features to use during training.
            additional_y (Optional[Dict[str, torch.Tensor]]): Additional labels to use during training.

        Returns:
            TabPFNBaseModel: The fitted model object (self).
        """
        timing_clear()
        timing_start(name="fit")

        # TODO: Transform input data to ordinals if not numbers (e.g. strings)

        # Prediction fit caching
        if self.cached_models_:
            for m in self.cached_models_:
                m.empty_trainset_representation_cache()
        self.cached_preprocessors_ = None
        self.cached_models_ = None
        self.cached_shuffles_ = None

        # Must not modify any input parameters
        self.init_model_and_get_model_config()

        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        X, y = self.check_training_data(self, X, y)
        self.n_features_in_ = X.shape[1]

        X = X.astype(np.float32)
        self.X_ = X if torch.is_tensor(X) else torch.tensor(X)
        self.y_ = y if torch.is_tensor(y) else torch.tensor(y)

        # Handle additional_x and additional_y
        self.additional_x_ = self._cast_numpy_dict_to_tensors(additional_x)
        self.additional_y_ = self._cast_numpy_dict_to_tensors(additional_y)

        self._pre_train_hook()

        timing_end(name="fit")

        self.infer_categorical_features(X)

        if not self.fit_at_predict_time:
            self._fit()

        # Return the classifier
        return self

    def infer_categorical_features(self, X: np.ndarray) -> Tuple[int, ...]:
        """
        Infer the categorical features from the input data.

        Parameters:
            X (ndarray): The input data.

        Returns:
            Tuple[int, ...]: The indices of the categorical features.
        """
        MAX_UNIQUE_VALUES_DISCARD = 30
        MAX_UNIQUE_VALUES_ADD = 5

        categorical_features = []
        for i in range(X.shape[-1]):
            # Filter categorical features, with too many unique values
            if i in self.categorical_features:
                if len(np.unique(X[:, i])) < MAX_UNIQUE_VALUES_DISCARD:
                    categorical_features += [
                        i,
                    ]

            # Filter non-categorical features, with few unique values
            elif (
                i not in self.categorical_features
                and len(np.unique(X[:, i])) < MAX_UNIQUE_VALUES_ADD
                and X.shape[0] > 100
            ):
                categorical_features += [
                    i,
                ]

        self._categorical_features = categorical_features

        return categorical_features

    def get_aux_clf(self):
        raise NotImplementedError

    @staticmethod
    def _cast_numpy_dict_to_tensors(data_dict):
        """Convert dictionary values to tensors"""
        if data_dict is not None:
            if not isinstance(data_dict, dict):
                raise TypeError("Additional data must be a dictionary.")
            return {
                k: torch.tensor(v, dtype=torch.float32) if not torch.is_tensor(v) else v
                for k, v in data_dict.items()
            }
        return data_dict

    def _transform_with_PCA(self, X):
        try:
            U, S, _ = torch.pca_lowrank(torch.squeeze(X))
            return torch.unsqueeze(U * S, 1)
        except:
            return X

    def _get_columns_with_nan(self, eval_xs: torch.Tensor) -> np.ndarray:
        nan_columns = np.isnan(eval_xs).any(axis=0)
        return np.where(nan_columns)[0]

    def get_feature_preprocessor(
        self,
        preprocess_transform: PreprocessorConfig,
        normalize_with_test: bool = False,
    ):
        assert not self.normalize_with_test, "not supported atm"

        poly_transformer = None
        if self.use_poly_features:
            poly_transformer = preprocessing.NanHandlingPolynomialFeaturesStep(
                max_poly_features=self.max_poly_features,
                rnd=self._rnd,
            )

        fingerprint_step = None
        if self.add_fingerprint_features:
            fingerprint_step = preprocessing.AddFingerprintFeaturesStep(self._rnd)

        if self.feature_shift_decoder != "none":
            assert [
                "onehot" not in preprocess_transform.categorical_name
            ], "Feature shift decoder is not compatible with one hot encoding"

        reshape_features_step = preprocessing.ReshapeFeatureDistributionsStep(
            preprocess_transform.name,
            apply_to_categorical=preprocess_transform.categorical_name == "numeric",
            append_to_original=preprocess_transform.append_original,
            subsample_features=preprocess_transform.subsample_features,
            global_transformer_name=preprocess_transform.global_transformer_name,
            rnd=self._rnd,
        )
        encode_categoricals_step = preprocessing.EncodeCategoricalFeaturesStep(
            preprocess_transform.categorical_name, rnd=self._rnd
        )

        preprocessor = preprocessing.SequentialFeatureTransformer(
            (
                ([poly_transformer] if poly_transformer is not None else [])
                + [
                    preprocessing.RemoveConstantFeaturesStep(),
                    reshape_features_step,
                    encode_categoricals_step,
                ]
                + ([fingerprint_step] if fingerprint_step is not None else [])
            )
        )
        return preprocessor

    def get_embeddings(
        self, X: torch.Tensor, additional_x: dict = None, additional_y: dict = None
    ) -> torch.Tensor:
        """
        Get the embeddings for the input data `X`.

        Parameters:
            X (torch.Tensor): The input data tensor.
            additional_x (dict, optional): Additional features as a dictionary. Defaults to None.
            additional_y (dict, optional): Additional labels as a dictionary. Defaults to None.

        Returns:
            torch.Tensor: The computed embeddings.
        """
        return self.predict_full(
            X,
            additional_x=additional_x,
            additional_y=additional_y,
            get_additional_outputs=["test_embeddings"],
        )["test_embeddings"]

    def init_style(self, style, differentiable_hps_as_style=True):
        if not differentiable_hps_as_style:
            style = None

        if style is not None:
            style = style
            style = style.unsqueeze(0) if len(style.shape) == 1 else style
            num_styles = style.shape[0]
        else:
            num_styles = 1
            style = None

        return num_styles, style

    @staticmethod
    def generate_shufflings(
        feature_n, n_configurations, rnd, shuffle_method="local_shuffle", max_step=10000
    ) -> torch.Tensor:
        if (
            max_step == 0
        ):  # this means that we use a per feature arch, which does not care about shuffling
            shuffle_method = "none"

        if shuffle_method == "auto_rotate" or shuffle_method == "rotate":

            def generate_shifting_permutations(n_features):
                initial = list(range(n_features))
                rotations = [initial]

                for i in range(1, n_features):
                    rotated = initial[i:] + initial[:i]
                    rotations.append(rotated)
                rnd.shuffle(rotations)

                return torch.tensor(rotations)

            feature_shift_configurations = generate_shifting_permutations(feature_n)
        elif shuffle_method == "shuffle":
            features_indices = list(range(feature_n))

            # Generate all permutations
            # all_permutations = itertools.permutations(features_indices)

            unique_shufflings = set()
            iterations = 0
            while (
                len(unique_shufflings) < n_configurations
                and iterations < n_configurations * 3
            ):
                shuffled = rnd.choice(
                    features_indices, size=len(features_indices), replace=False
                )
                unique_shufflings.add(tuple(shuffled))
                iterations += 1
            unique_shufflings = list(unique_shufflings)
            # Convert to PyTorch tensor
            feature_shift_configurations = torch.tensor(unique_shufflings)
        elif shuffle_method == "none":
            feature_shift_configurations = torch.tensor(
                list(range(feature_n))
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unknown feature_shift_decoder {shuffle_method}")

        return feature_shift_configurations

    def get_ensemble_configurations(
        self,
        eval_xs: torch.Tensor,
        eval_ys: torch.Tensor,
        base_configuration: EnsembleConfiguration | None = None,
        rnd: np.random.Generator = None,
        eval_position: int = None,
    ) -> List[EnsembleConfiguration]:
        """
        Generates a list of ensemble configurations for preprocessing.

        Each ensemble configuration contains:
        - class_shift_configuration: Permutation to apply to the classes. Only used for classification.
        - feature_shift_configuration: Permutation to apply to the features.
        - preprocess_transform_configuration: Preprocessing transform to apply.
        - styles_configuration: Style vector to use. Set by `differentiable_hps_as_style` in model config.
        - subsample_samples_configuration: Indices of samples to use. Used for bagging.

        The number of generated configurations is determined by `N_ensemble_configurations`, unless `base_configuration` is provided,
        in which case only one configuration is generated based on it.

        Parameters:
            eval_xs (torch.Tensor): The input features.
            eval_ys (torch.Tensor): The target labels.
            base_configuration (Optional[EnsembleConfiguration]): A base configuration to use. If provided, only one configuration will be generated.
            rnd (Optional[np.random.Generator]): A random number generator to use. If None, `self.rnd` is used.
            eval_position (Optional[int]): The position where the test set begins in the concatenated train+test dataset.

        Returns:
            List[EnsembleConfiguration]: A list of ensemble configurations.
        """
        timing_start(name="build_ensemble_configurations")

        if rnd is None:
            rnd = self._rnd

        use_base_config = base_configuration is not None
        # TODO: this is not used anymore but creates new configs, should we remove this? -> results in fitting two times the same config!
        styles_configurations = range(0, self.num_styles_)
        if use_base_config and base_configuration.styles_configuration is not None:
            styles_configurations = [base_configuration.styles_configuration]

        N_ensemble_configurations = self.get_max_N_ensemble_configurations(eval_xs)

        feature_start_shift = self._rnd.integers(
            0, 1000
        )  # Ensure that ensembles of single models are different
        feature_shift_configurations = list(
            range(
                feature_start_shift,
                feature_start_shift + N_ensemble_configurations,
            )
        )
        if (
            use_base_config
            and base_configuration.feature_shift_configuration is not None
        ):
            feature_shift_configurations = [
                base_configuration.feature_shift_configuration
            ]

        # Use all samples
        subsample_samples_configurations = [list(range(eval_position))]

        if 0 < self.subsample_samples < eval_position:
            subsample_samples = self.subsample_samples
            if self.subsample_samples < 1:
                subsample_samples = int(self.subsample_samples * eval_position) + 1

            subsample_samples_configurations = [
                np.sort(
                    rnd.choice(
                        list(range(eval_position)), subsample_samples, replace=False
                    )
                )
                for _ in range(N_ensemble_configurations)
            ]

        rnd.shuffle(subsample_samples_configurations)

        if self.is_classification_:
            class_shift_configurations = TabPFNBaseModel.generate_shufflings(
                self.num_classes_,
                rnd=rnd,
                n_configurations=N_ensemble_configurations,
                shuffle_method=self.multiclass_decoder,
                max_step=2,
            )
            # class_shift_configurations = torch.repeat_interleave(torch.tensor([[1, 4, 0, 5, 6, 2, 3]]), self.get_max_N_ensemble_configurations(eval_xs), dim=0)
        else:
            class_shift_configurations = self.regression_y_preprocess_transforms
        if use_base_config and base_configuration.class_shift_configuration is not None:
            class_shift_configurations = [base_configuration.class_shift_configuration]

        shift_configurations = list(
            itertools.product(class_shift_configurations, feature_shift_configurations)
        )
        preprocess_transforms = self.preprocess_transforms
        if (
            use_base_config
            and base_configuration.preprocess_transform_configuration is not None
        ):
            preprocess_transforms = [
                base_configuration.preprocess_transform_configuration
            ]

        preprocess_transforms = sum(
            [
                [p] * 9 if p.name == "per_feature" else [p]
                for p in preprocess_transforms
            ],
            [],
        )
        preprocess_transforms = sum(
            [
                [p] * 4 if p.name == "ordinal_shuffled" else [p]
                for p in preprocess_transforms
            ],
            [],
        )

        rnd.shuffle(shift_configurations)

        r = [
            EnsembleConfiguration(
                class_shift_configuration=class_shift_configuration,
                feature_shift_configuration=feature_shift_configuration,
                preprocess_transform_configuration=preprocess_transform_configuration,
                styles_configuration=styles_configuration,
                subsample_samples_configuration=subsample_samples_configuration,
            )
            for (
                class_shift_configuration,
                feature_shift_configuration,
            ), preprocess_transform_configuration, styles_configuration, subsample_samples_configuration in itertools.product(
                shift_configurations,
                preprocess_transforms,
                styles_configurations,
                subsample_samples_configurations,
            )
        ]

        timing_end("build_ensemble_configurations")

        return r

    def preprocess_regression_y(self, eval_ys, eval_position, configuration, bar_dist):
        assert (len(eval_ys.shape) == 2) and (
            eval_ys.shape[1] == 1
        ), f"only support (N, 1) shape, but got {eval_ys.shape}"

        pt = preprocessing.ReshapeFeatureDistributionsStep.get_all_preprocessors(
            eval_ys.shape[0], rnd=self._rnd
        )[configuration]

        logit_cancel_mask = None
        if pt is not None:
            try:
                if "adaptive" in configuration:
                    column_types = (
                        preprocessing.ReshapeFeatureDistributionsStep.get_column_types(
                            eval_ys[:eval_position].numpy()
                        )
                    )
                    pt = [
                        pt_[1]
                        for pt_ in pt.transformers
                        if pt_[2](
                            pd.DataFrame(eval_ys[:eval_position], columns=column_types)
                        )
                    ][0]

                pt.fit(eval_ys[:eval_position].numpy())
                eval_ys_ = pt.transform(eval_ys.numpy())
                new_borders = pt.inverse_transform(
                    bar_dist.borders[:, None].cpu().numpy()
                )[:, 0]
            except Exception as e:
                # print whole traceback
                traceback.print_exc()
                print_once("Failed to transform with", configuration)

            if hasattr(self, "cancel_nan_borders") and self.cancel_nan_borders:
                print_once("cancel nan borders")
                nan_borders_mask = (
                    np.isnan(new_borders)
                    | np.isinf(new_borders)
                    | (new_borders > 1e3)
                    | (new_borders < -1e3)
                )
                if nan_borders_mask.any():
                    # assert it is consecutive areas starting from both ends
                    num_right_borders = (
                        nan_borders_mask[:-1] > nan_borders_mask[1:]
                    ).sum()
                    num_left_borders = (
                        nan_borders_mask[1:] > nan_borders_mask[:-1]
                    ).sum()
                    assert num_left_borders <= 1
                    assert num_right_borders <= 1
                    if num_right_borders:
                        assert nan_borders_mask[0] == True
                        rightmost_nan_of_left_group = (
                            np.where(nan_borders_mask[:-1] > nan_borders_mask[1:])[0][0]
                            + 1
                        )
                        new_borders[:rightmost_nan_of_left_group] = new_borders[
                            rightmost_nan_of_left_group
                        ]
                        new_borders[0] = new_borders[1] - 1.0
                    if num_left_borders:
                        assert nan_borders_mask[-1] == True
                        leftmost_nan_of_right_group = np.where(
                            nan_borders_mask[1:] > nan_borders_mask[:-1]
                        )[0][0]
                        new_borders[leftmost_nan_of_right_group + 1 :] = new_borders[
                            leftmost_nan_of_right_group
                        ]
                        new_borders[-1] = new_borders[-2] + 1.0

                    # logit mask, mask out the nan positions, the borders are 1 more than the logits
                    logit_cancel_mask = nan_borders_mask[1:] | nan_borders_mask[:-1]

            try:
                # Try to repair a broken transformation of the borders:
                #   This is needed when a transformation of the ys leads to very extreme values in
                #   the transformed borders, since the borders spanned a very large range in the original space.
                #   Borders that were transformed to extreme values are all set to the same value, the maximum of the
                #   transformed borders. Thus probabilities predicted in these buckets have no effects. The outhermost
                #   border is set to the maximum of the transformed borders times 2, so still allow for some weight
                #   in the long tailed distribution and avoid infinite loss.
                if np.isnan(new_borders[-1]):
                    new_borders[np.isnan(new_borders)] = new_borders[
                        ~np.isnan(new_borders)
                    ].max()
                    new_borders[-1] = new_borders[-1] * 2
                if new_borders[-1] - new_borders[-2] < 1e-6:
                    new_borders[-1] = new_borders[-1] * 1.1
                if new_borders[0] == new_borders[1]:
                    new_borders[0] -= np.abs(new_borders[0] * 0.1)

                assert not np.isnan(
                    eval_ys_
                ).any(), f"NaNs in transformed ys: {eval_ys_}"
                assert not np.isnan(
                    new_borders
                ).any(), f"NaNs in transformed borders: {new_borders}"
                assert not np.isinf(
                    eval_ys_
                ).any(), f"Infs in transformed ys: {eval_ys_}"
                assert not np.isinf(
                    new_borders
                ).any(), f"Infs in transformed borders: {new_borders}"
                eval_ys, bar_dist = (
                    torch.tensor(eval_ys_.astype(np.float32)),
                    FullSupportBarDistribution(
                        torch.tensor(new_borders.astype(np.float32))
                    ),
                )
            except Exception:
                traceback.print_exc()
                print_once("Failed to go back with", configuration)
                # This can fail due to overflow errors which would end the entire evaluation
        return eval_ys, bar_dist, logit_cancel_mask

    def build_transformer_input_for_each_configuration(
        self,
        ensemble_configurations: List[EnsembleConfiguration],
        eval_xs: torch.Tensor,
        eval_ys: torch.Tensor,
        eval_position: int,
        additional_xs: dict = None,
        additional_ys: dict = None,
        bar_dist: Optional[FullSupportBarDistribution] = None,
        cache_trainset_transforms: bool = False,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[int]],
        List,
        List[torch.Tensor | None],
        Dict[List[torch.Tensor]],
        Dict[List[torch.Tensor]],
    ]:
        """
        Builds transformer inputs and labels based on given ensemble configurations.

        Parameters:
            ensemble_configurations (List[Tuple]): List of ensemble configurations.
            eval_xs (torch.Tensor): Input feature tensor.
            eval_ys (torch.Tensor): Input label tensor.
            eval_position (int): Position where the training set ends and the test set begins.

            cache_trainset_transforms (bool): Whether to cache the transforms applied to the training set.
                if cache_trainset_transforms is set to True:
                    if eval_position > 0:
                        the transforms applied to the training set are cached to be reused later for new test sets.
                    else:
                        the cached transforms are used to transform the test set.
                CAUTION: This is only safe if the training set is not changed between evaluations, thus only use this
                for a call to build_transformer_input_for_each_configuration in one place.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[list], List[BarDistribution], Dict[List[torch.Tensor]]]: Transformed inputs, labels, categorical_inds_list, adapted_bar_dists, additional_xs_out and additional_ys_out
        """
        eval_xs_and_categorical_inds_transformed = {}
        (
            inputs,
            labels,
            categorical_inds_list,
            adapted_bar_dists,
            logit_cancel_masks,
            created_transformers,
            created_shuffles,
            additional_xs_out,
            additional_ys_out,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            {k: [] for k in additional_xs} if additional_xs is not None else {},
            {k: [] for k in additional_ys} if additional_ys is not None else {},
        )

        cached_preprocessors = (
            self.cached_preprocessors_
            if cache_trainset_transforms and self.cached_preprocessors_ is not None
            else [None for _ in ensemble_configurations]
        )
        cached_shuffles = (
            self.cached_shuffles_
            if cache_trainset_transforms and self.cached_shuffles_ is not None
            else [None for _ in ensemble_configurations]
        )

        assert len(cached_preprocessors) == len(ensemble_configurations)
        for i, (ens_config, cached_preprocessor, cached_shuffle) in enumerate(
            zip(ensemble_configurations, cached_preprocessors, cached_shuffles)
        ):
            class_shift_configuration = ens_config.class_shift_configuration
            feature_shift_configuration = ens_config.feature_shift_configuration
            preprocess_transform_configuration = (
                ens_config.preprocess_transform_configuration
            )
            subsample_samples_configuration = ens_config.subsample_samples_configuration

            # The feature shift configuration is a permutation of the features, but
            # during preprocessing additional features could be added
            # assert feature_shift_configuration.shape[0] >= eval_xs.shape[-1]

            # Create a list for each key, containing n times the tensor stored at this
            # key with n being the number of ensemble configurations.
            if additional_xs is not None:
                additional_xs_out = {
                    k: additional_xs_out[k] + [additional_xs[k]] for k in additional_xs
                }
            else:
                additional_xs_out = {}

            if additional_ys is not None:
                additional_ys_out = {
                    k: additional_ys_out[k] + [additional_ys[k]] for k in additional_ys
                }
            else:
                additional_ys_out = {}

            if self.is_classification_:
                # The class shift configuration is a permutation of the classes
                assert class_shift_configuration is None or (
                    class_shift_configuration.shape[0] == self.num_classes_
                )

            eval_ys_ = eval_ys.clone()
            if (
                preprocess_transform_configuration
                in eval_xs_and_categorical_inds_transformed
                and preprocess_transform_configuration.can_be_cached()
            ):
                eval_xs_, categorical_feats = eval_xs_and_categorical_inds_transformed[
                    preprocess_transform_configuration
                ]
                eval_xs_ = eval_xs_.clone()
                categorical_feats = copy(categorical_feats)
                created_transformers.append(None)
            else:
                eval_xs_ = eval_xs.clone()
                if cache_trainset_transforms and eval_position == 0:
                    preprocessor = cached_preprocessor
                else:
                    preprocessor = self.get_feature_preprocessor(
                        preprocess_transform=preprocess_transform_configuration,
                    )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Fit and predict calls are split up, so that the fit call always receives the same data for preprocessing
                    #    no matter if in cache more or not.
                    concat_ = []
                    if eval_position > 0:
                        preprocessor.fit(
                            eval_xs_[:eval_position, 0], self._categorical_features
                        )
                        (
                            eval_xs_train_processed,
                            categorical_feats,
                        ) = preprocessor.transform(eval_xs_[:eval_position, 0])
                        concat_ += [eval_xs_train_processed]

                    if eval_xs_.shape[0] > eval_position:
                        (
                            eval_xs_test_processed,
                            categorical_feats,
                        ) = preprocessor.transform(
                            eval_xs_[eval_position:, 0], is_test=True
                        )
                        concat_ += [eval_xs_test_processed]

                eval_xs_ = np.concatenate(concat_, axis=0)

                try:
                    eval_xs_ = torch.tensor(eval_xs_.astype(np.float32)).float()[
                        :, None
                    ]
                except Exception as e:
                    # This can fail due to overflow errors which would end the entire evaluation
                    eval_xs_ = eval_xs.clone()

                eval_xs_and_categorical_inds_transformed[
                    preprocess_transform_configuration
                ] = (eval_xs_.clone(), copy(categorical_feats))
                created_transformers.append(preprocessor)
            if cached_shuffle and eval_position == 0:
                feature_shuffler = cached_shuffle
            else:
                feature_shuffler = preprocessing.ShuffleFeaturesStep(
                    shuffle_method=self.feature_shift_decoder,
                    shuffle_index=feature_shift_configuration,
                    rnd=self._rnd,
                )
                feature_shuffler.fit(eval_xs_[:eval_position, 0], categorical_feats)
                created_shuffles.append(feature_shuffler)
            eval_xs_, categorical_feats = feature_shuffler.transform(eval_xs_[:, 0])
            eval_xs_ = eval_xs_[:, None]

            new_bar_dist = bar_dist

            logit_cancel_mask = None
            if class_shift_configuration is not None:
                if self.is_classification_:
                    eval_ys_ = eval_ys_.long()
                    eval_ys_[
                        eval_ys_ != self.semisupervised_indicator
                    ] = self._map_classes(
                        eval_ys_[eval_ys_ != self.semisupervised_indicator],
                        class_shift_configuration,
                    )
                else:
                    assert bar_dist is not None
                    if cache_trainset_transforms and eval_position == 0:
                        new_bar_dist = self.cached_bar_dists_[i]
                    else:
                        (
                            eval_ys_,
                            new_bar_dist,
                            logit_cancel_mask,
                        ) = self.preprocess_regression_y(
                            eval_ys_, eval_position, class_shift_configuration, bar_dist
                        )

            if subsample_samples_configuration is not None and not (
                cache_trainset_transforms and eval_position == 0
            ):
                eval_xs_train = eval_xs_[subsample_samples_configuration, :]
                eval_ys_train = eval_ys_[subsample_samples_configuration]

                eval_xs_ = torch.cat(
                    [
                        eval_xs_train,
                        eval_xs_[eval_position:, :],
                    ]
                )
                eval_ys_ = torch.cat([eval_ys_train, eval_ys_[eval_position:]])

                for key in additional_xs_out:
                    additional_xs_out[key][i] = torch.cat(
                        [
                            additional_xs_out[key][i][
                                subsample_samples_configuration, :
                            ],
                            additional_xs_out[key][i][eval_position:, :],
                        ]
                    )

                # TODO: Shouldn't additional_ys be treated the same way as additional_xs?

            inputs += [eval_xs_]
            labels += [eval_ys_]
            categorical_inds_list += [categorical_feats]
            adapted_bar_dists += [new_bar_dist]
            logit_cancel_masks += [logit_cancel_mask]

        if cache_trainset_transforms and eval_position:
            self.cached_preprocessors_ = created_transformers
            self.cached_bar_dists_ = adapted_bar_dists
            self.cached_shuffles_ = created_shuffles

        return (
            inputs,
            labels,
            categorical_inds_list,
            adapted_bar_dists,
            logit_cancel_masks,
            additional_xs_out,
            additional_ys_out,
        )

    def build_transformer_input_in_batches(
        self,
        inputs: List[torch.Tensor],
        labels: List[torch.Tensor],
        categorical_inds_list: List[List[int]],
        batch_size_inference: int,
        additional_xs: dict = None,
        additional_ys: dict = None,
    ) -> Tuple[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor],
        torch.Tensor,
        tuple,
        List[dict],
        List[dict],
    ]:
        """
        Partition inputs and labels into batches for transformer.

        Parameters:
            inputs (List[torch.Tensor]): List of input tensors.
            labels (List[torch.Tensor]): List of label tensors.
            batch_size_inference (int): Size of each inference batch.

        Returns:
            Tuple[inputs: List[torch.Tensor], labels: List[torch.Tensor], implied_permutation: torch.Tensor]: Partitioned inputs, labels, and permutations.
            Each items contains a list of tensors, where each tensor is a batch, with maximally batch_size_inference elements and the same number of features.
        """
        inds_for_each_feat_dim = defaultdict(list)
        for i, inp in enumerate(inputs):
            inds_for_each_feat_dim[inp.shape[-1]].append(i)
        inds_for_each_feat_dim = OrderedDict(inds_for_each_feat_dim)
        implied_permutation = torch.tensor(
            sum((inds for inds in inds_for_each_feat_dim.values()), [])
        )
        inputs = [
            torch.cat([inputs[i] for i in inds], dim=1)
            for inds in inds_for_each_feat_dim.values()
        ]

        inputs = sum(
            (torch.split(inp, batch_size_inference, dim=1) for inp in inputs), tuple()
        )

        labels = [
            torch.cat([labels[i] for i in inds], dim=1)
            for inds in inds_for_each_feat_dim.values()
        ]
        labels = sum(
            (torch.split(lab, batch_size_inference, dim=1) for lab in labels), tuple()
        )

        def chunk_additional_data(
            additional_data: dict[str, torch.Tensor]
        ) -> list[dict[str, torch.Tensor]]:
            """
            Chunk additional data based on feature dimension indices and batch size.

            Returns:
            - list of dicts: A list of dictionaries containing the chunked additional data.
            """
            if additional_data is None:
                return []

            additional_data_chunked = {}
            for k in additional_data.keys():
                data_key = additional_data[k]
                data_key = [
                    torch.cat([data_key[i] for i in inds], dim=1)
                    for inds in inds_for_each_feat_dim.values()
                ]
                data_key = sum(
                    (
                        torch.split(data, batch_size_inference, dim=1)
                        for data in data_key
                    ),
                    tuple(),
                )
                additional_data_chunked[k] = data_key

            list_length = len(labels)

            # Convert to list of dictionaries
            additional_data_chunked = [
                {
                    key: additional_data_chunked[key][i]
                    for key in additional_data_chunked
                }
                for i in range(list_length)
            ]

            return additional_data_chunked

        additional_xs_chunked = chunk_additional_data(additional_xs)
        additional_ys_chunked = chunk_additional_data(additional_ys)

        categorical_inds_list = [
            [categorical_inds_list[i] for i in inds]
            for inds in inds_for_each_feat_dim.values()
        ]

        categorical_inds_list = sum(
            (
                tuple(utils.chunks(cis, batch_size_inference))
                for cis in categorical_inds_list
            ),
            tuple(),
        )

        return (
            inputs,
            labels,
            implied_permutation,
            categorical_inds_list,
            additional_xs_chunked,
            additional_ys_chunked,
        )

    def _reweight_probs_based_on_train(
        self,
        eval_ys: torch.Tensor,
        output: torch.Tensor,
        num_classes: int,
        device: str,
    ) -> torch.Tensor:
        """
        Reweight class probabilities based on training set.

        Parameters:
            eval_ys (torch.Tensor): Label tensor for evaluation.
            output (torch.Tensor): Output probability tensor.
            num_classes (int): Number of classes.
            device (str): Computing device.

        Returns:
            torch.Tensor: Reweighted output tensor.
        """
        # make histogram of counts of each class in train set
        train_class_probs = torch.zeros(num_classes, device=device)
        train_class_probs.scatter_add_(
            0,
            eval_ys.flatten().long().to(device),
            torch.ones_like(eval_ys.flatten(), device=device),
        )
        train_class_probs = (
            train_class_probs / train_class_probs.sum()
        )  # shape: num_classes
        output /= train_class_probs
        # make sure outputs last dim sums to 1
        output /= output.sum(dim=-1, keepdim=True)

        return output

    @staticmethod
    def _map_classes(tensor, permutation):
        permutation_tensor = permutation.long()
        return permutation_tensor[tensor.long()]

    @staticmethod
    def _reverse_permutation(output, permutation):
        """
        At the beginning of a prediction we create a permutation of the classes.
        We do this by simply indexing into the permutation tensor with the classes, i.e. `permutation[classes]`.

        Now output contains logits for the classes, but we want to get the logits for the original classes.
        That is why we reverse this permutation by indexing the other way around, `logits[permutation]`.

        The property we want is
        ```
        classes = torch.randint(100, (200,)) # 200 examples with random classes out of 100
        permutation = torch.randperm(100) # a random permutation of the classes

        # forward permutation
        permuted_classes = permutation[classes]
        perfect_predictions_on_permuted_classes = torch.nn.functional.one_hot(permuted_classes, 100)

        # backward permutation (this function)
        perfect_predictions_on_original_classes = perfect_predictions_on_permuted_classes[:, permutation]

        # now we want
        assert (perfect_predictions_on_original_classes == torch.nn.functional.one_hot(classes, 100)).all()
        ```

        We use the second dimension of the output tensor to index into the permutation tensor, as we have both batch and seq dimensions.

        Parameters:
            output: tensor with shape (seq_len, batch_size, num_classes)
            permutation: tensor with shape (num_classes)
        :return:
        """
        return output[..., permutation]

    def predict_common_setup(
        self, X_eval, additional_x_eval=None, additional_y_eval=None
    ):
        """
        Common setup steps for prediction methods.

        This method validates the input data, prepares it for prediction by the transformer model,
        and extends it with the test set if transductive learning is enabled.

        Parameters:
            X_eval (Union[ndarray, torch.Tensor]): The input features for prediction.
            additional_y_eval (Optional[Dict[str, torch.Tensor]]): Additional labels to use during prediction.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], int]: A tuple containing:
                - X_full (torch.Tensor): The full input feature matrix (train + test).
                - y_full (torch.Tensor): The full target label tensor (train + test).
                - additional_y_full (Dict[str, torch.Tensor]): The full additional label tensors (train + test).
                - eval_pos (int): The position where the test set begins in the full dataset.
        """
        # We save the seed before the prediction call to be able to restore it afterwards again
        self._init_rnd()

        if additional_x_eval is None:
            additional_x_eval = {}
        else:
            additional_x_eval = self._cast_numpy_dict_to_tensors(additional_x_eval)

        if additional_y_eval is None:
            additional_y_eval = {}
        else:
            additional_y_eval = self._cast_numpy_dict_to_tensors(additional_y_eval)

        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X_eval = check_array(
            X_eval, accept_sparse="csr", dtype=np.float32, force_all_finite=False
        )

        if X_eval.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in the input data ({X_eval.shape[1]}) must match the number of features during training ({self.n_features_in_})."
            )

        X_eval = torch.tensor(X_eval)

        X_train, y_train, additional_x_train, additional_y_train = (
            self.X_,
            self.y_,
            deepcopy(self.additional_x_),
            deepcopy(self.additional_y_),
        )
        if not self.fit_at_predict_time:
            # we empty all training tensors in this case, as they were already fed into the model in self._fit
            X_train = X_train[:0]
            y_train = y_train[:0]
            if additional_x_train:
                additional_x_train = {
                    key: additional_x_train[key][:0] for key in additional_x_train
                }
            if additional_y_train:
                additional_y_train = {
                    key: additional_y_train[key][:0] for key in additional_y_train
                }

        ### Extend X, y and additional_y with the test set for transductive learning
        if self.transductive:
            print("Transductive prediction enabled.")
            X_train = torch.cat([X_train, X_eval], dim=0)
            y_train = torch.cat(
                [
                    y_train,
                    torch.ones_like(X_eval[:, 0]) * self.semisupervised_indicator,
                ],
                dim=0,
            )
            if additional_x_train is not None and len(additional_x_train.keys()) > 0:
                for k, v in additional_x_train.items():
                    additional_x_train[k] = torch.cat([v, additional_x_eval[k]], dim=0)
            if additional_y_train is not None and len(additional_y_train.keys()) > 0:
                for k, v in additional_y_train.items():
                    additional_y_train[k] = torch.cat([v, additional_y_eval[k]], dim=0)

        ### Join train and test sets
        X_full = torch.cat([X_train, X_eval], dim=0).float().unsqueeze(1)
        y_full = (
            torch.cat([y_train, torch.zeros_like(X_eval[:, 0])], dim=0)
            .float()
            .unsqueeze(1)
        )

        if additional_x_eval:
            assert (
                additional_x_train
            ), "additional_x_train is None in fit but not predict"
            for k, v in additional_x_eval.items():
                additional_x_eval[k] = torch.cat(
                    [additional_x_train[k], v], dim=0
                ).float()
        else:
            assert (
                self.additional_x_ is None
            ), "additional_x is None in predict but not fit"

        if additional_y_eval:
            assert (
                additional_y_train
            ), "additional_y_train is None in fit but not predict"
            for k, v in additional_y_eval.items():
                additional_y_eval[k] = torch.cat(
                    [additional_y_train[k], v], dim=0
                ).float()
        else:
            assert (
                self.additional_y_ is None
            ), "additional_y is None in predict but not fit"

        eval_pos = X_train.shape[0]

        return X_full, y_full, additional_x_eval, additional_y_eval, eval_pos

    def predict_full(
        self, X, additional_x=None, additional_y=None, get_additional_outputs=None
    ) -> dict[str, torch.Tensor]:
        """
        Generates full predictions for the input data.

        In addition to the standard prediction probabilities or regression values, this method can
        return additional outputs from the transformer model, such as embeddings.

        The actual implementation of this method should be provided by subclasses (`TabPFNClassifier`,
        `TabPFNRegressor`, etc.).

        Parameters:
            X (Union[ndarray, torch.Tensor]): The input features for prediction.
            additional_x (Optional[Dict[str, torch.Tensor]]): Additional features to use during prediction.
            additional_y (Optional[Dict[str, torch.Tensor]]): Additional labels to use during prediction.
            get_additional_outputs (Optional[List[str]]): Names of additional outputs to return from the transformer.

        Returns:
            dict: A dictionary containing:
                - 'proba' (ndarray): The predicted class probabilities (for classifiers).
                - 'mean', 'median', etc. (ndarray): The predicted regression values (for regressors).
                - str(key) (torch.Tensor): Additional outputs from the transformer, with keys as specified in `get_additional_outputs`.
        """
        raise NotImplementedError("This method is not implemented yet.")

    def get_max_N_ensemble_configurations(self, eval_xs, **overwrite_params):
        """
        Computes the maximum number of ensemble configurations that can be evaluated given the computational budget.

        If the number of ensemble configurations (`N_ensemble_configurations`) is specified and positive, this is returned.
        If the number of ensemble configurations is specified and negative, the number of ensemble configurations is
            the number of ensemble configurations times the number of datasets that fit in one batch, but at most 16
        If the number of ensemble configurations is None, the number of ensemble configurations is
            computed based on the computational budget.

        Returns:
            int: The maximum number of ensemble configurations to use.
        """
        N_ensemble_configurations = overwrite_params.get(
            "N_ensemble_configurations", self.N_ensemble_configurations
        )

        if N_ensemble_configurations is not None:
            return N_ensemble_configurations

        max_iterations, min_iterations = 64, 8

        overwrite_params["N_ensemble_configurations"] = 1

        if "batch_size_inference" not in overwrite_params:
            overwrite_params["batch_size_inference"] = self.get_batch_size_inference(
                eval_xs, safety_factor=2.5, **overwrite_params
            )

        per_sample_cost = self.estimate_computation_usage(eval_xs, **overwrite_params)

        N_ensemble_configurations = int(self.computational_budget / per_sample_cost)

        N_ensemble_configurations = min(
            max_iterations, max(min_iterations, N_ensemble_configurations)
        )

        return N_ensemble_configurations

    def _batch_predict(
        self,
        inputs: list[torch.Tensor],
        labels: list[torch.Tensor],
        additional_xs: dict[list[torch.Tensor]],
        additional_ys: dict[list[torch.Tensor]],
        categorical_inds: list[list[int]],
        eval_position: int,
        cache_trainset_representations: bool = False,
        get_additional_outputs: list[str] = None,
    ) -> tuple[torch.Tensor, dict[list[torch.Tensor]]]:
        fill_cache = cache_trainset_representations and eval_position
        use_cache = cache_trainset_representations and not eval_position

        batch_size = self.get_batch_size_inference(
            inputs[0], save_peak_memory=self._save_peak_memory
        )

        ## Split inputs to model into chunks that can be calculated batchwise for faster inference
        ## We split based on the dimension of the features, so that we can feed in a batch
        (
            inputs,
            labels,
            implied_permutation,
            categorical_inds,
            additional_xs_inputs,
            additional_ys_inputs,
        ) = self.build_transformer_input_in_batches(
            inputs,
            labels,
            categorical_inds_list=categorical_inds,
            batch_size_inference=batch_size,
            additional_xs=additional_xs,
            additional_ys=additional_ys,
        )

        softmax_temperature = utils.to_tensor(
            self.softmax_temperature, device=self.device_
        )

        style_ = self.style_.to(self.device_) if self.style_ is not None else None

        if use_cache:
            cached_models = self.cached_models_
        else:
            cached_models = [None for _ in inputs]

        assert len(cached_models) == len(
            inputs
        ), "Cached transformers and inputs have different lengths"

        ## Get predictions and save in intermediate tensor
        outputs = []
        additional_outputs = (
            {}
            if get_additional_outputs is None
            else {k: [] for k in get_additional_outputs}
        )
        models_to_cache = []
        for (
            batch_input,
            batch_label,
            batch_categorical_inds,
            batch_additional_xs,
            batch_additional_ys,
            cached_preprocessor,
        ) in tqdm.tqdm(
            list(
                zip(
                    inputs,
                    labels,
                    categorical_inds,
                    additional_xs_inputs,
                    additional_ys_inputs,
                    cached_models,
                )
            ),
            desc="Heavy lifting",
            disable=not self.show_progress,
            unit="batch",
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="None of the inputs have requires_grad=True. Gradients will be None",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
                )
                with (
                    torch.amp.autocast(device_type="cuda", enabled=self.fp16_inference)
                    if version.parse(torch.__version__) >= version.parse("2.4.0")
                    else torch.cuda.amp.autocast(enabled=self.fp16_inference)
                ):
                    inference_mode_call = (
                        torch.inference_mode() if self.inference_mode else NOP()
                    )
                    with inference_mode_call:
                        style_expanded = (
                            style_.repeat(batch_input.shape[1], 1)
                            if self.style_ is not None
                            else None
                        )
                        batch_additional_xs = (
                            {
                                k: v.to(self.device_) if torch.is_tensor(v) else v
                                for k, v in batch_additional_xs.items()
                            }
                            if batch_additional_xs is not None
                            else {}
                        )
                        batch_additional_ys = (
                            {
                                k: v.to(self.device_) if torch.is_tensor(v) else v
                                for k, v in batch_additional_ys.items()
                            }
                            if batch_additional_ys is not None
                            else {}
                        )
                        if self.is_classification_:
                            batch_label = batch_label.float()
                            batch_label[
                                batch_label == self.semisupervised_indicator
                            ] = np.nan

                        features = {
                            "main": batch_input.to(self.device_),
                            **batch_additional_xs,
                        }

                        labels = {
                            "main": batch_label.float().to(self.device_),
                            **batch_additional_ys,
                        }

                        if use_cache:
                            assert cached_preprocessor is not None
                            labels = None
                            model = cached_preprocessor
                        elif fill_cache:
                            model = deepcopy(self.model_processed_)
                            assert hasattr(
                                model, "cache_trainset_representation"
                            ), "Model does not support caching"
                            model.cache_trainset_representation = True
                        else:
                            model = self.model_processed_

                        only_return_standard_out = (
                            get_additional_outputs is None
                            or len(get_additional_outputs) == 0
                        )

                        output = model(
                            (
                                style_expanded,
                                features,
                                labels,
                            ),
                            single_eval_pos=eval_position,
                            only_return_standard_out=only_return_standard_out,
                            categorical_inds=batch_categorical_inds,
                        )

                        if fill_cache:
                            models_to_cache.append(model)
                        if isinstance(output, tuple):
                            output, output_once = output
                        if additional_outputs:
                            standard_prediction_output = output["standard"]
                            for k in additional_outputs:
                                additional_outputs[k].append(output[k].cpu())
                        else:
                            standard_prediction_output = output

                        # cut off additional logits for classes that do not exist in the dataset

                    assert (
                        self.num_classes_ <= standard_prediction_output.shape[-1]
                    ), "More classes in dataset than in prediction."
                    standard_prediction_output = standard_prediction_output[
                        :, :, : self.num_classes_
                    ].float() / torch.exp(softmax_temperature)

            outputs += [standard_prediction_output]
        outputs = torch.cat(outputs, 1)
        # argsort of a permutation index yields the inverse
        if fill_cache:
            self.cached_models_ = models_to_cache

        if additional_outputs:
            for k in additional_outputs:
                additional_outputs[k] = torch.cat(additional_outputs[k], dim=1)[
                    :, torch.argsort(implied_permutation), :
                ]

        return outputs[:, torch.argsort(implied_permutation), :], additional_outputs

    def _parse_predictions_per_configuration(
        self,
        predictions: torch.Tensor,
        ensemble_configurations: list[EnsembleConfiguration],
    ) -> list[torch.Tensor]:
        # Combine predictions
        ensemble_outputs = []
        for i, ensemble_configuration in enumerate(ensemble_configurations):
            class_shift_configuration = ensemble_configuration.class_shift_configuration

            output_ = predictions[:, i : i + 1, :]

            if class_shift_configuration is not None:
                if self.is_classification_:
                    output_ = self._reverse_permutation(
                        output_, class_shift_configuration
                    )

            if not self.average_logits and self.is_classification_:
                output_ = torch.nn.functional.softmax(output_, dim=-1)
            ensemble_outputs += [output_]
        return ensemble_outputs

    def _aggregate_predictions(
        self,
        ensemble_outputs: list[torch.Tensor],
        reweight_probs_based_on_train: bool,
        eval_ys: torch.Tensor,
        bar_distribution: FullSupportBarDistribution | None,
        logit_cancel_masks: list[torch.Tensor | None],
        adapted_bar_dists: list[FullSupportBarDistribution],
    ) -> torch.Tensor:
        if self.is_classification_:
            outputs = torch.cat(ensemble_outputs, 1)
            output = torch.mean(outputs, 1, keepdim=True)

            if self.average_logits:
                output = torch.nn.functional.softmax(output, dim=-1)

            if reweight_probs_based_on_train:
                # This is used to reweight probabilities for when the optimized metric should give balanced predictions
                output = self._reweight_probs_based_on_train(
                    eval_ys, output, self.num_classes_, self.device_
                )

        else:
            # assert all(not torch.isnan(o).any() for o in ensemble_outputs)
            # assert all(not torch.isnan(d.borders).any() for d in adapted_bar_dists)
            borders_device = bar_distribution.borders.device
            for o, mask in zip(ensemble_outputs, logit_cancel_masks):
                if mask is not None:
                    o[..., mask] = float("-inf")

            # all_borders_equal = all([(adapted_bar_dists[0].borders == bd.borders).all() for bd in adapted_bar_dists])
            # if all_borders_equal:
            #     print('all borders equal')
            #     output = torch.stack(ensemble_outputs).softmax(-1).mean(0).log()
            #     bar_distribution.borders = adapted_bar_dists[0].borders
            # elif len(adapted_bar_dists) > 1:
            if (
                hasattr(self, "super_bar_dist_averaging")
                and self.super_bar_dist_averaging
            ):
                adapted_bar_dists = [deepcopy(d) for d in adapted_bar_dists]
                bar_distribution.borders = torch.sort(
                    torch.unique(
                        torch.cat(
                            [d.borders.to(borders_device) for d in adapted_bar_dists]
                        )
                    )
                ).values
            output = bar_distribution.average_bar_distributions_into_this(
                [d.to(borders_device) for d in adapted_bar_dists],
                ensemble_outputs,
                average_logits=self.average_logits,
            )
            # assert all(not torch.isnan(o).any() for o in output)
        return torch.transpose(output, 0, 1)

    def transformer_predict(
        self,
        eval_xs: torch.Tensor,  # shape (num_examples, [1], num_features)
        eval_ys: torch.Tensor,  # shape (num_examples, [1], [1])
        eval_position: int,
        bar_distribution: FullSupportBarDistribution | None = None,
        reweight_probs_based_on_train=False,
        additional_xs=None,
        additional_ys=None,
        cache_trainset_representations: bool = False,
        get_additional_outputs: list[str] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Generates predictions from the transformer model.

        This method builds the ensemble configurations, applies preprocessing, runs the transformer
        model on the preprocessed data, and then aggregates the predictions from each configuration.

        Parameters:
            eval_xs (torch.Tensor): The input feature matrix.
            eval_ys (torch.Tensor): The target labels. Only labels up to `eval_position` will be used as training labels.
            eval_position (int): The position where the test set begins in the concatenated train+test dataset.
            bar_distribution (Optional[FullSupportBarDistribution]): The bin border distribution for regression models.
            reweight_probs_based_on_train (bool): If True, prediction probabilities will be adjusted based on training set label frequencies.
            additional_xs (Optional[Dict[str, torch.Tensor]]): Additional features used for prediction.
            additional_ys (Optional[Dict[str, torch.Tensor]]): Additional labels used for prediction.
            cache_trainset_representations (bool): If True, the transformer representation of the training set will be cached for reuse.
            get_additional_outputs (Optional[List[str]]): Names of additional outputs to return from the transformer.

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing:
                - output (torch.Tensor): The aggregated predictions of shape (n_test_samples, n_classes).
                - additional_outputs (dict): Additional outputs from the transformer model.
        """
        # Input validation
        assert (
            self.normalize_with_test is False
        ), "not supported right now, look into preprocess_input"

        assert (
            bar_distribution is None
        ) == self.is_classification_, (
            "bar_distribution needs to be set if and only if the model is a regressor"
        )

        # Memory optimization, todo: this could be moved even closer to the transformer to take batching into account
        save_peak_memory = self.get_save_peak_memory(eval_xs)
        if save_peak_memory:
            self.model_processed_.reset_save_peak_mem_factor(SAVE_PEAK_MEM_FACTOR)
        else:
            self.model_processed_.reset_save_peak_mem_factor(None)
        self._save_peak_memory = save_peak_memory

        if (
            self.estimate_memory_usage(eval_xs, "gb", save_peak_memory=save_peak_memory)
            > self.get_max_free_memory_in_gb()
        ):
            raise ValueError(
                f"Memory usage of the model is too high (Approximated Memory Usage (GB): {self.estimate_memory_usage(eval_xs, 'gb')}, Capacity {self.get_max_free_memory_in_gb()})."
            )

        fill_cache = cache_trainset_representations and eval_position > 0
        use_cache = cache_trainset_representations and eval_position == 0

        # Store the classes seen during fit

        # Handle inputs with optional batch dim
        if len(eval_ys.shape) == 1:
            eval_ys = eval_ys.unsqueeze(1)
        if len(eval_xs.shape) == 2:
            eval_xs = eval_xs.unsqueeze(1)

        # Never look at the test labels
        eval_ys = eval_ys[:eval_position]

        # Initialize list of preprocessings to check
        if use_cache:
            ensemble_configurations = self.ensemble_configurations_
        else:
            ensemble_configurations = self.get_ensemble_configurations(
                eval_xs, eval_ys, eval_position=eval_position
            )
            N_ensemble_configurations = self.get_max_N_ensemble_configurations(
                eval_xs, save_peak_memory=save_peak_memory
            )

            ensemble_configurations = ensemble_configurations[
                0:N_ensemble_configurations
            ]
        if fill_cache:
            self.ensemble_configurations_ = ensemble_configurations

        timing_start(name="preprocess")
        # Compute and save transformed inputs for each configuration
        (
            inputs,
            labels,
            categorical_inds,
            adapted_bar_dists,
            logit_cancel_masks,
            additional_xs,
            additional_ys,
        ) = self.build_transformer_input_for_each_configuration(
            ensemble_configurations=ensemble_configurations,
            eval_xs=eval_xs,
            eval_ys=eval_ys,
            eval_position=eval_position,
            bar_dist=bar_distribution,
            additional_xs=additional_xs,
            additional_ys=additional_ys,
            cache_trainset_transforms=cache_trainset_representations,
        )
        timing_end(name="preprocess")

        # Update eval position after data was potentially subsampled
        # Eval_position = (length of the new input data) - (old length of eval fraction, which is not changed in subsampling)
        eval_position = inputs[0].shape[0] - (eval_xs.shape[0] - eval_position)

        outputs, additional_outputs = self._batch_predict(
            inputs=inputs,
            labels=labels,
            additional_xs=additional_xs,
            additional_ys=additional_ys,
            categorical_inds=categorical_inds,
            eval_position=eval_position,
            cache_trainset_representations=cache_trainset_representations,
            get_additional_outputs=get_additional_outputs,
        )

        ensemble_outputs = self._parse_predictions_per_configuration(
            predictions=outputs, ensemble_configurations=ensemble_configurations
        )

        output = self._aggregate_predictions(
            ensemble_outputs=ensemble_outputs,
            reweight_probs_based_on_train=reweight_probs_based_on_train,
            eval_ys=eval_ys,
            bar_distribution=bar_distribution,
            logit_cancel_masks=logit_cancel_masks,
            adapted_bar_dists=adapted_bar_dists,
        )

        return output, additional_outputs

    def get_max_free_memory_in_gb(self) -> float:
        """
        How much memory to use at most in GB, if None, the memory usage will be calculated based on
        an estimation of the systems free memory. For CUDA will use the free memory of the GPU.
        For CPU will default to 32 GB.

        Returns:
            float: The maximum memory usage in GB.
        """
        if self.maximum_free_memory_in_gb is None:
            # TODO: Get System Stats and adapt to free memory for default case

            if self.device_.startswith("cpu"):
                try:
                    total_memory = (
                        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                    )
                except ValueError:
                    utils.print_once(
                        "Could not determine size of memory of this system, using default 8 GB"
                    )
                    total_memory = 8
                return total_memory
            elif self.device_.startswith("cuda"):
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                f = t - a  # free inside reserved

                return f / 1e9
            else:
                raise ValueError(f"Unknown device {self.device_}")
        else:
            return self.maximum_free_memory_in_gb

    def estimate_memory_usage(
        self,
        X: np.ndarray | torch.tensor,
        unit: Literal["b", "mb", "gb"] = "gb",
        **overwrite_params,
    ) -> float | None:
        """
        Estimates the memory usage of the model.

        Peak memory usage is accurate for ´save_peak_mem_factor´ in O(n_feats, n_samples) on average but with
        significant outliers (2x). Also this calculation does not include baseline usage and constant offsets.
        Baseline memory usage can be ignored if we set the maximum memory usage to the default None which uses
        the free memory of the system. The constant offsets are not significant for large datasets.

        Parameters:
            X (ndarray): The feature matrix. X should represent the concat of train and test in if
                `self.fit_at_predict_time` and train only otherwise. If you add a batch dimension at position 1 to the
                table this is used as the batch size used during inference, otherwise this depends on the
                `batch_size_inference` and `N_ensemble_configurations`.
            unit (Literal["b", "mb", "gb"]): The unit to return the memory usage in (bytes, megabytes, or gigabytes).

        Returns:
            int: The estimated memory usage in bytes.
        """
        byte_usage = self._estimate_model_usage(X, "memory", **overwrite_params)
        if byte_usage is None:
            return None

        if unit == "mb":
            return byte_usage / 1e6
        elif unit == "gb":
            return byte_usage / 1e9
        elif unit == "b":
            return byte_usage
        else:
            raise ValueError(f"Unknown unit {unit}")

    def estimate_computation_usage(
        self,
        X: np.ndarray,
        unit: Literal["sequential_flops", "s"] = "sequential_flops",
        **overwrite_params,
    ) -> float | None:
        """
        Estimates the sequential computation usage of the model. Those are the operations that are not parallelizable
        and are the main bottleneck for the computation time.

        Parameters:
            X (ndarray): The feature matrix. X should represent the concat of train and test in if
            ` self.fit_at_predict_time` and train only otherwise. If you add a batch dimension at position 1 to the
              table this is used as the batch size used during inference, otherwise this depends on the
             `batch_size_inference` and `N_ensemble_configurations`.
            unit (str): The unit to return the computation usage in.

        Returns:
            int: The estimated computation usage in unit of choice.
        """
        computation = self._estimate_model_usage(X, "computation", **overwrite_params)
        if unit == "s":
            computation = computation / self.processor_speed / 2e11
        return computation

    def _estimate_model_usage(
        self,
        X: np.ndarray | torch.tensor,
        estimation_task: Literal["memory", "computation"],
        **overwrite_params,
    ) -> int | None:
        """
        Estimate the memory and compute needs of the model for a particular dataset.
        Internal function, see the docstring of `estimate_memory_usage` and `estimate_computation_usage`
        for more details.

        To re-fit time and memory estimations, you can use the workflows in ´utils_fit_memory_and_predict_time.py´.
        """
        if not self.is_initialized():
            self.init_model_and_get_model_config()
        fp_16 = overwrite_params.get(
            "fp_16", self.fp16_inference and not self.device_.startswith("cpu")
        )
        num_heads = overwrite_params.get("num_heads", self.c_processed_.get("nhead"))
        num_layers = overwrite_params.get(
            "num_layers", self.c_processed_.get("nlayers")
        )
        embedding_size = overwrite_params.get(
            "embedding_size", self.c_processed_.get("emsize")
        )
        features_per_group = overwrite_params.get(
            "features_per_group", self.c_processed_.get("features_per_group")
        )

        # These settings can be auto adapted
        if "save_peak_mem_factor" in overwrite_params:
            save_peak_mem_factor = overwrite_params["save_peak_mem_factor"]
        else:
            save_peak_mem_factor = self.get_save_peak_memory(X)
        overwrite_params["save_peak_mem_factor"] = save_peak_mem_factor

        per_feature_transformer = overwrite_params.get(
            "per_feature_transformer",
            "PerFeat" in type(self.model_processed_).__name__,
        )
        if "batch_size_inference" in overwrite_params:
            batch_size_inference = overwrite_params["batch_size_inference"]
        else:
            batch_size_inference = self.get_batch_size_inference(X, **overwrite_params)
        overwrite_params["batch_size_inference"] = batch_size_inference

        N_ensemble_configurations = overwrite_params.get(
            "N_ensemble_configurations",
            self.get_max_N_ensemble_configurations(X, **overwrite_params),
        )

        if len(X.shape) == 3:
            num_feature_groups = X.shape[2]
            num_samples = X.shape[0]
            batch_size = X.shape[1]
        elif len(X.shape) == 2:
            num_feature_groups = X.shape[1]
            num_samples = X.shape[0]
            batch_size = min(batch_size_inference, N_ensemble_configurations)
        else:
            raise ValueError(f"Unknown shape {X.shape}")

        if self.use_poly_features:
            num_feature_groups += self.max_poly_features

        num_feature_groups = int(np.ceil(num_feature_groups / features_per_group))

        num_cells = (num_feature_groups + 1) * num_samples
        bytes_per_float = 2 if fp_16 else 4

        if estimation_task == "memory":
            if per_feature_transformer:
                CONSTANT_MEMORY_OVERHEAD = 100000000
                if save_peak_mem_factor:
                    memory_factor = 2.5  # this is an approximative constant, which should give an upper bound
                    return CONSTANT_MEMORY_OVERHEAD + (
                        num_cells
                        * embedding_size
                        * bytes_per_float
                        * memory_factor
                        * batch_size
                    )
                else:
                    memory_factor = 2.5 * SAVE_PEAK_MEM_FACTOR
                    return CONSTANT_MEMORY_OVERHEAD + (
                        num_cells
                        * embedding_size
                        * bytes_per_float
                        * memory_factor
                        * batch_size
                    )
            else:
                # TODO: Check if this is correct
                print("Warning: memory usage is untested")
                return num_samples * embedding_size * bytes_per_float
        elif estimation_task == "computation":
            # TODO: Check if this is correct
            if per_feature_transformer:
                CONSTANT_COMPUTE_OVERHEAD = 8000
                NUM_SAMPLES_FACTOR = 4
                NUM_SAMPLES_PLUS_FEATURES = 6.5
                CELLS_FACTOR = 0.25
                CELLS_SQUARED_FACTOR = 1.3e-7

                compute_cost = (embedding_size**2) * num_heads * num_layers

                return (
                    N_ensemble_configurations
                    * compute_cost
                    * (
                        CONSTANT_COMPUTE_OVERHEAD
                        + num_samples * NUM_SAMPLES_FACTOR
                        + (num_samples + num_feature_groups) * NUM_SAMPLES_PLUS_FEATURES
                        + num_cells * CELLS_FACTOR
                        + num_cells**2 * CELLS_SQUARED_FACTOR
                    )
                )
            else:
                return (
                    num_samples
                    * num_samples
                    * embedding_size
                    * embedding_size
                    * num_heads
                    * num_layers
                )


RegressionOptimizationMetricType = Literal[
    "mse", "rmse", "mae", "r2", "mean", "median", "mode", "exact_match", None
]


class TabPFNRegressor(TabPFNBaseModel, RegressorMixin):
    metric_type = RegressionOptimizationMetricType

    _predict_criterion = None

    data_mean_ = None
    data_std_ = None
    criterion_ = None

    predict_function_for_shap = "predict"

    def __init__(
        self,
        model: Optional[Any] = None,
        model_string: str = "",
        c: Optional[Dict] = None,
        N_ensemble_configurations: int = 10,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        feature_shift_decoder: str = "shuffle",
        normalize_with_test: bool = False,
        average_logits: bool = True,
        optimize_metric: RegressionOptimizationMetricType = "rmse",
        transformer_predict_kwargs: Optional[Dict] = None,
        softmax_temperature: Optional[float] = 0.0,
        use_poly_features=False,
        max_poly_features=None,
        transductive=False,
        remove_outliers=0.0,
        regression_y_preprocess_transforms: Optional[Tuple[Optional[str], ...]] = (
            None,
            "power",
        ),
        add_fingerprint_features: bool = False,
        cancel_nan_borders: bool = False,
        super_bar_dist_averaging: bool = False,
        subsample_samples: float = -1,
        # The following parameters are not tunable, but specify the execution mode
        fit_at_predict_time: bool = True,
        device: tp.Literal["cuda", "cpu", "auto"] = "auto",
        seed: Optional[int] = 0,
        show_progress: bool = True,
        batch_size_inference: int = None,
        fp16_inference: bool = True,
        save_peak_memory: Literal["True", "False", "auto"] = "True",
    ):
        """
        According to Sklearn API we need to pass all parameters to the super class constructor without **kwargs or *args

        Parameters:
            regression_y_preprocess_transforms (None): Preprocessing transforms for the target variable. Not used in classification.
            super_bar_dist_averaging: TODO
            cancel_nan_borders: TODO
        """
        # print(f"{optimize_metric=}")
        assert optimize_metric in tp.get_args(
            self.metric_type
        ), f"Unknown metric {optimize_metric}"

        self.regression_y_preprocess_transforms = regression_y_preprocess_transforms
        self.cancel_nan_borders = cancel_nan_borders
        self.super_bar_dist_averaging = super_bar_dist_averaging

        super().__init__(
            model=model,
            device=device,
            model_string=model_string,
            batch_size_inference=batch_size_inference,
            fp16_inference=fp16_inference,
            c=c,
            N_ensemble_configurations=N_ensemble_configurations,
            preprocess_transforms=preprocess_transforms,
            feature_shift_decoder=feature_shift_decoder,
            normalize_with_test=normalize_with_test,
            average_logits=average_logits,
            seed=seed,
            optimize_metric=optimize_metric,
            transformer_predict_kwargs=transformer_predict_kwargs,
            show_progress=show_progress,
            save_peak_memory=save_peak_memory,
            softmax_temperature=softmax_temperature,
            use_poly_features=use_poly_features,
            max_poly_features=max_poly_features,
            transductive=transductive,
            remove_outliers=remove_outliers,
            add_fingerprint_features=add_fingerprint_features,
            fit_at_predict_time=fit_at_predict_time,
            subsample_samples=subsample_samples,
        )

    def get_optimization_mode(self):
        if self.optimize_metric is None:
            return "mean"
        elif self.optimize_metric in ["rmse", "mse", "r2", "mean"]:
            return "mean"
        elif self.optimize_metric in ["mae", "median"]:
            return "median"
        elif self.optimize_metric in ["mode", "exact_match"]:
            return "mode"
        else:
            raise ValueError(f"Unknown metric {self.optimize_metric}")

    def score(self, X, y, additional_x=None, additional_y=None, sample_weight=None):
        y_pred = self.predict(X, additional_x, additional_y)

        opt_metric = (
            self.optimize_metric if self.optimize_metric is not None else "rmse"
        )

        return score_regression(opt_metric, y, y_pred, sample_weight=sample_weight)

    def init_model_and_get_model_config(self):
        super().init_model_and_get_model_config()
        assert not self.is_classification_, "This should not be a classification model"

    def _fit(self):
        y_train, (self.data_mean_, self.data_std_) = normalize_data(
            self.y_[:, None].float(),
            normalize_positions=len(self.y_),
            return_scaling=True,
            std_only=self.normalize_std_only_,
        )

        self.criterion_ = deepcopy(self.model_processed_.criterion)

        self.transformer_predict(
            self.X_[:, None].float(),
            y_train,
            len(self.X_),
            additional_xs=self.additional_x_,
            additional_ys=self.additional_y_,
            bar_distribution=self.criterion_,
            cache_trainset_representations=not self.fit_at_predict_time,  # this will always be true here
            **get_params_from_config(self.c_processed_),
        )

    def predict(self, X, additional_x=None, additional_y=None) -> np.ndarray:
        timing_start("predict")
        prediction = self.predict_full(
            X=X, additional_x=additional_x, additional_y=additional_y
        )
        timing_end("predict")

        return prediction[self.get_optimization_mode()]

    def predict_y_proba(
        self, X: np.ndarray, y: np.ndarray, additional_x=None, additional_y=None
    ) -> np.ndarray:
        """
        Predicts the probability of the target y given the input X.
        """
        prediction = self.predict_full(X, additional_x, additional_y)
        return prediction["criterion"].pdf(
            torch.tensor(prediction["logits"]), torch.tensor(y)
        )

    def predict_full(
        self, X, additional_x=None, additional_y=None, get_additional_outputs=None
    ) -> dict:
        """
        Predicts the target y given the input X.

        Parameters:
            X:
            additional_x: Additional features
            additional_y: Additional inputs
            get_additional_outputs: Keys for additional outputs to return.


        Returns:
             (dict: The predictions, dict: Additional outputs)
        """
        (
            X_full,
            y_full,
            additional_x,
            additional_y,
            eval_position,
        ) = self.predict_common_setup(
            X_eval=X, additional_x_eval=additional_x, additional_y_eval=additional_y
        )

        y_full, (data_mean, data_std) = normalize_data(
            data=y_full,
            normalize_positions=eval_position,
            return_scaling=True,
            std_only=self.normalize_std_only_,
            mean=None if self.fit_at_predict_time else self.data_mean_,
            std=None if self.fit_at_predict_time else self.data_std_,
            clip=False,
        )

        if self.fit_at_predict_time:
            criterion = deepcopy(self.model_processed_.criterion)
        else:
            criterion = self.criterion_

        prediction, additional_outputs = self.transformer_predict(
            eval_xs=X_full,
            eval_ys=y_full,
            eval_position=eval_position if self.fit_at_predict_time else 0,
            additional_xs=additional_x,
            additional_ys=additional_y,
            bar_distribution=criterion,
            cache_trainset_representations=not self.fit_at_predict_time,
            get_additional_outputs=get_additional_outputs,
            **get_params_from_config(self.c_processed_),
        )
        self._predict_criterion = self._post_process_predict_criterion(
            criterion=criterion, data_mean=data_mean, data_std=data_std
        )
        return {
            **self._post_process_predict_full(
                prediction=prediction,
                criterion=self._predict_criterion,
            ),
            **additional_outputs,
        }

    @staticmethod
    def _post_process_predict_full(
        prediction: torch.Tensor,
        criterion: FullSupportBarDistribution,
    ) -> dict:
        prediction_ = prediction.squeeze(0)

        predictions = {
            "criterion": criterion.cpu(),
            "mean": criterion.mean(prediction_.cpu()).detach().numpy(),
            "median": criterion.median(prediction_.cpu()).detach().numpy(),
            "mode": criterion.mode(prediction_.cpu()).detach().numpy(),
            "logits": prediction_.cpu().detach().numpy(),
            "buckets": torch.nn.functional.softmax(prediction_.cpu(), dim=-1)
            .detach()
            .numpy(),
        }

        predictions.update(
            {
                f"quantile_{q:.2f}": criterion.icdf(prediction_.cpu(), q)
                .detach()
                .numpy()
                for q in tuple(i / 10 for i in range(1, 10))
            }
        )

        return predictions

    def _post_process_predict_criterion(
        self, criterion: FullSupportBarDistribution, data_mean, data_std
    ) -> FullSupportBarDistribution:
        data_mean_added = (
            data_mean.to(criterion.borders.device)
            if not self.normalize_std_only_
            else 0
        )
        criterion.borders = (
            criterion.borders * data_std.to(criterion.borders.device) + data_mean_added
        ).float()
        return criterion


ClassificationOptimizationMetricType = Literal[
    "auroc", "roc", "auroc_ovo", "balanced_acc", "acc", "log_loss", None
]


class TabPFNClassifier(TabPFNBaseModel, ClassifierMixin):
    semisupervised_indicator = -100
    metric_type = ClassificationOptimizationMetricType

    predict_function_for_shap = "predict_proba"

    def __init__(
        self,
        model: Optional[Any] = None,
        model_string: str = "",
        c: Optional[Dict] = None,
        N_ensemble_configurations: int = 10,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        feature_shift_decoder: str = "shuffle",
        normalize_with_test: bool = False,
        average_logits: bool = False,
        optimize_metric: ClassificationOptimizationMetricType = "roc",
        transformer_predict_kwargs: Optional[Dict] = None,
        multiclass_decoder="shuffle",
        softmax_temperature: Optional[float] = math.log(0.8),
        use_poly_features=False,
        max_poly_features=None,
        transductive=False,
        remove_outliers=0.0,
        add_fingerprint_features=False,
        subsample_samples=-1,
        # The following parameters are not tunable, but specify the execution mode
        fit_at_predict_time: bool = True,
        device: tp.Literal["cuda", "cpu", "auto"] = "auto",
        seed: Optional[int] = 0,
        show_progress: bool = True,
        batch_size_inference: int = None,
        fp16_inference: bool = True,
        save_peak_memory: Literal["True", "False", "auto"] = "True",
    ):
        """
        You need to specify a model either by setting the `model_string` or by setting `model` and `c`,
        where the latter is the config.

        Parameters:
            model (Optional[Any]): The model, if you want to specify it directly, this is used in combination with c.
            device: The device to use for inference, "auto" means that it will use cuda if available, otherwise cpu
            model_string (str): The model string is the path to the model.
            batch_size_inference (int): The batch size to use for inference, this does not affect the results, just the
                memory usage and speed. A higher batch size is faster but uses more memory. Setting the batch size to None
                means that the batch size is automatically determined based on the memory usage and the maximum free memory
                specified with `maximum_free_memory_in_gb`.
            fp16_inference (bool): Whether to use fp16 for inference on GPU, does not affect CPU inference.
            inference_mode (bool): Whether to use inference mode, which does not allow to backpropagate through the model.
            c (Optional[Dict]): The config, if you want to specify it directly, this is used in combination with model.
            N_ensemble_configurations (int): The number of ensemble configurations to use, the most important setting.
            preprocess_transforms (Tuple[PreprocessorConfig, ...]): A tuple of strings, specifying the preprocessing steps to use.
                You can use the following strings as elements '(none|power|quantile|robust)[_all][_and_none]', where the first
                part specifies the preprocessing step and the second part specifies the features to apply it to and
                finally '_and_none' specifies that the original features should be added back to the features in plain.
                Finally, you can combine all strings without `_all` with `_onehot` to apply one-hot encoding to the categorical
                features specified with `self.fit(..., categorical_features=...)`.
            feature_shift_decoder (str): ["False", "True", "auto"] Whether to shift features for each ensemble configuration.
            normalize_with_test (bool): If True, the test set is used to normalize the data, otherwise the training set is used only.
            average_logits (bool): Whether to average logits or probabilities for ensemble members.
            optimize_metric (ClassificationOptimizationMetricType): The optimization metric to use.
            seed (Optional[int]): The default seed to use for the order of the ensemble configurations, a seed of None will not.
            transformer_predict_kwargs (Optional[Dict]): Additional keyword arguments to pass to the transformer predict method.
            show_progress (bool): Whether to show progress bars during training and inference.
            multiclass_decoder (str): The multiclass decoder to use.
            save_peak_memory (Literal["True", "False", "auto"]): Whether to save the peak memory usage of the model, can enable up to 8 times larger datasets to fit into memory.
                "True", means always enabled, "False", means always disabled, "auto" means that it will be set based on the memory usage.
            use_poly_features (bool): Whether to use polynomial features as the last preprocessing step.
            max_poly_features (int): Maximum number of polynomial features to use, None means unlimited.
            transductive (bool): Whether to use transductive learning.
            remove_outliers (float): If not 0.0, will remove outliers from the input features, where values with a standard deviation
                larger than remove_outliers will be removed.
            add_fingerprint_features (bool): If True, will add one feature of random values, that will be added to
                the input features. This helps discern duplicated samples in the transformer model.
            subsample_samples (float): If not None, will use a random subset of the samples for training in each ensemble configuration.
        """
        assert optimize_metric in tp.get_args(self.metric_type)
        self.multiclass_decoder = multiclass_decoder

        # Pass all parameters to super class constructor
        super().__init__(
            model=model,
            device=device,
            model_string=model_string,
            batch_size_inference=batch_size_inference,
            fp16_inference=fp16_inference,
            c=c,
            N_ensemble_configurations=N_ensemble_configurations,
            preprocess_transforms=preprocess_transforms,
            feature_shift_decoder=feature_shift_decoder,
            normalize_with_test=normalize_with_test,
            average_logits=average_logits,
            optimize_metric=optimize_metric,
            seed=seed,
            transformer_predict_kwargs=transformer_predict_kwargs,
            show_progress=show_progress,
            softmax_temperature=softmax_temperature,
            save_peak_memory=save_peak_memory,
            use_poly_features=use_poly_features,
            max_poly_features=max_poly_features,
            transductive=transductive,
            remove_outliers=remove_outliers,
            add_fingerprint_features=add_fingerprint_features,
            subsample_samples=subsample_samples,
            fit_at_predict_time=fit_at_predict_time,
        )

    def _validate_targets(self, y) -> np.ndarray:
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)

        # Get classes and encode before type conversion to guarantee correct class labels.
        not_nan_mask = ~np.isnan(y)
        cls, y_[not_nan_mask] = np.unique(y_[not_nan_mask], return_inverse=True)

        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        # convert type to align with the negative value of the indicator (e.g., avoid uint8)
        y_ = y_.astype(np.float32)
        y_[~not_nan_mask] = TabPFNClassifier.semisupervised_indicator

        return np.asarray(y_, dtype=np.float32, order="C")

    @staticmethod
    def check_training_data(
        clf: TabPFNClassifier, X: np.ndarray, y: np.ndarray
    ) -> Tuple:
        y[np.isnan(y)] = clf.semisupervised_indicator

        unique_labels_ = unique_labels(y)

        has_nan_class_count = 0
        if clf.semisupervised_indicator in unique_labels_:
            assert clf.c_processed_.get(
                "semisupervised_enabled", False
            ), "Semisupervised not enabled for this model"
            has_nan_class_count = 1
            print_once(
                "Found nan class in training data, will be used as semisupervsied"
            )

        if len(unique_labels_) > clf.max_num_classes_ + has_nan_class_count:
            raise ValueError(
                f"The number of classes for this classifier is restricted to {clf.max_num_classes_}. "
                f"Consider wrapping the estimator with `tabpfn.estimator.ManyClassClassifier` to be "
                f"able to predict more classes"
            )

        clf.num_classes_ = len(unique_labels_) - has_nan_class_count

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, force_all_finite=False)

        # Store the classes seen during fit
        y = clf._validate_targets(y)
        clf.label_encoder_ = LabelEncoder()
        y[
            y != TabPFNClassifier.semisupervised_indicator
        ] = clf.label_encoder_.fit_transform(
            y[y != TabPFNClassifier.semisupervised_indicator]
        )

        X, y = TabPFNBaseModel.check_training_data(clf, X, y)

        return X, y

    def init_model_and_get_model_config(self):
        super().init_model_and_get_model_config()
        assert self.is_classification_, "This should be a classification model"
        self.max_num_classes_ = self.c_processed_["max_num_classes"]

    def _post_process_predict_proba(self, prediction: torch.Tensor) -> np.ndarray:
        prediction = prediction.squeeze(0)
        prediction = prediction.detach().cpu().numpy()

        if self.sklearn_compatible_precision:
            print("SKLEARN COMPATIBLE PREDICTION FOR DEBUG PURPOSE")
            prediction = np.around(
                prediction, decimals=16
            )  # TODO: Do we want this, its just for sklearn
            prediction[prediction < 0.001] = 0
            prediction = prediction / prediction.sum(axis=1, keepdims=True)
        return prediction

    def fit(self, X, y, additional_x=None, additional_y=None) -> TabPFNClassifier:
        """
        Fits the TabPFNClassifier model to the input data `X` and `y`.

        The actual training logic is delegated to the `_fit` method, which should be implemented by subclasses.

        Parameters:
            X (Union[ndarray, torch.Tensor]): The input feature matrix of shape (n_samples, n_features).
            y (Union[ndarray, torch.Tensor]): The target labels of shape (n_samples,).
            additional_x (Optional[Dict[str, torch.Tensor]]): Additional features to use during training.
            additional_y (Optional[Dict[str, torch.Tensor]]): Additional labels to use during training.

        Returns:
            TabPFNClassifier: The fitted model object (self).
        """
        return super().fit(X, y, additional_x=additional_x, additional_y=additional_y)

    def _fit(self):
        self.transformer_predict(
            self.X_[:, None].float(),
            self.y_[:, None].float(),
            len(self.X_),
            additional_xs=self.additional_x_,
            additional_ys=self.additional_y_,
            cache_trainset_representations=not self.fit_at_predict_time,  # this will always be true here
            **get_params_from_config(self.c_processed_),
            **(self.transformer_predict_kwargs or {}),
        )

    def predict_full(
        self, X, additional_x=None, additional_y=None, get_additional_outputs=None
    ) -> dict:
        timing_start("predict")
        (
            X_full,
            y_full,
            additional_x,
            additional_y,
            eval_pos,
        ) = self.predict_common_setup(
            X_eval=X, additional_x_eval=additional_x, additional_y_eval=additional_y
        )

        prediction, additional_outputs = self.transformer_predict(
            X_full,
            y_full,
            eval_pos if self.fit_at_predict_time else 0,
            additional_xs=additional_x,
            additional_ys=additional_y,
            cache_trainset_representations=not self.fit_at_predict_time,
            reweight_probs_based_on_train=self.optimizes_balanced_metric(),
            get_additional_outputs=get_additional_outputs,
            **get_params_from_config(self.c_processed_),
            **(self.transformer_predict_kwargs or {}),
        )

        prediction = self._post_process_predict_proba(prediction)
        timing_end("predict")

        return {"proba": prediction, **additional_outputs}

    def predict_proba(self, X, additional_x=None, additional_y=None):
        """
        Calls the transformer to predict the probabilities of the classes of the X test inputs given the previous set
        training dataset

        Parameters:
            X: test datapoints
        """
        return self.predict_full(
            X, additional_x=additional_x, additional_y=additional_y
        )["proba"]

    def predict(
        self, X, additional_x=None, additional_y=None, return_winning_probability=False
    ):
        """
        Predict the class labels for the input samples.

        Parameters:
            X (array-like): The input samples.
            return_winning_probability (bool): Whether to return the winning probability.

        Returns:
            array: The predicted class labels.
        """
        p = self.predict_proba(X, additional_x, additional_y)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=int))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y

    def predict_y_proba(self, X, y, additional_x=None, additional_y=None):
        """
        Predict the probability of the target labels `y` given the input samples `X`.

        Parameters:
            X (array-like): The input samples.
            y (array-like): The target labels.

        Returns:
            array: The predicted probabilities of the target labels.
        """
        prediction = self.predict_proba(X, additional_x, additional_y)
        y_prob = prediction[np.arange(len(y)), y.astype(int)]
        return y_prob

    def score(self, X, y, additional_x=None, additional_y=None, sample_weight=None):
        """
        Compute the score of the model on the given test data and labels.

        Parameters:
            X (array-like): The input samples.
            y (array-like): The true labels for `X`.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            float: The computed score.
        """
        if self.optimize_metric in ["roc", "auroc", "log_loss"]:
            pred = self.predict_proba(
                X, additional_x=additional_x, additional_y=additional_y
            )

            if len(np.unique(y)) == 2:
                pred = pred[:, 1]
        else:
            pred = self.predict(X, additional_x=additional_x, additional_y=additional_y)

        return score_classification(
            self.optimize_metric, y, pred, sample_weight=sample_weight
        )


def get_single_tabpfn(config: TabPFNConfig, **kwargs):
    """Get a single TabPFN model given the task type."""
    return tabpfn_task_type_models[config.task_type](**config.to_kwargs(), **kwargs)


from .base_dist_shift import TabPFNDistShiftClassifier

tabpfn_task_type_models = {
    "multiclass": TabPFNClassifier,
    "regression": TabPFNRegressor,
    "dist_shift_multiclass": TabPFNDistShiftClassifier,
}
