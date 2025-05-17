from __future__ import annotations

import torch
import math

import typing as tp
from typing import Optional, Tuple, Any, Dict, Literal

from .configs import PreprocessorConfig
from .base import TabPFNClassifier, ClassificationOptimizationMetricType


class TabPFNDistShiftClassifier(TabPFNClassifier):
    semisupervised_indicator = -100
    metric_type = ClassificationOptimizationMetricType

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
        device: Literal["cuda", "cpu", "auto"] = "auto",
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

        self.predict_function_for_shap = self.predict_proba

        # Define whether the dist shift domain information should be appended to the input data or not.
        # Only takes effect if 'dist_shift_active' is False.
        self.append_domain = True

    def _preprocess_dist_shift_domain(self, X, additional_x):
        # In case the dist shift config is active, we just check that the corresponding information is available in additional_x
        # in order to be encoded correctly in the respective encoder.
        assert (
            "dist_shift_domain" in additional_x
        ), "The distribution shift domain information is missing."

        # Cast to expected shape of (n_samples, 1, 1) if necessary.
        # overall shape: (n_samples, batch_size, num_features)
        if len(additional_x["dist_shift_domain"].shape) == 1:
            additional_x["dist_shift_domain"] = additional_x[
                "dist_shift_domain"
            ].unsqueeze(1)
        if len(additional_x["dist_shift_domain"].shape) == 2:
            additional_x["dist_shift_domain"] = additional_x[
                "dist_shift_domain"
            ].unsqueeze(1)
        assert (
            additional_x["dist_shift_domain"].shape[1] == 1
            and additional_x["dist_shift_domain"].shape[2] == 1
            and additional_x["dist_shift_domain"].shape[0] == X.shape[0]
        ), f"Invalid dist shift domain shape {additional_x['dist_shift_domain'].shape}"

        # Otherwise, we have a base model, for which we concatenate the dist shift domain information to the input data. And ensemble
        # as well as encode it as a typical feature.
        if not self.c_processed_["dist_shift_active"]:
            if self.append_domain:
                # As casting would be done at a later point in the super() call, we need to do it here manually, beforehand.
                X = X if torch.is_tensor(X) else torch.tensor(X)

                dist_shift_domain = additional_x["dist_shift_domain"]
                dist_shift_domain = (
                    dist_shift_domain
                    if torch.is_tensor(dist_shift_domain)
                    else torch.tensor(dist_shift_domain)
                )

                # We concatenate the dist_shift_domain to the end in this case in order to not mess up the feature order. This might
                # be relevant for categorical feature indices etc.
                X = torch.cat([X, dist_shift_domain.squeeze(-1)], dim=-1)

            # Remove the dist_shift_domain from the additional_x dictionary, as it is not needed anymore.
            # This is done in order to let any access to it when dist_shift_active is False to raise an error.
            del additional_x["dist_shift_domain"]

            if len(additional_x) == 0:
                additional_x = None

        return X, additional_x

    def set_append_domain(self, append_domain: bool):
        """
        Set whether to append the domain information to the input data or not. Only takes effect if 'dist_shift_active'
        is False.

        Parameters:
            append_domain (bool): Whether to append the domain information to the input data or not.
        """
        self.append_domain = append_domain

    def _pre_train_hook(self):
        """
        Preprocess the domain information before training the model in fit().
        """
        # Store the original value of append_domain, as it might be changed after training. We want to catch this change.
        self.append_domain_ = self.append_domain

        self.X_, self.additional_x_ = self._preprocess_dist_shift_domain(
            self.X_, self.additional_x_
        )
        self.n_features_in_ = self.X_.shape[1]

    def predict_proba(self, X, additional_x=None, additional_y=None):
        """
        Preprocess the domain information before prediction.
        """
        assert (
            self.append_domain == self.append_domain_
        ), "The append_domain parameter has been changed since fit()."

        X, additional_x = self._preprocess_dist_shift_domain(X, additional_x)

        return super().predict_proba(
            X, additional_x=additional_x, additional_y=additional_y
        )
