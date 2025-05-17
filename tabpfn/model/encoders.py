from __future__ import annotations

from abc import ABCMeta

import torch
import torch.nn as nn
import numpy as np

from tabpfn.utils import (
    normalize_data,
    to_ranking_low_mem,
    remove_outliers,
    torch_nanmean,
    print_once,
    min_max_scale_data,
)
from .utils import select_features


class InputEncoder(nn.Module):
    """Base class for input encoders.

    All input encoders should subclass this class and implement the `forward` method.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
    ) -> torch.Tensor:
        """Encode the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor to encode.
            single_eval_pos (int): The position to use for single evaluation.

        Returns:
            torch.Tensor: The encoded tensor.
        """
        raise NotImplementedError


## ENCODER COMPONENTS


class SequentialEncoder(torch.nn.Sequential, InputEncoder):
    """An encoder that applies a sequence of encoder steps.

    SequentialEncoder allows building an encoder from a sequence of EncoderSteps.
    The input is passed through each step in the provided order.
    """

    def __init__(self, *args, output_key: str = "output", **kwargs):
        """Initialize the SequentialEncoder.

        Parameters:
            *args: A list of SeqEncStep instances to apply in order.
            output_key (str): The key to use for the output of the encoder in the state dict.
                              Defaults to "output", i.e. `state["output"]` will be returned.
            **kwargs: Additional keyword arguments passed to `torch.nn.Sequential`.
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(self, input: dict, **kwargs) -> torch.Tensor:
        """Apply the sequence of encoder steps to the input.

        Parameters:
            input (dict): The input state dictionary.
                          If the input is not a dict and the first layer expects one input key,
                          the input tensor is mapped to the key expected by the first layer.
            **kwargs: Additional keyword arguments passed to each encoder step.

        Returns:
            torch.Tensor: The output of the final encoder step.
        """
        if type(input) != dict:
            # If the input is not a dict and the first layer expects one input, mapping the
            #   input to the first input key must be correct
            if len(self[0].in_keys) == 1:
                input = {self[0].in_keys[0]: input}

        for module in self:
            input = module(input, **kwargs)

        return input[self.output_key] if self.output_key is not None else input


class LinearInputEncoder(torch.nn.Module):
    """A simple linear input encoder."""

    def __init__(
        self,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
    ):
        """Initialize the LinearInputEncoder.

        Parameters:
            num_features (int): The number of input features.
            emsize (int): The embedding size, i.e. the number of output features.
            replace_nan_by_zero (bool): Whether to replace NaN values in the input by zero. Defaults to False.
            bias (bool): Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, *x, **kwargs):
        """Apply the linear transformation to the input.

        Parameters:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return (self.layer(x),)


class SeqEncStep(torch.nn.Module, metaclass=ABCMeta):
    """Abstract base class for sequential encoder steps.

    SeqEncStep is a wrapper around a module that defines the expected input keys
    and the produced output keys. The outputs are assigned to the output keys
    in the order specified by `out_keys`.

    Subclasses should either implement `_forward` or `_fit` and `_transform`.
    Subclasses that transform `x` should always use `_fit` and `_transform`,
    creating any state that depends on the train set in `_fit` and using it in `_transform`.
    This allows fitting on data first and doing inference later without refitting.
    Subclasses that work with `y` can alternatively use `_forward` instead.
    """

    def __init__(
        self, in_keys: tuple[str] = ("main",), out_keys: tuple[str] = ("main",)
    ):
        """Initialize the SeqEncStep.

        Parameters:
            in_keys (tuple[str]): The keys of the input tensors. Defaults to ("main",).
            out_keys (tuple[str]): The keys to assign the output tensors to. Defaults to ("main",).
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    # Either implement _forward:

    def _forward(self, *x, **kwargs):
        """Forward pass of the encoder step. Implement this if not implementing _fit and _transform.

        Parameters:
            *x: The input tensors. A single tensor or a tuple of tensors.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The output tensor or a tuple of output tensors.
        """
        raise NotImplementedError

    # Or implement _fit and _transform:

    def _fit(self, *x, single_eval_pos: int = None, **kwargs):
        """Fit the encoder step on the training set.

        Parameters:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos (int): The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.
        """
        raise NotImplementedError

    def _transform(self, *x, single_eval_pos: int = None, **kwargs):
        """Transform the data using the fitted encoder step.

        Parameters:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos (int): The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The transformed output tensor or a tuple of output tensors.
        """
        raise NotImplementedError

    def forward(
        self, state: dict, cache_trainset_representation: bool = False, **kwargs
    ):
        """Perform the forward pass of the encoder step.

        Parameters:
            state (dict): The input state dictionary containing the input tensors.
            cache_trainset_representation (bool): Whether to cache the training set representation.
                                                  Only supported for _fit and _transform (not _forward).
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The updated state dictionary with the output tensors assigned to the output keys.
        """
        try:
            args = [state[in_key] for in_key in self.in_keys]
        except KeyError:
            raise KeyError(
                f"EncoderStep expected input keys {self.in_keys}, but got {list(state.keys())}"
            )

        if hasattr(self, "_fit"):
            if kwargs["single_eval_pos"] or not cache_trainset_representation:
                self._fit(*args, **kwargs)
            out = self._transform(*args, **kwargs)
        else:
            assert (
                not cache_trainset_representation
            ), f"cache_trainset_representation is not supported for _forward, as implemented in {self.__class__.__name__}"
            out = self._forward(*args, **kwargs)

        assert (
            type(out) == tuple
        ), "EncoderStep must return a tuple of values (can be size 1)"
        assert len(out) == len(
            self.out_keys
        ), f"EncoderStep outputs don't match out_keys {len(out)} (out) != {len(self.out_keys)} (out_keys = {self.out_keys})"

        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state


class LinearInputEncoderStep(SeqEncStep):
    """A simple linear input encoder step."""

    def __init__(
        self,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str] = ("main",),
        out_keys: tuple[str] = ("output",),
    ):
        """Initialize the LinearInputEncoderStep.

        Parameters:
            num_features (int): The number of input features.
            emsize (int): The embedding size, i.e. the number of output features.
            replace_nan_by_zero (bool): Whether to replace NaN values in the input by zero. Defaults to False.
            bias (bool): Whether to use a bias term in the linear layer. Defaults to True.
            in_keys (tuple[str]): The keys of the input tensors. Defaults to ("main",).
            out_keys (tuple[str]): The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *x, **kwargs):
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""
        pass

    def _transform(self, *x, **kwargs):
        """Apply the linear transformation to the input.

        Parameters:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return (self.layer(x),)


class NanHandlingEncoderStep(SeqEncStep):
    """Encoder step to handle NaN and infinite values in the input."""

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        keep_nans: bool = True,
        in_keys: tuple[str] = ("main",),
        out_keys: tuple[str] = ("main", "nan_indicators"),
    ):
        """Initialize the NanHandlingEncoderStep.

        Parameters:
            keep_nans (bool): Whether to keep NaN values as separate indicators. Defaults to True.
            in_keys (tuple[str]): The keys of the input tensors. Must be a single key.
            out_keys (tuple[str]): The keys to assign the output tensors to.
                                   Defaults to ("main", "nan_indicators").
        """
        assert len(in_keys) == 1, "NanHandlingEncoderStep expects a single input key"
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        """Compute the feature means on the training set for replacing NaNs.

        Parameters:
            x (torch.Tensor): The input tensor.
            single_eval_pos (int): The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        self.feature_means_ = torch_nanmean(
            x[:single_eval_pos], axis=0, eps=1e-10, include_inf=True
        )

    def _transform(self, x: torch.Tensor, **kwargs):
        """Replace NaN and infinite values in the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor and optionally the NaN indicators.
        """
        nans_indicator = None
        if self.keep_nans:
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations
        x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]

        return x, nans_indicator


class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove empty (constant) features."""

    def __init__(self, **kwargs):
        """Initialize the RemoveEmptyFeaturesEncoderStep.

        Parameters:
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.sel = None

    def _fit(self, x: torch.Tensor, **kwargs):
        """Compute the feature selection mask on the training set.

        Parameters:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        self.sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

    def _transform(self, x: torch.Tensor, **kwargs):
        """Remove empty features from the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor with empty features removed.
        """
        return (select_features(x, self.sel),)


class RemoveDuplicateFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove duplicate features."""

    def __init__(self, normalize_on_train_only: bool = True, **kwargs):
        """Initialize the RemoveDuplicateFeaturesEncoderStep.

        Parameters:
            normalize_on_train_only (bool): Whether to normalize only on the training set. Defaults to True.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        """Currently does nothing. Fit functionality not implemented."""
        pass

    def _transform(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        """Remove duplicate features from the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            single_eval_pos (int): The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the input tensor (removal not implemented).
        """
        # TODO: This uses a lot of memory, as it computes the covariance matrix for each batch
        #   This could be done more efficiently, models go OOM with this
        return (x,)
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        x_norm = normalize_data(x[:, :normalize_position])
        sel = torch.zeros(x.shape[1], x.shape[2], dtype=torch.bool, device=x.device)
        for B in range(x_norm.shape[1]):
            cov_mat = (torch.cov(x_norm[:, B].transpose(1, 0)) > 0.999).float()
            cov_mat_sum_below_trace = torch.triu(cov_mat).sum(dim=0)
            sel[B] = cov_mat_sum_below_trace == 1.0

        new_x = select_features(x, sel)

        return (new_x,)


class VariableNumFeaturesEncoderStep(SeqEncStep):
    """Encoder step to handle variable number of features.

    Transforms the input to a fixed number of features by appending zeros.
    Also normalizes the input by the number of used features to keep the variance
    of the input constant, even when zeros are appended.
    """

    def __init__(
        self,
        num_features: int,
        normalize_by_used_features: bool = True,
        normalize_by_sqrt: bool = True,
        **kwargs,
    ):
        """Initialize the VariableNumFeaturesEncoderStep.

        Parameters:
            num_features (int): The number of features to transform the input to.
            normalize_by_used_features (bool): Whether to normalize by the number of used features. Defaults to True.
            normalize_by_sqrt (bool): Legacy option to normalize by sqrt instead of the number of used features. Defaults to True.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features = num_features
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features = None

    def _fit(self, x: torch.Tensor, **kwargs):
        """Compute the number of used features on the training set.

        Parameters:
            x (torch.Tensor): The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

        # Get the number of non-constant features in each batch.
        # Clip with min 1 to prevent division by zero.
        self.number_of_used_features = torch.clip(sel.sum(-1).unsqueeze(-1), min=1)

    def _transform(self, x: torch.Tensor, **kwargs):
        """Transform the input tensor to have a fixed number of features.

        Parameters:
            x (torch.Tensor): The input tensor of shape (seq_len, batch_size, num_features_old).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor of shape (seq_len, batch_size, num_features).
        """
        if x.shape[2] == 0:
            return torch.zeros(
                x.shape[0],
                x.shape[1],
                self.num_features,
                device=x.device,
                dtype=x.dtype,
            )
        if self.normalize_by_used_features:
            if self.normalize_by_sqrt:
                # Verified that this gives indeed unit variance with appended zeros
                x = x * torch.sqrt(self.num_features / self.number_of_used_features)
            else:
                x = x * (self.num_features / self.number_of_used_features)

        zeros_appended = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros_appended], -1)
        return (x,)


class InputNormalizationEncoderStep(SeqEncStep):
    """Encoder step to normalize the input in different ways.

    Can be used to normalize the input to a ranking, remove outliers,
    or normalize the input to have unit variance.
    """

    def __init__(
        self,
        normalize_on_train_only: bool,
        normalize_to_ranking: bool,
        normalize_x: bool,
        remove_outliers: bool,
        remove_outliers_sigma: float = 4.0,
        seed: int = 0,
        **kwargs,
    ):
        """Initialize the InputNormalizationEncoderStep.

        Parameters:
            normalize_on_train_only (bool): Whether to compute normalization only on the training set.
            normalize_to_ranking (bool): Whether to normalize the input to a ranking.
            normalize_x (bool): Whether to normalize the input to have unit variance.
            remove_outliers (bool): Whether to remove outliers from the input.
            remove_outliers_sigma (float): The number of standard deviations to use for outlier removal. Defaults to 4.0.
            seed (int): Random seed for reproducibility. Defaults to 0.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only
        self.normalize_to_ranking = normalize_to_ranking
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.remove_outliers_sigma = remove_outliers_sigma
        self.seed = seed
        self.reset_seed()
        self.lower_for_outlier_removal = None
        self.upper_for_outlier_removal = None
        self.mean_for_normalization = None
        self.std_for_normalization = None

    def reset_seed(self):
        """Reset the random seed."""
        pass

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        """Compute the normalization statistics on the training set.

        Parameters:
            x (torch.Tensor): The input tensor.
            single_eval_pos (int): The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1
        if self.remove_outliers and not self.normalize_to_ranking:
            x, (
                self.lower_for_outlier_removal,
                self.upper_for_outlier_removal,
            ) = remove_outliers(
                x,
                normalize_positions=normalize_position,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            x, (
                self.mean_for_normalization,
                self.std_for_normalization,
            ) = normalize_data(
                x, normalize_positions=normalize_position, return_scaling=True
            )

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs,
    ):
        """Normalize the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            single_eval_pos (int): The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the normalized tensor.
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        if self.normalize_to_ranking:
            assert (
                False
            ), "Not implemented currently as it was not used in a long time and hard to move out the state."
            x = to_ranking_low_mem(x)

        elif self.remove_outliers:
            assert (
                self.remove_outliers_sigma > 1.0
            ), "remove_outliers_sigma must be > 1.0"

            x, _ = remove_outliers(
                x,
                normalize_positions=normalize_position,
                lower=self.lower_for_outlier_removal,
                upper=self.upper_for_outlier_removal,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            x = normalize_data(
                x,
                normalize_positions=normalize_position,
                mean=self.mean_for_normalization,
                std=self.std_for_normalization,
            )

        return (x,)


class MinMaxScalingEncoderStep(SeqEncStep):
    """
    Linearly transforms the input to a certain range given by the min and max value.
    """

    def __init__(
        self,
        normalize_on_train_only: bool,
        min: float = 0,
        max: float = 1,
        **kwargs,
    ):
        """
        :param normalize_on_train_only: If True, the normalization is calculated only on the training set and applied
        to the test set as well. If False, the normalization is calculated on the entire dataset.
        :param min: The minimum value to which the minimum value in the data columns gets mapped to.
        :param max: The maximum value to which the maximum value in the data columns gets mapped to.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only
        self.min = min
        self.max = max

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        pass

    def _transform(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        x = min_max_scale_data(
            x, min=self.min, max=self.max, normalize_positions=normalize_position
        )

        return (x,)


class Time2VecEncoderStep(SeqEncStep):
    """
    Projects data representing time into an time_emb_size dimensional space using a linear and multiple sinusoid components each
    using a scale and bias parameter. Those parameters are learnable, thereby aiming to learn a good embedding of the different
    recurring patterns encoded in time information.

    For details, see:
    @misc{
        kazemi2020timevec,
        title={Time2Vec: Learning a Vector Representation of Time},
        author={Seyed Mehran Kazemi and Rishab Goel and Sepehr Eghbali and Janahan Ramanan and Jaspreet Sahota and Sanjay Thakur and Stella Wu and Cathal Smyth and Pascal Poupart and Marcus Brubaker},
        year={2020},
        url={https://openreview.net/forum?id=rklklCVYvB}
    }

    """

    def __init__(
        self,
        time_emb_size: int = 20,
        weight_init_alpha: float = 1.0,
        weight_init_beta: float = 2.0,
        gradient_multiplier: float = 1.0,
        **kwargs,
    ):
        """
        :param time_emb_size: The dimension of the time embedding the time feature is projected to.
        :weight_init_alpha: The alpha parameter of the sine scale initialization.
        :weight_init_beta: The beta parameter of the sine scale initialization.
        :gradient_multiplier: The factor by which the gradients are scaled.
        """
        super().__init__(**kwargs)

        assert (
            time_emb_size >= 2
        ), "There should be at least one linear and one sinusoid component."

        self.linear_time_transform = nn.Linear(1, time_emb_size)
        self.num_features_out = time_emb_size

        ## Initialize the scaling factors and biases:
        # The linear projection should be the identity in the beginning.
        self.linear_time_transform.weight.data[0] = 1.0
        self.linear_time_transform.bias.data[0] = 0.0

        # For the sine scale, we initialize the scales using a beta function in the range of [0, 2000].
        self.linear_time_transform.weight.data[1:] = (
            torch.Tensor(
                np.random.beta(
                    weight_init_alpha, weight_init_beta, (time_emb_size - 1, 1)
                )
            )
            * 2000
        )  # initializing weights

        # For the sine bias, we uniformly choose the bias in the range [0, 2pi].
        self.linear_time_transform.bias.data[1:] = torch.Tensor(
            np.random.uniform(0.0, 2 * np.pi, (time_emb_size - 1,))
        )  # initializing biases

        # Register hooks to potentially scale gradients
        self.linear_time_transform.weight.register_hook(
            self.multiply_gradient_hook(gradient_multiplier)
        )
        self.linear_time_transform.bias.register_hook(
            self.multiply_gradient_hook(gradient_multiplier)
        )

    def multiply_gradient_hook(self, factor: float):
        def hook_function(grad):
            return grad * factor

        return hook_function

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        pass

    def _transform(self, x: torch.Tensor, single_eval_pos: int, **kwargs):
        assert (
            x.shape[-1] == 1
        ), "Time2Vec expects single features that encode a certain time. Use separate encoders for different time features."

        transformed_time = self.linear_time_transform(x)

        # The first dimension is kept as a linear component, the rest are transformed using sin.
        x = torch.cat(
            (transformed_time[:, :, :1], torch.sin(transformed_time[:, :, 1:])), dim=-1
        )

        return (x,)


class FrequencyFeatureEncoderStep(SeqEncStep):
    """Encoder step to add frequency-based features to the input."""

    def __init__(
        self,
        num_features: int,
        num_frequencies: int,
        freq_power_base: float = 2.0,
        max_wave_length: float = 4.0,
        **kwargs,
    ):
        """Initialize the FrequencyFeatureEncoderStep.

        Parameters:
            num_features (int): The number of input features.
            num_frequencies (int): The number of frequencies to add (both sin and cos).
            freq_power_base (float): The base of the frequencies. Defaults to 2.0.
                                     Frequencies will be `freq_power_base`^i for i in range(num_frequencies).
            max_wave_length (float): The maximum wave length. Defaults to 4.0.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.num_features = num_features
        self.num_features_out = num_features + 2 * num_frequencies * num_features
        self.freq_power_base = freq_power_base
        # We add frequencies with a factor of freq_power_base
        wave_lengths = torch.tensor(
            [freq_power_base**i for i in range(num_frequencies)], dtype=torch.float
        )
        wave_lengths = wave_lengths / wave_lengths[-1] * max_wave_length
        # After this adaption, the last (highest) wavelength is max_wave_length
        self.register_buffer("wave_lengths", wave_lengths)

    def _fit(
        self,
        x: torch.Tensor,
        single_eval_pos: int = None,
        categorical_inds: list[int] = None,
    ):
        """Fit the encoder step. Does nothing for FrequencyFeatureEncoderStep."""
        pass

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int = None,
        categorical_inds: list[int] = None,
        **kwargs,
    ):
        """Add frequency-based features to the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor of shape (seq_len, batch_size, num_features).
            single_eval_pos (int): The position to use for single evaluation. Not used.
            categorical_inds (list[int]): The indices of categorical features. Not used.

        Returns:
            A tuple containing the transformed tensor of shape (seq_len, batch_size, num_features + 2 * num_frequencies * num_features).
        """
        extended = x[..., None] / self.wave_lengths[None, None, None, :] * 2 * torch.pi
        new_features = torch.cat(
            (x[..., None], torch.sin(extended), torch.cos(extended)), -1
        )
        new_features = new_features.reshape(*x.shape[:-1], -1)
        return (new_features,)


class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
    """
    Expects input of size 1.
    """

    def __init__(self, num_features, emsize, base_encoder, num_embs=1_000, **kwargs):
        super().__init__(**kwargs)
        assert num_features == 1
        self.num_features = num_features
        self.emsize = emsize
        self.num_embs = num_embs
        self.embedding = nn.Embedding(num_embs, emsize)
        self.base_encoder = base_encoder

    def _fit(self, x, single_eval_pos: int, categorical_inds: list[int], **kwargs):
        pass

    def _transform(
        self, x, single_eval_pos: int, categorical_inds: list[int], **kwargs
    ):
        if categorical_inds is None:
            is_categorical = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        else:
            print_once("using cateogircal inds")
            assert all(ci in ([0], []) for ci in categorical_inds), categorical_inds
            is_categorical = torch.tensor(
                [ci == [0] for ci in categorical_inds], device=x.device
            )
        if is_categorical.any():
            lx = x[:, is_categorical]
            nan_mask = torch.isnan(lx) | torch.isinf(lx)
            lx = lx.long().clamp(0, self.num_embs - 2)
            lx[nan_mask] = self.num_embs - 1
            categorical_embs = self.embedding(lx.squeeze(-1))
        else:
            categorical_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        if (~is_categorical).any():
            lx = x[:, ~is_categorical]
            continuous_embs = self.base_encoder(lx, single_eval_pos=single_eval_pos)[0]
        else:
            continuous_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        # return (torch.cat((continuous_embs, categorical_embs), dim=1),)
        # above is wrong as we need to preserve order in the batch dimension
        embs = torch.zeros(
            x.shape[0], x.shape[1], self.emsize, device=x.device, dtype=torch.float
        )
        embs[:, is_categorical] = categorical_embs.float()
        embs[:, ~is_categorical] = continuous_embs.float()
        return (embs,)


class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


def get_linear_encoder_generator(in_keys):
    def get_linear_encoder(num_features, emsize):
        encoder = SequentialEncoder(
            LinearInputEncoderStep(
                num_features, emsize, in_keys=in_keys, out_keys=["output"]
            ),
            output_key="output",
        )
        return encoder

    return get_linear_encoder


##### TARGET ENCODERS #####


class MulticlassClassificationTargetEncoder(SeqEncStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unique_ys_ = None

    def _fit(self, y: torch.Tensor, single_eval_pos: int, **kwargs):
        assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
        self.unique_ys_ = [
            torch.unique(y[:single_eval_pos, b_i]) for b_i in range(y.shape[1])
        ]

    @staticmethod
    def flatten_targets(y: torch.Tensor, unique_ys: torch.Tensor | None = None):
        if unique_ys is None:
            unique_ys = torch.unique(y)
        y = (y.unsqueeze(-1) > unique_ys).sum(axis=-1)
        return y

    def _transform(self, y: torch.Tensor, single_eval_pos: int = None, **kwargs):
        assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
        assert not (
            y.isnan().any() and self.training
        ), "NaNs are not allowed in the target at this point during training (set to model.eval() if not in training)"
        y_new = y.clone()
        for B in range(y.shape[1]):
            y_new[:, B, :] = self.flatten_targets(y[:, B, :], self.unique_ys_[B])
        return (y_new,)
