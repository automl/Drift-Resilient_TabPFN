from __future__ import annotations

import contextlib
import dataclasses
import os
import math
import argparse
import random
import itertools
import typing as tp
import warnings
import hashlib

import torch
from torch import nn
import numpy as np
import scipy.stats as stats
import pandas as pd
import re
import networkx as nx
import time
import scipy.stats
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
import logging

if tp.TYPE_CHECKING:
    import model.transformer

logger = logging.getLogger(__name__)


def mean_nested_structures(nested_structures):
    """
    Computes the mean of a list of nested structures. Supports lists, tuples, and dicts.
    E.g. mean_nested_structures([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]) == {'a': 2, 'b': 3}
    It also works for torch.tensor in the leaf nodes.
    :param nested_structures: List of nested structures
    :return: Mean of nested structures
    """
    if isinstance(nested_structures[0], dict):
        assert all(
            [
                set(arg.keys()) == set(nested_structures[0].keys())
                for arg in nested_structures
            ]
        )
        return {
            k: mean_nested_structures([arg[k] for arg in nested_structures])
            for k in nested_structures[0]
        }
    elif isinstance(nested_structures[0], list):
        assert all([len(arg) == len(nested_structures[0]) for arg in nested_structures])
        return [mean_nested_structures(elems) for elems in zip(*nested_structures)]
    elif isinstance(nested_structures[0], tuple):
        assert all([len(arg) == len(nested_structures[0]) for arg in nested_structures])
        return tuple(mean_nested_structures(elems) for elems in zip(*nested_structures))
    else:  # Assume leaf node is a tensor-like object that supports addition
        return sum(nested_structures) / len(nested_structures)


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    else:
        return obj


# TODO: Is there a better way to do this?
#   1. Comparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def torch_nanmean(x, axis=0, return_nanshare=False, eps=1e-16, include_inf=False):
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / (num + eps), 1.0 - num / x.shape[axis]
    return value / (num + eps)


def torch_nanstd(x, axis=0, eps=1e-16):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / (num + eps)
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis), x.shape[axis], dim=axis
    )
    return torch.sqrt(
        torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num + eps - 1)
    )


def normalize_data(
    data,
    normalize_positions=-1,
    return_scaling=False,
    clip=True,
    std_only=False,
    mean=None,
    std=None,
):
    """
    Normalize data to mean 0 and std 1

    :param data: T,B,H
    :param normalize_positions: If > 0, only use the first `normalize_positions` positions for normalization
    :param return_scaling: If True, return the scaling parameters as well (mean, std)
    :param clip: If True, clip values to [-100, 100]
    :param std_only: If True, only divide by std
    :param mean, std: If given, use these values instead of computing them
    """
    assert (mean is None) == (
        std is None
    ), "Either both or none of mean and std must be given"
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)
            std = torch_nanstd(data[:normalize_positions], axis=0) + 0.000001
        else:
            mean = torch_nanmean(data, axis=0)
            std = torch_nanstd(data, axis=0) + 0.000001

        if len(data) == 1 or normalize_positions == 1:
            std[:] = 1.0

        if std_only:
            mean[:] = 0
    data = (data - mean) / std

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def hash_tensor(tensor):
    return int(hashlib.sha256(tensor.tobytes()).hexdigest(), 16) % 10**12 / 10**12


def min_max_scale_data(
    data, min=0, max=1, normalize_positions=-1, return_scaling=False, clip=True
):
    """
    Linearly transform data to range between [min, max].
    If the data in a certain column is constant this column is shifted to min.

    :param data: T,B,H
    :param min: The minimum value to which the minimum value in the data columns gets mapped to.
    :param max: The maximum value to which the maximum value in the data columns gets mapped to.
    :param normalize_positions: If > 0, only use the first `normalize_positions` positions for normalization
    :param return_scaling: If True, return the scaling parameters as well (min, max)
    :param clip: If True, clip values to [min-5, max+5]
    """
    assert max > min, "The given range is invalid. Max has to be greater than min."

    # In case the normalize position is 0, we can't infer the min and max from the data. In that case this is a no-op.
    if normalize_positions is not None and normalize_positions == 0:
        return data
    elif normalize_positions is not None and normalize_positions > 0:
        min_val = torch.min(data[:normalize_positions], dim=0, keepdim=True)[0]
        max_val = torch.max(data[:normalize_positions], dim=0, keepdim=True)[0]
    else:
        min_val = torch.min(data, dim=0, keepdim=True)[0]
        max_val = torch.max(data, dim=0, keepdim=True)[0]

    # If a column in a batch is constant, the scaler for this column is 1. Then only the min is subtracted below.
    scaler = max_val - min_val + (max_val - min_val < 1e-16).float()

    data = min + ((data - min_val) * (max - min)) / scaler

    if clip:
        data = torch.clip(data, min=min - 5, max=max + 5)

    if return_scaling:
        return data, (min_val, max_val)

    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1, lower=None, upper=None):
    # Expects T, B, H
    assert (lower is None) == (upper is None), "Either both or none of lower and upper"
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    if lower is None:
        data = X if normalize_positions == -1 else X[:normalize_positions]
        data_clean = data[:].clone()
        data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

        data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
        data_mean, data_std = torch_nanmean(data_clean, axis=0), torch_nanstd(
            data_clean, axis=0
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X, (lower, upper)


# NOP decorator for python with statements (x = NOP(); with x:)
class NOP:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


class SerializableGenerator(torch.Generator):
    """
    A serializable version of the torch.Generator, that can be saved and pickled.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x, device=device)


printed_already = set()


def print_once(*msgs: tp.Any):
    """
    Print a message or multiple messages, but only once.
    It has the same behavior on the first call as standard print.
    If you call it again with the exact same input, it won't print anymore, though.
    """
    msg = " ".join([(m if isinstance(m, str) else repr(m)) for m in msgs])
    if msg not in printed_already:
        print(msg)
        printed_already.add(msg)


def compare_nested_dicts(old_dict, new_dict):
    """
    Compare two nested dicts with each other and give verbose prints of their diffs.
    :param old_dict: dict
    :param new_dict: dict
    """
    print("only in new\n")
    for k in new_dict.keys() - old_dict.keys():
        print(k, new_dict[k])

    print("only in old\n")
    for k in old_dict.keys() - new_dict.keys():
        print(k, old_dict[k])

    print("in both\n")
    for k in old_dict.keys() & new_dict.keys():
        if old_dict[k] != new_dict[k]:
            if isinstance(new_dict[k], dict):
                print("\ngoing into", k, "\n")
                compare_nested_dicts(old_dict[k], new_dict[k])
            else:
                print(k, "old:", old_dict[k], "new:", new_dict[k])


def np_load_if_exists(path):
    """Checks if a numpy file exists. Returns None if not, else returns the loaded file."""
    if os.path.isfile(path):
        # print(f'loading results from {path}')
        with open(path, "rb") as f:
            try:
                return np.load(f, allow_pickle=True).tolist()
            except Exception as e:
                logger.warning(f"Could not load {path} because {e}")
                return None
    return None


def skew(x: np.ndarray):
    """
    skewness: 3 * (mean - median) / std
    """
    return 3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0)


"""timings with GPU involved are potentially wrong.
TODO: a bit of documentation on how to use these.
maybe include torch.cuda.synchronize!? but might make things slower..
maybe better write that timings with GPU involved are potentially wrong.
"""

timing_dict_aggregation, timing_dict, timing_meta_dict = {}, {}, {}


def timing_clear():
    timing_dict_aggregation.clear()
    timing_dict.clear()
    timing_meta_dict.clear()


def timing_start(tag="", name="", enabled=True, meta=None):
    if not enabled:
        return
    id = tag + name
    timing_dict[id] = time.time()
    timing_meta_dict[id] = {"tag": tag, "name": name, "meta": meta}


def timing_end(tag="", name="", enabled=True, collect=True, print_enabled=False):
    if not enabled:
        return

    id = tag + name

    if collect:
        timing_dict_aggregation[id] = (
            timing_dict_aggregation.get(id, 0) + time.time() - timing_dict[id]
        )
        return timing_dict_aggregation[id]
    else:
        t = time.time() - timing_dict[id]
        if print_enabled:
            timing_print(id, t)

        del timing_dict[id]
        del timing_meta_dict[id]

        if id in timing_dict_aggregation:
            del timing_dict_aggregation[id]

        return t


def timing_print(id, t):
    tag, name, meta = (
        timing_meta_dict[id]["tag"],
        timing_meta_dict[id]["name"],
        timing_meta_dict[id]["meta"],
    )
    print(
        "timing | "
        + (f"[{tag}] " if tag else "")
        + f"{name} {t:>3.9f}s"
        + (f" | {meta}" if meta is not None else "")
    )


def lambda_time(f, name="", tag="", enabled=True, collect=False, meta=None):
    timing_start(tag, name, enabled, meta)
    r = f()
    timing_end(tag, name, enabled, collect)
    return r


def timing_collect(name="", tag="", enabled=True, print_enabled=False):
    id = tag + name

    if not enabled or id not in timing_dict_aggregation:
        return
    if print_enabled:
        timing_print(id, timing_dict_aggregation[id])

    t = timing_dict_aggregation[id]

    timing_dict_aggregation[id] = 0

    return t


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _save_stratified_splits(
    _splitter: StratifiedKFold | RepeatedStratifiedKFold,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    auto_fix_stratified_splits: bool = False,
) -> list[list[list[int], list[int]]]:
    """Fix from AutoGluon to avoid unsafe splits for classification if less than n_splits instances exist for all classes.

    https://github.com/autogluon/autogluon/blob/0ab001a1193869a88f7af846723d23245781a1ac/core/src/autogluon/core/utils/utils.py#L70.
    """
    try:
        splits = [
            [train_index, test_index]
            for train_index, test_index in _splitter.split(x, y)
        ]
    except ValueError as e:
        x = pd.DataFrame(x)
        y = pd.Series(y)
        y_dummy = pd.concat([y, pd.Series([-1] * n_splits)], ignore_index=True)
        X_dummy = pd.concat([x, x.head(n_splits)], ignore_index=True)
        invalid_index = set(y_dummy.tail(n_splits).index)
        splits = [
            [train_index, test_index]
            for train_index, test_index in _splitter.split(X_dummy, y_dummy)
        ]
        len_out = len(splits)
        for i in range(len_out):
            train_index, test_index = splits[i]
            splits[i][0] = [
                index for index in train_index if index not in invalid_index
            ]
            splits[i][1] = [index for index in test_index if index not in invalid_index]

        # only rais afterward because only now we know that we cannot fix it
        if not auto_fix_stratified_splits:
            raise AssertionError(
                "Cannot split data in a stratifed way with each class in each subset of the data."
            ) from e
    except UserWarning as e:
        # Assume UserWarning for not enough classes for correct stratified splitting.
        raise e

    return splits


def fix_split_by_dropping_classes(
    x: np.ndarray, y: np.ndarray, n_splits: int, spliter_kwargs: dict
) -> list[list[list[int], list[int]]]:
    """Fixes stratifed splits for edge case.

    For each class that has fewer instances than number of splits, we oversample before split to n_splits and then remove all oversamples and
    original samples from the splits; effectively removing the class from the data without touching the indices.
    """
    val, counts = np.unique(y, return_counts=True)
    too_low = val[counts < n_splits]
    too_low_counts = counts[counts < n_splits]

    y_dummy = pd.Series(y.copy())
    X_dummy = pd.DataFrame(x.copy())
    org_index_max = len(X_dummy)
    invalid_index = []

    for c_val, c_count in zip(too_low, too_low_counts, strict=True):
        fill_missing = n_splits - c_count
        invalid_index.extend(np.where(y == c_val)[0])
        y_dummy = pd.concat(
            [y_dummy, pd.Series([c_val] * fill_missing)], ignore_index=True
        )
        X_dummy = pd.concat(
            [X_dummy, pd.DataFrame(x).head(fill_missing)], ignore_index=True
        )

    invalid_index.extend(list(range(org_index_max, len(y_dummy))))
    splits = _save_stratified_splits(
        _splitter=StratifiedKFold(**spliter_kwargs),
        x=X_dummy,
        y=y_dummy,
        n_splits=n_splits,
    )
    len_out = len(splits)
    for i in range(len_out):
        train_index, test_index = splits[i]
        splits[i][0] = [index for index in train_index if index not in invalid_index]
        splits[i][1] = [index for index in test_index if index not in invalid_index]

    return splits


def assert_valid_splits(
    splits: list[list[list[int], list[int]]],
    y: np.ndarray,
    *,
    non_empty: bool = True,
    each_selected_class_in_each_split_subset: bool = True,
    same_length_training_splits: bool = True,
):
    """Verify that the splits are valid."""
    if non_empty:
        assert len(splits) != 0, "No splits generated!"
        for split in splits:
            assert len(split) != 0, "Some split is empty!"
            assert len(split[0]) != 0, "A train subset of a split is empty!"
            assert len(split[1]) != 0, "A test subset of a split is empty!"

    if each_selected_class_in_each_split_subset:
        # As we might drop classes, we first need to build the set of classes that are in the splits.
        #   - 2nd unique is for speed up purposes only.
        _real_y = set(
            np.unique([c for split in splits for c in np.unique(y[split[1]])])
        )
        # Now we need to check that each class that exists in all splits is in each split.
        for split in splits:
            assert _real_y == (
                set(np.unique(y[split[0]]))
            ), "A class is missing in a train subset!"
            assert _real_y == (
                set(np.unique(y[split[1]]))
            ), "A class is missing in a test subset!"

    if same_length_training_splits:
        for split in splits:
            assert len(split[0]) == len(
                splits[0][0]
            ), "A train split has different amount of samples!"


def _equalize_training_splits(
    input_splits: list[list[list[int], list[int]]], rng: np.random.RandomState
) -> list[list[list[int], list[int]]]:
    """Equalize training splits by duplicating samples in too small splits."""
    splits = input_splits[:]
    n_max_train_samples = max(len(split[0]) for split in splits)
    for split in splits:
        curr_split_len = len(split[0])
        if curr_split_len < n_max_train_samples:
            missing_samples = n_max_train_samples - curr_split_len
            split[0].extend(
                [int(dup_i) for dup_i in rng.choice(split[0], size=missing_samples)]
            )
            split[0] = sorted(split[0])

    return splits


def get_cv_split_for_data(
    x: np.ndarray,
    y: np.ndarray,
    splits_seed: int,
    n_splits: int,
    *,
    stratified_split: bool,
    safety_shuffle: bool = True,
    auto_fix_stratified_splits: bool = False,
    force_same_length_training_splits: bool = False,
) -> list[list[list[int], list[int]]] | str:
    """Safety shuffle and generate (safe) splits.

    If it returns str at the first entry, no valid split could be generated and the str is the reason why.
    Due to the safety shuffle, the original x and y are also returned and must be used.

    Note: the function does not support repeated splits at this point.
    Simply call this function multiple times with different seeds to get repeated splits.

    Test with:

    ```python
        if __name__ == "__main__":
        print(
            get_cv_split_for_data(
                x=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T,
                y=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]),
                splits_seed=42,
                n_splits=3,
                stratified_split=True,
                auto_fix_stratified_splits=True,
            )
        )
    ```

    Args:
        x: The data to split.
        y: The labels to split.
        splits_seed: The seed to use for the splits. Or a RandomState object.
        n_splits: The number of splits to generate.
        stratified_split: Whether to use stratified splits.
        safety_shuffle: Whether to shuffle the data before splitting.
        auto_fix_stratified_splits: Whether to try to fix stratified splits automatically.
            Fix by dropping classes with less than n_splits samples.
        force_same_length_training_splits: Whether to force the training splits to have the same amount of samples.
            Force by duplicating random instance in the training subset of a too small split until all training splits have the same amount of samples.
    Out:
        A list of pairs of indexes, where in each pair first come the train examples, then test. So we get something like
        `[[TRAIN_INDICES_0, TEST_INDICES_0], [TRAIN_INDICES_1, TRAIN_INDICES_1]]` for 2 splits.
        Or a string if no valid split could be generated whereby the string gives the reason.
    """
    assert len(x) == len(y), "x and y must have the same length!"

    rng = np.random.RandomState(splits_seed)
    if safety_shuffle:
        p = rng.permutation(len(x))
        x, y = x[p], y[p]
    spliter_kwargs = {"n_splits": n_splits, "shuffle": True, "random_state": rng}

    if not stratified_split:
        splits = [list(tpl) for tpl in KFold(**spliter_kwargs).split(x, y)]
    else:
        warnings.filterwarnings("error")
        try:
            splits = _save_stratified_splits(
                _splitter=StratifiedKFold(**spliter_kwargs),
                x=x,
                y=y,
                n_splits=n_splits,
                auto_fix_stratified_splits=auto_fix_stratified_splits,
            )
        except UserWarning as e:
            logger.debug(e)
            if auto_fix_stratified_splits:
                logger.debug("Trying to fix stratified splits automatically...")
                splits = fix_split_by_dropping_classes(
                    x=x, y=y, n_splits=n_splits, spliter_kwargs=spliter_kwargs
                )
            else:
                splits = "Cannot generate valid stratified splits for dataset without losing classes in some subsets!"
        except AssertionError as e:
            logger.debug(e)
            splits = "Cannot generate valid stratified splits for dataset without losing classes in some subsets!"

        warnings.resetwarnings()

    if isinstance(splits, str):
        return splits

    if force_same_length_training_splits:
        splits = _equalize_training_splits(splits, rng)

    assert_valid_splits(
        splits=splits,
        y=y,
        non_empty=True,
        same_length_training_splits=force_same_length_training_splits,
        each_selected_class_in_each_split_subset=stratified_split,
    )

    if safety_shuffle:
        # Revert to correct outer scope indices
        for split in splits:
            split[0] = sorted(p[split[0]])
            split[1] = sorted(p[split[1]])

    return splits


def get_submodule_from_statedict(state_dict: dict, submodule: str):
    return {
        k[len(submodule) + 1 :]: v
        for k, v in state_dict.items()
        if k.startswith(submodule + ".")
    }


def set_submodule_statedict(
    state_dict: dict, submodule: str, submodule_state_dict: dict
):
    for k in list(state_dict.keys()):
        if k.startswith(submodule + "."):
            del state_dict[k]
    for k, v in submodule_state_dict.items():
        state_dict[submodule + "." + k] = v


def apply_to_nested_structure(structure, func):
    if isinstance(structure, dict):
        return {k: apply_to_nested_structure(v, func) for k, v in structure.items()}
    elif isinstance(structure, (list, tuple)):
        return [apply_to_nested_structure(v, func) for v in structure]
    else:
        return func(structure)


def default_task_settings(task_type=None):
    max_samples = 10000
    max_features = 500
    max_times = [1, 5, 15, 30, 60, 60 * 5, 60 * 15, 60 * 60]
    max_classes = 10

    return max_samples, max_features, max_times, max_classes


def target_is_multiclass(task_type: str) -> bool:
    return task_type in {"multiclass", "dist_shift_multiclass"}


def target_is_continuous(task_type: str) -> bool:
    return not target_is_multiclass(task_type)
