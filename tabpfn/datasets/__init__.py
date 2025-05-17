from __future__ import annotations

import copy
import pandas as pd
import tqdm
import os

import math
import random
from .dist_shift_datasets import (
    get_datasets_dist_shift_multiclass_test,
    get_datasets_dist_shift_multiclass_valid,
)
import torch
import numpy as np
import openml
import warnings
from scipy.sparse._csr import csr_matrix
from typing import Optional, Literal, List
from .benchmark_dids import (
    automl_dids_classification,
    openml_cc_18_classification,
    valid_dids_classification,
    tabpfn_dids_classification,
    automl_dids_regression,
    openml_ctr23_regression,
    valid_dids_regression,
    automl_dids_regression,
    grinzstjan_numerical_regression,
    grinzstjan_categorical_regression,
    grinzstjan_numerical_regression_without_automl_overlap,
    grinzstjan_categorical_regression_without_automl_overlap,
    grinzstjan_categorical_classification_without_automl_overlap,
    grinzstjan_numerical_classification_without_automl_overlap,
    grinzstjan_numerical_classification,
    grinzstjan_categorical_classification,
)
from tabpfn.utils import default_task_settings
from tabpfn.local_settings import openml_path
from tabpfn.utils import target_is_multiclass, target_is_continuous


class DatasetModifications:
    def __init__(self, classes_capped: bool, feats_capped: bool, samples_capped: bool):
        """
        DEPRECATED, ONLY HERE TO KEEP PICKLE COMPATIBILITY
        :param classes_capped: Whether the number of classes was capped
        :param feats_capped: Whether the number of features was capped
        :param samples_capped: Whether the number of samples was capped
        """
        self.classes_capped = classes_capped
        self.feats_capped = feats_capped
        self.samples_capped = samples_capped


class TabularDataset:
    def __init__(
        self,
        name: str,
        x: torch.tensor,
        y: torch.tensor,
        task_type: str,
        attribute_names: list[str],
        categorical_feats: Optional[list[int]] = None,
        modifications: Optional[list[str]] = None,
        splits: Optional[list[tuple[torch.tensor, torch.tensor]]] = None,
        benchmark_name: Optional[str] = None,
        extra_info: Optional[dict] = None,
        description: Optional[str] = None,
    ):
        """
        :param name: Name of the dataset
        :param x: The data matrix
        :param y: The labels
        :param categorical_feats: A list of indices of categorical features
        :param attribute_names: A list of attribute names
        :param modifications: A DatasetModifications object
        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices)
        """
        if categorical_feats is None:
            categorical_feats = []

        self.name = name
        self.x = x
        self.y = y
        self.categorical_feats = categorical_feats
        self.attribute_names = attribute_names
        self.modifications = modifications if modifications is not None else []
        self.splits = splits
        self.task_type = task_type
        self.benchmark_name = benchmark_name
        self.extra_info = extra_info
        self.description = description
        self.cov_cache = {}
        self.test_portions = {"main"}

        if target_is_multiclass(self.task_type):
            from tabpfn.model.encoders import MulticlassClassificationTargetEncoder

            self.y = MulticlassClassificationTargetEncoder.flatten_targets(self.y)

    def __setstate__(self, state):
        self.__dict__ = copy.deepcopy(state)
        if isinstance(state["modifications"], DatasetModifications):
            self.modifications = []
            if state["modifications"].classes_capped:
                self.modifications.append("classes_capped")
            if state["modifications"].feats_capped:
                self.modifications.append("feats_capped")
            if state["modifications"].samples_capped:
                self.modifications.append("samples_capped")

    def get_dataset_identifier(self, with_identifier: bool = True) -> str:
        if self.extra_info is None or (
            "openml_tid" not in self.extra_info and "openml_did" not in self.extra_info
        ):
            return self.name

        tid = self.extra_info.get("openml_tid", "notask")
        did = self.extra_info.get("openml_did", None)

        if did is None:
            did = self.extra_info.get("did", "nodataset")

        modification_str = "__".join(sorted(self.modifications))
        if modification_str:
            modification_str = f"__{modification_str}"
        if not with_identifier:
            modification_str = ""
        return f"{did}_{tid}" + modification_str

    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(self.x.numpy(), columns=self.attribute_names)
        df = df.astype({name: "category" for name in self.categorical_names})
        df.loc[:, "target"] = self.y.numpy()
        return df

    @property
    def categorical_names(self) -> list[str]:
        return [self.attribute_names[i] for i in self.categorical_feats]

    def infer_and_set_categoricals(self) -> None:
        """
        Infers and sets categorical features from the data and sets the categorical_feats attribute. Don't use this
        method if the categorical indicators are already known from a predefined source. This method is used to infer
        categorical features from the data itself and only an approximation.
        """
        dummy_df = pd.DataFrame(self.x.numpy(), columns=self.attribute_names)
        encoded_with_categoricals = infer_categoricals(
            dummy_df,
            max_unique_values=20,
            max_percentage_of_all_values=0.1,
        )

        categorical_idxs = [
            i
            for i, dtype in enumerate(encoded_with_categoricals.dtypes)
            if dtype == "category"
        ]
        self.categorical_feats = categorical_idxs

    def __getitem__(self, indices):
        # convert a simple index x[y] to a tuple for consistency
        # if not isinstance(indices, tuple):
        #    indices = tuple(indices)
        ds = copy.deepcopy(self)
        ds.x = ds.x[indices]
        ds.y = ds.y[indices]

        return ds

    @staticmethod
    def check_is_valid_split(task_type, ds, index_train, index_test):
        if target_is_continuous(task_type):
            return True

        # Checks if the set of classes are the same in dataset and its subsets
        if set(torch.unique(ds.y[index_train]).tolist()) != set(
            torch.unique(ds.y).tolist()
        ):
            return False
        if set(torch.unique(ds.y[index_test]).tolist()) != set(
            torch.unique(ds.y).tolist()
        ):
            return False

        return True

    def generate_valid_split(
        self,
        n_splits: int | None = None,
        splits: list[list[list[int], list[int]]] | None = None,
        split_number: int = 1,
        auto_fix_stratified_splits: bool = False,
    ) -> tuple[TabularDataset, TabularDataset] | tuple[None, None]:
        """Generates a deterministic train-(test/valid) split.

        Both splits must contain the same classes and all classes in the entire datasets.
        If no such split can be sampled, returns None.

        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices) or None. If None, we generate the splits.
        :param n_splits: The number of splits to generate. Only required if splits is None.
        :param split_number: The split id. n_splits are coming from the same split and are disjoint. Further splits are
            generated by changing the seed. Only used if splits is None.
        :param auto_fix_stratified_splits: If True, we try to fix the splits if they are not valid. Only used if splits is None.

        :return: the train and test split in format of TabularDataset or None, None if no valid split could be generated.
        """
        if split_number == 0:
            raise ValueError("Split number 0 is not used, we index starting from 1.")
        # We are using split numbers from 1 to 5 to legacy reasons
        split_number = split_number - 1

        if splits is None:
            if n_splits is None:
                raise ValueError("If `splits` is None, `n_splits` must be set.")
            # lazy import as not needed elsewhere.
            from tabpfn.utils import get_cv_split_for_data

            # assume torch tensor as nothing else possible according to typing.
            x = self.x if isinstance(self.x, np.ndarray) else self.x.numpy()
            y = self.y if isinstance(self.y, np.ndarray) else self.y.numpy()

            splits = get_cv_split_for_data(
                x=x,
                y=y,
                n_splits=n_splits,
                splits_seed=(split_number // n_splits)
                + 1,  # deterministic for all splits from one seed/split due to using //
                stratified_split=target_is_multiclass(self.task_type),
                safety_shuffle=False,  # if ture, shuffle in the split function, and you have to update x and y
                auto_fix_stratified_splits=auto_fix_stratified_splits,
            )
            if isinstance(splits, str):
                print(f"Valid split could not be generated {self.name} due to {splits}")
                return None, None

            split_number_parsed = split_number % n_splits
            train_inds, test_inds = splits[split_number_parsed]
            train_ds = self[train_inds]
            test_ds = self[test_inds]
        else:
            train_inds, test_inds = splits[split_number]
            train_ds = self[train_inds]
            test_ds = self[test_inds]

        return train_ds, test_ds

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},id={self.get_dataset_identifier()})[num_samples={len(self.x)}, num_features={self.x.shape[1]}]"

    def get_duplicated_samples_(self, data, var_thresh=0.999999):
        from tabpfn.utils import normalize_data

        key = float(data.nansum() + var_thresh)
        if key in self.cov_cache:
            return self.cov_cache[key]

        data = data.to("cpu")
        data[torch.isnan(data)] = 0.0

        x = normalize_data(data.transpose(1, 0))
        cov_mat = torch.cov(x.transpose(1, 0))
        cov_mat = torch.logical_or(cov_mat == 1.0, cov_mat > var_thresh).float()
        duplicated_samples = cov_mat.sum(axis=0)

        self.cov_cache[key] = (
            (duplicated_samples, cov_mat),
            (duplicated_samples > 0).float().mean().item(),
        )

        return self.cov_cache[key]

    def get_duplicated_samples(self, features_only=False, var_thresh=0.999999):
        """
        Calculates duplicated samples based on the covariance matrix of the data.

        :param features_only: Whether to only consider the features for the calculation
        :param var_thresh: The threshold for the variance to be considered a duplicate

        :return: Tuple of ((covariance matrix, duplicated_samples indices), fraction_duplicated_samples)
        """
        if features_only:
            data = self.x.clone()
        else:
            data = torch.cat([self.x, self.y.unsqueeze(1)], dim=1)

        return self.get_duplicated_samples_(data, var_thresh=var_thresh)


def get_openml_dataset(
    did: int | str, max_samples: int
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str], str]:
    """
    :param did: The dataset id
    :param max_samples: The maximum number of samples to return
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = openml.datasets.get_dataset(did)

    target = dataset.default_target_attribute
    description = dataset.description
    assert isinstance(target, str)

    if "," in target:
        # TODO: Should really make this an error and skip...
        target = target.split(",")[0]
        print(
            f"Using only the first target for {dataset.name, dataset.dataset_id},"
            f" namely {target}"
        )

    with warnings.catch_warnings():
        # TODO: Using `"array"` will error in the next version of OpenML.
        # Trying to naively fix this and set `dataset_format="dataframe"`
        # will result in a different benchmark checksum.
        warnings.simplefilter("ignore")
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="array",
            target=target,
        )

    X = X.toarray() if isinstance(X, csr_matrix) else X
    y = y.toarray() if isinstance(y, csr_matrix) else y

    msg = (
        f"Dataset {dataset.name} did not return np arrays for X and/or y"
        f" but {type(X)} and {type(y)}"
    )
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), msg

    X, y = torch.tensor(X), torch.tensor(y)

    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names, description


def cap_dataset(
    X,
    y,
    categorical_feats,
    is_regression,
    num_feats,
    max_num_classes,
    min_samples,
    max_samples,
    num_cells,
    return_capped=False,
    min_targets_regression=10,
):
    modifications = {
        "samples_capped": False,
        "classes_capped": False,
        "feats_capped": False,
    }

    def stratified_sub_sample(x, y, new_num_samples):
        if is_regression:
            return x[:new_num_samples], y[:new_num_samples]
        else:
            # while there is a stratified sampling function in sklearn, it does not work if one of the two subsets (train/test) gets too small
            inds_per_class = [np.where(y == c)[0] for c in np.unique(y)]
            randomness = np.random.RandomState(42)
            inds = np.concatenate(
                [
                    randomness.choice(
                        inds, round(len(inds) / len(y) * new_num_samples), replace=False
                    )
                    for inds in inds_per_class
                ]
            )
            return x[inds], y[inds]

    if (
        not is_regression
        and max_num_classes is not None
        and len(np.unique(y)) > max_num_classes
    ):
        if return_capped:
            classes, counts = np.unique(y, return_counts=True)
            sorted_classes = classes[np.argsort(-counts)]
            mask = np.isin(y, sorted_classes[:max_num_classes])
            X = X[mask]
            y = y[mask]
            modifications["classes_capped"] = True
        else:
            raise ValueError("Too many classes")

    if num_feats is not None and X.shape[1] > num_feats:
        if return_capped:
            X = X[:, 0:num_feats]
            categorical_feats = [c for c in categorical_feats if c < num_feats]
            modifications["feats_capped"] = True
        else:
            raise ValueError(f"Too many features! ({X.shape} vs {num_feats}), skipping")
    elif X.shape[1] == 0:
        raise ValueError(f"No features! ({X.shape}), skipping")

    if max_samples is not None and X.shape[0] > max_samples:
        if return_capped:
            modifications["samples_capped"] = True
            X, y = stratified_sub_sample(X, y, max_samples)
        else:
            raise ValueError("Too many samples")

    if num_cells is not None and X.shape[0] * X.shape[1] >= num_cells:
        if return_capped:
            modifications["samples_capped"] = True
            X, y = stratified_sub_sample(X, y, num_cells // X.shape[1])
        else:
            raise ValueError("Too many cells")

    if X.shape[0] < min_samples:
        raise ValueError("Too few samples left")

    if is_regression and len(np.unique(y)) < min_targets_regression:
        raise ValueError("Too few targets for regression")

    return X, y, categorical_feats, modifications


def load_openml_list(
    dids: list[int | str],
    filter_for_nan: bool = False,
    num_feats: Optional[int] = None,
    min_samples: int = 50,
    max_samples: Optional[int] = None,
    max_num_classes: Optional[int] = None,
    max_num_cells: Optional[int] = None,
    min_targets_regression: int = 10,
    return_capped: bool = False,
    load_data: bool = True,
    return_as_lists: bool = True,
    benchmark_name: str = "",
    n_max: int = -1,
    load_dummy_data: bool = False,
):
    datasets = []
    tids, dids = zip(
        *[did.split("@") if type(did) == str else (None, did) for did in dids]
    )

    # Reload file information from disk cache if possible, this speeds up loading significantly
    path = os.path.join(
        openml_path, "cached_lists", f"{np.sum(np.array(dids).astype(float))}.csv"
    )

    if not os.path.isfile(path):
        openml_list = openml.datasets.list_datasets(dids)
        datalist = pd.DataFrame.from_dict(openml_list, orient="index")
        datalist.reset_index(drop=True).to_csv(path, index=False)
    else:
        datalist = pd.read_csv(path)
        datalist = datalist.set_index("did", drop=False)
        datalist.index.name = None

    print(f"Number of datasets: {len(datalist)}")

    if filter_for_nan:
        datalist = datalist[datalist["NumberOfInstancesWithMissingValues"] == 0]
        print(f"No. of datasets after Nan & feature number filtering: {len(datalist)}")

    if not load_data:
        return [[] for _ in range(len(datalist))], datalist

    do_load_dummy_data = load_dummy_data and all(
        c is None for c in [max_num_classes, max_samples, max_num_cells, num_feats]
    )
    if do_load_dummy_data:
        print("Skip loading data and only return dummy TabularDataset object.")

    keep_idx = []
    for i in (pbar := tqdm.tqdm(list(range(len(dids))))):
        if n_max > 0 and len(datasets) >= n_max:
            break
        entry = datalist.loc[int(dids[i])]
        pbar.set_description(f"Loading {entry['name']} (openml id: {entry.did})")

        if not return_capped and (
            (max_samples is not None and entry.NumberOfInstances > max_samples)
            or (num_feats is not None and entry.NumberOfFeatures > num_feats)
        ):
            # print("Skipping: too many features or samples")
            continue

        splits = None
        is_regression = entry["NumberOfClasses"] == 0.0 or np.isnan(
            entry["NumberOfClasses"]
        )

        if do_load_dummy_data:
            # Avoid loading the data if not necessary
            # Only works for no constraints as filter is applied only after loading.
            # Otherwise, the dataset list would change and the current way of referencing the dataset by order would be wrong!
            datasets += [
                TabularDataset(
                    name=entry["name"],
                    x=None,
                    y=torch.tensor([0, 1]),
                    attribute_names=["dummy"],
                    splits=splits,
                    task_type="regression" if is_regression else "multiclass",
                    benchmark_name=benchmark_name,
                    extra_info={"openml_did": int(dids[i]), "openml_tid": tids[i]},
                )
            ]
        else:
            if tids[i] is not None and not return_capped:  # If task id is provided
                splits = []

                # Used to ignore deprecation warning we can't circumvent
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    task = openml.tasks.get_task(int(tids[i]), download_splits=True)

                    for fold in range(0, 10):
                        train_indices, test_indices = task.get_train_test_split_indices(
                            repeat=0,
                            fold=fold,
                            sample=0,
                        )
                        splits += [(train_indices, test_indices)]
            try:
                (
                    X,
                    y,
                    categorical_feats,
                    attribute_names,
                    description,
                ) = get_openml_dataset(int(entry.did), None)
                X, y, categorical_feats, modifications = cap_dataset(
                    X,
                    y,
                    categorical_feats,
                    is_regression,
                    num_feats,
                    max_num_classes,
                    min_samples,
                    max_samples,
                    max_num_cells,
                    min_targets_regression=min_targets_regression,
                    return_capped=return_capped,
                )
            except ValueError as e:
                print(f'Skipping {entry["name"]} due to {e}')
                continue

            if return_as_lists:
                datasets += [
                    [
                        entry["name"],
                        X,
                        y,
                        categorical_feats,
                        attribute_names,
                        modifications,
                        splits,
                    ]
                ]
            else:
                modifications = [
                    mod for mod, applied in modifications.items() if applied
                ]
                datasets += [
                    TabularDataset(
                        name=entry["name"],
                        x=X,
                        y=y,
                        categorical_feats=categorical_feats,
                        attribute_names=attribute_names,
                        modifications=modifications,
                        splits=splits,
                        task_type="regression" if is_regression else "multiclass",
                        benchmark_name=benchmark_name,
                        extra_info={"openml_did": int(dids[i]), "openml_tid": tids[i]},
                        description=description,
                    )
                ]
        keep_idx += [i]

    return datasets, datalist.iloc[keep_idx]


benchmarks = {
    ("multiclass", "test"): [
        # (tabpfn_dids_classification, "tabpfn"),
        (automl_dids_classification, "automl"),
        # (openml_cc_18_classification, "openml_cc18"),
        # (grinzstjan_categorical_classification, "grinzstjan_categorical"),
        # (grinzstjan_numerical_classification, "grinzstjan_numerical"),
    ],
    ("multiclass", "test_extended"): [
        (tabpfn_dids_classification, "tabpfn"),
        (automl_dids_classification, "automl"),
        (openml_cc_18_classification, "openml_cc18"),
        (grinzstjan_categorical_classification, "grinzstjan_categorical"),
        (grinzstjan_numerical_classification, "grinzstjan_numerical"),
    ],
    ("multiclass", "grinzstjan"): [
        (grinzstjan_categorical_classification, "grinzstjan_categorical"),
        (grinzstjan_numerical_classification, "grinzstjan_numerical"),
    ],
    ("multiclass", "valid"): [(valid_dids_classification, "valid")],
    ("multiclass", "valid_hard"): [
        (valid_dids_classification, "valid"),
        (
            grinzstjan_categorical_classification_without_automl_overlap,
            "grinzstjan_categorical",
        ),
        (
            grinzstjan_numerical_classification_without_automl_overlap,
            "grinzstjan_numerical",
        ),
    ],
    ("multiclass", "debug"): [
        (
            [
                459,  # binary
                468,  # multiclass
            ],
            "debug",
        )
    ],  # Tiny dataset from valid set
    ("regression", "test"): [
        (automl_dids_regression, "automl"),
        (openml_ctr23_regression, "openml_ctr23"),
        # (grinzstjan_categorical_regression, "grinzstjan_categorical"),
        # (grinzstjan_numerical_regression, "grinzstjan_numerical"),
    ],
    ("regression", "test_extended"): [
        (automl_dids_regression, "automl"),
        (openml_ctr23_regression, "openml_ctr23"),
        (grinzstjan_categorical_regression, "grinzstjan_categorical"),
        (grinzstjan_numerical_regression, "grinzstjan_numerical"),
    ],
    ("regression", "valid"): [(valid_dids_regression, "valid")],
    ("regression", "valid_hard"): [
        (valid_dids_regression, "valid"),
        (
            grinzstjan_categorical_regression_without_automl_overlap,
            "grinzstjan_categorical",
        ),
        (
            grinzstjan_numerical_regression_without_automl_overlap,
            "grinzstjan_numerical",
        ),
    ],
    ("regression", "debug"): [
        (
            [660],
            "debug",
        )
    ],  # Tiny dataset from valid set
    ("regression", "grinzstjan"): [
        (grinzstjan_categorical_regression, "grinzstjan_categorical"),
        (grinzstjan_numerical_regression, "grinzstjan_numerical"),
    ],
}
did_to_benchmark = {
    did: benchmark_name
    for task_type, split in benchmarks
    for dids, benchmark_name in benchmarks[(task_type, split)]
    for did in dids
}


def get_benchmark_dids_for_task(
    task_type: Literal["multiclass", "regression"],
    split: Literal["train", "valid", "debug", "test"] = "train",
) -> list[tuple[list[int | str], str]]:
    """Returns a list of datasets for a given task and split.

    :param task_type: The task type, either "multiclass" or "regression".
    :param split: The split, either "train", "valid", "debug" or "test".
    :return: A list of tuples, where the first element
        is a list of dataset ids and the second element is the benchmark name
    """
    if (retrieved := benchmarks.get((task_type, split))) is not None:  # type: ignore
        return retrieved

    raise ValueError(
        f"No benchmark found for {task_type=}, {split=}"
        f"\nChoose from: {benchmarks.keys()}"
    )


max_samples, max_features, max_times, max_classes = default_task_settings()


def get_benchmark_for_task(
    task_type: Literal[
        "regression",
        "multiclass",
        "quantile_regression",
        "dist_shift_multiclass",
    ],
    split: Literal["train", "valid", "debug", "test", "kaggle"] = "test",
    max_samples: Optional[int] = max_samples,
    max_features: Optional[int] = max_features,
    max_classes: Optional[int] = max_classes,
    max_num_cells: Optional[int] = None,
    min_samples: int = 50,
    min_targets_regression: int = 10,
    filter_for_nan: bool = False,
    return_capped: bool = False,
    return_as_lists: bool = True,
    n_max: int = 200,
    load_data: bool = True,
    load_dummy_data: bool = False,
) -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
    if task_type in {"regression", "multiclass", "quantile_regression"}:
        if task_type == "quantile_regression":
            task_type = "regression"
        benchmarks = get_benchmark_dids_for_task(task_type, split)
        openml_datasets, openml_datasets_df = [], []
        for dids, benchmark_name in benchmarks:
            openml_datasets_, openml_datasets_df_ = load_openml_list(
                dids,
                filter_for_nan=filter_for_nan,
                max_samples=max_samples,
                num_feats=max_features,
                return_capped=return_capped,
                max_num_classes=max_classes,
                max_num_cells=max_num_cells,
                return_as_lists=return_as_lists,
                benchmark_name=benchmark_name,
                min_samples=min_samples,
                load_data=load_data,
                n_max=n_max,
                load_dummy_data=load_dummy_data,
                min_targets_regression=min_targets_regression,
            )
            openml_datasets += openml_datasets_
            openml_datasets_df += [openml_datasets_df_]
        openml_datasets_df = pd.concat(openml_datasets_df)
    elif task_type == "dist_shift_multiclass":
        openml_datasets, openml_datasets_df = get_dist_shift_benchmark(
            split=split,
            min_samples=min_samples,
            max_samples=max_samples,
            num_feats=max_features,
        )

        for dataset in openml_datasets:
            dataset.benchmark_name = "dist_shift_set"

    else:
        raise NotImplementedError(f"Unknown task type {task_type}")

    return openml_datasets, openml_datasets_df


def print_benchmark_datasets(datasets, task_type, print_description=False):
    automl_datasets = [df.name for df in datasets if df.benchmark_name == "automl"]
    # Remove duplicates
    datasets = {
        df.name: df
        for df in datasets
        if df.name not in automl_datasets or df.benchmark_name == "automl"
    }
    datasets = list(datasets.values())

    df = pd.DataFrame(
        [
            {
                "Name": ds.name,
                "OpenML ID": ds.extra_info.get("openml_did", None),
                "\\# Features": ds.x.shape[1],
                "\\# Samples": ds.x.shape[0],
                "\\# Targets": len(ds.y.unique()),
                "\\# Categorical Feats.": len(ds.categorical_feats),
                "\\# Numerical Feats.": (
                    len(ds.attribute_names) - len(ds.categorical_feats)
                ),
                "Benchmark": ds.benchmark_name,
                "Description": ds.description if hasattr(ds, "description") else "",
            }
            for ds in datasets
        ]
    )
    df.sort_values("\\# Samples")
    if not print_description:
        df = df.drop(columns=["Description"])

    if task_type == "dist_shift_multiclass":
        df["\\# Domains"] = [
            torch.unique(ds.dist_shift_domain).shape[0] for ds in datasets
        ]

    return df


def get_dist_shift_benchmark(
    split, min_samples, max_samples, num_feats, append_domain=False
):
    if split == "test":
        datasets = get_datasets_dist_shift_multiclass_test()
    elif split == "valid":
        datasets = get_datasets_dist_shift_multiclass_valid()
    elif split == "debug":
        from .dist_shift_datasets import (
            get_rotated_blobs,
        )

        datasets = [
            get_rotated_blobs(
                num_domains=10,
                num_samples_per_blob=40,
                num_blobs=5,
                rotation_sampler=lambda domain: domain * np.deg2rad(-20),
                noise_standard_dev=4.5,
                radius=25,
                name="Rotated Five Blobs - -20 deg",
                random_state=0,
                center=(-5, 30),
            ),
        ]
    else:
        raise ValueError(f"Unknown split {split}.")

    datasets = rename_duplicated_datasets(datasets)

    datasets = [
        dataset
        for dataset in datasets
        if dataset.x.shape[0] >= min_samples
        and dataset.x.shape[0] <= max_samples
        and dataset.x.shape[1] <= num_feats
    ]

    if append_domain:
        for dataset in datasets:
            dataset.append_domain()

    return datasets, None


class DistributionShiftDataset(TabularDataset):
    def __init__(
        self,
        dist_shift_domain: torch.tensor,
        task_type: str,
        dataset_source: Literal["synthetic", "real-world"],
        **kwargs,
    ):
        assert (
            task_type == "dist_shift_multiclass" or task_type == "dist_shift_regression"
        ), "The task type for this dataset is not for distribution shifts."

        super().__init__(task_type=task_type, **kwargs)

        self.dataset_source = dataset_source
        self.dist_shift_domain = dist_shift_domain

        self.test_portions = {"id", "ood"}

        # Determine whether the domain has already been concatenated or not
        self.concatenated_domain = False

    def __getitem__(self, indices):
        # convert a simple index x[y] to a tuple for consistency
        # if not isinstance(indices, tuple):
        #    indices = tuple(indices)
        ds = copy.deepcopy(self)
        ds.x = ds.x[indices]
        ds.y = ds.y[indices]
        ds.dist_shift_domain = ds.dist_shift_domain[indices]

        ds.modifications = None
        ds.splits = None

        return ds

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name},id={self.get_dataset_identifier()})[num_samples={len(self.x)}, num_features={self.x.shape[1]}, num_domains={torch.unique(self.dist_shift_domain).shape[0]}]"

    @staticmethod
    def _shuffle_indices_by_domain(dist_shift_domain):
        # Filter for unique values in the domain and their appearance in the domain vector.
        unique_values, value_indices = torch.unique(
            dist_shift_domain, sorted=True, return_inverse=True
        )
        # Create an empty vector of the same size.
        shuffled_indices = torch.empty_like(dist_shift_domain, dtype=torch.long)

        for value_index in range(len(unique_values)):
            # Find the indices in the original domain where the current domain value appears
            indices = (value_indices == value_index).nonzero(as_tuple=True)[0]
            # Shuffle the found indices and place them back into the shuffled_indices tensor
            shuffled_indices[indices] = indices[torch.randperm(len(indices))]

        return shuffled_indices

    def append_domain(self):
        assert (
            self.concatenated_domain == False
        ), "Domain already has been concatenated."

        # Since we concatenated the domain to the front, we need to shift the
        # indices of the categorical features.
        if self.categorical_feats and len(self.categorical_feats) > 0:
            self.categorical_feats = [i + 1 for i in self.categorical_feats]

        # Also add the attribute name to include the new domain feature
        self.attribute_names = ["dist_shift_domain"] + self.attribute_names

        self.x = DistributionShiftDataset.append_domain_to_x(
            self.x, self.dist_shift_domain
        )

        self.concatenated_domain = True

    @staticmethod
    def append_domain_to_x(x, dist_shift_domain):
        x = torch.cat([dist_shift_domain.unsqueeze(-1), x], dim=1)

        return x

    @staticmethod
    def extract_domain_from_x(x):
        # returns (x, dist_shift_domain)
        return x[:, 1:], x[:, 0]

    def plot(self, alpha=0.85, num_steps=None, eps=0.1, animate=False, filename=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
        from tabpfn.scripts.decision_boundary import get_plot_visuals
        from IPython.display import Image
        import tempfile

        assert self.x.shape[1] == 2, "Can only plot 2D data."

        domains = self.dist_shift_domain.unique()
        classes = np.unique(self.y)
        _, node_colors, markers = get_plot_visuals(len(classes))

        # Determine global limits for all plots
        min_feature1, max_feature1 = (
            self.x[:, 0].min().item(),
            self.x[:, 0].max().item(),
        )
        min_feature2, max_feature2 = (
            self.x[:, 1].min().item(),
            self.x[:, 1].max().item(),
        )
        min_glob = min(min_feature1, min_feature2) - eps
        max_glob = max(max_feature1, max_feature2) + eps

        def setup_axis(ax):
            """Setup plot axis properties"""
            ax.set_xlim(min_glob, max_glob)
            ax.set_ylim(min_glob, max_glob)
            ax.set_aspect("equal", "box")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(
                labelbottom=False, labelleft=False, labelright=False, labeltop=False
            )

        def plot_domain_data(ax, domain_idx):
            """Plot data for a specific domain"""
            cur_dom = domains[domain_idx]
            domain_mask = self.dist_shift_domain == cur_dom
            x_dom = self.x[domain_mask, :]

            for cls in classes:
                class_mask = self.y[domain_mask] == cls
                ax.scatter(
                    x_dom[class_mask, 0],
                    x_dom[class_mask, 1],
                    s=70,
                    marker=markers[cls],
                    label=f"Class {cls}",
                    alpha=alpha,
                    color=node_colors[cls],
                    edgecolor="k",
                )

        if animate:
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.set_dpi(300)
            fig.subplots_adjust(top=0.9, left=0.17)
            plt.rcParams.update({"font.size": 16})
            setup_axis(ax)

            def update(frame):
                ax.clear()
                setup_axis(ax)
                plot_domain_data(ax, frame)

                ax.set_title(
                    self.name + f" - Domain {domains[frame]}", fontsize=16, wrap=True
                )
                ax.set_xlabel("$x_1$", fontsize=16)
                ax.set_ylabel("$x_2$", fontsize=16)
                ax.tick_params(labelleft=True, labelbottom=True)
                # ax.legend(fontsize=14)

                return (ax,)

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(domains),
                interval=500,
                repeat=True,
                repeat_delay=2000,
            )

            file_name = (
                filename
                if filename is not None
                else tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
            )
            ani.save(file_name, writer="pillow", dpi=100)  # Save as GIF

            plt.close(fig)
            return Image(filename=file_name)  # Display the GIF in the notebook

        else:
            if num_steps is None:
                num_steps = len(domains)

            columns = int(np.ceil(np.sqrt(num_steps)))
            rows = int(np.ceil(num_steps / columns))
            fig, axs = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
            fig.set_dpi(300)
            fig.subplots_adjust(
                bottom=0.05, top=0.95, left=0.05, right=0.95, wspace=0.03, hspace=0.03
            )
            plt.rcParams.update({"font.size": 16})

            fig.suptitle(self.name, fontsize=20)
            for i, idx in enumerate(
                np.linspace(0, len(domains) - 1, num_steps).astype(int)
            ):
                ax = axs.flatten()[i]
                setup_axis(ax)
                plot_domain_data(ax, idx)

                # Place text inside the axis area, e.g., top-left corner
                ax.text(
                    0.025,
                    0.975,
                    f"Domain {domains[idx]}",
                    transform=ax.transAxes,
                    fontsize=16,
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
                )

                if i == 0:
                    ax.legend(fontsize=14)

                if i % columns == 0:
                    ax.set_ylabel("$x_2$", fontsize=16)
                    ax.tick_params(labelleft=True, labelsize=14)

                if i + columns >= num_steps:
                    ax.set_xlabel("$x_1$", fontsize=16)
                    ax.tick_params(labelbottom=True, labelsize=14)

            # Hide unused axes
            for j in range(i + 1, rows * columns):
                axs.flatten()[j].set_visible(False)

            plt.show()

    def generate_valid_split(
        self,
        splits=None,
        all_preceding_data=True,
        all_remaining_data=True,
        max_predict_domains=None,
        test_set_start_index=None,
        num_predict_domains=None,
        split_number=0,
        minimize_num_train_domains=False,
        previous_domain_splits_on_ds=[],
        shuffle_per_domain=False,
    ):
        """
        Generates a valid train-test split for a dataset by selecting unique domain values as
        splitting points, ensuring that all classes are represented in both splits.

        Thereby the following conditions are ensured:
            - We have seen at least 30% of the domains in the training set.
            - The set domain has size of at least 10% of the total number of domains.


        :param splits: A list of tuples containing predefined train and test index tensors. Default is None.
        :param all_preceding_data: If True, all data before the testing set will be included in the training set.
                                    Default is True.
        :param max_predict_domains: The maximum number of domains to include in the test set. Default is None.
        :param split_number: The index of the split to use if 'splits' is provided. Default is 0.
                                Is also used as a seed for the random number generator.

        :returns: A tuple containing the train and test datasets as separate Dataset objects.
        """

        assert torch.equal(
            self.dist_shift_domain, self.dist_shift_domain.sort(stable=True)[0]
        ), "Domains must be sorted."

        # In case we have a predefined split, we can just return the corresponding datasets
        if splits is not None:
            train_inds, id_inds, ood_inds = (
                splits[split_number][0],
                splits[split_number][1],
                splits[split_number][1],
            )

            train_ds = self[train_inds]
            id_ds = self[id_inds]
            ood_ds = self[ood_inds]

            return train_ds, {"ood": ood_ds, "id": id_ds}

        initial_seed = split_number * 100
        done = False
        seed = initial_seed

        unique_values, count = self.dist_shift_domain.unique_consecutive(
            return_counts=True
        )
        domain_boundaries = torch.cat((torch.tensor([0]), count.cumsum(dim=0)))

        # We want to see at least 30 % of the samples and domains in the training set
        smart_start_test_index = max(math.ceil(unique_values.shape[0] * 0.3), 2)
        for idx, value in enumerate(domain_boundaries):
            if value.item() >= self.x.shape[0] * 0.3:
                smart_start_test_index = max(idx, smart_start_test_index)
                break

        # We want to see at least 20 % of the samples and domains in the test set
        max_start_test_index = math.floor(
            unique_values.shape[0] * 0.8
        )  # max number of domains in train
        for idx, value in enumerate(domain_boundaries):
            if value.item() > self.x.shape[0] * 0.8:
                # if the current value is larger than 80 %
                max_start_test_index = min(idx - 1, max_start_test_index)
                break

        # Remove the splits that have already been used from the pool of possible splits
        start_test_index_value_pool = [
            i
            for i in range(smart_start_test_index, max_start_test_index + 1)
            if i not in set(previous_domain_splits_on_ds)
        ]

        # If there are no possible splits left, return None
        if test_set_start_index is None and not start_test_index_value_pool:
            return None, None

        # Iterate until we found a valid split
        while not done:
            if seed > initial_seed + 100:
                return (
                    None,
                    None,
                )  # No split could be generated in 100 passes, return None

            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # Shuffle the indices by domain
            if shuffle_per_domain:
                perm = self._shuffle_indices_by_domain(self.dist_shift_domain)
                ds = self[perm]
            else:
                ds = self

            # Choose a random domain out of the possible positions that defines where we split between train and test.
            # We exclude the first and last index to ensure that we have at least one domain in train and test.
            start_test_index = (
                random.choice(start_test_index_value_pool)
                if test_set_start_index is None
                else test_set_start_index
            )

            # Choose the start domain of the training set randomly in case all_preceding_data is False.
            # Otherwise include all domains before the evaluation position.
            start_train_index = (
                random.choice(range(0, start_test_index))
                if not all_preceding_data
                else 0
            )

            # Choose the end domain of the test set randomly in case all_remaining_data is False.
            # Otherwise include all domains after the evaluation position.
            if all_remaining_data:
                end_test_index = unique_values.shape[0] - 1
            else:
                # In case the number of domains to predict is not specified, choose a random number of domains.
                # Otherwise choose the number of domains to predict, capped by the number of domains left.
                if num_predict_domains is None:
                    end_test_index = random.choice(
                        range(start_test_index, len(unique_values))
                    )
                else:
                    end_test_index = min(
                        start_test_index + num_predict_domains - 1,
                        len(unique_values) - 1,
                    )

                # In case the maximum number of domains to predict is specified, cap the number of domains to predict.
                if (
                    max_predict_domains is not None
                    and end_test_index - start_test_index >= max_predict_domains
                ):
                    end_test_index = start_test_index + max_predict_domains - 1

            # Convert the domain indices to the corresponding indices in the domain vector
            start_train = domain_boundaries[start_train_index]
            eval_position = domain_boundaries[start_test_index]
            end_test = domain_boundaries[end_test_index + 1]

            # Check if the conditions for this split are met
            done = self._check_conditions(
                ds.y, ds.y[start_train:end_test], eval_position - start_train
            )

            if done:
                done = False
                iterations = 20

                while not done and iterations <= 20:
                    iterations += 1

                    id_indices_mask = []
                    for i, domain in enumerate(
                        unique_values[start_train_index:start_test_index]
                    ):
                        # 10% of each domain should be kept as an in-distribution validation set
                        id_perm = torch.randperm(count[i]) >= count[i] * 0.9
                        id_indices_mask += [id_perm]

                    id_indices_mask = torch.cat(id_indices_mask)

                    y = ds.y[start_train:eval_position]
                    train_y = y[~id_indices_mask]
                    id_y = y[id_indices_mask]

                    done = self._check_conditions(
                        y, torch.cat([train_y, id_y]), train_y.shape[0]
                    )

            # Increase the seed to get a different split in the next iteration
            seed += 1

            if target_is_multiclass(self.task_type):
                # Heuristic to adjust the smart_start_test_index in the correct direction
                if (
                    torch.unique(ds.y[:eval_position]).shape[0]
                    < torch.unique(ds.y[eval_position:]).shape[0]
                    and torch.unique(self.dist_shift_domain).shape[0]
                    > self.dist_shift_domain.shape[0] // 5
                ):
                    smart_start_test_index = start_test_index

        # Minimize the number of domains in the training set such that the conditions are still met
        # This has to be done after the initial split has been found. Otherwise, the test portion would be
        # different opposed to the original split where minimize_num_train_domains=False.
        if minimize_num_train_domains:
            done = False
            # The inital value is off by one in order to allow an initial subtraction at the beginning of the loop
            min_start_train_index = start_test_index

            while not done:
                min_start_train_index -= 1

                start_train = domain_boundaries[min_start_train_index]

                # Account for the fact that we dropped some domains in the minimization process
                delta_start_train_index = (
                    start_train - domain_boundaries[start_train_index]
                )
                reduced_id_indices_mask = id_indices_mask[delta_start_train_index:]

                # Check that the reduced train portion has the same classes as the test portion
                done = self._check_conditions(
                    ds.y, ds.y[start_train:end_test], eval_position - start_train
                )

                # Split y into the train and id valid portion
                y = ds.y[start_train:eval_position]
                train_y = y[~reduced_id_indices_mask]
                id_y = y[reduced_id_indices_mask]

                # Check that the train portion has the same classes as the id valid portion
                done = done and self._check_conditions(
                    y, torch.cat([train_y, id_y]), train_y.shape[0], 25
                )

            # print(f"Were able to reduce dataset {self.name} in split {split_number} to {start_test_index - min_start_train_index} domains in the train set.")

            id_indices_mask = reduced_id_indices_mask

        # Remap the target categories to be consecutive beginning at 0
        if target_is_multiclass(self.task_type):
            ds.y = (ds.y.unsqueeze(1) > torch.unique(ds.y).unsqueeze(0)).sum(axis=1)

        # Slice the final split dataset into train and test
        train_ds = ds[start_train:eval_position]
        ood_ds = ds[eval_position:end_test]

        # Split the train portion again into train and in distribution validation
        id_ds = train_ds[id_indices_mask]
        train_ds = train_ds[~id_indices_mask]

        previous_domain_splits_on_ds.append(start_test_index)

        # print(f"Dataset {self.name} - Train-part: {round(train_ds.x.shape[0]/self.x.shape[0],2)*100}%, "
        #      f"doms {round(start_test_index/unique_values.shape[0], 2)*100}%")

        return train_ds, {"ood": ood_ds, "id": id_ds}

    def _check_conditions(
        self, whole_y, subset_y, eval_position, min_samples_train=None
    ):
        min_samples = min_samples_train is None or eval_position >= min_samples_train

        if target_is_multiclass(self.task_type):
            all_classes_present = (
                torch.unique(subset_y).shape[0] == torch.unique(whole_y).shape[0]
            )
            all_classes_equal = (
                all_classes_present
                and torch.all(torch.unique(subset_y) == torch.unique(whole_y)).item()
            )

            train_test_classes_present = (
                torch.unique(subset_y[:eval_position]).shape[0]
                == torch.unique(subset_y[eval_position:]).shape[0]
            )
            train_test_classes_equal = (
                train_test_classes_present
                and torch.all(
                    torch.unique(subset_y[:eval_position])
                    == torch.unique(subset_y[eval_position:])
                ).item()
            )

            return min_samples and all_classes_equal and train_test_classes_equal
        else:
            return min_samples


def infer_categoricals(
    df: pd.DataFrame,
    max_unique_values: int = 9,
    max_percentage_of_all_values: float = 0.1,
) -> pd.DataFrame:
    """
    Infers categorical features from the data and sets the categorical_feats attribute.
    :param df: Pandas dataframe
    :param max_unique_values: Maximum number of unique values for a feature to be considered categorical
    :param max_percentage_of_all_values: Maximum percentage of all values for a feature to be considered categorical
    :return: Pandas dataframe with categorical features encoded as category dtype
    """
    for column in df.columns:
        unique_values = df[column].nunique()
        unique_percentage = unique_values / len(df)

        if (
            unique_values <= max_unique_values
            and unique_percentage < max_percentage_of_all_values
        ):
            df[column] = df[column].astype("category")

    return df


def subsample(tensor, label_value, fraction):
    if label_value is None:
        num_samples = int(fraction * len(tensor))
        indices = torch.randperm(len(tensor))[:num_samples]
        return indices

    # Split tensor indices based on the label value
    matching_indices = (tensor[:, 0] == label_value).nonzero().squeeze()
    nonmatching_indices = (tensor[:, 0] != label_value).nonzero().squeeze()

    # Calculate how many matching rows we need to achieve the desired fraction
    num_matching = int(fraction * (len(nonmatching_indices) / (1.0 - fraction)))

    # If we need more matching rows than we have, adjust the number of non-matching rows
    if num_matching > len(matching_indices):
        num_matching = len(matching_indices)
        num_nonmatching = int(num_matching / fraction - num_matching)

        # Randomly select num_nonmatching rows
        indices_nonmatching = torch.randperm(len(nonmatching_indices))[:num_nonmatching]
        nonmatching_indices = nonmatching_indices[indices_nonmatching]

    # Randomly select num_matching rows
    indices_matching = torch.randperm(len(matching_indices))[:num_matching]
    selected_matching_indices = matching_indices[indices_matching]

    # Concatenate selected_matching_indices with nonmatching_indices
    result_indices = torch.cat((selected_matching_indices, nonmatching_indices), 0)

    # Shuffle the result tensor to avoid having all matching indices at the top
    result_indices = result_indices[torch.randperm(len(result_indices))]

    return result_indices


def remove_duplicated_datasets(dataset_list: List[TabularDataset]):
    """
    Removes datasets with duplicated names from the list of datasets.

    :param dataset_list: List
    :return:
    """
    seen = {}
    unique_objects = []
    for ds in dataset_list:
        if ds.name not in seen:
            unique_objects.append(ds)
            seen[ds.name] = True
    return unique_objects


def rename_duplicated_datasets(dataset_list):
    """
    Renames datasets with duplicated names from the list of datasets
    to be unique. E.g. ['dataset', 'dataset'] => ['dataset', 'dataset1']

    :param dataset_list: List
    :return:
    """
    name_count = {}
    for dataset in dataset_list:
        if dataset.name in name_count:
            name_count[dataset.name] += 1
            dataset.name += str(name_count[dataset.name])
        else:
            name_count[dataset.name] = 0
    return dataset_list
