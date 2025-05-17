from __future__ import annotations
import copy
import time
import tqdm
import os
import numpy as np
import warnings
import datetime

from typing import Any, Literal, TYPE_CHECKING, Callable, Optional

from tabpfn.utils import print_once

from tabpfn import local_settings
from . import tabular_metrics
from .tabular_metrics import (
    calculate_score_per_method,
    get_main_eval_metric,
    get_standard_eval_metrics,
    is_imbalanced,
    check_metric_fits_task_type,
)
from tabpfn.utils import np_load_if_exists
from tabpfn.datasets import (
    remove_duplicated_datasets,
    DistributionShiftDataset,
)
from .tabular_evaluation_utils import (
    DatasetEvaluation,
    DatasetEvaluationCollection,
)
from .estimator import tabpfn_model_type_getters

N_SPLITS = 5

if TYPE_CHECKING:
    from tabpfn.datasets import TabularDataset
    from tabpfn.scripts.tabular_metrics import MetricDefinition


def evaluate_and_score(
    valid_datasets: list[TabularDataset],
    valid_metrics: Optional[dict] = None,
    metric_with_model: Callable[..., dict[str, DatasetEvaluation]] = None,
    metric_used: Optional[Callable] = None,
    split_name: str = "test",
    log_per_dataset_metrics: bool = False,
    log_per_split_metrics: bool = False,
    manual_dataset_groups: dict[str, list[str]] | None = None,
    num_splits: int = 5,
    evaluate_subsets: Literal[
        True, "all", "property_based_subsets", False
    ] = "property_based_subsets",
    max_time: int = 300,
    random_state: int = 0,
    save: bool = False,
    overwrite: bool = False,
    fetch_only: bool = False,
    base_path: str = local_settings.base_path,
    path_interfix: str = "",
    method_name: str = "test",
    rename_gpu_runs: bool = False,  # If True, we rename the gpu results to not overwrite the cpu results
    device: str = "cpu",
    eval_kwargs: dict = {},
):
    """
    This function evaluates a model on multiple datasets and calculates aggregate results over multiple splits. These results
    are then aggregated into an overall score for each metric across all datasets. Manual groups can also be defined to report
    the aggregated scores for them separately. The function can also log metrics per dataset and store the results. It returns the
    log message as well as all calculated scores.

    Example for the usage of manual dataset groups:
    ```python
    # Assume we have the following datasets represented by their names here
    valid_datasets = ['ds1', 'ds2', 'ds3']

    # Define manual dataset groups
    manual_dataset_groups = {
        'group1': ['ds1', 'ds2'],
        'group2': ['ds2', 'ds3']
    }
    ```

    :param valid_datasets: List of datasets to evaluate on
    :param valid_metrics: List of metrics to evaluate on
    :param metric_with_model: Model function, i.e. partial(transfromer_metric, model=model)
    :param metric_used: Metric function to use for evaluation
    :param split_name: Name of the split to evaluate on, this is only used for logging
    """

    assert metric_with_model is not None, "metric_with_model must be defined"

    if valid_metrics is None:
        valid_metrics = get_standard_eval_metrics(valid_datasets[0].task_type)

    if metric_used is None:
        metric_used = get_main_eval_metric(valid_datasets[0].task_type)

    # Get the tasks for the datasets
    if evaluate_subsets is True or evaluate_subsets == "all":
        tasks = get_benchmark_tasks(valid_datasets)
    elif evaluate_subsets == "property_based_subsets":
        tasks = get_benchmark_tasks_for_subset(valid_datasets)
    elif evaluate_subsets is False:
        tasks = {}
    else:
        raise ValueError(f"{evaluate_subsets} not allowed as evaluate_subsets keyword.")

    # Build a mapping from dataset name to its index in the list of datasets
    ds_name_to_index_mapping = {
        ds.get_dataset_identifier(): i for i, ds in enumerate(valid_datasets)
    }

    # Create a new dictionary that stores the defined dataset objects for each group
    manual_dataset_groups = (
        {} if manual_dataset_groups is None else manual_dataset_groups
    )
    dataset_groups = {group: [] for group in manual_dataset_groups}

    # Map the dataset names to their respective dataset objects
    for group in manual_dataset_groups:
        for ds_name in manual_dataset_groups[group]:
            # Raise an error if the manually defined dataset name is not in the list of provided datasets
            assert (
                ds_name in ds_name_to_index_mapping
            ), f"Dataset {ds_name} defined in the group {group} could not be found in the dataset list."

            # Get the index of this dataset in the list of datasets
            index = ds_name_to_index_mapping[ds_name]

            # Add the dataset object to the respective group
            dataset_groups[group] += [valid_datasets[index]]

    assert tasks.keys().isdisjoint(
        dataset_groups.keys()
    ), "The benchmark task and manually defined dataset group names should not overlap."

    # Define the subgroups of datasets to be aggregated separately
    subgroups = {**tasks, **dataset_groups}

    # Define a dictionary that is passed to evaluate() and evaluate_position()
    # and can therefore be used to store data across all splits. This is for
    # example used when evaluating the distribution shift datasets, in which
    # we need to keep track of the previous domain splits.
    shared_kwargs = {}

    split_results = {}
    # Iterate over the splits
    for split in (pbar := tqdm.tqdm(list(range(1, num_splits + 1)))):
        pbar.set_description(f"Running split {split - 1}/{num_splits}")
        import os, psutil

        print(
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            "MiB",
            "split",
            split,
        )
        # Evaluate the split
        split_result = evaluate(
            datasets=valid_datasets,
            model=metric_with_model,
            method=method_name,
            metric_used=metric_used,
            overwrite=overwrite,
            save=save,
            return_tensor=False,
            verbose=False,
            num_splits=num_splits,
            split_number=split,
            max_time=max_time,
            random_state=random_state,
            fetch_only=fetch_only,
            base_path=base_path,
            path_interfix=path_interfix,
            rename_gpu_runs=rename_gpu_runs,
            multi_test_portion_support=True,
            shared_kwargs=shared_kwargs,
            device=device,
            **eval_kwargs,
        )

        split_results[split] = split_result

    # Get the set of keys from the first split
    test_portions = set(next(iter(split_results.values())).keys())

    # Check if all other splits have the same set of keys
    for d in split_results.values():
        assert set(d.keys()) == test_portions, "Splits must have the same keys."

    global_results = {test_portion: {} for test_portion in test_portions}

    log_msg = {}
    for test_portion in test_portions:
        for ds in valid_datasets:
            # Note: Some of the split_results can be None in case no split was found.
            global_results[test_portion][
                ds.get_dataset_identifier()
            ] = DatasetEvaluationCollection(
                ds.get_dataset_identifier(),
                {
                    k: split_results[k][test_portion][ds.get_dataset_identifier()]
                    for k in split_results
                },
            )

        # Aggregate the metrics of each dataset over the number of splits
        # into a total metric, group metrics as well as task metrics.
        for metric in (pbar := tqdm.tqdm(valid_metrics)):
            pbar.set_description(f"Calculating {metric['aggregator']}_{metric['name']}")
            calculate_score_per_method(
                metric["func"],
                metric["name"],
                global_results[test_portion],
                valid_datasets,
                aggregator=metric["aggregator"],
                subgroups=subgroups,
            )

            if log_per_split_metrics:
                for split_result in split_results.values():
                    calculate_score_per_method(
                        metric["func"],
                        metric["name"],
                        split_result[test_portion],
                        valid_datasets,
                        aggregator=metric["aggregator"],
                        subgroups=subgroups,
                    )

        aggr_log_msg = {
            f"{split_name}/{test_portion}/{num_splits}_splits/{metric['aggregator']}_{metric['name']}": float(
                global_results[test_portion][f"{metric['aggregator']}_{metric['name']}"]
            )
            for metric in valid_metrics
        }

        print(str(aggr_log_msg))

        log_msg = {
            **log_msg,
            **aggr_log_msg,
        }

        if log_per_split_metrics:
            log_msg = {
                **log_msg,
                **{
                    f"{split_name}/{test_portion}/{num_splits}_splits/split_{i}/{metric['aggregator']}_{metric['name']}": float(
                        split_result[test_portion][
                            f"{metric['aggregator']}_{metric['name']}"
                        ]
                    )
                    for i, split_result in split_results.items()
                    for metric in valid_metrics
                },
            }

        for task in tasks:
            if len(tasks[task]) == 0:
                continue

            log_msg = {
                **log_msg,
                **{
                    f"{split_name}/{test_portion}/{num_splits}_splits/per_task/benchmark_{task}/{metric['aggregator']}_{metric['name']}": float(
                        global_results[test_portion][
                            f"{task}_{metric['aggregator']}_{metric['name']}"
                        ]
                    )
                    for metric in valid_metrics
                },
            }

            if log_per_split_metrics:
                log_msg = {
                    **log_msg,
                    **{
                        f"{split_name}/{test_portion}/{num_splits}_splits/split_{i}/per_task/benchmark_{task}/{metric['aggregator']}_{metric['name']}": float(
                            split_result[test_portion][
                                f"{task}_{metric['aggregator']}_{metric['name']}"
                            ]
                        )
                        for i, split_result in split_results.items()
                        for metric in valid_metrics
                    },
                }

        for group in dataset_groups:
            log_msg = {
                **log_msg,
                **{
                    f"{split_name}/{test_portion}/{num_splits}_splits/per_group/benchmark_{group}/{metric['aggregator']}_{metric['name']}": float(
                        global_results[test_portion][
                            f"{group}_{metric['aggregator']}_{metric['name']}"
                        ]
                    )
                    for metric in valid_metrics
                },
            }

            if log_per_split_metrics:
                log_msg = {
                    **log_msg,
                    **{
                        f"{split_name}/{test_portion}/{num_splits}_splits/split_{i}/per_group/benchmark_{group}/{metric['aggregator']}_{metric['name']}": float(
                            split_result[test_portion][
                                f"{group}_{metric['aggregator']}_{metric['name']}"
                            ]
                        )
                        for i, split_result in split_results.items()
                        for metric in valid_metrics
                    },
                }

        if log_per_dataset_metrics:
            for ds in valid_datasets:
                log_msg = {
                    **log_msg,
                    **{
                        f"{split_name}/{test_portion}/{num_splits}_splits/per_dataset/{ds.name}_{ds.get_dataset_identifier()}/{metric['aggregator']}_{metric['name']}": float(
                            global_results[test_portion][
                                ds.get_dataset_identifier()
                            ].metrics[f"{metric['aggregator']}_{metric['name']}"]
                        )
                        for metric in valid_metrics
                    },
                }

                if log_per_split_metrics:
                    log_msg = {
                        **log_msg,
                        **{
                            f"{split_name}/{test_portion}/{num_splits}_splits/split_{i}/per_dataset/{ds.name}_{ds.get_dataset_identifier()}/{metric['aggregator']}_{metric['name']}": float(
                                split_result[test_portion][
                                    ds.get_dataset_identifier()
                                ].metrics[f"{metric['aggregator']}_{metric['name']}"]
                            )
                            for i, split_result in split_results.items()
                            for metric in valid_metrics
                        },
                    }

    return log_msg, global_results


def evaluate(
    datasets: list[TabularDataset],
    verbose: bool = False,
    shared_kwargs: dict[str, Any] = None,
    multi_test_portion_support: bool = False,  # As long as not all methods are updated, we need to support the old method, which only works for one test dataset per fitted model.
    **eval_kwargs: Any,
) -> dict[str, dict[str, DatasetEvaluation]] | dict[str, DatasetEvaluation]:
    """
    Evaluates a list of datasets for a model function.

    :param datasets: List of datasets
    :param eval_kwargs: Keyword arguments for `evaluate_position`
    :return: Dictionary of dataset names and their evaluation results
    """
    overall_results = {}
    it = tqdm.tqdm if verbose else lambda x: x

    test_portions = {}

    for i, ds in enumerate(it(datasets)):
        # uncomment, if you want to debug why your tests are failing..
        # print("evaluating", ds.name, "...")
        assert type(ds) is not list, ValueError("Datasets must be Dataset objects")
        ds_results = evaluate_position(
            dataset=ds, verbose=verbose, shared_kwargs=shared_kwargs, **eval_kwargs
        )

        if len(test_portions) == 0:
            test_portions = {*ds.test_portions}
            overall_results = {test_portion: {} for test_portion in test_portions}

        # In case the split fails, set the dataset result to None.
        if ds_results is None:
            print(
                f"{ds.name} could not be evaluated on split {eval_kwargs.get('split_number', -1)}. Skipping."
            )

            for test_portion in test_portions:
                overall_results[test_portion][ds.get_dataset_identifier()] = None
        else:
            # We should assert that each dataset contains the same evaluated test datasets.
            # e.g. for distribution shift datasets:
            #   "dataset": {"id": DatasetEvaluation}, {"ood": DatasetEvaluation}
            # or for other datasets:
            #   "dataset": {"main": DatasetEvaluation}
            assert ds.test_portions == {
                *ds_results.keys()
            }, "Datasets must have the same keys."
            assert test_portions == {
                *ds_results.keys()
            }, "Datasets must have the same keys."

            for test_portion in test_portions:
                overall_results[test_portion][ds.get_dataset_identifier()] = ds_results[
                    test_portion
                ]

    # For backwards compatibility, we return a single dictionary if there is only one key.
    # TODO: Adapt the code affected by this change to support multiple evaluated test datasets on the same fitted model.
    if not multi_test_portion_support:
        assert (
            len(test_portions) == 1
        ), "Old method only supports one test dataset per fitted model."
        return overall_results[test_portions.pop()]

    return overall_results


"""
===============================
INTERNAL HELPER FUNCTIONS
===============================
"""


def evaluate_position(
    dataset: TabularDataset,
    model: Callable[..., dict[str, DatasetEvaluation]],
    method: str,
    metric_used: Callable,
    base_path: str = ".",
    path_interfix: str = "",
    fetch_only: bool = False,
    overwrite_splits: bool = False,
    overwrite: bool = True,
    save: bool = True,
    max_time: int = 300,
    num_splits: int = 5,
    split_number: int = 1,
    random_state: int = 0,
    raise_if_result_not_found: bool = False,
    allow_remap_time: bool = True,
    shared_kwargs: dict[str, Any] = None,
    **eval_kwargs,
) -> dict[str, DatasetEvaluation] | None:
    """
    Evaluates a dataset with a 'bptt' number of training samples.

    :param dataset: Dataset to evaluate on
    :param model: A function taking in (train_ds=, test_ds=, metric_used=, max_time=)
    :param method: Name of the method, "transformer" or other...
    """
    # Generates a string that identifies the settings of the evaluation result
    max_time_mapped = get_mapped_time(
        method, max_time, allow_remap_time=allow_remap_time
    )
    time_string = "_time_" + str(max_time_mapped) if max_time else ""
    metric_used_string = "_" + tabular_metrics.get_scoring_string(metric_used, usage="")
    method = method + time_string + metric_used_string

    assert check_metric_fits_task_type(
        metric_used, dataset.task_type
    ), f"Metric {metric_used} does not fit task type {dataset.task_type}"

    bptt = len(dataset.x)

    if dataset.splits is None or overwrite_splits:
        num_splits_or_official_split = (
            f"{N_SPLITS}"  # Hardcoded to 5 splits for any number of splits
        )
    else:
        num_splits_or_official_split = "official"

    prefix_for_device = (
        ""
        if (
            (eval_kwargs.get("device", "cpu") == "cpu")
            and (not eval_kwargs.get("rename_gpu_runs", False))
        )
        else "gpu_"
    )
    old_type_path = os.path.join(
        base_path,
        f"results/tabular/{path_interfix}/{prefix_for_device}results_{method}_{dataset.get_dataset_identifier()}_{split_number}_{num_splits_or_official_split}.npy",
    )
    if os.path.exists(old_type_path):
        warnings.warn(
            f"You are using a base_path with the old type of results. Please switch to the new base_path as in local_settings. Your currently used base_path is {base_path}",
        )
        path = old_type_path
    else:
        method_path = os.path.join(
            base_path, f"results/tabular/{path_interfix}", prefix_for_device + method
        )
        os.makedirs(method_path, exist_ok=True)
        path = os.path.join(
            method_path,
            f"dataset_{dataset.get_dataset_identifier()}_split_{split_number}_{num_splits_or_official_split}.npy",
        )

    ## Try loading results from disk
    if not overwrite or fetch_only:
        result = np_load_if_exists(path)
        if result is not None:
            # Backwards compatibility to load results which were saved with the old method.
            if "pred" in result:
                result = {"main": result}

            ds_results = {}

            for test_portion, test_portion_dict in result.items():
                # print(f"Loaded saved result for {path}")
                test_portion_dict.update({"algorithm_name": method})
                ds_results[test_portion] = DatasetEvaluation(**test_portion_dict)

            return ds_results
        elif fetch_only:
            print(f"Could not load saved result for {path} but in fetch only mode")
            return {
                test_portion: DatasetEvaluation(y=None, pred=None)
                for test_portion in dataset.test_portions
            }

    ## Generate data splits
    if isinstance(dataset, DistributionShiftDataset):
        assert (
            type(shared_kwargs) is dict
        ), "shared_kwargs must be a dictionary that is kept across all splits."

        shared_kwargs["previous_domain_splits"] = shared_kwargs.get(
            "previous_domain_splits", {}
        )
        previous_domain_splits = shared_kwargs["previous_domain_splits"]

        previous_domain_splits[
            dataset.get_dataset_identifier()
        ] = previous_domain_splits.get(dataset.get_dataset_identifier(), [])
        previous_domain_splits_on_ds = previous_domain_splits[
            dataset.get_dataset_identifier()
        ]

        train_ds, test_ds = dataset.generate_valid_split(
            all_preceding_data=True,
            all_remaining_data=True,
            max_predict_domains=None,
            splits=None if overwrite_splits else dataset.splits,
            split_number=split_number,
            previous_domain_splits_on_ds=previous_domain_splits_on_ds,
            minimize_num_train_domains=eval_kwargs["minimize_num_train_domains"],
        )

        model_kwargs = {
            "pipeline_kwargs": {
                "dist_shift_append_domain": eval_kwargs["append_domain_as_feature"]
            }
        }
    else:
        train_ds, test_ds = dataset.generate_valid_split(
            n_splits=N_SPLITS,
            # n_splits=num_splits,  # Causes KFold to fail for n_splits=1. @3bd2cecf is the fixed value above intended here?
            splits=None if overwrite_splits else dataset.splits,
            split_number=split_number,
        )

        model_kwargs = {}

    if train_ds is None:
        print(f"No dataset could be generated {dataset.name} {bptt}")
        return None

    device = eval_kwargs.get("device", "cpu")
    print(
        f"Running model with: device={device}, max_time={max_time}, metric={metric_used_string}, rng={random_state}, split_number={split_number}, dataset={dataset.get_dataset_identifier()}, eval_kwargs={eval_kwargs}."
    )

    # There might be multiple test datasets that should be evaluated using the
    # model underlying the same training dataset and training process.
    if type(test_ds) is not dict:
        test_ds = {"main": test_ds}

    start_time = time.time()

    ds_results = model(
        train_ds=copy.deepcopy(train_ds),
        test_ds=copy.deepcopy(test_ds),
        metric_used=metric_used,
        max_time=max_time,  # Device?
        random_state=random_state,
        device=device,
        **model_kwargs,
    )

    timestamp = datetime.datetime.now().isoformat(" ", "seconds")
    delta_time = time.time() - start_time
    ds_task_type = dataset.task_type
    ds_name = dataset.name
    ds_identifier = dataset.get_dataset_identifier()

    def update_results_metadata(test_ds, result):
        result.timestamp = timestamp
        result.time = delta_time

        result.task_type = ds_task_type
        result.name = ds_name
        result.identifier = ds_identifier

        result.y = test_ds.y

    for test_portion, test_portion_result in ds_results.items():
        update_results_metadata(test_ds[test_portion], test_portion_result)

    if save:
        with open(path, "wb") as f:
            np.save(
                f,
                {
                    test_portion: test_portion_result.to_dict()
                    for test_portion, test_portion_result in ds_results.items()
                },
            )
            print(f"Saved results to {path}")

    return ds_results


def get_mapped_time(
    method: str, max_time: int, *, allow_remap_time: bool = True
) -> int:
    """Remap time string.

    Maps the time string to -1 for methods which do not accept a time parameter. Thus we reload and save all results
    to the same file and don't redo calculations for multiple time budgets if the methods ignore these budgets anyways.

    :param method:
    :param max_time:
    :param allow_remap_time: If True, the time will be remapped based on the method name if {`transformer`,`default`,`tabpfn`}
        in the name or it the metod it is TabPFN's tpye getters, it is mapped to -1.
    """
    if max_time == -1:
        return -1

    remap = (
        "transformer" in method
        or "default" in method
        or "tabpfn" in method
        or method in tabpfn_model_type_getters
    )
    remap = remap and allow_remap_time

    max_time_mapped = -1 if remap else max_time
    if remap:
        print_once(f"Remapped time based on name to: {max_time_mapped}")
    return max_time_mapped


from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkGroup:
    groups: dict[str, Callable]
    name: str
    # Setup function that can use all datasets to generate statistics (e.g. quantiles for feature numbers)
    setup_func: Callable = field(default_factory=lambda: lambda datasets: {})
    task_types: list = field(
        default_factory=lambda: [
            "regression",
            "multiclass",
            "dist_shift_multiclass",
        ]
    )


samples_fine_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def in_percentile(fi, quantiles, v):
    if fi == 0:
        return v <= quantiles[fi]
    return quantiles[fi - 1] < v <= quantiles[fi]


benchmark_groups = [
    BenchmarkGroup(
        name="Nans",
        groups={
            "has_nans": lambda ds, _: ds.x.isnan().sum() > 0,
            "no_nans": lambda ds, _: ds.x.isnan().sum() == 0,
        },
    ),
    BenchmarkGroup(
        name="BinaryVsMulticlass",
        groups={
            "binary_classification": lambda ds, _: len(ds.y.unique()) <= 2,
            "nonbinary_classification": lambda ds, _: len(ds.y.unique()) > 2,
        },
        task_types=["multiclass", "dist_shift_multiclass"],
    ),
    BenchmarkGroup(
        name="Balanced",
        groups={
            "balanced": lambda ds, _: not is_imbalanced(ds.y),
            "imbalanced": lambda ds, _: is_imbalanced(ds.y),
        },
        task_types=["multiclass", "dist_shift_multiclass"],
    ),
    BenchmarkGroup(
        name="Features",
        groups={
            "small_features": (
                lambda ds, params: ds.x.shape[1] <= params["quantile_33"]
            ),
            "medium_features": lambda ds, params: (
                params["quantile_33"] < ds.x.shape[1] < params["quantile_66"]
            ),
            "large_features": (
                lambda ds, params: ds.x.shape[1] >= params["quantile_66"]
            ),
        },
        setup_func=lambda datasets: {
            "quantile_33": np.quantile([ds_.x.shape[1] for ds_ in datasets], 0.33),
            "quantile_66": np.quantile([ds_.x.shape[1] for ds_ in datasets], 0.66),
        },
    ),
    BenchmarkGroup(
        name="Samples",
        groups={
            "small_sample_size": (
                lambda ds, params: ds.x.shape[0] <= params["quantile_33"]
            ),
            "medium_sample_size": lambda ds, params: (
                params["quantile_33"] < ds.x.shape[0] < params["quantile_66"]
            ),
            "large_sample_size": (
                lambda ds, params: ds.x.shape[0] >= params["quantile_66"]
            ),
        },
        setup_func=lambda datasets: {
            "quantile_33": np.quantile([ds_.x.shape[0] for ds_ in datasets], 0.33),
            "quantile_66": np.quantile([ds_.x.shape[0] for ds_ in datasets], 0.66),
        },
    ),
    BenchmarkGroup(
        name="Samples_fine",
        groups={
            f"Samples in {round(samples_fine_fractions[fi] * 100)}th Percentile": (
                lambda ds, params, fi=fi: in_percentile(fi, params, ds.x.shape[0])
            )
            for fi in range(len(samples_fine_fractions))
        },
        setup_func=lambda datasets: {
            fi: np.quantile(
                [ds_.x.shape[0] for ds_ in datasets], samples_fine_fractions[fi]
            )
            for fi in range(len(samples_fine_fractions))
        },
    ),
    BenchmarkGroup(
        name="Features_fine",
        groups={
            f"Feats in {round(samples_fine_fractions[fi] * 100)}th Percentile": (
                lambda ds, params, fi=fi: in_percentile(fi, params, ds.x.shape[1])
            )
            for fi in range(len(samples_fine_fractions))
        },
        setup_func=lambda datasets: {
            fi: np.quantile(
                [ds_.x.shape[1] for ds_ in datasets], samples_fine_fractions[fi]
            )
            for fi in range(len(samples_fine_fractions))
        },
    ),
    BenchmarkGroup(
        name="Samples_to_features",
        groups={
            "small_features_per_sample": (
                lambda ds, params: ds.x.shape[1] / ds.x.shape[0]
                <= params["quantile_33"]
            ),
            "medium_features_per_sample": lambda ds, params: (
                params["quantile_33"]
                > (ds.x.shape[1] / ds.x.shape[0])
                > params["quantile_66"]
            ),
            "large_features_per_sample": (
                lambda ds, params: ds.x.shape[1] / ds.x.shape[0]
                >= params["quantile_66"]
            ),
        },
        setup_func=lambda datasets: {
            "quantile_33": np.quantile(
                [ds_.x.shape[1] / ds_.x.shape[0] for ds_ in datasets], 0.33
            ),
            "quantile_66": np.quantile(
                [ds_.x.shape[1] / ds_.x.shape[0] for ds_ in datasets], 0.66
            ),
        },
    ),
    BenchmarkGroup(
        name="Categoricals",
        groups={
            "purely_numerical": lambda ds, _: len(ds.categorical_feats) == 0,
            "purely_categorical": lambda ds, _: len(ds.categorical_feats)
            == ds.x.shape[1],
            "has_numericals": lambda ds, _: ds.x.shape[1] - len(ds.categorical_feats)
            > 0,
            "has_categoricals": lambda ds, _: len(ds.categorical_feats) > 0,
            "no_categorical_no_nan": lambda ds, _: (
                len(ds.categorical_feats) == 0 and ds.x.isnan().sum() == 0
            ),
        },
    ),
    # less extensive version of Categorical Benchmark Group
    BenchmarkGroup(
        name="Feature Type",
        groups={
            "no_categorical_feat_type": lambda ds, _: len(ds.categorical_feats) == 0,
            "has_categoricals_feat_type": lambda ds, _: len(ds.categorical_feats) > 0,
        },
    ),
    BenchmarkGroup(
        name="Y-outlier",
        setup_func=lambda datasets: {},
        groups={
            "Has > 50 Std": (lambda ds, params: (ds.y > ds.y.std() * 50).any()),
            "Has > 10 Std": (lambda ds, params: (ds.y > ds.y.std() * 10).any()),
            "None > 10 Std": lambda ds, params: (
                lambda ds, params: not (ds.y > ds.y.std() * 10).any()
            ),
        },
        task_types=["regression"],
    ),
    BenchmarkGroup(
        name="synthetic-vs-real-world",
        setup_func=lambda datasets: {},
        groups={
            "synthetic": (lambda ds, params: ds.dataset_source == "synthetic"),
            "real-world": (lambda ds, params: ds.dataset_source == "real-world"),
        },
        task_types=["dist_shift_multiclass"],
    ),
]


def get_benchmark_tasks_for_subset(
    valid_datasets: list[TabularDataset],
) -> dict[str, list[TabularDataset]]:
    results = {}

    for benchmark_group in benchmark_groups:
        if valid_datasets[0].task_type not in benchmark_group.task_types:
            continue
        params = benchmark_group.setup_func(valid_datasets)
        assert (
            len(set(results.keys()).intersection(set(benchmark_group.groups))) == 0
        ), f"A benchmark sub group {benchmark_group.groups} already exists, change the group names to avoid duplicates"
        results.update(
            {
                group: [
                    ds
                    for ds in valid_datasets
                    if benchmark_group.groups[group](ds, params)
                ]
                for group in benchmark_group.groups
            }
        )

    return results


def get_benchmark_tasks(valid_datasets: list[TabularDataset]) -> dict[str, Any]:
    subsets = set([ds.benchmark_name for ds in valid_datasets])
    results = {}
    for subset in subsets:
        subset_datasets = [ds for ds in valid_datasets if ds.benchmark_name == subset]
        benchmark_dict = get_benchmark_tasks_for_subset(subset_datasets)
        benchmark_dict = {f"{subset}_{k}": benchmark_dict[k] for k in benchmark_dict}
        results.update(benchmark_dict)
        results[f"{subset}"] = subset_datasets

    # Add one unified benchmark
    r = get_benchmark_tasks_for_subset(remove_duplicated_datasets(valid_datasets))
    r = {f"all_{k}": r[k] for k in r}  # prepend "all" to results for all datasets
    results.update(r)

    return results
