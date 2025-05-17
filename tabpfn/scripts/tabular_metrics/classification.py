from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
)

"""
===============================
Classification calculation
===============================
"""


def automl_benchmark_metric(target, pred, numpy=False, should_raise=False):
    lib = np if numpy else torch

    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    if len(lib.unique(target)) > 2:
        return -cross_entropy(target, pred)
    else:
        return auc_metric_ovr(target, pred, numpy=numpy, should_raise=should_raise)


def auc_metric_ovr(target, pred, numpy=False, should_raise=False):
    return auc_metric(
        target, pred, multi_class="ovr", numpy=numpy, should_raise=should_raise
    )


def auc_metric_ovo(target, pred, numpy=False, should_raise=False, labels=None):
    return auc_metric(
        target,
        pred,
        multi_class="ovo",
        numpy=numpy,
        should_raise=should_raise,
        labels=labels,
    )


def remove_classes_not_in_target_from_pred(target, pred):
    assert torch.is_tensor(target) == torch.is_tensor(
        pred
    ), "target and pred must be both torch tensors or both numpy arrays"
    convert_to_torch = False
    if torch.is_tensor(target):
        convert_to_torch = True
        target = target.numpy()
        pred = pred.numpy()

    unique_targets = np.unique(target)
    assert all(
        unique_targets[:-1] <= unique_targets[1:]
    ), "target must be sorted after unique"

    # assumption is that target is 0-indexed before removing classes
    if len(unique_targets) < pred.shape[1]:
        assert (
            unique_targets < pred.shape[1]
        ).all(), "target must be smaller than pred.shape[1]"
        pred = pred[:, unique_targets]
        # Can't add an eps here to prevent nans as then in binary classification
        # we would bias the result in saying even though both classes had 0 probability
        # the negative class now has probability 1. In multiclass we would have the issue
        # of the predictions not adding up to 1.
        pred = pred / pred.sum(axis=1, keepdims=True)

        # make target 0-indexed again, just for beauty
        # sklearn would handle it anyway
        mapping = {c: i for i, c in enumerate(unique_targets)}
        target = np.array([mapping[c] for c in target])
    if convert_to_torch:
        target = torch.tensor(target)
        pred = torch.tensor(pred)
    return target, pred


def auc_metric(
    target, pred, multi_class="ovr", numpy=False, should_raise=False, labels=None
):
    lib = np if numpy else torch

    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    else:
        target = np.array(target)
        pred = np.array(pred)

    # In 'ovr' we cannot deal with the case in which we have predictions for classes that do not appear in the ground
    # truth. In this case just drop the predictions of theses classes. The issue hereby is that it could lead to nans
    # due to normalization, in case the probability of the ground truth classes is 0.
    # When using sklearn's cross val score with this function, it expects the metric to accept 1D prediction in the binary case
    # Hence, we only apply our fix here, if pred shape is 2D (a.k.a. multiclass or called not from sklearn)
    if len(pred.shape) > 1 and (
        multi_class == "ovr" or (multi_class == "ovo" and labels is None)
    ):
        target, pred = remove_classes_not_in_target_from_pred(target, pred)

        assert (
            len(lib.unique(target)) == pred.shape[1]
        ), "target and pred must have the same number of classes"

        if pred.shape[1] == 2:
            pred = pred[:, 1]

    # Catch nan errors due to normalization or weird params. Prevents sklearn from crashing in cross_val_score.
    #
    #     Exemplary Stacktrace of a crashed xgb run:
    #     scripts/tabular_baselines/utils.py, line 108, in eval_f
    #     scores = cross_val_score(
    #     ...
    #     sklearn/metrics/_ranking.py, line 606, in roc_auc_score
    #     y_score = check_array(y_score, ensure_2d=False)
    #     ...
    #     sklearn/utils/validation.py, line 171, in _assert_all_finite_element_wise
    #     raise ValueError("Input contains NaN.")
    #
    # Instead of crashing, we can now catch this error e.g. in hpo to skip these params.
    if (numpy and np.isnan(pred).any()) or (
        not numpy and torch.isnan(pred).any().item()
    ):
        raise ValueError("Encountered NaN in y_score.")

    if multi_class == "ovo" and labels is not None:
        score = roc_auc_score(target, pred, multi_class=multi_class, labels=labels)
    else:
        score = roc_auc_score(target, pred, multi_class=multi_class)

    if not numpy:
        return torch.tensor(score)
    return score


def accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))


def f1_metric(target, pred, multi_class="micro"):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(
            f1_score(target, torch.argmax(pred, -1), average=multi_class)
        )
    else:
        return torch.tensor(f1_score(target, pred[:, 1] > 0.5))


def average_precision_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(average_precision_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(average_precision_score(target, pred[:, 1] > 0.5))


def balanced_accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(balanced_accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(balanced_accuracy_score(target, pred[:, 1] > 0.5))


def cross_entropy(target, pred, numpy=False):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        ce = torch.nn.CrossEntropyLoss()
        return ce(pred.float().log(), target.long())
    else:
        bce = torch.nn.BCELoss()
        return bce(pred[:, 1].float(), target.float())


def is_classification(metric_used):
    if metric_used == auc_metric or metric_used == cross_entropy:
        return True
    return False


def nll_bar_dist(target, pred, bar_dist):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    target, pred = target.unsqueeze(0).to(bar_dist.borders.device), pred.unsqueeze(
        1
    ).to(bar_dist.borders.device)

    l = bar_dist(pred.log(), target).mean().cpu()
    return l


def expected_calibration_error(target, pred, norm="l1", n_bins=10):
    import torchmetrics

    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    target, pred = remove_classes_not_in_target_from_pred(target, pred)

    ece = torchmetrics.classification.MulticlassCalibrationError(
        n_bins=n_bins,
        norm=norm,
        num_classes=len(torch.unique(target)),
    )
    return ece(
        target=target,
        preds=pred,
    )


def is_imbalanced(y, threshold=0.8):
    """
    Determine if a numpy array of class labels is imbalanced based on Gini impurity.

    Parameters:
    - y (numpy.ndarray): A 1D numpy array containing class labels.
    - threshold (float): Proportion of the maximum Gini impurity to consider as the boundary
                         between balanced and imbalanced. Defaults to 0.8.

    Returns:
    - bool: True if the dataset is imbalanced, False otherwise.

    Example:
    >>> y = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    >>> is_imbalanced(y)
    True
    """

    # Calculate class proportions
    _, class_counts = np.unique(y, return_counts=True)
    class_probs = class_counts / len(y)

    # Calculate Gini impurity
    gini = 1 - np.sum(class_probs**2)

    # Determine max possible Gini for the number of classes
    C = len(class_probs)
    max_gini = 1 - 1 / C

    # Check if the Gini impurity is less than the threshold of the maximum possible Gini
    return gini < threshold * max_gini
