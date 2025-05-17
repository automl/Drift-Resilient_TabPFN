from sklearn.model_selection._split import _BaseKFold

from sklearn.utils import (
    indexable,
)

from sklearn.utils.validation import _num_samples

import numpy as np

import warnings


class DistributionShiftSplit(_BaseKFold):
    """Time Series cross-validator

    Customized for our distribution shift datasets in which we have additional
    restrictions for the split positions.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <time_series_split>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value
        with ``gap=0``.

    gap : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i`` th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """

    def __init__(
        self,
        domain_indicators,
        n_splits=5,
        *,
        max_train_size=None,
        test_size=None,
        gap=0,
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)

        self.domain_indicators = indexable(domain_indicators)[0]

        self.unique_domains, self.domain_counts = np.unique(
            self.domain_indicators, return_counts=True
        )
        self.num_domains = self.unique_domains.shape[0]

        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        assert n_samples == _num_samples(
            self.domain_indicators
        ), "The number of domain indicators does not match the number of instances."

        indices = np.arange(n_samples)

        # In case there are more than two domains, reserve the first two pure for training
        # such that the model is able to extrapolate the shift.

        # More than 2 domains
        if self.num_domains > 2:
            # Reserve indices of the first two domains for training
            first_two_domains_count = np.sum(self.domain_counts[:2])
            initial_train_indices = indices[:first_two_domains_count]
            indices = indices[
                first_two_domains_count:
            ]  # update indices to exclude the first two domains

        # More than 1 domain but not more than 2
        elif self.num_domains > 1:
            # Reserve indices of the first domain for training
            first_domain_count = self.domain_counts[0]
            initial_train_indices = indices[:first_domain_count]
            indices = indices[
                first_domain_count:
            ]  # update indices to exclude the reserved parts of first domain

        # Only 1 domain
        else:
            initial_train_indices = np.array(
                [], dtype=int
            )  # no initial training data, all data can be used for splits

        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else len(indices) // n_folds
        )

        # Make sure we have enough samples for the given split parameters
        if n_folds > len(indices):
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples after reserving for domains."
            )

        if len(indices) - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples after reserving for domains"
                f" with test_size={test_size} and gap={gap}."
            )

        test_starts = range(
            len(indices) - n_splits * test_size, len(indices), test_size
        )

        for test_start in test_starts:
            train_end = test_start - gap

            if self.max_train_size and self.max_train_size < train_end:
                train_indices = np.concatenate(
                    [
                        initial_train_indices,
                        indices[train_end - self.max_train_size : train_end],
                    ]
                )
                test_indices = indices[test_start : test_start + test_size]
            else:
                train_indices = np.concatenate(
                    [initial_train_indices, indices[:train_end]]
                )
                test_indices = indices[test_start : test_start + test_size]

            unique_classes = np.unique(y[test_indices])

            # If fewer than two unique classes, find the minimum index which includes another class
            if len(unique_classes) < 2:
                # Try increasing the test size first
                additional_range = np.where(
                    np.isin(y[indices], unique_classes, invert=True)
                )[0]
                additional_range = additional_range[
                    additional_range >= test_start + test_size
                ]

                if additional_range.size == 0:
                    # If no additional classes ahead, decrease test_start if possible
                    additional_range = np.where(
                        np.isin(y[indices], unique_classes, invert=True)
                    )[0]
                    additional_range = additional_range[additional_range < test_start]

                    if additional_range.size == 0:
                        raise ValueError(
                            "Unable to find enough classes for ROC AUC calculation for this fold."
                        )

                    new_test_start = additional_range[
                        -1
                    ]  # use the last index that introduces a new class
                    test_indices = indices[new_test_start : new_test_start + test_size]

                    actual_decrease = test_start - new_test_start
                    warnings.warn(
                        f"CV Fold: Test start decreased by {actual_decrease} indices to include more than one class.",
                        UserWarning,
                    )

                    # Adjust train indices to avoid leakage
                    train_end = new_test_start - gap
                    if self.max_train_size and self.max_train_size < train_end:
                        train_indices = np.concatenate(
                            [
                                initial_train_indices,
                                indices[train_end - self.max_train_size : train_end],
                            ]
                        )
                    else:
                        train_indices = np.concatenate(
                            [initial_train_indices, indices[:train_end]]
                        )
                else:
                    new_test_end = additional_range[0] + 1  # include this new class
                    test_indices = indices[test_start:new_test_end]
                    actual_increase = new_test_end - (test_start + test_size)
                    warnings.warn(
                        f"CV Fold: Test set increased by {actual_increase} indices to include more than one class.",
                        UserWarning,
                    )

            yield train_indices, test_indices
