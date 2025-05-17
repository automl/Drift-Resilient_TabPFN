from __future__ import annotations

from sklearn.preprocessing import (
    PowerTransformer,
)
import numpy as np
import torch


try:
    from kditransform import KDITransformer

    # This import fails on some systems, due to problems with numba
except Exception as e:
    print("Could not import KDITransformer, error:", e)


class KDITransformerWithNaN(KDITransformer):
    """
    KDI transformer that can handle NaN values. It performs KDI with NaNs replaced by mean values and then
    fills the NaN values with NaNs after the transformation.
    """

    def _more_tags(self):
        return {"allow_nan": True}

    def fit(self, X, y=None):
        # if tensor convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        return super().fit(X, y)

    def transform(self, X):
        # if tensor convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Calculate the NaN mask for the current dataset
        nan_mask = np.isnan(X)
        # Replace NaNs with the mean of columns
        imputation = np.nanmean(X, axis=0)
        imputation = np.nan_to_num(imputation, nan=0)
        X = np.nan_to_num(X, nan=imputation)
        # Apply the transformation
        X = super().transform(X)

        # Reintroduce NaN values based on the current dataset's mask
        X[nan_mask] = np.nan

        return X


def get_all_kdi_transformers(rnd):
    all_preprocessors = {}

    try:
        all_preprocessors.update(
            {
                "kdi": KDITransformerWithNaN(alpha=1.0, output_distribution="normal"),
                "kdi_uni": KDITransformerWithNaN(
                    alpha=1.0, output_distribution="uniform"
                ),
            }
        )
        for alpha in [
            0.05,
            0.1,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.8,
            1.0,
            1.2,
            1.5,
            1.8,
            2.0,
            2.5,
            3.0,
            5.0,
        ]:
            all_preprocessors[f"kdi_alpha_{alpha}"] = KDITransformerWithNaN(
                alpha=alpha, output_distribution="normal"
            )
            all_preprocessors[f"kdi_alpha_{alpha}_uni"] = KDITransformerWithNaN(
                alpha=alpha, output_distribution="uniform"
            )
    except:
        pass

    return all_preprocessors


class SafePowerTransformer(PowerTransformer):
    """
    Power Transformer which reverts features back to their original values if they are
    transformed to very large values or the output column does not have unit variance.
    This happens e.g. when the input data has a large number of outliers.
    """

    def __init__(self, variance_threshold=1e-3, large_value_threshold=100, **kwargs):
        super().__init__(**kwargs)
        self.variance_threshold = variance_threshold
        self.large_value_threshold = large_value_threshold

        self.revert_indices_ = None

    def _find_features_to_revert_because_of_failure(self, transformed_X):
        # Calculate the variance for each feature in the transformed data
        variances = np.nanvar(transformed_X, axis=0)

        # Identify features where the variance is not close to 1
        non_unit_variance_indices = np.where(
            np.abs(variances - 1) > self.variance_threshold
        )[0]

        # Identify features with values greater than the large_value_threshold
        large_value_indices = np.any(transformed_X > self.large_value_threshold, axis=0)
        large_value_indices = np.nonzero(large_value_indices)[0]

        # Identify features to revert based on either condition
        self.revert_indices_ = np.unique(
            np.concatenate([non_unit_variance_indices, large_value_indices])
        )

    def _revert_failed_features(self, transformed_X, original_X):
        # Replace these features with the original features
        if self.revert_indices_ and self.revert_indices_ > 0:
            transformed_X[:, self.revert_indices_] = original_X[:, self.revert_indices_]

        return transformed_X

    def fit(self, X, y=None):
        # Fit the model
        super().fit(X, y)

        # Check and revert features as necessary
        self._find_features_to_revert_because_of_failure(super().transform(X))

        return self

    def transform(self, X):
        # Transform the input data
        transformed_X = super().transform(X)

        # Check and revert features as necessary
        return self._revert_failed_features(transformed_X, X)
