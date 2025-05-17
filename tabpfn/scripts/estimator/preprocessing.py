from __future__ import annotations

from collections import UserList
from copy import deepcopy
import numpy as np
import torch
import typing as tp
import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    OrdinalEncoder,
    OneHotEncoder,
    FunctionTransformer,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import make_column_selector
from sklearn.decomposition import TruncatedSVD
from scipy.stats import shapiro

from .feature_transformers import (
    KDITransformerWithNaN,
    get_all_kdi_transformers,
    SafePowerTransformer,
)
from tabpfn.utils import print_once, skew, hash_tensor


class BadDataError(Exception):
    """Raised when a data is in a bad state for predictions.

    Namely:
        - All columns are constant.
    """


class CustomColumnTransformer(ColumnTransformer):
    def fit(self, X, y=None):
        # remove columns from self.transformers that are not in X
        self.transformers = [
            (name, transformer, cols)
            for name, transformer, cols in self.transformers
            if set(cols).issubset(set(*X.columns.values))
        ]
        return super().fit(X, y)


class TransformResult(tp.NamedTuple):
    X: np.ndarray
    categorical_features: tp.List[int]


class FeaturePreprocessingTransformerStep(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.categorical_features_after_transform_ = None

    def fit_transform(
        self, X: np.ndarray, categorical_features: tp.List[int]
    ) -> TransformResult:
        self.fit(X, categorical_features)
        return self.transform(X)

    @abstractmethod
    def _fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :param categorical_features: list of indices of categorical features, e.g. [0, 1, 2]
        :return: new categorical_features
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :param categorical_features: list of indices of categorical features, e.g. [0, 1, 2]
        :return: self
        """
        self.categorical_features_after_transform_ = self._fit(X, categorical_features)
        assert (
            self.categorical_features_after_transform_ is not None
        ), "_fit must return a list of the indexes of the categorical features after the transform."
        return self

    @abstractmethod
    def _transform(self, X: np.array, is_test: bool = False):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :return: new X: 2d np.array of shape (n_samples, new n_features)
        """
        raise NotImplementedError

    def transform(self, X: np.array, is_test: bool = False):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :return: new X: 2d np.array of shape (n_samples, new n_features), new categorical_features: list of indices of categorical features, e.g. [0, 1, 2]
        """
        # print(self.__class__.__name__, sum([hash_tensor(X[i]) for i, row in enumerate(X)]), X.sum())
        result = self._transform(X, is_test=is_test)
        # print(self.__class__.__name__,sum([hash_tensor(result[i]) for i, row in enumerate(result)]), result.sum())
        return TransformResult(result, self.categorical_features_after_transform_)


class SequentialFeatureTransformer(UserList):
    """
    A transformer that applies a sequence of feature preprocessing steps.
    This is very related to sklearn's Pipeline, but it is designed to work with
    categorical_features lists that are always passed on.

    Currently this class is only used once, thus this could also be made less general if needed.
    """

    def __init__(self, steps: tp.List[FeaturePreprocessingTransformerStep]):
        super().__init__(steps)
        self.categorical_features_ = None

    def fit_transform(
        self, X: np.ndarray | torch.Tensor, categorical_features: tp.List[int]
    ):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :param categorical_features: list of indices of categorical features, e.g. [0, 1, 2]
        :return: X
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        for step in self:
            X, categorical_features = step.fit_transform(X, categorical_features)
            assert isinstance(
                categorical_features, list
            ), f"The FeaturePreprocessingTransformerStep must return a list of categorical features, but {type(step)} returned {categorical_features}"
        self.categorical_features_ = categorical_features
        return TransformResult(X, categorical_features)

    def fit(self, X: np.ndarray | torch.Tensor, categorical_features: tp.List[int]):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :param categorical_features: list of indices of categorical features, e.g. [0, 1, 2]
        :return: self
        """
        assert (
            len(self) > 0
        ), "The SequentialFeatureTransformer must have at least one step."
        self.fit_transform(X, categorical_features)
        return self

    def transform(self, X: np.array, is_test: bool = False):
        """
        :param X: 2d array of shape (n_samples, n_features)
        :return: new X: 2d np.array of shape (n_samples, new n_features)
        """
        assert (
            len(self) > 0
        ), "The SequentialFeatureTransformer must have at least one step."
        assert (
            self.categorical_features_ is not None
        ), "The SequentialFeatureTransformer must be fit before it can be used to transform."
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        for step in self:
            X, categorical_features = step.transform(X, is_test=is_test)
        assert (
            categorical_features == self.categorical_features_
        ), f"Expected categorical features {self.categorical_features_}, but got {categorical_features}"
        return TransformResult(X, categorical_features)


class RemoveConstantFeaturesStep(FeaturePreprocessingTransformerStep):
    def __init__(self):
        super().__init__()
        self.sel_ = None

    def _fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        self.sel_ = ((X == X[0:1, :]).mean(axis=0) < 1.0).tolist()

        if not any(self.sel_):
            raise BadDataError(
                "All features are constant and would have been removed! Unable to predict using TabPFN."
            )

        new_categorical_features = [
            new_idx
            for new_idx, idx in enumerate(np.where(self.sel_)[0])
            if idx in categorical_features
        ]
        return new_categorical_features

    def _transform(self, X: np.array, is_test: bool = False):
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_]


class AddFingerprintFeaturesStep(FeaturePreprocessingTransformerStep):
    """
    This step adds a fingerprint feature to the features based on the hash of each row.
    If is_test = True, it keeps the first hash even if there are collisions.
    If is_test = False, it handles hash collisions by counting up and rehashing until a unique hash is found.
    """

    def __init__(self, rnd=np.random.default_rng()):
        self.rnd = rnd
        super().__init__()

    def _fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        self.rnd_salt_ = self.rnd.integers(0, 2**32)
        return [*categorical_features]

    def _transform(self, X: np.array, is_test: bool = False):
        X_h = np.zeros(X.shape[0], dtype=np.float32)

        if is_test:
            # Keep the first hash even if there are collisions
            for i, row in enumerate(X):
                h = hash_tensor(row + self.rnd_salt_)
                X_h[i] = h
        else:
            # Handle hash collisions by counting up and rehashing
            seen_hashes = set()
            for i, row in enumerate(X):
                h = hash_tensor(row + self.rnd_salt_)
                add_to_hash = 0
                while h in seen_hashes:
                    add_to_hash += 1
                    h = hash_tensor(row + self.rnd_salt_ + add_to_hash)
                X_h[i] = h
                seen_hashes.add(h)

        return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)


class ShuffleFeaturesStep(FeaturePreprocessingTransformerStep):
    def __init__(
        self, shuffle_method="rotate", shuffle_index=0, rnd=np.random.default_rng()
    ):
        super().__init__()
        self.shuffle_method = shuffle_method
        self.shuffle_index = shuffle_index
        self.rnd = rnd

        self.index_permutation_ = None

    def _fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        if self.shuffle_method == "rotate":
            # this yields a permutation where shuffle index indicates the position of the first feature
            # and we wrap around the features at the end
            self.index_permutation_ = np.roll(np.arange(X.shape[1]), self.shuffle_index)
        elif self.shuffle_method == "shuffle":
            self.index_permutation_ = self.rnd.permutation(X.shape[1])
        elif self.shuffle_method == "none":
            self.index_permutation_ = np.arange(X.shape[1])
        else:
            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")

        categorical_features = [
            new_idx
            for new_idx, idx in enumerate(self.index_permutation_)
            if idx in categorical_features
        ]
        # print('feature permutation:', self.index_permutation_)

        return categorical_features

    def _transform(self, X: np.array, is_test: bool = False):
        assert self.index_permutation_ is not None, "You must call fit first"
        assert (
            len(self.index_permutation_) == X.shape[1]
        ), "The number of features must not change after fit"
        return X[:, self.index_permutation_]


class ReshapeFeatureDistributionsStep(FeaturePreprocessingTransformerStep):
    none_transformer = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    @staticmethod
    def get_column_types(X: np.ndarray) -> tp.List[str]:
        """
        Returns a list of column types for the given data, that indicate how the data should be preprocessed.
        """
        column_types = []
        for col in range(X.shape[1]):
            if np.unique(X[:, col]).size < 10:
                column_types.append(f"ordinal_{col}")
            elif (
                skew(X[:, col]) > 1.1
                and np.min(X[:, col]) >= 0
                and np.max(X[:, col]) <= 1
            ):
                column_types.append(f"skewed_pos_1_0_{col}")
            elif skew(X[:, col]) > 1.1 and np.min(X[:, col]) > 0:
                column_types.append(f"skewed_pos_{col}")
            elif skew(X[:, col]) > 1.1:
                column_types.append(f"skewed_{col}")
            elif shapiro(X[0:3000, col]).statistic > 0.95:
                column_types.append(f"normal_{col}")
            else:
                column_types.append(f"other_{col}")
        return column_types

    @staticmethod
    def get_adaptive_preprocessors(num_examples: int = 100):
        """
        Returns a dictionary of adaptive column transformers that can be used to preprocess the data.
        Adaptive column transformers are used to preprocess the data based on the column type, they receive
        a pandas dataframe with column names, that indicate the column type. Column types are not datatypes,
        but rather a string that indicates how the data should be preprocessed.

        :param num_examples:
        :return:
        """
        preprocessings = {
            "adaptive": ColumnTransformer(
                [
                    (
                        "skewed_pos_1_0",
                        FunctionTransformer(
                            func=lambda x: np.exp(x),
                            inverse_func=lambda x: np.log(x),
                            check_inverse=False,
                        ),
                        make_column_selector("skewed_pos_1_0*"),
                    ),
                    (
                        "skewed_pos",
                        SafePowerTransformer(standardize=True, method="box-cox"),
                        make_column_selector("skewed_pos*"),
                    ),
                    (
                        "skewed",
                        SafePowerTransformer(standardize=True, method="yeo-johnson"),
                        make_column_selector("skewed*"),
                    ),
                    (
                        "other",
                        QuantileTransformer(
                            output_distribution="normal", n_quantiles=num_examples // 10
                        ),
                        # "other" or "ordinal"
                        make_column_selector("other*"),
                    ),
                    (
                        "ordinal",
                        ReshapeFeatureDistributionsStep.none_transformer,
                        # "other" or "ordinal"
                        make_column_selector("ordinal*"),
                    ),
                    (
                        "normal",
                        ReshapeFeatureDistributionsStep.none_transformer,
                        make_column_selector("normal*"),
                    ),
                ],
                remainder="passthrough",
            )
        }
        assert all(
            ["adaptive" in p for p in preprocessings.keys()]
        ), "Adaptive preprocessing must be named 'adaptive'"

        return preprocessings

    @staticmethod
    def get_all_preprocessors(num_examples: int = 100, rnd=np.random.default_rng()):
        all_preprocessors = {
            # "none": ReshapeFeatureDistributionsStep.none_transformer,  -> Duplicate entry, removed first as the latter had precedence.
            "power": PowerTransformer(standardize=True),
            "safepower": SafePowerTransformer(standardize=True),
            "power_box": PowerTransformer(standardize=True, method="box-cox"),
            "safepower_box": SafePowerTransformer(standardize=True, method="box-cox"),
            "log": FunctionTransformer(
                func=lambda x: np.log(x), inverse_func=lambda x: np.exp(x)
            ),
            "1_plus_log": FunctionTransformer(
                func=lambda x: np.log(1 + x), inverse_func=lambda x: np.exp(x) - 1
            ),
            "norm_and_kdi": FeatureUnion(
                [
                    (
                        "norm",
                        QuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=max(num_examples // 10, 2),
                        ),
                    ),
                    (
                        "kdi",
                        KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
                    ),
                ]
            ),
            "exp": FunctionTransformer(
                func=lambda x: np.exp(x), inverse_func=lambda x: np.log(x)
            ),
            "quantile_uni_coarse": QuantileTransformer(
                output_distribution="uniform", n_quantiles=max(num_examples // 10, 2)
            ),
            "quantile_norm_coarse": QuantileTransformer(
                output_distribution="normal", n_quantiles=max(num_examples // 10, 2)
            ),
            "quantile_uni": QuantileTransformer(
                output_distribution="uniform", n_quantiles=max(num_examples // 5, 2)
            ),
            "quantile_norm": QuantileTransformer(
                output_distribution="normal", n_quantiles=max(num_examples // 5, 2)
            ),
            "quantile_uni_fine": QuantileTransformer(
                output_distribution="uniform", n_quantiles=num_examples
            ),
            "quantile_norm_fine": QuantileTransformer(
                output_distribution="normal", n_quantiles=num_examples
            ),
            "robust": RobustScaler(unit_variance=True),
            "none": None,
            # "robust": RobustScaler(unit_variance=True),
            **get_all_kdi_transformers(rnd),
        }

        assert all(
            ["adaptive" not in p for p in all_preprocessors.keys()]
        ), "Adaptive preprocessing must be named 'adaptive'"

        all_preprocessors.update(
            ReshapeFeatureDistributionsStep.get_adaptive_preprocessors(num_examples)
        )

        return all_preprocessors

    def get_all_global_transformers(
        self, num_examples: int, num_features: int, rnd=np.random.default_rng()
    ):
        inf_to_nan_transformer = FunctionTransformer(
            lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)
        )
        nan_impute_transformer = SimpleImputer(missing_values=np.nan, strategy="mean")

        _make_finite_transformer = [
            (
                "inf_to_nan",
                inf_to_nan_transformer,
            ),
            (
                "nan_impute",
                nan_impute_transformer,
            ),
        ]

        def make_standard_scaler_save(_name_scaler_tuple):
            # Make sure that all data that enters and leaves a scaler is finite.
            # This is needed in edge cases where, for example, a division by zero occurs while scaling or when the input contains not number values.
            return Pipeline(
                steps=[
                    *_make_finite_transformer,
                    _name_scaler_tuple,
                    *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
                ]
            )

        def get_svd_transformer(n_components=100):
            return Pipeline(
                steps=[
                    (
                        "save_standard",
                        make_standard_scaler_save(
                            ("standard", StandardScaler(with_mean=False))
                        ),
                    ),
                    (
                        "svd",
                        TruncatedSVD(
                            n_components=n_components,
                            random_state=self.rnd.integers(0, 2**32),
                        ),
                    ),
                ]
            )

        return {
            "scaler": make_standard_scaler_save(("standard", StandardScaler())),
            "svd": FeatureUnion(
                [
                    ("passthrough", FunctionTransformer(func=lambda x: x)),
                    (
                        "svd",
                        get_svd_transformer(
                            n_components=max(
                                1, min(num_examples // 10 + 1, num_features // 2)
                            )
                        ),
                    ),
                ]
            ),
        }

    def __init__(
        self,
        transform_name: str = "safepower",
        apply_to_categorical: bool = False,
        append_to_original: bool = False,
        subsample_features: float = -1,
        global_transformer_name: str = None,
        rnd: np.random.Generator = np.random.default_rng(),
    ):
        super().__init__()
        self.transform_name = transform_name
        self.apply_to_categorical = apply_to_categorical
        self.append_to_original = append_to_original
        self.rnd = rnd
        self.subsample_features = float(subsample_features)
        self.global_transformer_name = global_transformer_name

        self.applied_transforms_ = {}

    def _fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        X = X.copy()
        num_examples, num_feats = X.shape

        self.global_transformer_ = None
        if (
            self.global_transformer_name is not None
            and self.global_transformer_name != "None"
            and not (self.global_transformer_name == "svd" and num_feats < 2)
        ):
            self.global_transformer_ = self.get_all_global_transformers(
                num_examples, num_feats, rnd=self.rnd
            )[self.global_transformer_name]

        all_preprocessors = self.get_all_preprocessors(num_examples, rnd=self.rnd)
        if self.subsample_features > 0:
            subsample_features = int(self.subsample_features * num_feats) + 1
            replace = (
                subsample_features > num_feats
            )  # sampling more features than exist

            self.subsampled_features_ = self.rnd.choice(
                list(range(X.shape[1])), subsample_features, replace=replace
            )
            categorical_features = [
                new_idx
                for new_idx, idx in enumerate(self.subsampled_features_)
                if idx in categorical_features
            ]
            num_feats = subsample_features
        else:
            self.subsampled_features_ = list(range(X.shape[1]))

        X = X[:, self.subsampled_features_]

        if "per_feature" != self.transform_name:
            transformers = [
                deepcopy(all_preprocessors[self.transform_name])
                for _ in range(num_feats)
            ]
        else:
            transformers = [
                deepcopy(self.rnd.choice(list(all_preprocessors.values())))
                for _ in range(num_feats)
            ]

        self.applied_transforms_ = {}

        self.column_types_ = [f"feature_{i}" for i in range(num_feats)]
        # If an adaptive preprocessing is used, we need to determine the column types. All other preprocessing
        # methods are applied to all columns the same, so we can skip this step.
        if "adaptive" in self.transform_name:
            self.column_types_ = self.get_column_types(X)

        X = pd.DataFrame(
            X, columns=self.column_types_
        )  # Use column names to indicate column types for column transformer

        X_new = []
        # going through the columns one by one here is almost as fast as doing all at once (around 10% slower)
        categorical_features_ = []
        for col in range(X.shape[1]):
            data = X.iloc[:, col : col + 1]

            if not self.apply_to_categorical and col in categorical_features:
                transformers[col] = self.none_transformer
                categorical_features_ += [
                    int(np.sum(np.array([x.shape[1] for x in X_new])))
                ]

            # Check if preprocessor is working, if not use none_transformer for this column
            try:
                transformers[col].fit(data)
                # this line is here to catch errors during transform
                X_new += [transformers[col].transform(data)]
                self.applied_transforms_[col] = transformers[col]
            except Exception as e:
                print_once("failed to fit feature with error", e, "skipping")
                X_new += [data]
                self.applied_transforms_[col] = self.none_transformer

            if (
                self.append_to_original
                and self.applied_transforms_[col] != self.none_transformer
            ):
                X_new += [data]
                if col in categorical_features:
                    categorical_features_ += [
                        int(np.sum(np.array([x.shape[1] for x in X_new])))
                    ]

        X = np.concatenate(X_new, axis=1)

        # Apply a global transformer which accepts the entire dataset instead of one column
        if self.global_transformer_:
            self.global_transformer_.fit(X)

        return categorical_features_

    def _transform(self, X: np.array, is_test: bool = False):
        X = X.copy()
        X_new = []

        X = X[:, self.subsampled_features_]

        X = pd.DataFrame(X, columns=self.column_types_)

        for col, transform in self.applied_transforms_.items():
            data = X.iloc[:, col : col + 1]
            try:
                trans = transform.transform(data)
                X_new += [trans]
            except Exception as e:
                # TODO: This will yield bad performance, is there a way to recover better?
                print_once("failed to transform feature with error", e, "skipping")
                X_new += [data]

            if self.append_to_original and transform != self.none_transformer:
                X_new += [data]
        X = np.concatenate(X_new, axis=1)

        if self.global_transformer_:
            X = self.global_transformer_.transform(X)

        return X


class EncodeCategoricalFeaturesStep(FeaturePreprocessingTransformerStep):
    def __init__(
        self,
        categorical_transform_name: str = "ordinal",
        rnd: np.random.Generator = np.random.default_rng(),
    ):
        super().__init__()
        self.categorical_transform_name = categorical_transform_name
        self.rnd = rnd

        self.categorical_transformer_ = None

    @staticmethod
    def get_least_common_category_count(x_column: np.array):
        counts: pd.Series = pd.Series(x_column).value_counts()
        return counts.iloc[-1] if len(counts) else 0

    def _fit(self, X: np.array, categorical_features: tp.List[int]):
        if self.categorical_transform_name.startswith("ordinal"):
            categorical_transform_name_without_prefix = self.categorical_transform_name[
                len("ordinal") :
            ]
            # Create a column transformer
            if categorical_transform_name_without_prefix.startswith(
                "_common_categories"
            ):
                categorical_transform_name_without_prefix = (
                    categorical_transform_name_without_prefix[
                        len("_common_categories") :
                    ]
                )
                categorical_features = [
                    i
                    for i in range(X.shape[1])
                    if i in categorical_features
                    and self.get_least_common_category_count(X[:, i]) >= 10
                ]
            elif categorical_transform_name_without_prefix.startswith(
                "_very_common_categories"
            ):
                categorical_transform_name_without_prefix = (
                    categorical_transform_name_without_prefix[
                        len("_very_common_categories") :
                    ]
                )
                categorical_features = [
                    i
                    for i in range(X.shape[1])
                    if i in categorical_features
                    and self.get_least_common_category_count(X[:, i]) >= 10
                    and pd.unique(X[:, i]).size < (len(X) // 10)
                ]

            ct = ColumnTransformer(
                [
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),  # 'sparse' has been deprecated
                        categorical_features,
                    )
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            ct.fit(X[:, :])
            categorical_features = list(range(len(categorical_features)))
            self.random_mappings_ = {}
            if categorical_transform_name_without_prefix == "_shuffled":
                for col in categorical_features:
                    self.random_mappings_[col] = self.rnd.permutation(
                        len(ct.named_transformers_["ordinal_encoder"].categories_[col])
                    )
            else:
                assert (
                    categorical_transform_name_without_prefix == ""
                ), f'unknown categorical transform name, should be "ordinal" or "ordinal_shuffled" it was {self.categorical_transform_name}'
        elif self.categorical_transform_name == "onehot":
            # Create a column transformer
            ct = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            sparse_output=False, handle_unknown="ignore"
                        ),  # 'sparse' has been deprecated
                        categorical_features,
                    )
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            ct.fit(X)
            eval_xs = ct.transform(X)
            if eval_xs.size < 1_000_000:
                categorical_features = list(range(eval_xs.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
            else:
                ct = None
        elif self.categorical_transform_name in ("numeric", "none"):
            ct = None
        else:
            raise ValueError(
                f"Unknown categorical transform {self.categorical_transform_name}"
            )

        self.categorical_transformer_ = ct
        return categorical_features

    def _transform(self, X: np.array, is_test: bool = False):
        if self.categorical_transformer_ is None:
            return X
        else:
            transformed = self.categorical_transformer_.transform(X)
            if self.categorical_transform_name == "ordinal_shuffled":
                for col, mapping in self.random_mappings_.items():
                    not_nan_mask = ~np.isnan(transformed[:, col])
                    transformed[:, col][not_nan_mask] = mapping[
                        transformed[:, col][not_nan_mask].astype(int)
                    ].astype(transformed[:, col].dtype)
            return transformed


class NanHandlingPolynomialFeaturesStep(FeaturePreprocessingTransformerStep):
    def __init__(
        self,
        *,
        max_poly_features: tp.Optional[int] = None,
        rnd=np.random.default_rng(),
    ):
        # super().__init__(degree=degree, interaction_only=interaction_only, include_bias=include_bias, order=order)
        super().__init__()

        self.max_poly_features = max_poly_features
        self.rnd = rnd

        self.poly_factor_1_idx = None
        self.poly_factor_2_idx = None

        self.standardizer = StandardScaler(with_mean=False)

    def _fit(self, X: np.ndarray, categorical_features: tp.List[int]):
        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"

        if X.shape[0] == 0 or X.shape[1] == 0:
            return [*categorical_features]

        # How many polynomials can we create?
        n_polynomials = (X.shape[1] * (X.shape[1] - 1)) // 2 + X.shape[1]
        n_polynomials = (
            min(self.max_poly_features, n_polynomials)
            if self.max_poly_features
            else n_polynomials
        )

        X = self.standardizer.fit_transform(X)

        # Randomly select the indices of the factors
        self.poly_factor_1_idx = self.rnd.choice(
            np.arange(0, X.shape[1]), size=n_polynomials, replace=True
        )
        self.poly_factor_2_idx = np.ones_like(self.poly_factor_1_idx) * -1
        for i in range(len(self.poly_factor_1_idx)):
            while self.poly_factor_2_idx[i] == -1:
                poly_factor_1_ = self.poly_factor_1_idx[i]
                # indices of the factors that have already been used
                used_indices = self.poly_factor_2_idx[
                    self.poly_factor_1_idx == poly_factor_1_
                ]
                # remaining indices, only factors with higher index can be selected to avoid duplicates
                indices_ = set(range(poly_factor_1_, X.shape[1])) - set(
                    used_indices.tolist()
                )
                if len(indices_) == 0:
                    self.poly_factor_1_idx[i] = self.rnd.choice(
                        np.arange(0, X.shape[1]), size=1
                    )
                    continue
                self.poly_factor_2_idx[i] = self.rnd.choice(list(indices_), size=1)

        return categorical_features

    def _transform(self, X: np.array, is_test: bool = False):
        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"

        if X.shape[0] == 0 or X.shape[1] == 0:
            return X

        X = self.standardizer.transform(X)

        poly_features_xs = X[:, self.poly_factor_1_idx] * X[:, self.poly_factor_2_idx]

        X = np.hstack((X, poly_features_xs))

        return X
