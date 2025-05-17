from . import preprocessing
from .base import (
    TabPFNClassifier,
    TabPFNRegressor,
    TabPFNBaseModel,
    PreprocessorConfig,
    get_single_tabpfn,
    ClassificationOptimizationMetricType,
    RegressionOptimizationMetricType,
)
from .configs import (
    TabPFNClassificationConfig,
    TabPFNRegressionConfig,
    TabPFNConfig,
    TabPFNModelPathsConfig,
    TabPFNDistShiftClassificationConfig,
    EnsembleConfiguration,
)


### MODEL CREATION ###
def get_tabpfn(config: TabPFNConfig, **kwargs):
    """Get a combined TabPFN Model."""
    return tabpfn_model_type_getters[config.model_type](config, **kwargs)


tabpfn_model_type_getters = {
    "single": get_single_tabpfn,
    "single_fast": get_single_tabpfn,
}
