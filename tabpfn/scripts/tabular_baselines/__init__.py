from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np

from ..estimator import TabPFNModelPathsConfig

from .methods import (
    catboost_metric,
    lightgbm_metric,
    xgb_metric,
    wildtime_metric,
)

from .methods import transformer_metric

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

param_grid, param_grid_hyperopt = {}, {}


def get_clf_dict(task_type):
    if task_type in {"multiclass", "regression", "dist_shift_multiclass"}:
        clf_dict = {
            # ---- Part of the Benchmark
            # -- Default Baselines
            "catboost_default": partial(catboost_metric, no_tune={}),
            "lgb_default": partial(lightgbm_metric, no_tune={}),
            "xgb_default": partial(xgb_metric, no_tune={}),
            "xgb": xgb_metric,
            "catboost": catboost_metric,
            "lightgbm": lightgbm_metric,
            "transformer": transformer_metric,
        }
        if task_type in {"dist_shift_multiclass"}:
            wildtime_network_dict = {
                "MLP",
                "FTT",
            }
            wildtime_method_dict = {
                "coral",
                "groupdro",
                "erm",
                "erm_lisa",
                "erm_mixup",
                "ft",
                "si",
                "swa",
                "irm",
                "ewc",
                "agem",
            }
            for network, method in product(wildtime_network_dict, wildtime_method_dict):
                clf_dict[f"wildtime_{network}_{method}"] = partial(
                    wildtime_metric, network=network, method_name=method
                )
                clf_dict[f"wildtime_{network}_{method}_default"] = partial(
                    wildtime_metric, network=network, method_name=method, no_tune={}
                )
    else:
        raise NotImplementedError(f"Unknown task type {task_type}")
    return clf_dict


clf_relabeler = {
    "transformer": "Tabular PFN",
    "tabpfn": "TabPFN",
    "catboost": "Catboost",
    "catboost_default": "Catboost (default)",
    "xgb": "XGB",
    "xgb_default": "XGB (default)",
    "lightgbm": "LightGBM",
    "lgb_default": "LightGBM (default)",
    "wildtime_MLP_coral": "WildTime (MLP, CORAL)",
    "wildtime_MLP_groupdro": "WildTime (MLP, GroupDRO)",
    "wildtime_MLP_erm": "WildTime (MLP, ERM)",
    "wildtime_MLP_erm_lisa": "WildTime (MLP, ERM_LISA)",
    "wildtime_MLP_erm_mixup": "WildTime (MLP, ERM_MIXUP)",
    "wildtime_MLP_ft": "WildTime (MLP, FT)",
    "wildtime_MLP_si": "WildTime (MLP, SI)",
    "wildtime_MLP_swa": "WildTime (MLP, SWA)",
    "wildtime_MLP_irm": "WildTime (MLP, IRM)",
    "wildtime_MLP_ewc": "WildTime (MLP, EWC)",
    "wildtime_MLP_agem": "WildTime (MLP, AGEM)",
    "wildtime_FTT_coral": "WildTime (FTT, CORAL)",
    "wildtime_FTT_groupdro": "WildTime (FTT, GroupDRO)",
    "wildtime_FTT_erm": "WildTime (FTT, ERM)",
    "wildtime_FTT_erm_lisa": "WildTime (FTT, ERM_LISA)",
    "wildtime_FTT_erm_mixup": "WildTime (FTT, ERM_MIXUP)",
    "wildtime_FTT_ft": "WildTime (FTT, FT)",
    "wildtime_FTT_si": "WildTime (FTT, SI)",
    "wildtime_FTT_swa": "WildTime (FTT, SWA)",
    "wildtime_FTT_irm": "WildTime (FTT, IRM)",
    "wildtime_FTT_ewc": "WildTime (FTT, EWC)",
    "wildtime_FTT_agem": "WildTime (FTT, AGEM)",
    "wildtime_MLP_coral_default": "WildTime (MLP, CORAL, default)",
    "wildtime_MLP_groupdro_default": "WildTime (MLP, GroupDRO, default)",
    "wildtime_MLP_erm_default": "WildTime (MLP, ERM, default)",
    "wildtime_MLP_erm_lisa_default": "WildTime (MLP, ERM_LISA, default)",
    "wildtime_MLP_erm_mixup_default": "WildTime (MLP, ERM_MIXUP, default)",
    "wildtime_MLP_ft_default": "WildTime (MLP, FT, default)",
    "wildtime_MLP_si_default": "WildTime (MLP, SI, default)",
    "wildtime_MLP_swa_default": "WildTime (MLP, SWA, default)",
    "wildtime_MLP_irm_default": "WildTime (MLP, IRM, default)",
    "wildtime_MLP_ewc_default": "WildTime (MLP, EWC, default)",
    "wildtime_MLP_agem_default": "WildTime (MLP, AGEM, default)",
    "wildtime_FTT_coral_default": "WildTime (FTT, CORAL, default)",
    "wildtime_FTT_groupdro_default": "WildTime (FTT, GroupDRO, default)",
    "wildtime_FTT_erm_default": "WildTime (FTT, ERM, default)",
    "wildtime_FTT_erm_lisa_default": "WildTime (FTT, ERM_LISA, default)",
    "wildtime_FTT_erm_mixup_default": "WildTime (FTT, ERM_MIXUP, default)",
    "wildtime_FTT_ft_default": "WildTime (FTT, FT, default)",
    "wildtime_FTT_si_default": "WildTime (FTT, SI, default)",
    "wildtime_FTT_swa_default": "WildTime (FTT, SWA, default)",
    "wildtime_FTT_irm_default": "WildTime (FTT, IRM, default)",
    "wildtime_FTT_ewc_default": "WildTime (FTT, EWC, default)",
    "wildtime_FTT_agem_default": "WildTime (FTT, AGEM, default)",
}


def clf_relabeler_with_time(clf, time):
    def time_mapper(time):
        if time == 3600:
            return "1h "
        else:
            return ""

    if "default" in clf or "single" in clf or "tabpfn" in clf:
        return clf_relabeler.get(clf, clf)
    return clf_relabeler.get(clf, clf) + f" ({time_mapper(time)}tuned)"
