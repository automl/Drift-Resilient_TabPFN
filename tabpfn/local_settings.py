"""By importing this file you setup you configuration for the cluster.

You set this with the environment variable `TABPFN_CLUSTER_SETUP`. If there
is no such variable, it will use the local setup.
"""
import os
from pathlib import Path
from typing import Union

global base_path
global log_folder
global wandb_project_prefix
global wandb_entity
global openml_path
global kaggle_cache_path
global model_string_config
global local_model_path

setup = os.environ.get("TABPFN_CLUSTER_SETUP")
wandb_entity = "WANDB_ENTITY"
wandb_project_prefix = "WANDB_PROJECT_PREFIX"

if setup == "LOCAL" or setup is None:
    # Local setup for usage of the repo on, e.g., a laptop.
    model_string_config = "LOCAL"
    root_dir = Path(__file__).parent.resolve()
    openml_path = root_dir / "openml_cache"
    base_path = root_dir / "results"
    kaggle_cache_path = root_dir / "kaggle_cache"
    local_model_path = root_dir / "model_cache"
    container_path = None
    log_folder = root_dir / "logs"
else:
    raise ValueError(f"Unknown Cluster Setup {setup=}")


def get_wandb_project(task_type="multiclass"):
    if task_type == "multiclass":
        return wandb_project_prefix
    else:
        return wandb_project_prefix + f"-{task_type}"


DEFAULT_OPENML_PATH = openml_path


def set_openml_config_path(path: Union[str, Path] = DEFAULT_OPENML_PATH) -> None:
    """Sets the path where openml stores its cache.

    :param path: The path to the openml cache
    """
    path = Path(path).absolute()
    if not path.exists():
        path.mkdir(parents=True)
    # Future proofing for openml>=0.14.0
    # See https://github.com/openml/automlbenchmark/pull/579/files
    try:
        openml.config.set_cache_directory(str(path))  # type: ignore
    except AttributeError:
        openml.config.set_root_cache_directory(str(path))  # type: ignore


try:
    os.makedirs(f"{base_path}/results", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/multiclass", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/regression", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/quantile_regression", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/dist_shift_multiclass", exist_ok=True)
    os.makedirs(f"{base_path}/results/models_diff", exist_ok=True)
    os.makedirs(f"{base_path}/wandb", exist_ok=True)
    os.makedirs(f"{openml_path}/cached_lists", exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(openml_path, exist_ok=True)
    if kaggle_cache_path is not None:
        os.makedirs(kaggle_cache_path, exist_ok=True)
except PermissionError as e:
    print(
        f"Could not create the necessary folders. Please make sure that you have the necessary permissions. {e}"
    )
    # raise e

try:
    import openml

    set_openml_config_path(openml_path)
except ImportError:
    pass
