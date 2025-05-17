import warnings

import numpy as np
from hyperopt import hp

from copy import deepcopy

from tabpfn.datasets import DistributionShiftDataset
from ...tabular_baselines.utils import (
    eval_complete_f,
    preprocess_and_impute,
    get_random_seed,
)

from tabpfn.utils import target_is_multiclass

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

## Wildtime
param_grid_hyperopt = {
    "train_update_iter": hp.pchoice(
        "train_update_iter",
        [
            (0.1, 500),
            (0.1, 1000),
            (0.1, 2000),
            (0.15, 3000),
            (0.15, 4000),
            (0.25, 5000),
            (0.075, 6000),
            (0.075, 7000),
        ],
    ),
    "lr": hp.loguniform("lr", -14, -4),  # hp.uniform("lr", 0.000001, 0.00005),
    "use_scheduler": hp.pchoice("use_scheduler", [(0.7, True), (0.3, False)]),
    "ft_scheduler_gamma": hp.uniform("ft_scheduler_gamma", 0.9, 1.0),
    "weight_decay": hp.pchoice(
        "weight_decay", [(0.25, 0.0), (0.5, 1e-5), (0.25, 1e-2)]
    ),
    "early_stop": hp.pchoice("early_stop", [(0.4, True), (0.6, False)]),
    "early_stop_holdout": hp.pchoice(
        "early_stop_holdout", [(0.5, 0.1), (0.25, 0.15), (0.25, 0.2)]
    ),
    "early_stop_patience": hp.randint("early_stop_patience", 10, 30),
    "mini_batch_size": hp.choice("mini_batch_size", [32, 64, 128, 256]),
}


def wildtime_metric(
    train_ds,
    test_ds,
    metric_used,
    method_name,
    network="MLP",
    device="cpu",
    max_time=300,
    no_tune=None,
    random_state=0,
    pipeline_kwargs=None,
    **kwargs,
):
    assert (
        type(train_ds).__name__ == DistributionShiftDataset.__name__
    ), "Dataset must be a DistributionShift Dataset"

    from vendor.wildtime.baseline_trainer import get_model as wildtime_get_model

    method_dict = {
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
    assert (
        method_name in method_dict
    ), f"Method {method_name} not found in {method_dict.keys()}"

    network_dict = {"MLP", "FTT"}
    assert (
        network in network_dict
    ), f"Network {network} not found in {network_dict.keys()}"

    base_config = {
        "train_update_iter": 5000,
        "mini_batch_size": 128,
        "weight_decay": 1e-5,
        "device": device,
        "regression": False,
        "load_model": False,
        "save_model": False,
        "log_dir": ".",
        "num_workers": 0,
        "network": network,
        "step_callback_frequency": 50,
        "early_stop": False,
        "early_stop_holdout": 0.1,
        "early_stop_patience": 30,
        "use_scheduler": True,
        "ft_scheduler_gamma": 0.96,
        "append_domain_as_feature": pipeline_kwargs["dist_shift_append_domain"],
    }

    if network == "MLP":
        use_one_hot = True

        base_config.update(
            {
                "lr": 0.00002,
                "hparams": {
                    "mlp_width": 128,
                    "mlp_dropout": 0.5,
                    "mlp_hidden_layers": 2,
                    "use_batch_norm": True,
                    "activation": "leaky_relu",
                },
            }
        )
        param_grid_hyperopt.update(
            {
                "hparams": {
                    "mlp_width": hp.pchoice(
                        "mlp_width", [(0.2, 32), (0.2, 64), (0.4, 128), (0.2, 256)]
                    ),
                    "mlp_dropout": hp.pchoice(
                        "mlp_dropout", [(0.5, 0.0), (0.25, 0.2), (0.25, 0.5)]
                    ),
                    "mlp_hidden_layers": hp.randint("mlp_hidden_layers", 0, 5),
                    "activation": hp.pchoice(
                        "activation",
                        [
                            (0.2, "relu"),
                            (0.5, "leaky_relu"),
                            (0.1, "prelu"),
                            (0.2, "tanh"),
                        ],
                    ),
                    "use_batch_norm": hp.choice("use_batch_norm", [True, False]),
                }
            }
        )
    elif network == "FTT":
        use_one_hot = False

        base_config.update(
            {
                "lr": 0.000003,
                "hparams": {
                    "cat_embed_dropout": 0.1,
                    "use_cat_bias": True,
                    "cat_embed_activation": None,
                    "full_embed_dropout": False,
                    "shared_embed": False,
                    "add_shared_embed": False,
                    "frac_shared_embed": 0.0,
                    "cont_norm_layer": "batchnorm",
                    "cont_embed_dropout": 0.1,
                    "use_cont_bias": True,
                    "cont_embed_activation": None,
                    "input_dim": 64,
                    "kv_compression_factor": 1.0,
                    "kv_sharing": False,
                    "use_qkv_bias": False,
                    "n_heads": 4,
                    "n_blocks": 4,
                    "attn_dropout": 0.2,
                    "ff_dropout": 0.1,
                    "ff_factor": 1.33,
                    "transformer_activation": "relu",
                    "mlp_hidden_dims": [128, 64, 32],
                    "mlp_activation": "tanh",
                    "mlp_dropout": 0.5,
                    "mlp_batchnorm": True,
                    "mlp_batchnorm_last": True,
                    "mlp_linear_first": True,
                },
            }
        )
        param_grid_hyperopt.update(
            {
                "hparams": {
                    # categorical embeddings
                    "cat_embed_dropout": hp.uniform("cat_embed_dropout", 0, 0.5),
                    "use_cat_bias": hp.pchoice(
                        "use_cat_bias", [(0.75, True), (0.25, False)]
                    ),
                    "cat_embed_activation": hp.pchoice(
                        "cat_embed_activation",
                        [
                            (0.6, None),
                            (0.1, "tanh"),
                            (0.1, "relu"),
                            (0.1, "leaky_relu"),
                            (0.1, "gelu"),
                        ],
                    ),
                    "full_embed_dropout": hp.choice(
                        "full_embed_dropout", [True, False]
                    ),
                    # Shared embeddings are currently causing errors in the training
                    # They are therefore disables for now
                    #'shared_embed': hp.choice('shared_embed', [True, False]),
                    #'add_shared_embed': hp.choice('add_shared_embed', [True, False]),
                    #'frac_shared_embed': hp.uniform('frac_shared_embed', 0, 1),
                    # continuous embeddings
                    "cont_norm_layer": hp.pchoice(
                        "cont_norm_layer",
                        [(0.25, "layernorm"), (0.5, "batchnorm"), (0.25, None)],
                    ),
                    "cont_embed_dropout": hp.uniform("cont_embed_dropout", 0, 0.5),
                    "use_cont_bias": hp.pchoice(
                        "use_cont_bias", [(0.75, True), (0.25, False)]
                    ),
                    "cont_embed_activation": hp.pchoice(
                        "cont_embed_activation",
                        [
                            (0.6, None),
                            (0.1, "tanh"),
                            (0.1, "relu"),
                            (0.1, "leaky_relu"),
                            (0.1, "gelu"),
                        ],
                    ),
                    # FTTransformer hyperparameters
                    # Values less than 0.5 can crash the training on CUDA specifying that optimizer.step() failed.
                    # For the crash to happen kv_sharing has to be False, however.
                    # Exemplary params for SWA method:
                    # params = {'hparams': {'attn_dropout': 0.2, 'cat_embed_activation': 'gelu', 'cat_embed_dropout': 0.1, 'cont_embed_activation': 'gelu', 'cont_embed_dropout': 0.1, 'cont_norm_layer': 'layernorm', 'ff_dropout': 0.1, 'ff_factor': 1.33, 'full_embed_dropout': True, 'input_dim': 32, 'kv_compression_factor': 0.49, 'kv_sharing': False, 'mlp_activation': 'gelu', 'mlp_batchnorm': False, 'mlp_batchnorm_last': True, 'mlp_dropout': 0.31257425301562447, 'mlp_hidden_dims': (32,), 'mlp_linear_first': False, 'n_blocks': 6, 'n_heads': 2, 'transformer_activation': 'leaky_relu', 'use_cat_bias': True, 'use_cont_bias': True, 'use_qkv_bias': False}, 'lr': 0.005, 'scheduler_frequency': 700, 'scheduler_gamma': 1, 'swa_portion': 0.75, 'train_update_iter': 5000, 'use_scheduler': True, 'weight_decay': 0.01}
                    # Traceback:
                    #   File ".../lib/python3.10/site-packages/torch/optim/adamw.py", line 446, in _multi_tensor_adamw
                    #     torch._foreach_add_(device_exp_avgs, device_grads, alpha=1 - beta1)
                    # RuntimeError: CUDA error: an illegal memory access was encountered
                    "kv_compression_factor": hp.uniform(
                        "kv_compression_factor", 0.5, 1
                    ),
                    "kv_sharing": hp.pchoice(
                        "kv_sharing", [(0.25, True), (0.75, False)]
                    ),
                    "use_qkv_bias": hp.pchoice(
                        "use_qkv_bias", [(0.25, True), (0.75, False)]
                    ),
                    "input_dim": hp.pchoice(
                        "input_dim", [(0.25, 32), (0.5, 64), (0.25, 128)]
                    ),
                    "n_heads": hp.pchoice(
                        "n_heads", [(0.25, 2), (0.5, 4), (0.25, 8)]
                    ),  # input_dim must be divisible by n_heads
                    "n_blocks": hp.pchoice(
                        "n_blocks", [(0.25, 2), (0.5, 4), (0.25, 6)]
                    ),
                    "attn_dropout": hp.uniform("attn_dropout", 0, 0.3),
                    "ff_dropout": hp.uniform("ff_dropout", 0, 0.5),
                    "ff_factor": hp.uniform("ff_factor", 1, 5),
                    "transformer_activation": hp.choice(
                        "transformer_activation",
                        ["tanh", "relu", "leaky_relu", "gelu", "geglu", "reglu"],
                    ),
                    # MLP hyperparameters
                    "mlp_hidden_dims": hp.pchoice(
                        "mlp_hidden_dims",
                        [(0.25, [64, 32]), (0.25, [128, 64]), (0.5, [128, 64, 32])],
                    ),
                    # Example sizes. Adjust if needed.
                    "mlp_activation": hp.choice(
                        "mlp_activation", ["tanh", "relu", "leaky_relu", "gelu"]
                    ),
                    "mlp_dropout": hp.pchoice(
                        "mlp_dropout", [(0.25, 0.0), (0.75, 0.5)]
                    ),
                    "mlp_batchnorm": hp.pchoice(
                        "mlp_batchnorm", [(0.75, True), (0.25, False)]
                    ),
                    "mlp_batchnorm_last": hp.choice(
                        "mlp_batchnorm_last", [True, False]
                    ),
                    "mlp_linear_first": hp.pchoice(
                        "mlp_linear_first", [(0.75, True), (0.25, False)]
                    ),
                }
            }
        )

    if method_name in {"coral", "groupdro", "irm"}:
        base_config.update({"group_size": 4, "non_overlapping": False})
        param_grid_hyperopt.update(
            {
                "group_size": hp.randint("group_size", 1, 6),
                "non_overlapping": hp.choice("non_overlapping", [False, True]),
            }
        )

    if method_name == "erm":
        base_config.update({"method": "erm"})

    elif method_name == "erm_lisa":
        base_config.update(
            {
                "method": "erm",
                "lisa": True,
                "lisa_start_time": 0,
                "mix_alpha": 2.0,
                "cut_mix": True,
            }
        )
        param_grid_hyperopt.update(
            {
                "mix_alpha": hp.uniform("mix_alpha", 0.5, 4.0),
                "cut_mix": hp.choice("cut_mix", [False, True]),
            }
        )

    elif method_name == "erm_mixup":
        base_config.update({"method": "erm", "mixup": True, "mix_alpha": 2.0})
        param_grid_hyperopt.update({"mix_alpha": hp.uniform("mix_alpha", 0.5, 4.0)})

    elif method_name == "ewc":
        base_config.update(
            {
                "method": "ewc",
                "gamma": 1.0,
                "fisher_n": None,
                "emp_FI": False,
                "ewc_lambda": 0.5,
                "online": True,
            }
        )
        param_grid_hyperopt.update(
            {
                "gamma": hp.uniform("gamma", 1.0, 2.0),
                "ewc_lambda": hp.uniform("ewc_lambda", 0.5, 2.0),
            }
        )

    elif method_name == "agem":
        base_config.update({"method": "agem", "buffer_size": 1000})
        # param_grid_hyperopt.update({'buffer_size': 1000})

    elif method_name == "ft":
        base_config.update({"method": "ft"})

    elif method_name == "groupdro":
        base_config.update(
            {
                "method": "groupdro",
                "group_loss_adjustments": 0.5,
                "group_loss_btl": False,
            }
        )
        param_grid_hyperopt.update(
            {
                "group_loss_adjustments": hp.pchoice(
                    "group_loss_adjustments",
                    [(0.4, None), (0.2, 0.1), (0.2, 0.5), (0.2, 1.0)],
                ),
                "group_loss_btl": hp.pchoice(
                    "group_loss_btl", [(0.25, True), (0.75, False)]
                ),
            }
        )

    elif method_name == "swa":
        base_config.update({"method": "swa", "swa_portion": 0.75, "swa_lr_factor": 4})
        param_grid_hyperopt.update(
            {
                "swa_portion": hp.uniform("swa_portion", 0.5, 0.9),
                "swa_lr_factor": hp.randint("swa_lr_factor", 1, 6),
            }
        )

    elif method_name == "irm":
        base_config.update(
            {"method": "irm", "irm_lambda": 1.0, "irm_penalty_anneal_iters": 0}
        )
        param_grid_hyperopt.update(
            {
                "irm_lambda": hp.randint("irm_lambda", 1, 100),
                "irm_penalty_anneal_iters": hp.choice(
                    "irm_penalty_anneal_iters", [0, 250, 500, 750, 1000]
                ),
            }
        )

    elif method_name == "si":
        base_config.update({"method": "si", "si_c": 0.1, "epsilon": 0.001})
        param_grid_hyperopt.update(
            {
                "si_c": hp.uniform("si_c", 0.05, 0.2),
                "epsilon": hp.uniform("epsilon", 0.0005, 0.002),
            }
        )

    elif method_name == "coral":
        base_config.update({"method": "coral", "coral_lambda": 0.9})
        param_grid_hyperopt.update(
            {"coral_lambda": hp.uniform("coral_lambda", 0.1, 1.0)}
        )

    task_type = train_ds.task_type

    (
        x,
        y,
        test_xs,
        attribute_names,
        categorical_feats,
        preprocess_kwargs,
    ) = preprocess_and_impute(
        train_ds=train_ds,
        test_ds=test_ds,
        impute=True,
        one_hot=use_one_hot,
        standardize=True,
        is_classification=target_is_multiclass(task_type),
        dist_shift_append_domain=True,  # We need to force append here, but will remove the indices in Wild-Time again
    )

    all_domains = np.concatenate(
        [x["dist_shift_domain"]]
        + [cur_x["dist_shift_domain"] for cur_x in test_xs.values()],
        axis=0,
    )
    unique_domains = np.sort(np.unique(all_domains))

    categorical_mask = np.zeros(len(attribute_names), dtype=bool)
    categorical_mask[categorical_feats] = True

    ds_meta_data = {
        "name": train_ds.name,
        "domains": unique_domains.tolist(),
        "num_domains": unique_domains.shape[0],
        "categorical_cols": [
            attr for attr, mask in zip(attribute_names, categorical_mask) if mask
        ],
        "continuous_cols": [
            attr for attr, mask in zip(attribute_names, ~categorical_mask) if mask
        ],
    }

    random_seed = get_random_seed(random_state, y)

    # TODO: currently incompatible with ray, that does not inherit the wandb run of the outer scope
    """
    import wandb

    if wandb.run:

        def step_callback(trainer, **metrics):
            log_msg = {
                f"wildtime/{method_name}/{train_ds.name}/seed_{random_seed}/{trainer.id}/{metric_name}": metric
                for metric_name, metric in metrics.items()
            }

            wandb.log(log_msg)

    else:
        step_callback = None
    """
    step_callback = None

    base_config.update(
        {
            "dataset": ds_meta_data,
            "random_seed": random_seed,
            "step_callback": step_callback,
        }
    )

    def recursive_update(d, u):
        """
        Recursively update dictionary d with keys from u.
        If key from u exists in d and both u[key] and d[key] are dict, update d[key] recursively.
        Otherwise, set d[key] = u[key].
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def clf_(**params):
        params = recursive_update(deepcopy(base_config), params)

        return wildtime_get_model(
            **params,
        )

    return eval_complete_f(
        x,
        y,
        test_xs,
        task_type,
        preprocess_kwargs,
        "wildtime",
        param_grid_hyperopt,
        clf_,
        metric_used,
        max_time,
        no_tune,
        random_state,
    )
