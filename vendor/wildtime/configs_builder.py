import argparse

# TODO: outdated, update
parser = argparse.ArgumentParser(description="Wild-Time")

parser.add_argument("--dataset", default="yearbook", choices=["drug", "mimic"])
parser.add_argument(
    "--regression",
    dest="regression",
    action="store_true",
    help="regression task for mimic datasets",
)
parser.add_argument(
    "--prediction_type", type=str, help='MIMIC: "mortality" or "readmission"'
)
parser.add_argument(
    "--method",
    default="ft",
    choices=[
        "agem",
        "coral",
        "ensemble",
        "ewc",
        "ft",
        "groupdro",
        "irm",
        "si",
        "erm",
        "simclr",
    ],
)
parser.add_argument("--device", default=0, type=int, help="gpu id")
parser.add_argument("--random_seed", default=1, type=int, help="random seed number")

# Training hyperparameters
parser.add_argument(
    "--train_update_iter", default=10, type=int, help="train update iter"
)
parser.add_argument(
    "--lr", default=0.01, type=float, help="the base learning rate of the generator"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
parser.add_argument(
    "--mini_batch_size", type=int, default=32, help="mini batch size for SGD"
)
parser.add_argument(
    "--reduced_train_prop",
    type=float,
    default=None,
    help="proportion of samples allocated to train at each time step",
)

# Evaluation
parser.add_argument("--split_time", type=int, help="timestamp to split ID vs OOD")
parser.add_argument(
    "--eval_next_timestamps",
    default=1,
    type=int,
    help="number of future timestamps to evaluate on",
)
parser.add_argument(
    "--load_model",
    dest="load_model",
    action="store_true",
    help="load trained model for evaluation only",
)
parser.add_argument(
    "--eval_all_timestamps",
    dest="eval_all_timestamps",
    action="store_true",
    help="evaluate at ID and OOD time steps",
)

# FT
parser.add_argument(
    "--K", default=1, type=int, help="number of previous timestamps to finetune on"
)

# LISA and Mixup
parser.add_argument("--lisa", dest="lisa", action="store_true", help="train with LISA")
parser.add_argument(
    "--lisa_intra_domain",
    dest="lisa_intra_domain",
    action="store_true",
    help="train with LISA intra domain",
)
parser.add_argument(
    "--mixup", dest="mixup", action="store_true", help="train with vanilla mixup"
)
parser.add_argument("--lisa_start_time", type=int, default=0, help="lisa_start_time")
parser.add_argument("--mix_alpha", type=float, default=2.0, help="mix alpha for LISA")
parser.add_argument(
    "--cut_mix", dest="cut_mix", action="store_true", help="use cut mix up"
)

# GroupDRO
parser.add_argument(
    "--num_groups",
    type=int,
    default=4,
    help="number of windows for Invariant Learning baselines",
)
parser.add_argument(
    "--group_size",
    type=int,
    default=4,
    help="window size for Invariant Learning baselines",
)
parser.add_argument(
    "--non_overlapping",
    dest="non_overlapping",
    action="store_true",
    help="non-overlapping time windows",
)

# EWC
parser.add_argument(
    "--ewc_lambda",
    type=float,
    default=1.0,
    help='how strong to weigh EWC-loss ("regularisation strength")',
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.0,
    help="decay-term for old tasks (contribution to quadratic term)",
)
parser.add_argument(
    "--online",
    dest="online",
    action="store_true",
    help='"online" (=single quadratic term) or "offline" (=quadratic term per task) EWC',
)
parser.add_argument(
    "--fisher_n",
    type=int,
    default=None,
    help='sample size for estimating FI-matrix (if "None", full pass over dataset)',
)
parser.add_argument(
    "--emp_FI",
    dest="emp_FI",
    action="store_true",
    help='if True, use provided labels to calculate FI ("empirical FI"); else predicted labels',
)

# A-GEM
parser.add_argument(
    "--buffer_size", type=int, default=100, help="buffer size for A-GEM"
)

# CORAL
parser.add_argument(
    "--coral_lambda", type=float, default=1.0, help="how strong to weigh CORAL loss"
)

# IRM
parser.add_argument(
    "--irm_lambda", type=float, default=1.0, help="how strong to weigh IRM penalty loss"
)
parser.add_argument(
    "--irm_penalty_anneal_iters",
    type=int,
    default=0,
    help="number of iterations after which we anneal IRM penalty loss",
)

# SI
parser.add_argument(
    "--si_c", type=float, default=0.1, help="SI: regularisation strength"
)
parser.add_argument(
    "--epsilon",
    type=float,
    default=0.001,
    help='dampening parameter: bounds "omega" when squared parameter-change goes to 0',
)

## Logging, saving, and testing options
parser.add_argument(
    "--data_dir", default="./Data", type=str, help="directory for datasets."
)
parser.add_argument(
    "--log_dir",
    default="./checkpoints",
    type=str,
    help="directory for summaries and checkpoints.",
)
parser.add_argument(
    "--results_dir",
    default="./results",
    type=str,
    help="directory for summaries and checkpoints.",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="number of workers in data generator"
)


args = parser.parse_args()

config_dicts = vars(args)

with open(f"./configs/configs_{args.dataset}.py", "a") as dict_file:
    dict_file.write(f"configs_{args.dataset}_{args.method} = ")
    dict_file.write(repr(config_dicts))
    dict_file.write("\n")
