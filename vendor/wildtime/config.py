# TODO: outdated, update

config = {
    "dataset": {
        "name": "mimic",
        "domains": [0, 0.5, 1],
        "num_domains": 3,
        "num_classes": 2,
    },
    "method": "erm",  # choices=['agem', 'coral', 'ensemble', 'ewc', 'ft', 'groupdro', 'irm', 'si', 'erm', 'swa']
    "random_seed": 1,  # 'random seed number'
    "device": "cpu",  # choices=['cpu', 'cuda']
    # Training hyperparameters
    "train_update_iter": 10,  # 'train update iter'
    "lr": 0.01,  # 'the base learning rate of the generator'
    "weight_decay": 0.0,  # 'weight decay'
    "mini_batch_size": 32,  # 'mini batch size for SGD'
    "reduced_train_prop": None,  # 'proportion of samples allocated to train at each time step'
    # MIMIC
    "regression": False,  # help='regression task for mimic datasets')
    # Evaluation
    # todo: set value of split_time
    "split_time": 0,  # 'timestep to split ID vs OOD' #
    "load_model": False,  # 'load trained model for evaluation only'
    "save_model": False,  # 'save trained model'
    # LISA and Mixup
    "lisa": False,  # 'train with LISA'
    "mixup": False,  # 'train with vanilla mixup'
    "lisa_start_time": 0,  # 'lisa_start_time'
    "mix_alpha": 2.0,  # 'mix alpha for LISA'
    "cut_mix": False,  # 'use cut mix up'
    # GroupDRO
    "num_groups": 4,  # 'number of windows for Invariant Learning baselines'
    "group_size": 4,  # 'window size for Invariant Learning baselines'
    "non_overlapping": False,  # 'non-overlapping time windows'
    # EWC
    "ewc_lambda": 1.0,  # help='how strong to weigh EWC-loss ("regularisation strength")'
    "gamma": 1.0,  # help='decay-term for old tasks (contribution to quadratic term)'
    "online": False,  # help='"online" (=single quadratic term) or "offline" (=quadratic term per task) EWC'
    "fisher_n": None,  # help='sample size for estimating FI-matrix (if "None", full pass over dataset)'
    "emp_FI": False,  # help='if True, use provided labels to calculate FI ("empirical FI"); else predicted labels'
    # A-GEM
    "buffer_size": 100,  # 'buffer size for A-GEM'
    # CORAL
    "coral_lambda": 1.0,  # 'how strong to weigh CORAL loss'
    # IRM
    "irm_lambda": 1.0,  # 'how strong to weigh IRM penalty loss'
    "irm_penalty_anneal_iters": 0,  # 'number of iterations after which we anneal IRM penalty loss'
    # SI
    "si_c": 0.1,  # 'SI: regularisation strength'
    "epsilon": 0.001,  # 'dampening parameter: bounds "omega" when squared parameter-change goes to 0'
    # Logging, saving, and testing options
    "log_dir": "./checkpoints",  # 'directory for summaries and checkpoints.'
    "num_workers": 0,  # 'number of workers in data generator'
}
