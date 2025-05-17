import torch
import numpy as np
import random

from sklearn.metrics import roc_auc_score, accuracy_score
from torch.autograd import Variable

from .lisa import lisa
from .mixup import mixup_data, mixup_criterion


def prepare_data(x, y, dataset_name: str, device: str):
    x = x.to(device)
    y = y.to(device)

    return x, y


def forward_pass(
    x,
    y,
    dataset,
    network,
    criterion,
    use_lisa: bool,
    use_mixup: bool,
    cut_mix: bool,
    device: str,
    mix_alpha=2.0,
):
    if use_lisa:
        sel_x, sel_y = lisa(
            x,
            y,
            dataset=dataset,
            mix_alpha=mix_alpha,
            num_classes=dataset.num_classes,
            time_idx=dataset.current_time,
            device=device,
            cut_mix=cut_mix,
        )
        logits = network(sel_x)
        y = torch.argmax(sel_y, dim=1)
        loss = criterion(logits, y)

    elif use_mixup:
        x, y_a, y_b, lam = mixup_data(x, y, device=device, mix_alpha=mix_alpha)
        x, y_a, y_b = map(Variable, (x, y_a, y_b))
        logits = network(x)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

    else:
        logits = network(x)
        loss = criterion(logits, y)

    return loss, logits, y


def split_into_groups(g):
    """
    From https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/wilds/common/utils.py#L40.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def get_collate_functions(args, train_dataset):
    train_collate_fn = None
    eval_collate_fn = None

    return train_collate_fn, eval_collate_fn


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auc_metric(pred, target, multi_class="ovo", numpy=False, should_raise=False):
    lib = np if numpy else torch
    try:
        if not numpy:
            target = torch.tensor(target) if not torch.is_tensor(target) else target
            pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

            target = target.cpu()
            pred = pred.cpu()

        if len(lib.unique(target)) > 2:
            score = roc_auc_score(target, pred, multi_class=multi_class)
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]

            score = roc_auc_score(target, pred)

        return score if numpy else torch.tensor(score)
    except ValueError as e:
        if should_raise:
            raise e
        print(e)
        return np.nan if numpy else torch.tensor(np.nan)


def accuracy_metric(pred, target):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    target = target.cpu()
    pred = pred.cpu()

    if len(pred.shape) == 2:
        score = accuracy_score(target, torch.argmax(pred, -1))
    else:
        score = accuracy_score(target, pred[:, 1] > 0.5)

    return torch.tensor(score)
