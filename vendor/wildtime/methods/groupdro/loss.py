import numbers

import torch


class LossComputer:
    """
    Adapted from https://github.com/kohpangwei/group_DRO/blob/master/loss.py
    """

    def __init__(
        self,
        dataset,
        criterion,
        is_robust,
        device,
        alpha=0.2,
        gamma=0.1,
        adj=None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
    ):
        self.criterion = criterion
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.dataset = dataset

        self.device = device

        self.num_groups = dataset.num_groups
        self.group_counts = dataset.group_counts().to(device)
        self.group_frac = self.group_counts / self.group_counts.sum()
        self.group_str = str(dataset)

        if adj is not None:
            if isinstance(adj, numbers.Number):
                adj = torch.ones(self.num_groups, dtype=torch.float) * adj
            else:
                adj = torch.tensor(adj, dtype=torch.float)

                assert (
                    adj.shape[0] == self.num_groups
                ), f"The adjustments need to be specified for all {self.num_groups} groups, only specified {adj.shape[0]}."

            self.adj = adj.to(device)
        else:
            self.adj = torch.zeros(self.num_groups).float().to(device)

        if is_robust:
            assert alpha, "alpha must be specified"

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.num_groups).to(device) / self.num_groups
        self.exp_avg_loss = torch.zeros(self.num_groups).to(device)
        self.exp_avg_initialized = torch.zeros(self.num_groups).byte().to(device)

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx
        )

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        elif self.is_robust and self.btl:
            actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(
                self.group_counts
            )  # Why is self.group_counts and not group_count used here?
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(
            self.group_counts
        )  # Why is self.group_counts and not group_count used here?
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (
            1 - self.min_var_weight
        )

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        # group_idx is of shape (bs,) containing the group index of each example in the batch
        assert (
            (group_idx < self.num_groups).all().item()
        ), "The group indices reported contain indices beyond the number of groups defined."

        group_map = (
            group_idx
            == torch.arange(self.num_groups).unsqueeze(1).long().to(self.device)
        ).float()  # (num_groups, bs)
        group_count = group_map.sum(1)  # (4)
        group_denom = (
            group_count + (group_count == 0).float()
        )  # avoid nans, (num_groups)
        group_loss = (group_map @ losses.view(-1)) / group_denom  # (num_groups)
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0
        ).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.num_groups).to(self.device)
        self.update_data_counts = torch.zeros(self.num_groups).to(self.device)
        self.update_batch_counts = torch.zeros(self.num_groups).to(self.device)
        self.avg_group_loss = torch.zeros(self.num_groups).to(self.device)
        self.avg_group_acc = torch.zeros(self.num_groups).to(self.device)
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def update_stats(
        self, actual_loss, group_loss, group_acc, group_count, weights=None
    ):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = (
            prev_weight * self.avg_group_loss + curr_weight * group_loss
        )

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (
            1 / denom
        ) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.0
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict["model_norm_sq"] = model_norm_sq.item()
        stats_dict["reg_loss"] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.num_groups):
            stats_dict[f"avg_loss_group:{idx}"] = self.avg_group_loss[idx].item()
            stats_dict[f"exp_avg_loss_group:{idx}"] = self.exp_avg_loss[idx].item()
            stats_dict[f"avg_acc_group:{idx}"] = self.avg_group_acc[idx].item()
            stats_dict[
                f"processed_data_count_group:{idx}"
            ] = self.processed_data_counts[idx].item()
            stats_dict[f"update_data_count_group:{idx}"] = self.update_data_counts[
                idx
            ].item()
            stats_dict[f"update_batch_count_group:{idx}"] = self.update_batch_counts[
                idx
            ].item()

        stats_dict["avg_actual_loss"] = self.avg_actual_loss.item()
        stats_dict["avg_per_sample_loss"] = self.avg_per_sample_loss.item()
        stats_dict["avg_acc"] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(
            f"Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n"
        )
        logger.write(f"Average sample loss: {self.avg_actual_loss.item():.3f}  \n")
        logger.write(f"Average acc: {self.avg_acc.item():.3f}  \n")
        for group_idx in range(self.num_groups):
            logger.write(
                f"[n = {int(self.processed_data_counts[group_idx])}]:\t"
                f"loss = {self.avg_group_loss[group_idx]:.3f}  "
                f"exp loss = {self.exp_avg_loss[group_idx]:.3f}  "
                f"adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx] / torch.sqrt(self.group_counts)[group_idx]:.3f}  "
                f"adv prob = {self.adv_probs[group_idx]:3f}   "
                f"acc = {self.avg_group_acc[group_idx]:.3f}\n"
            )
        logger.flush()
