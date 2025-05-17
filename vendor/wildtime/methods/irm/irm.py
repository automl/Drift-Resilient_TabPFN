import time
import torch
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, split_into_groups


class IRM(BaseTrainer):
    """
    Invariant risk minimization.

    Original paper:
        @article{arjovsky2019invariant,
          title={Invariant risk minimization},
          author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
          journal={arXiv preprint arXiv:1907.02893},
          year={2019}
        }

    Code adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/IRM.py.
    """

    def __init__(self, args):
        super().__init__(args)
        self.irm_lambda = args.irm_lambda
        self.irm_penalty_anneal_iters = args.irm_penalty_anneal_iters
        self.scale = torch.tensor(1.0).requires_grad_()

    def __str__(self):
        return (
            f"IRM-irm_lambda={self.irm_lambda}-irm_penalty_anneal_iters={self.irm_penalty_anneal_iters}"
            f"-{self.base_trainer_str}"
        )

    def irm_penalty(self, losses):
        grad_1 = torch.autograd.grad(
            losses[0::2].mean(), [self.scale], create_graph=True
        )[0]
        grad_2 = torch.autograd.grad(
            losses[1::2].mean(), [self.scale], create_graph=True
        )[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def train_step(self, dataloader):
        self.network.train()
        self.early_stop_reset()
        loss_sum = 0
        step_time_sum = 0
        tqdm_iter = tqdm(total=self.train_update_iter)

        step_scheduler = self.scheduler(self.ft_scheduler.get_last_lr()[0])
        for step, (x, y, g) in enumerate(dataloader):
            tqdm_iter.update()

            start_time = time.time()

            x, y = prepare_data(x, y, str(self.train_dataset), self.args.device)
            g = g.squeeze(1).to(self.args.device)

            self.network.zero_grad()
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()
            avg_loss = 0.0
            penalty = 0.0
            _, logits, y = forward_pass(
                x,
                y,
                self.train_dataset,
                self.network,
                self.criterion,
                self.lisa,
                self.mixup,
                self.cut_mix,
                self.args.device,
                self.mix_alpha,
            )
            for i_group in group_indices:
                group_losses = self.criterion(self.scale * logits[i_group], y[i_group])
                if group_losses.numel() > 0:
                    avg_loss += group_losses.mean()
                penalty += self.irm_penalty(group_losses)
            avg_loss /= n_groups_per_batch
            penalty /= n_groups_per_batch

            if step >= self.irm_penalty_anneal_iters - 1:
                penalty_weight = self.irm_lambda
            else:
                penalty_weight = 1.0

            loss = avg_loss + penalty * penalty_weight
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()

            step_time = time.time() - start_time
            step_time_sum += step_time

            early_stop = False
            if step != 0 and step % self.step_callback_frequency == 0:
                early_stop, holdout_loss = self.early_stop_callback(self.network, step)

                if self.step_callback:
                    callback_metrics = {
                        "trainer": self,
                        "step": step,
                        "step_time": step_time,
                        "mean_loss": loss.item(),
                        "lr": step_scheduler.get_last_lr()[0],
                        "holdout_loss": holdout_loss,
                    }

                    self.step_callback(**callback_metrics)

            tqdm_iter.set_postfix(
                {
                    "loss": loss.item(),
                    "mean_loss": loss_sum / (step + 1),
                    "step_time": f"{step_time:5.6f}",
                    "mean_step_time": f"{step_time_sum / (step + 1):5.6f}",
                    "lr": step_scheduler.get_last_lr()[0],
                }
            )

            step_scheduler.step()

            if step == self.train_update_iter - 1 or early_stop:
                self.early_stop_load_checkpoint()
                self.ft_scheduler.step()
                break
