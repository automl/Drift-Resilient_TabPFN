import time
import torch
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, split_into_groups


class DeepCORAL(BaseTrainer):
    """
    Deep CORAL

    This algorithm was originally proposed as an unsupervised domain adaptation algorithm.

    Original paper:
        @inproceedings{sun2016deep,
          title={Deep CORAL: Correlation alignment for deep domain adaptation},
          author={Sun, Baochen and Saenko, Kate},
          booktitle={European Conference on Computer Vision},
          pages={443--450},
          year={2016},
          organization={Springer}
        }

    Code adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/deepCORAL.py.
    """

    def __init__(self, args):
        super().__init__(args)
        self.coral_lambda = args.coral_lambda

    def __str__(self):
        return f"DeepCORAL-coral_lambda={self.coral_lambda}-{self.base_trainer_str}"

    def coral_penalty(self, x, y):
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

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
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()

            classification_loss, logits, y = forward_pass(
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
            coral_loss = torch.zeros(1).to(self.args.device)
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group + 1, n_groups_per_batch):
                    coral_loss += self.coral_penalty(
                        logits[group_indices[i_group]].squeeze(0),
                        logits[group_indices[j_group]].squeeze(0),
                    )
            if n_groups_per_batch > 1:
                coral_loss /= n_groups_per_batch * (n_groups_per_batch - 1) / 2

            loss = classification_loss + self.coral_lambda * coral_loss
            loss_sum += loss.item()

            self.optimizer.zero_grad()
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
