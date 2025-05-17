import time
import torch
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..groupdro.loss import LossComputer
from ..utils import prepare_data, forward_pass


class GroupDRO(BaseTrainer):
    """
    Group distributionally robust optimization.

    Original paper:
        @inproceedings{sagawa2019distributionally,
          title={Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }
    """

    def __init__(self, args):
        super().__init__(args)
        self.group_size = args.group_size

    def __str__(self):
        return f"GroupDRO-group_size={self.group_size}-{self.base_trainer_str}"

    def init_model(self, X, y):
        super().init_model(X, y)

        self.train_dataset.current_time = self.train_dataset.ENV[0]
        self.loss_computer = LossComputer(
            self.train_dataset,
            self.criterion,
            is_robust=True,
            device=self.args.device,
            adj=self.args.group_loss_adjustments,
            btl=self.args.group_loss_btl,
        )

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
            loss = self.loss_computer.loss(logits, y, g, is_training=True)
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
