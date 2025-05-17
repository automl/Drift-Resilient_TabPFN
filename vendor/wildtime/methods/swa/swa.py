import time
from tqdm import tqdm
import torch
from ..utils import prepare_data, forward_pass
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..base_trainer import BaseTrainer


class SWA(BaseTrainer):
    """
    Stochastic Weighted Averaging

    Original paper:
        @article{izmailov2018averaging,
            title={Averaging weights leads to wider optima and better generalization},
            author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
            journal={arXiv preprint arXiv:1803.05407},
            year={2018}
        }
    """

    def __init__(self, args):
        super().__init__(args)

    def init_model(self, X, y):
        super().init_model(X, y)
        self.base_network = self.network
        self.base_scheduler = self.scheduler

        self.network = torch.optim.swa_utils.AveragedModel(
            self.network, device=self.args.device
        )
        self.scheduler = torch.optim.swa_utils.SWALR(
            self.optimizer,
            swa_lr=self.args.swa_lr_factor * self.args.lr,
            anneal_strategy="cos",
        )

    def train_step(self, dataloader):
        self.base_network.train()
        self.network.train()
        self.early_stop_reset()
        loss_sum = 0
        step_time_sum = 0
        tqdm_iter = tqdm(total=self.train_update_iter)

        step_scheduler = self.base_scheduler(self.ft_scheduler.get_last_lr()[0])
        base_early_stop = False
        init_swa_training = True

        for step, (x, y) in enumerate(dataloader):
            tqdm_iter.update()

            swa_training = step > int(self.train_update_iter * self.args.swa_portion)

            if base_early_stop and not swa_training:
                continue

            if swa_training and init_swa_training:
                init_swa_training = False
                self.early_stop_load_checkpoint()
                self.early_stop_reset()

            start_time = time.time()

            x, y = prepare_data(x, y, str(self.train_dataset), self.args.device)
            loss, logits, y = forward_pass(
                x,
                y,
                self.train_dataset,
                self.base_network,
                self.criterion,
                self.lisa,
                self.mixup,
                self.cut_mix,
                self.args.device,
                self.mix_alpha,
            )
            loss_sum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if swa_training:
                lr = self.scheduler.get_last_lr()[0]

                self.network.update_parameters(self.base_network)
                self.scheduler.step()
            else:
                lr = step_scheduler.get_last_lr()[0]

                step_scheduler.step()

            step_time = time.time() - start_time
            step_time_sum += step_time

            early_stop = False
            if step != 0 and step % self.step_callback_frequency == 0:
                if swa_training:
                    self.update_swa_bn()
                    early_stop, holdout_loss = self.early_stop_callback(
                        self.network, step
                    )
                else:
                    base_early_stop, holdout_loss = self.early_stop_callback(
                        self.base_network, step
                    )

                if self.step_callback:
                    callback_metrics = {
                        "trainer": self,
                        "step": step,
                        "step_time": step_time,
                        "mean_loss": loss.item(),
                        "lr": lr,
                        "holdout_loss": holdout_loss,
                    }

                    self.step_callback(**callback_metrics)

            tqdm_iter.set_postfix(
                {
                    "loss": loss.item(),
                    "mean_loss": loss_sum / (step + 1),
                    "step_time": f"{step_time:5.6f}",
                    "mean_step_time": f"{step_time_sum / (step + 1):5.6f}",
                    "lr": lr,
                }
            )

            if step == self.train_update_iter - 1 or early_stop:
                self.early_stop_load_checkpoint()
                self.network.update_parameters(self.base_network)
                self.update_swa_bn()

                self.ft_scheduler.step()
                break

    def get_checkpoint(self):
        return {
            "network": self.network.state_dict(),
            "base_network": self.base_network.state_dict(),
        }

    def load_checkpoint(self, checkpoint):
        self.network.load_state_dict(checkpoint["network"], strict=False)
        self.base_network.load_state_dict(checkpoint["base_network"], strict=False)

    def save_model(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa"
        torch.save(self.get_checkpoint(), swa_model_path)
        print(f"Saving model at timestamp {timestamp} to path {swa_model_path}...\n")

    def load_model(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa"
        self.load_checkpoint(torch.load(swa_model_path))

    def train_online(self):
        raise NotImplementedError(
            "SWA for online training is currently not implemented."
        )

    def get_swa_model_copy(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa_copy"
        torch.save(self.network, swa_model_path)
        return torch.load(swa_model_path)

    def update_swa_bn(self):
        finite_train_dataloader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.mini_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_collate_fn,
            drop_last=True,
        )
        torch.optim.swa_utils.update_bn(
            finite_train_dataloader, self.network, self.args.device
        )

    def __str__(self):
        return f"SWA-{self.base_trainer_str}"
