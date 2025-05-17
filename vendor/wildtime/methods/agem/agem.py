import time
import numpy as np
import torch
from tqdm import tqdm

from .buffer import Buffer
from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass


class AGEM(BaseTrainer):
    """
    Averaged Gradient Episodic Memory (A-GEM)

    Code adapted from https://github.com/aimagelab/mammoth.

    Original Paper:

        @article{chaudhry2018efficient,
        title={Efficient lifelong learning with a-gem},
        author={Chaudhry, Arslan and Ranzato, Marc'Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
        journal={arXiv preprint arXiv:1812.00420},
        year={2018}
        }
    """

    def __init__(self, args):
        super().__init__(args)

        self.buffer = Buffer(self.args.buffer_size, self.args.device)

    def __str__(self):
        return f"AGEM-buffer_size={self.args.buffer_size}-{self.base_trainer_str}"

    def init_model(self, X, y):
        super().init_model(X, y)

        self.grad_dims = []
        for param in self.network.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.args.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.args.device)

    def end_task(self, dataloader):
        sample = next(iter(dataloader))
        cur_x, cur_y = sample

        cur_x, cur_y = prepare_data(
            cur_x, cur_y, str(self.train_dataset), self.args.device
        )
        self.buffer.add_data(examples=cur_x, labels=cur_y)

    def train_step(self, dataloader):
        self.network.train()
        self.early_stop_reset()
        loss_sum = 0
        step_time_sum = 0
        tqdm_iter = tqdm(total=self.train_update_iter)

        step_scheduler = self.scheduler(self.ft_scheduler.get_last_lr()[0])
        for step, (x, y) in enumerate(dataloader):
            tqdm_iter.update()

            start_time = time.time()

            x, y = prepare_data(x, y, str(self.train_dataset), self.args.device)

            self.network.zero_grad()
            loss, logits, y = forward_pass(
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
            loss_sum += loss.item()

            loss.backward()

            if not self.buffer.is_empty():
                store_grad(self.network.parameters, self.grad_xy, self.grad_dims)

                buf_data = self.buffer.get_data(self.mini_batch_size, transform=None)
                if len(buf_data) > 2:
                    buf_inputs = [buf_data[0], buf_data[1]]
                    buf_labels = buf_data[2]
                else:
                    buf_inputs, buf_labels = buf_data
                buf_inputs, buf_labels = prepare_data(
                    buf_inputs, buf_labels, str(self.train_dataset), self.args.device
                )

                self.network.zero_grad()
                penalty, buff_outputs, buf_labels = forward_pass(
                    buf_inputs,
                    buf_labels,
                    self.train_dataset,
                    self.network,
                    self.criterion,
                    self.lisa,
                    self.mixup,
                    self.cut_mix,
                    self.args.device,
                    self.mix_alpha,
                )
                penalty.backward()
                store_grad(self.network.parameters, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                    overwrite_grad(self.network.parameters, g_tilde, self.grad_dims)
                else:
                    overwrite_grad(
                        self.network.parameters, self.grad_xy, self.grad_dims
                    )

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
                self.end_task(dataloader)
                break


def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[: count + 1])
            this_grad = newgrad[begin:end].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger
