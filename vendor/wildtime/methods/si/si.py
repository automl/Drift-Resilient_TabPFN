import time

import torch
import torch.utils.data
from tqdm import tqdm
from ..base_trainer import BaseTrainer
from ..dataloaders import InfiniteDataLoader
from ..utils import prepare_data, forward_pass


class SI(BaseTrainer):
    """
    Synaptic Intelligence

    Original paper:
        @inproceedings{zenke2017continual,
            title={Continual learning through synaptic intelligence},
            author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
            booktitle={International Conference on Machine Learning},
            pages={3987--3995},
            year={2017},
            organization={PMLR}
        }

    Code adapted from https://github.com/GMvandeVen/continual-learning.
    """

    def __init__(self, args):
        super().__init__(args)
        self.si_c = (
            args.si_c
        )  # -> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = (
            args.epsilon
        )  # -> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

    def __str__(self):
        str_all = f"SI-si_c={self.si_c}-epsilon={self.epsilon}-{self.base_trainer_str}"
        return str_all

    def update_omega(self, W, epsilon):
        """After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)
        """

        # Loop over all parameters
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")

            # Find/calculate new values for quadratic penalty on parameters
            p_prev = getattr(self.network, "{}_SI_prev_task".format(n))
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            omega_add = W[n] / (p_change**2 + epsilon)
            try:
                omega = getattr(self.network, "{}_SI_omega".format(n))
            except AttributeError:
                omega = p.detach().clone().zero_()
            omega_new = omega + omega_add

            # Store these new values in the model
            self.network.register_buffer("{}_SI_prev_task".format(n), p_current)
            self.network.register_buffer("{}_SI_omega".format(n), omega_new)

    def surrogate_loss(self):
        """
        Calculate SI's surrogate loss.
        """
        try:
            losses = []
            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace(".", "__")
                    prev_values = getattr(self.network, "{}_SI_prev_task".format(n))
                    omega = getattr(self.network, "{}_SI_omega".format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)

        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0.0, device=self.args.device)

    def train_step(self, dataloader):
        # Prepare <dicts> to store running importance estimates and parameter-values before update
        W = {}
        p_old = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()

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
            loss = loss + self.si_c * self.surrogate_loss()
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

                # Update running parameter importance estimates in W
                for n, p in self.network.named_parameters():
                    if p.requires_grad:
                        # n = "network." + n
                        n = n.replace(".", "__")
                        if p.grad is not None:
                            W[n].add_(-p.grad * (p.detach() - p_old[n]))
                        p_old[n] = p.detach().clone()
                self.update_omega(W, self.epsilon)
                break

    def _train_online_hook_pre(self):
        # Register starting param-values (needed for "intelligent synapses").
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                self.network.register_buffer(
                    "{}_SI_prev_task".format(n), p.detach().clone()
                )
