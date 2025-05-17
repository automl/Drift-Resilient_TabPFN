import copy
import time
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from tqdm import tqdm

from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass


class EWC(BaseTrainer):
    """
    Elastic Weight Consolidation

    Original paper:
        @article{kirkpatrick2017overcoming,
            title={Overcoming catastrophic forgetting in neural networks},
            author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
            journal={Proceedings of the national academy of sciences},
            volume={114},
            number={13},
            pages={3521--3526},
            year={2017},
            publisher={National Acad Sciences}
        }

    Code adapted from https://github.com/GMvandeVen/continual-learning.
    """

    def __init__(self, args):
        super().__init__(args)
        self.ewc_lambda = (
            args.ewc_lambda
        )  # -> hyperparam: how strong to weigh EWC-loss ("regularization strength")
        self.gamma = (
            args.gamma
        )  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = (
            args.online
        )  # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = (
            args.fisher_n
        )  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = (
            args.emp_FI
        )  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = (
            0  # -> keeps track of number of quadratic loss terms (for "offline EWC")
        )

    def __str__(self):
        str_all = (
            f"EWC-lambda={self.ewc_lambda}-gamma={self.gamma}-online={self.online}-fisher_n={self.fisher_n}"
            f"-emp_FI={self.emp_FI}-{self.base_trainer_str}"
        )
        return str_all

    def estimate_fisher(self):
        """
        After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        """
        est_fisher_info = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                est_fisher_info[n] = p.detach().clone().zero_()

        self.network.eval()

        data_loader = get_data_loader(
            self.train_dataset,
            batch_size=self.mini_batch_size,
            collate_fn=self.train_collate_fn,
        )

        ind = 0
        for index, (x, y) in enumerate(data_loader):
            ind = index
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            x, y = prepare_data(x, y, str(self.train_dataset), self.args.device)
            loss, output, y = forward_pass(
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
            if self.emp_FI:
                label = torch.LongTensor([y]) if type(y) == int else y
                label = label.to(self.args.device)
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            else:
                if self.args.regression:
                    negloglikelihood = F.mse_loss(output, y)
                else:
                    label = output.max(1)[1]
                    negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            self.network.zero_grad()
            negloglikelihood.backward()

            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    n = n.replace(".", "__")
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        est_fisher_info = {n: p / (ind + 1) for n, p in est_fisher_info.items()}

        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                self.network.register_buffer(
                    "{}_EWC_prev_task{}".format(
                        n, "" if self.online else self.EWC_task_count + 1
                    ),
                    p.detach().clone(),
                )
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(
                        self.network, "{}_EWC_estimated_fisher".format(n)
                    )
                    est_fisher_info[n] += self.gamma * existing_values
                self.network.register_buffer(
                    "{}_EWC_estimated_fisher{}".format(
                        n, "" if self.online else self.EWC_task_count + 1
                    ),
                    est_fisher_info[n],
                )

        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        self.network.train()

    def ewc_loss(self):
        if self.EWC_task_count > 0:
            losses = []
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.network.named_parameters():
                    if p.requires_grad:
                        n = n.replace(".", "__")
                        mean = getattr(
                            self.network,
                            "{}_EWC_prev_task{}".format(n, "" if self.online else task),
                        )
                        fisher = getattr(
                            self.network,
                            "{}_EWC_estimated_fisher{}".format(
                                n, "" if self.online else task
                            ),
                        )
                        fisher = self.gamma * fisher if self.online else fisher
                        losses.append((fisher * (p - mean) ** 2).sum())
            return (1.0 / 2) * sum(losses)
        else:
            return torch.tensor(0.0, device=self.args.device)

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
            loss = loss + self.ewc_lambda * self.ewc_loss()
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
                self.estimate_fisher()
                break


def get_data_loader(
    dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False
):
    """
    Return <DataLoader>-object for the provided <DataSet>-object [dataset].
    """
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose(
            [dataset.transform, *dataset.AVAILABLE_TRANSFORMS["augment"]]
        )
    else:
        dataset_ = dataset

    rand_sampler = torch.utils.data.RandomSampler(
        dataset, replacement=True, num_samples=min(batch_size, len(dataset_))
    )
    return DataLoader(
        dataset_,
        sampler=rand_sampler,
        collate_fn=(collate_fn or default_collate),
        drop_last=drop_last,
        **({"num_workers": 1, "pin_memory": True} if cuda else {}),
    )
