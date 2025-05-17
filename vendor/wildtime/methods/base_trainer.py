import copy
import os
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data.datasets import DatasetMode
from ..networks.mlp import MLP
from ..networks.ftt import FTT

from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import (
    prepare_data,
    forward_pass,
    get_collate_functions,
    fix_seeds,
    accuracy_metric,
)

group_datasets = ["coral", "groupdro", "irm"]


class BaseTrainer:
    def __init__(self, args):
        # Logging parameters
        self.id = str(uuid.uuid4())[:8]
        self.step_callback = (
            args.step_callback if hasattr(args, "step_callback") else None
        )
        self.step_callback_frequency = (
            args.step_callback_frequency
            if hasattr(args, "step_callback_frequency")
            else 50
        )

        # Training hyperparameters
        self.args = args
        self.train_update_iter = args.train_update_iter
        self.early_stop = args.early_stop
        self.early_stop_patience = args.early_stop_patience if self.early_stop else None
        self.early_stop_holdout = args.early_stop_holdout if self.early_stop else 0.0

        self.use_lisa = args.lisa if hasattr(args, "lisa") else False
        self.lisa = self.use_lisa
        self.lisa_start_time = (
            args.lisa_start_time if hasattr(args, "lisa_start_time") else 0
        )
        self.mixup = args.mixup if hasattr(args, "mixup") else False
        self.cut_mix = args.cut_mix if hasattr(args, "cut_mix") else False
        self.mix_alpha = args.mix_alpha if hasattr(args, "mix_alpha") else None
        self.mini_batch_size = args.mini_batch_size
        self.num_workers = args.num_workers
        self.base_trainer_str = self.get_base_trainer_str()

        self.append_domain_as_feature = args.append_domain_as_feature

        self.fitted = False

        # SKlearn compatibility
        self._estimator_type = "classifier"

    def __str__(self):
        pass

    def get_params(self, deep=True):
        return {"args": self.args}

    @classmethod
    def set_params(cls, **parameters):
        return cls(**parameters)

    def get_base_trainer_str(self):
        base_trainer_str = (
            f"train_update_iter={self.train_update_iter}-lr={self.args.lr}-"
            f"mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}"
        )
        if self.lisa:
            base_trainer_str += f"-lisa-mix_alpha={self.mix_alpha}"
        elif self.mixup:
            base_trainer_str += f"-mixup-mix_alpha={self.mix_alpha}"
        if self.cut_mix:
            base_trainer_str += f"-cut_mix"

        base_trainer_str += f"-eval_fix"
        return base_trainer_str

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

    def early_stop_reset(self):
        self.train_dataset.get_item_mode = DatasetMode.HOLDOUT
        len_holdout = len(self.train_dataset)
        self.train_dataset.get_item_mode = DatasetMode.TRAIN

        if not self.early_stop or len_holdout == 0:
            return

        self.early_stop_best_score = np.inf
        self.early_stop_counter = 1

    def early_stop_callback(self, model, step):
        self.train_dataset.get_item_mode = DatasetMode.HOLDOUT
        len_holdout = len(self.train_dataset)

        if not self.early_stop or len_holdout == 0:
            self.train_dataset.get_item_mode = DatasetMode.TRAIN

            return False, np.nan

        model.eval()

        dataloader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.mini_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.eval_collate_fn,
        )

        all_logits = []
        all_y = []
        for x, y in dataloader:
            x, y = prepare_data(x, y, str(self.train_dataset), self.args.device)

            with torch.no_grad():
                logits = model(x)
                all_logits.append(logits)
                all_y.append(y)

        logits = torch.cat(all_logits, dim=0)

        y = torch.cat(all_y, dim=0)

        # Cross Entropy Loss
        # loss = self.criterion(logits, y)

        # ROC AUC Loss
        # pred_proba = F.softmax(logits, dim=1)
        # loss = -1 * auc_metric(pred_proba, y).item()

        # ACC loss
        pred_proba = F.softmax(logits, dim=1)
        loss = -1 * accuracy_metric(pred_proba, y).item()

        print(loss)

        stop = False
        if loss < self.early_stop_best_score:
            self.early_stop_best_score = loss
            self.early_stop_best_step = step
            self.early_stop_best_checkpoint = copy.deepcopy(self.get_checkpoint())
            self.early_stop_counter = 1
        else:
            self.early_stop_counter += 1

            if self.early_stop_counter >= self.early_stop_patience:
                print(
                    f"Early stopping with best score {self.early_stop_best_score} after {self.early_stop_patience * self.step_callback_frequency} steps."
                )
                stop = True

        model.train()
        self.train_dataset.get_item_mode = DatasetMode.TRAIN

        return stop, loss

    def early_stop_load_checkpoint(self):
        self.train_dataset.get_item_mode = DatasetMode.HOLDOUT
        len_holdout = len(self.train_dataset)
        self.train_dataset.get_item_mode = DatasetMode.TRAIN

        if not self.early_stop or len_holdout == 0:
            return

        print(
            f"Loading checkpoint saved during step {self.early_stop_best_step} which yielded score {self.early_stop_best_score}."
        )
        self.load_checkpoint(self.early_stop_best_checkpoint)

    def _train_online_hook_pre(self):
        pass

    def train_online(self):
        # Hook to be overwritten by methods that require online training to
        # initialize some variables.
        self._train_online_hook_pre()

        for i, timestamp in enumerate(self.train_dataset.ENV):
            if self.args.load_model and self.model_path_exists(timestamp):
                self.load_model(timestamp)
            else:
                self.lisa = self.use_lisa and i >= self.lisa_start_time

                self.train_dataset.update_current_timestamp(timestamp)

                if len(self.train_dataset) == 0:
                    continue

                train_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset,
                    weights=None,
                    batch_size=self.mini_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.train_collate_fn,
                )
                self.train_step(train_dataloader)

                if self.args.save_model:
                    self.save_model(timestamp)

                if timestamp == self.split_time:
                    break

                if self.args.method in ["coral", "groupdro", "irm", "erm"]:
                    self.train_dataset.update_historical(
                        i + 1, data_del=True, exclude_holdout=True
                    )

                self.train_dataset.update_historical_holdout(i + 1)

    def train_offline(self):
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                ## Concatenates all historical data to the current data of this timestamp
                # train portion
                # D_t+1 = [D_t+1, D_t, ..., D_1]
                # Keeps the old aggregated timestamps
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1)

            elif timestamp == self.split_time:
                self.train_dataset.update_current_timestamp(timestamp)
                train_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset,
                    weights=None,
                    batch_size=self.mini_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.train_collate_fn,
                )
                if self.args.load_model and self.model_path_exists(timestamp):
                    self.load_model(timestamp)
                else:
                    self.train_step(train_dataloader)

                    if self.args.save_model:
                        self.save_model(timestamp)
                break

    def init_model(self, X, y):
        if self.args.method in group_datasets:
            from ..data.datasets import TabularDatasetGroup

            dataset = TabularDatasetGroup(self.args, X, y, self.early_stop_holdout)
        else:
            from ..data.datasets import TabularDataset

            dataset = TabularDataset(self.args, X, y, self.early_stop_holdout)

        if self.args.network == "MLP":
            self.network = MLP(
                n_inputs=dataset.num_features,
                n_outputs=dataset.num_classes,
                hparams=self.args.hparams,
            ).to(self.args.device)
        elif self.args.network == "FTT":
            self.network = FTT(
                column_idx=dataset.processor.column_idx,
                cat_embed_input=dataset.processor.cat_embed_input
                if dataset.categorical_cols
                else None,
                continuous_cols=dataset.processor.continuous_cols,
                n_outputs=dataset.num_classes,
                hparams=self.args.hparams,
            ).to(self.args.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss(reduction=self.args.reduction).to(
            self.args.device
        )

        if self.args.use_scheduler:
            # Reduce the learning rate for each fine tuning step
            self.ft_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=self.args.ft_scheduler_gamma
            )

            # Use the OneCycleLR scheduler within each training step
            self.scheduler = lambda lr: torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                total_steps=self.args.train_update_iter,
                anneal_strategy="cos",
            )
        else:
            self.ft_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )
            self.scheduler = lambda lr: torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )

        # Dataset settings
        self.train_dataset = dataset
        self.num_classes = dataset.num_classes
        self.num_tasks = dataset.num_tasks
        self.split_time = dataset.split_time
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(
            self.args, self.train_dataset
        )

        # Store the class labels for each class
        self.classes_, y = np.unique(y, return_inverse=True)

    def fit(self, X, y):
        # from sklearn.utils.estimator_checks import check_estimator
        # check_estimator(self)

        start_time = time.time()

        torch.cuda.empty_cache()

        # Fix the seeds each time fit() is called to restore the random state. This is
        # necessary to get the same results when calling fit() multiple times.
        fix_seeds(self.args.random_seed)

        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must have the same number of instances."

        self.init_model(X, y)

        print(
            "=========================================================================================="
        )
        print("Fitting network...\n")

        if self.args.method in ["agem", "ewc", "ft", "si"]:
            self.train_online()
        else:
            self.train_offline()

        self.fitted = True

        runtime = time.time() - start_time

        print(f"Runtime: {runtime:.2f}\n")

        return self

    def predict_proba(self, X):
        assert self.fitted, "This model has not been fitted yet."

        self.network.eval()

        if not self.append_domain_as_feature:
            X = X.drop(["dist_shift_domain"], axis=1)

        X = self.train_dataset.processor.transform(X).astype(np.float32)

        dataloader = FastDataLoader(
            dataset=X,
            batch_size=self.mini_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.eval_collate_fn,
        )

        all_logits = []
        for batch in dataloader:
            batch = batch.to(self.args.device)

            with torch.no_grad():
                logits = self.network(batch)
                all_logits.append(logits)

        logits = torch.cat(all_logits, dim=0)
        pred_proba = F.softmax(logits, dim=1)

        self.network.train()

        return pred_proba.detach().cpu().numpy()

    def predict(self, X):
        with torch.no_grad():
            pred_proba = self.predict_proba(X)
            pred = np.argmax(pred_proba, axis=1)

        # Return the predicted class labels
        return self.classes_[pred]

    def get_model_path(self, timestamp):
        model_str = (
            f'{str(self.train_dataset).replace(" ", "_")}_{str(self)}_time={timestamp}'
        )
        path = os.path.join(self.args.log_dir, model_str)
        return path

    def model_path_exists(self, timestamp):
        return os.path.exists(self.get_model_path(timestamp))

    def get_checkpoint(self):
        return {"network": self.network.state_dict()}

    def load_checkpoint(self, checkpoint):
        self.network.load_state_dict(checkpoint["network"], strict=False)

    def save_model(self, timestamp):
        path = self.get_model_path(timestamp)
        torch.save(self.get_checkpoint(), path)
        print(f"Saving model at timestamp {timestamp} to path {path}...\n")

    def load_model(self, timestamp):
        path = self.get_model_path(timestamp)
        self.load_checkpoint(torch.load(path))
