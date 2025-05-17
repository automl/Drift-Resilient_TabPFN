# Import necessary libraries
import numpy as np
import torch
from pytorch_widedeep.preprocessing import TabPreprocessor
from torch.utils import data
import warnings

from enum import Enum


class DatasetMode(Enum):
    TRAIN = 0
    HOLDOUT = 1


# Define the base class for the TabularDataset dataset
class TabularDatasetBase(data.Dataset):
    def __init__(self, args, X, y, holdout_portion=0.0):
        super().__init__()

        # Separate the domain from the features
        self.dist_shift_train_domain = X["dist_shift_domain"].values
        self.unique_train_domains = np.sort(np.unique(self.dist_shift_train_domain))

        start_index = 0
        if not args.append_domain_as_feature:
            X = X.drop(["dist_shift_domain"], axis=1)
            start_index = 1

        self.processor = TabPreprocessor(
            cat_embed_cols=args.dataset["categorical_cols"]
            if len(args.dataset["categorical_cols"]) > 0
            else None,
            continuous_cols=args.dataset["continuous_cols"][start_index:]
            if len(args.dataset["continuous_cols"][start_index:]) > 0
            else None,
            with_attention=True,
            with_cls_token=False,
        )

        X = self.processor.fit_transform(X)

        self.args = args
        self.name = args.dataset["name"]
        self.ENV = args.dataset["domains"]
        self.ENV_idx = {domain: i for i, domain in enumerate(self.ENV)}
        self.num_tasks = args.dataset["num_domains"]
        self.num_classes = np.unique(y).shape[0]
        self.num_features = X.shape[1]
        self.split_time = self.unique_train_domains[-1]

        self.categorical_cols = self.processor.cat_embed_cols
        self.continuous_cols = self.processor.continuous_cols

        self.num_train_instances = 0
        self.get_item_mode = DatasetMode.TRAIN

        self.train_dataset = {}
        self.train_domain_mapping = {}
        self.holdout_dataset = {}
        self.holdout_domain_mapping = {}

        # Beware: Those won't be updated during update_historical.
        self.train_instances_per_domain = {}
        self.holdout_instances_per_domain = {}

        for domain in self.ENV:
            domain_ds_x = X[self.dist_shift_train_domain == domain].astype(np.float32)
            domain_ds_y = y[self.dist_shift_train_domain == domain]

            unique_labels = np.unique(domain_ds_y)

            train_indices = []
            holdout_indices = []

            # Use stratified sampling for the holdout
            for label in unique_labels:
                label_indices = np.where(domain_ds_y == label)[0]
                num_label_instances = len(label_indices)

                # Calculate sizes for train and holdout sets
                holdout_size = int(num_label_instances * holdout_portion)
                train_size = num_label_instances - holdout_size

                shuffled_label_indices = np.random.permutation(label_indices)

                train_indices.extend(shuffled_label_indices[:train_size])
                holdout_indices.extend(shuffled_label_indices[train_size:])

            train_indices = np.array(train_indices, dtype=int)
            holdout_indices = np.array(holdout_indices, dtype=int)

            train_indices.sort()
            holdout_indices.sort()

            train_ds_x = domain_ds_x[train_indices]
            train_ds_y = domain_ds_y[train_indices]
            train_ds_instances = train_ds_x.shape[0]

            holdout_ds_x = domain_ds_x[holdout_indices]
            holdout_ds_y = domain_ds_y[holdout_indices]
            holdout_ds_instances = holdout_ds_x.shape[0]

            self.num_train_instances += train_ds_instances + holdout_ds_instances

            train_ds_x = train_ds_x if domain <= self.split_time else None
            train_ds_y = train_ds_y if domain <= self.split_time else None

            holdout_ds_x = holdout_ds_x if domain <= self.split_time else None
            holdout_ds_y = holdout_ds_y if domain <= self.split_time else None

            self.train_dataset[domain] = {"x": train_ds_x, "y": train_ds_y}
            self.train_domain_mapping[domain] = np.full(train_ds_instances, domain)
            self.train_instances_per_domain[domain] = train_ds_instances

            self.holdout_dataset[domain] = {"x": holdout_ds_x, "y": holdout_ds_y}
            self.holdout_domain_mapping[domain] = np.full(holdout_ds_instances, domain)
            self.holdout_instances_per_domain[domain] = holdout_ds_instances

        assert (
            self.num_train_instances == X.shape[0]
        ), "Something went wrong! The number of training instances is not correct!"

        # Initialize the current time
        self.current_time = self.ENV[0]

        self.class_id_list = {i: {} for i in range(self.num_classes)}

        # Build a mapping of all the indices for a certain class in the training dataset
        # This is used for the method lisa
        for i in self.ENV:
            for classid in range(self.num_classes):
                if self.train_dataset[i]["y"] is None:
                    sel_idx = np.array([])
                else:
                    sel_idx = np.nonzero(self.train_dataset[i]["y"] == classid)[0]
                self.class_id_list[classid][i] = sel_idx

            # print(f'Domain {str(i)} loaded')

    def update_historical_holdout(self, idx, data_del=False):
        # Concatenates the time stated with idx with the previous time
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]

        # Concatenate the holdout dataset timepoints
        self.holdout_dataset[time]["x"] = np.concatenate(
            (self.holdout_dataset[time]["x"], self.holdout_dataset[prev_time]["x"]),
            axis=0,
        )
        self.holdout_dataset[time]["y"] = np.concatenate(
            (self.holdout_dataset[time]["y"], self.holdout_dataset[prev_time]["y"]),
            axis=0,
        )
        self.holdout_domain_mapping[time] = np.concatenate(
            (self.holdout_domain_mapping[time], self.holdout_domain_mapping[prev_time]),
            axis=0,
        )

        if data_del:
            del self.holdout_dataset[prev_time]
            del self.holdout_domain_mapping[prev_time]

    # Define method for updating historical data
    def update_historical(self, idx, data_del=False, exclude_holdout=False):
        # Concatenates the time stated with idx with the previous time
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]

        if not exclude_holdout:
            self.update_historical_holdout(idx, data_del)

        # Concatenate the train dataset timepoints
        self.train_dataset[time]["x"] = np.concatenate(
            (self.train_dataset[time]["x"], self.train_dataset[prev_time]["x"]), axis=0
        )
        self.train_dataset[time]["y"] = np.concatenate(
            (self.train_dataset[time]["y"], self.train_dataset[prev_time]["y"]), axis=0
        )
        self.train_domain_mapping[time] = np.concatenate(
            (self.train_domain_mapping[time], self.train_domain_mapping[prev_time]),
            axis=0,
        )

        # Delete the old time point in case we should
        if data_del:
            del self.train_dataset[prev_time]
            del self.train_domain_mapping[prev_time]

        # Update the mapping of all corresponding indices in a certain class of the train dataset
        for classid in range(self.num_classes):
            if self.train_dataset[time]["y"] is None:
                sel_idx = np.array([])
            else:
                sel_idx = np.nonzero(self.train_dataset[time]["y"] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    # Define method for updating the current timestamp
    def update_current_timestamp(self, timestamp):
        self.current_time = timestamp

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        feats = self.train_dataset[time_idx]["x"][sel_idx]
        label = self.train_dataset[time_idx]["y"][sel_idx]

        return torch.FloatTensor(feats).to(self.args.device), torch.LongTensor(
            label
        ).unsqueeze(-1).to(self.args.device)

    def get_mode_dataset(self):
        if self.get_item_mode == DatasetMode.TRAIN:
            return self.train_dataset
        elif self.get_item_mode == DatasetMode.HOLDOUT:
            return self.holdout_dataset
        else:
            raise ValueError(f"Unsupported dataset mode: {self.get_item_mode}")

    # Define method for getting an item from the dataset
    def __getitem__(self, index):
        pass

    # Define method for getting the length of the dataset
    def __len__(self):
        pass

    # Define method for getting the string representation of the dataset
    def __str__(self):
        return self.name


# Define the class for the TabularDataset dataset, which extends the TabularDatasetBase
class TabularDataset(TabularDatasetBase):
    def __init__(self, args, X, y, holdout_portion=0.0):
        super().__init__(
            args=args, X=X, y=y, holdout_portion=holdout_portion
        )  # Initialize the base class

    def __getitem__(self, index):
        dataset = self.get_mode_dataset()
        feats = dataset[self.current_time]["x"][index]
        label = int(dataset[self.current_time]["y"][index])

        return feats, label

    # Override the __len__ method to get the length of the dataset
    def __len__(self):
        dataset = self.get_mode_dataset()
        return len(
            dataset[self.current_time]["y"]
        )  # Return the length of the dataset for the current time


# Define the class for the TabularDatasetGroup dataset, which extends the TabularDatasetBase and groups the time domains
# into moving windows
class TabularDatasetGroup(TabularDatasetBase):
    def __init__(self, args, X, y, holdout_portion=0.0):
        super().__init__(args=args, X=X, y=y, holdout_portion=holdout_portion)
        self.group_size = args.group_size
        self.non_overlapping = args.non_overlapping

        if self.non_overlapping:
            self.num_groups = int(
                np.ceil(self.unique_train_domains.shape[0] / self.group_size)
            )
        else:
            self.num_groups = max(
                1, self.unique_train_domains.shape[0] - self.group_size + 1
            )

        idx = self.ENV_idx[self.split_time]
        start_indices = self.get_group_indices(idx)
        group_counts = []
        for group_start_idx in start_indices:
            group_end_idx = min(group_start_idx + self.group_size, idx + 1)
            domains_in_group = self.ENV[group_start_idx:group_end_idx]

            samples_in_group = 0
            for domain in domains_in_group:
                samples_in_group += self.train_instances_per_domain[domain]

            group_counts += [samples_in_group]

        self._group_counts = torch.LongTensor(group_counts)

    def get_group_indices(self, idx):
        if self.non_overlapping:
            # Get start indices for non-overlapping groups
            start_indices = list(range(0, idx + 1, self.group_size))
        else:
            # Get start indices for overlapping groups
            start_indices = list(range(0, max(1, idx - self.group_size + 2)))

        return start_indices

    def __getitem__(self, index):
        dataset = self.get_mode_dataset()

        if self.get_item_mode == DatasetMode.TRAIN:
            # Create local random number generator with the given seed that doesn't affect the global seed
            rng = np.random.default_rng(seed=self.args.random_seed ^ index)

            # Select group ID
            idx = self.ENV_idx[self.current_time]

            # 1. Get the start indices of the groups
            group_start_indices = self.get_group_indices(idx)

            assert (
                len(group_start_indices) > 0
            ), "There are no groups to select indices from."

            # 2. Randomly select a group index
            group_id = rng.integers(len(group_start_indices))
            group_start_idx = group_start_indices[group_id]
            group_end_idx = min(group_start_idx + self.group_size, idx + 1)

            # 3. Get all valid domains within the group
            domains = self.ENV[group_start_idx:group_end_idx]
            valid_domains = [
                domain
                for domain in domains
                if np.any(self.train_domain_mapping[self.current_time] == domain)
            ]

            assert (
                valid_domains
            ), "No valid domains available within the selected group."

            # 4. Randomly select one sample from a valid domain
            if not valid_domains:
                # Fallback to a random selection from all training data of the current time if no valid domain found
                warnings.warn(
                    f"No valid domains available within the selected group {group_id}; falling back to random selection."
                )
                sel_idx = rng.integers(
                    len(self.train_domain_mapping[self.current_time])
                )
            else:
                # Randomly select one valid domain and then one example from that domain
                domain = rng.choice(valid_domains)
                candidates = np.nonzero(
                    self.train_domain_mapping[self.current_time] == domain
                )[0]
                sel_idx = rng.choice(candidates)

            # Pick the example in the time step
            feats = dataset[self.current_time]["x"][sel_idx]
            label = int(dataset[self.current_time]["y"][sel_idx])
            group_tensor = torch.LongTensor([group_id])

            return feats, label, group_tensor
        elif self.get_item_mode == DatasetMode.HOLDOUT:
            feats = dataset[self.current_time]["x"][index]
            label = int(dataset[self.current_time]["y"][index])

            return feats, label

    def group_counts(self):
        return self._group_counts

    def __len__(self):
        dataset = self.get_mode_dataset()
        return len(dataset[self.current_time]["y"])
