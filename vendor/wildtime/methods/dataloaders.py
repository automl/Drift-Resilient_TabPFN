# PyTorch is a popular open-source machine learning library for Python.
import torch

# DataLoader is a utility that provides the ability to batch, shuffle and load the data in parallel using multiprocessing workers.
from torch.utils.data import DataLoader


# This class is used to create an infinite stream of data samples from a given sampler.
class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    # The initializer takes a sampler as input.
    def __init__(self, sampler):
        self.sampler = sampler

    # The iterator method is defined to yield an infinite stream of data.
    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


# This class creates a DataLoader that provides an infinite stream of data samples.
class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, collate_fn=None):
        super().__init__()

        # If weights are provided, a WeightedRandomSampler is used, which samples elements from
        # [0,..,len(weights)-1] with given probabilities (weights).
        # Otherwise, a RandomSampler is used, which samples elements randomly from the dataset.
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        # A BatchSampler is used to batch the data from the sampler.
        # The number of samples per batch is the minimum of the provided batch_size and the length of the dataset.
        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=min(batch_size, len(dataset)), drop_last=True
        )

        # DataLoader is initialized with the batch_sampler wrapped in the _InfiniteSampler.
        self._infinite_iterator = iter(
            DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
                collate_fn=collate_fn,
            )
        )

    # The iterator method is defined to yield an infinite stream of data batches.
    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    # The length method is not defined for an infinite data loader, so it raises a ValueError.
    def __len__(self):
        raise ValueError


# This class creates a DataLoader that provides a stream of data samples with slightly improved speed
# by not respawning worker processes at every epoch.
class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""

    def __init__(
        self, dataset, batch_size, num_workers, collate_fn=None, drop_last=False
    ):
        super().__init__()

        # A BatchSampler is used to batch the data from a SequentialSampler.
        # The number of samples per batch is the minimum of the provided batch_size and the length of the dataset.
        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset),
            batch_size=min(batch_size, len(dataset)),
            drop_last=drop_last,
        )

        # DataLoader is initialized with the batch_sampler wrapped in the _InfiniteSampler.
        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
                collate_fn=collate_fn,
            )
        )

        # The length of the data loader is the number of batches, which is the length of the batch_sampler.
        self._length = len(batch_sampler)

    # The iterator method is defined to yield a stream of data batches for the number of batches in the data loader.
    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    # The length method returns the number of batches in the data loader.
    def __len__(self):
        return self._length
