from typing import Tuple

import numpy as np
import torch
from torchvision import transforms


class Buffer:
    """
    The memory buffer of the rehearsal method.

    Code adapted from https://github.com/aimagelab/mammoth.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode="reservoir"):
        assert mode in ["ring", "reservoir"]
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == "ring":
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ["examples", "labels", "logits", "task_labels"]

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(
        self,
        examples,
        labels: torch.Tensor,
        logits: torch.Tensor,
        task_labels: torch.Tensor,
    ) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith("els") else torch.float32
                if isinstance(attr, list) or isinstance(attr, tuple):
                    if isinstance(attr[0], tuple):
                        tensor_list = attr
                    else:
                        tensor_list = [
                            torch.zeros(
                                (self.buffer_size, *a.shape[1:]),
                                dtype=typ,
                                device=self.device,
                            )
                            for a in attr
                        ]
                    setattr(self, attr_str, tensor_list)
                else:
                    setattr(
                        self,
                        attr_str,
                        torch.zeros(
                            (self.buffer_size, *attr.shape[1:]),
                            dtype=typ,
                            device=self.device,
                        ),
                    )

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels, logits, task_labels)

        if isinstance(examples, list) or isinstance(examples, tuple):
            if isinstance(examples[0], tuple):
                batch_size = labels.shape[0]
            else:
                batch_size = examples[0].shape[0]
        else:
            batch_size = examples.shape[0]

        for i in range(batch_size):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                if isinstance(examples, list) or isinstance(examples, tuple):
                    if isinstance(examples[0], tuple):
                        continue
                    else:
                        for j in range(len(self.examples)):
                            self.examples[j][index] = examples[j][i].to(self.device)
                else:
                    self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(
        self, size: int, transform: transforms = None, return_index=False
    ) -> Tuple:
        """
        Randomly samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        """
        if isinstance(self.examples, list) or isinstance(self.examples, tuple):
            if isinstance(self.examples[0], tuple):
                batch_size = len(self.examples)
            else:
                batch_size = self.examples[0].shape[0]
            if size > min(self.num_seen_examples, batch_size):
                size = min(self.num_seen_examples, batch_size)

            choice = np.random.choice(
                min(self.num_seen_examples, batch_size), size=size, replace=False
            )
            if transform is None:
                transform = lambda x: x
            ret_tuple = ()
            for input in self.examples:
                if isinstance(input, tuple):
                    ret_tuple += (np.array(self.examples)[choice],)
                else:
                    ret_tuple += (
                        torch.stack([transform(ee) for ee in input[choice].cpu()]).to(
                            self.device
                        ),
                    )
            for attr_str in self.attributes[1:]:
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    ret_tuple += (attr[choice].to(self.device),)

            if not return_index:
                return ret_tuple
            else:
                return (torch.tensor(choice).to(self.device),) + ret_tuple

            return ret_tuple
        else:
            if size > min(self.num_seen_examples, self.examples.shape[0]):
                size = min(self.num_seen_examples, self.examples.shape[0])

            choice = np.random.choice(
                min(self.num_seen_examples, self.examples.shape[0]),
                size=size,
                replace=False,
            )
            if transform is None:
                transform = lambda x: x
            ret_tuple = (
                torch.stack([transform(ee) for ee in self.examples[choice].cpu()]).to(
                    self.device
                ),
            )
            for attr_str in self.attributes[1:]:
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    ret_tuple += (attr[choice].to(self.device),)

            if not return_index:
                return ret_tuple
            else:
                return (torch.tensor(choice).to(self.device),) + ret_tuple

            return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def empty(self) -> None:
        """
        Sets all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size
