from enum import Enum
from typing import Union, Optional
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from tabpfn.model.save_peak_mem_factor import support_save_peak_mem_factor


class Activation(Enum):
    """
    Enum for activation functions.
    """

    GELU = 1
    RELU = 2


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) module.

    This module consists of two linear layers with an activation function in between.
    It supports various configurations such as the hidden size, activation function,
    initializing the output to zero, and recomputing the forward pass during backpropagation.

    Args:
        size (int): The input and output size of the MLP.
        hidden_size (int): The size of the hidden layer.
        activation (Union[Activation, str]): The activation function to use.
            Can be either an Activation enum or a string representing the activation name.
        device (torch.device): The device to use for the linear layers.
        dtype (torch.dtype): The data type to use for the linear layers.
        initialize_output_to_zero (bool): Whether to initialize the output layer weights to zero.
            Default is False.
        recompute (bool): Whether to recompute the forward pass during backpropagation.
            This can save memory but increase computation time. Default is False.

    Attributes:
        linear1 (torch.nn.Linear): The first linear layer.
        linear2 (torch.nn.Linear): The second linear layer.
        activation (Activation): The activation function to use.

    Methods:
        forward(x, add_input=False, allow_inplace=False, save_peak_mem_factor=None):
            Performs the forward pass of the MLP.
            - x (torch.Tensor): The input tensor.
            - add_input (bool): Whether to add the input to the output. Default is False.
            - allow_inplace (bool): Indicates that 'x' is not used after the call and its buffer
                can be reused for the output. The operation is not guaranteed to be inplace.
                Default is False.
            - save_peak_mem_factor (Optional[int]): If provided, enables a memory-saving technique
                that reduces peak memory usage during the forward pass. This requires 'add_input'
                and 'allow_inplace' to be True. See the documentation of the decorator
                'support_save_peak_mem_factor' for details. Default is None.

    Example:
        >>> mlp = MLP(size=128, hidden_size=256, activation='gelu', device='cuda', dtype=torch.float32)
        >>> x = torch.randn(32, 128, device='cuda', dtype=torch.float32)
        >>> output = mlp(x)
    """

    linear1: torch.nn.Linear
    linear2: torch.nn.Linear
    activation: Activation

    def __init__(
        self,
        size: int,
        hidden_size: int,
        activation: Union[Activation, str],
        device,
        dtype,
        initialize_output_to_zero: bool = False,
        recompute: bool = False,
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(
            size, hidden_size, bias=False, device=device, dtype=dtype
        )
        self.linear2 = torch.nn.Linear(
            hidden_size, size, bias=False, device=device, dtype=dtype
        )
        if isinstance(activation, str):
            activation = Activation[activation.upper()]
        self.activation = activation
        if initialize_output_to_zero:
            torch.nn.init.zeros_(self.linear2.weight)
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)

    @support_save_peak_mem_factor
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.activation is Activation.GELU:
            x = torch.nn.functional.gelu(x)
        elif self.activation is Activation.RELU:
            x = torch.nn.functional.relu(x)
        else:
            raise NotImplementedError(
                f"Activation Function {self.activation} is not implemented."
            )
        x = self.linear2(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: Optional[int] = None,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self._compute(
            x,
            add_input=add_input,
            allow_inplace=allow_inplace,
            save_peak_mem_factor=save_peak_mem_factor,
        )
        x = x.reshape(input_shape)
        return x
