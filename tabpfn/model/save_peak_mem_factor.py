from typing import Optional

import torch


def support_save_peak_mem_factor(method):
    """
    Can be applied to a method acting on a tensor 'x' whose first dimension is a flat batch dimension (i.e. the operation is trivially parallel over the first dimension).
    For additional tensor arguments, it is assumed that the first dimension is again the batch dimension, and that non-tensor arguments can be passed as-is to splits when parallelizing over the batch dimension.
    The decorator adds options 'add_input' to add the principal input 'x' to the result of the method and 'allow_inplace'.
    By setting 'allow_inplace', the caller indicates that 'x' is not used after the call and its buffer can be reused for the output.
    Setting 'allow_inplace' does not ensure that the operation will be inplace, and the return value should be used for clarity and simplicity.
    Moreover, it adds an optional int parameter 'save_peak_mem_factor' that is only supported in combination with 'allow_inplace' during inference and subdivides the operation into the specified number of chunks to reduce peak memory consumption.
    """

    def method_(
        self: torch.nn.Module,
        x: torch.Tensor,
        *args: Optional[torch.Tensor],
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(self, torch.nn.Module)
        assert (
            save_peak_mem_factor is None or allow_inplace
        ), "The parameter save_peak_mem_factor is only supported with 'allow_inplace' set."
        assert isinstance(x, torch.Tensor)
        tensor_inputs = [
            t
            for t in tuple(self.parameters()) + tuple(args)
            if isinstance(t, torch.Tensor)
        ]
        assert (
            save_peak_mem_factor is None
            or not any(t.requires_grad for t in tensor_inputs)
            or not torch.is_grad_enabled()
        ), "The parameter save_peak_mem_factor is only supported during inference."

        if save_peak_mem_factor is not None:
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor

            def split_args(*args):
                return zip(
                    *[
                        torch.split(arg, split_size)
                        if isinstance(arg, torch.Tensor)
                        else [arg] * save_peak_mem_factor
                        for arg in args
                    ]
                )

            for x_, *args_ in split_args(x, *args):
                if add_input:
                    x_[:] += method(self, x_, *args_, **kwargs)
                else:
                    x_[:] = method(self, x_, *args_, **kwargs)
            return x
        elif add_input:
            return x + method(self, x, *args, **kwargs)
        else:
            return method(self, x, *args, **kwargs)

    return method_
