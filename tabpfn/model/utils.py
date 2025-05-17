import torch


def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Select features from the input tensor based on the selection mask.

    Parameters:
        x (torchTensor): The input tensor.
    sel (torch.Tensor): The boolean selection mask indicating which features to keep.

    Returns:
        torch.Tensor: The tensor with selected features.
    """
    new_x = x.clone()
    for B in range(x.shape[1]):
        if x.shape[1] > 1:
            new_x[:, B, :] = torch.cat(
                [
                    x[:, B, sel[B]],
                    torch.zeros(
                        x.shape[0],
                        x.shape[-1] - sel[B].sum(),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                ],
                -1,
            )
        else:
            # If B == 1, we don't need to append zeros, as the number of features can change
            new_x = x[:, :, sel[B]]
    return new_x
