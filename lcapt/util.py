"""Package-wide utilities."""
import torch


Tensor = torch.Tensor


def to_5d_from_3d(inputs: Tensor) -> Tensor:
    """Reshape input from 3 dimensions to 5 dimensions.

    Args:
        inputs (Tensor): Original input.

    Returns:
        Tensor: Reshaped input.
    """
    assert len(inputs.shape) == 3
    return inputs.unsqueeze(-1).unsqueeze(-1)


def to_5d_from_4d(inputs: Tensor) -> Tensor:
    """Reshape input from 4 dimensions to 5 dimensions.

    Args:
        inputs (Tensor): Original input.

    Returns:
        Tensor: Reshaped input.
    """
    assert len(inputs.shape) == 4
    return inputs.unsqueeze(-3)


def to_3d_from_5d(inputs: Tensor) -> Tensor:
    """Reshape input from 5 dimensions to 3 dimensions.

    Args:
        inputs (Tensor): Original input.

    Returns:
        Tensor: Reshaped input.
    """
    assert len(inputs.shape) == 5
    assert inputs.shape[-2] == 1 and inputs.shape[-1] == 1
    return inputs[..., 0, 0]


def to_4d_from_5d(inputs: Tensor) -> Tensor:
    """Reshape input from 5 dimensions to 4 dimensions.

    Args:
        inputs (Tensor): Original input.

    Returns:
        Tensor: Reshaped input.
    """
    assert len(inputs.shape) == 5
    assert inputs.shape[-3] == 1
    return inputs[..., 0, :, :]
