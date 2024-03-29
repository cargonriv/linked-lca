"""Module to analyze the LCA training and testing."""
from math import sqrt

import torch
from torchvision.utils import make_grid


Tensor = torch.Tensor


def _make_feature_grid_1D(weights: Tensor, scale_each: bool = False) -> Tensor:
    """Plot feature grid from a trained model in 1D input.

    Args:
        weights (Tensor): trained model's weights.
        scale_each (bool): Scaling mode of each weight. Default = False.

    Returns:
        Tensor: Feature grid output.
    """
    assert len(weights.shape) == 3
    if weights.shape[1] == 1:
        return weights[:, 0]
    else:
        nrow = int(sqrt(weights.shape[0]))
        feat_grid = make_grid(
            weights.unsqueeze(1),
            nrow,
            normalize=True,
            scale_each=scale_each,
            pad_value=0.5,
        )
        return feat_grid[0]


def _make_feature_grid_2D(weights: Tensor, scale_each: bool = False) -> Tensor:
    """Plot feature grid from a trained model in 2D input.

    Args:
        weights (Tensor): trained model's weights.
        scale_each (bool): Scaling mode of each weight. Default = False.

    Returns:
        Tensor: Feature grid output.
    """
    assert len(weights.shape) == 4
    nrow = int(sqrt(weights.shape[0]))
    feat_grid = make_grid(weights, nrow, normalize=True, scale_each=scale_each, pad_value=0.5)
    return feat_grid.permute(1, 2, 0)


def _make_feature_grid_3D(weights: Tensor, scale_each: bool = False) -> Tensor:
    """Plot feature grid from a trained model in 3D input.

    Args:
        weights (Tensor): trained model's weights.
        scale_each (bool): Scaling mode of each weight. Default = False.

    Returns:
        Tensor: Feature grid output.
    """
    assert len(weights.shape) == 5
    if weights.shape[2] == 1:
        return _make_feature_grid_2D(weights[:, :, 0], scale_each)
    nrow = int(sqrt(weights.shape[0]))
    T = weights.shape[2]
    grids = []
    for t in range(T):
        grid = make_grid(weights[:, :, t], nrow, normalize=True, scale_each=scale_each, pad_value=0.5)
        grids.append(grid)
    return torch.stack(grids).permute(0, 2, 3, 1)


def make_feature_grid(weights: Tensor, scale_each: bool = False) -> Tensor:
    """Select feature grid shape from the trained model's input and corresponding weights.

    Args:
        weights (Tensor): trained model's weights.
        scale_each (bool): Scaling mode of each weight. Default = False.

    Returns:
        Tensor: Feature grid output.
    """
    if len(weights.shape) == 3:
        return _make_feature_grid_1D(weights, scale_each)
    elif len(weights.shape) == 4:
        print("2D grid here")
        return _make_feature_grid_2D(weights, scale_each)
    elif len(weights.shape) == 5:
        print("3D grid here")
        return _make_feature_grid_3D(weights, scale_each)
