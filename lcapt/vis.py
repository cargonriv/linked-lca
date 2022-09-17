"""Module to generate visualizations."""
import math
from os import PathLike
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision.utils import make_grid

from lcapt.help import import_module


Tensor = torch.Tensor

PROJECT_DIR = Path(__file__).parents[2]
DEFUALT_CONFIG_PATH = PROJECT_DIR / "conf" / "default.py"

if __name__ != "__main__":
    CONFIG = import_module(DEFUALT_CONFIG_PATH)


def plot_1input_2recon_3error(
    windowed_frames_batch: Tensor, recon: Tensor, save: bool, show: bool, path: PathLike, ind: int
) -> None:
    """Visualization of windowed-frames input, reconstruction, and the reconstruction's error.

    Args:
        windowed_frames_batch (Tensor): Batch sample of windowed-frames input.
        recon (Tensor): Batch sample of windowed-frames reconstruction.
        save (bool): Whether to save the figure to disk.
        show (bool): Whether to show the figure.
        path (PathLike): A path object.
        ind (int): Batch index.
    """
    fig1, axs1 = plt.subplots(1, 7, figsize=(16, 9))
    fig2, axs2 = plt.subplots(1, 7, figsize=(16, 9))
    fig3, axs3 = plt.subplots(1, 7, figsize=(16, 9))

    for ind, (a1, a2, a3) in enumerate(zip(axs1, axs2, axs3)):
        recon_error = windowed_frames_batch - recon

        recon_samples = recon[0].squeeze().squeeze().cpu().numpy()

        recon_error_samples = recon_error[0].squeeze().squeeze().cpu().numpy()

        recon_sample = recon_samples[ind]
        recon_error_sample = recon_error_samples[ind]
        inputs_sample = recon_error_sample + recon_sample

        recon_sample = (recon_sample - recon_sample.min()) / (recon_sample.max() - recon_sample.min())
        inputs_sample = (inputs_sample - inputs_sample.min()) / (inputs_sample.max() - inputs_sample.min())

        a1.imshow(inputs_sample)
        a2.imshow(recon_sample)
        a3.imshow(recon_error_sample)

        a1.set_title(f"Input frame{ind+1}")
        a2.set_title(f"Recon frame{ind+1}")
        a3.set_title(f"Recon Error frame{ind+1}")

        a1.get_xaxis().set_visible(False)
        a1.get_yaxis().set_visible(False)
        a2.get_xaxis().set_visible(False)
        a2.get_yaxis().set_visible(False)
        a3.get_xaxis().set_visible(False)
        a3.get_yaxis().set_visible(False)

    if show:
        plt.show()

    if save:
        fig1.savefig(path / Path(f"input_window_sample{ind}.png"), bbox_inches="tight")
        fig2.savefig(path / Path(f"recon_window_sample{ind}.png"), bbox_inches="tight")
        fig3.savefig(path / Path(f"recon_error_window_sample{ind}.png"), bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def activation_histogram(code: Tensor, save: bool, show: bool, path: PathLike) -> None:
    """Visualization of activation histogram over the features.

    Args:
        code (Tensor): feature activations.
        save (bool): Whether to save the figure to disk.
        show (bool): Whether to show the figure.
        path (PathLike): A path object.
    """
    bool_freq_df = pd.DataFrame(columns=["neuron", "frequency"])

    v1_batch_mean_arr = code.cpu().sum(axis=(0, 2, 3))

    for n in range(v1_batch_mean_arr.shape[1]):
        bool_freq_df.loc[n, "neuron"] = n + 1
        bool_freq_df.loc[n, "frequency"] = (
            sum(v1_batch_mean_arr[:, n]) / (v1_batch_mean_arr.shape[0] * v1_batch_mean_arr.shape[1])
        ).item()

    mean_sparsity = bool_freq_df["frequency"].mean()
    mean_sparsity_df = bool_freq_df * 0 + mean_sparsity

    ax1 = bool_freq_df.plot.bar(x="neuron", y="frequency", figsize=(25, 10), legend=False)
    mean_sparsity_df.plot(ax=ax1, color="r", title=f"average sparsity: {mean_sparsity}", legend=False)

    ax2 = bool_freq_df.sort_values(by=["frequency"], ascending=False).plot.bar(
        x="neuron", y="frequency", figsize=(25, 10), legend=False
    )
    mean_sparsity_df.plot(ax=ax2, color="r", title=f"average sparsity: {mean_sparsity}", legend=False)

    if save:
        ax1.get_figure().savefig(f"{path}/orderedActivityHist.png")
        ax2.get_figure().savefig(f"{path}/sortedActivityHist.png")

    if show:
        plt.show()


def spatiotemporal_dictionary(lca: torch.nn.Module, save: bool, show: bool, path: PathLike) -> None:
    """Visualization of 3D dictionary grid.

    Args:
        lca (torch.nn.Module): Trained model of the Locally Competitive Algorithm.
        save (bool): Whether to save the figure to disk.
        show (bool): Whether to show the figure.
        path (PathLike): A path object.
    """
    T = lca.module.weights.shape[2]
    grids = []

    for t in range(T):
        grids.append(
            make_grid(
                lca.module.weights[:, :, t],
                int(math.sqrt(lca.module.weights.shape[0])),
                normalize=False,
                scale_each=False,
                pad_value=0.5,
            ).cpu()
        )

    final_grids = torch.stack(grids).permute(0, 2, 3, 1)

    if save:
        imageio.mimwrite(f"{path}/linked_dictionary.gif", final_grids, "gif")

    if show:
        with Image.open("{path}/linked_dictionary.gif") as im:
            im.seek(1)  # skip to the second frame

            try:
                while 1:
                    im.seek(im.tell() + 1)
                    # do something to im
            except EOFError:
                pass  # end of sequence
