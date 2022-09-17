"""Module containing data handling functions."""
import glob
import os
from pathlib import Path
from typing import Optional, PathLike, Union

import imageio
import numpy as np
import pandas as pd
import torch

from lcapt.help import import_module


Tensor = torch.Tensor

PROJECT_DIR: Optional[Union[str, Path]] = os.environ.get("PROJECT_DIR")
if PROJECT_DIR is None:
    PROJECT_DIR = Path.cwd().parent
else:
    PROJECT_DIR = Path(PROJECT_DIR)

CONFIG = import_module(os.environ.get("PDYN_ML_CONFIG", PROJECT_DIR / "conf" / "default.py"))


def read_windowed_movie_frames(imgs_path: PathLike) -> Tensor:
    """Load and preprocess movie frames into sliding windows of equitative frame numbers.

    Args:
        imgs_path (PathLike): A path object.

    Returns:
        Tensor: Windowed frames tensor.
    """
    imgs = []

    for img_path in sorted(glob.glob(f"{imgs_path}/*.png")):
        # print(img_path, imageio.imread(img_path).shape)
        imgs.append(imageio.imread(img_path))

    torch_wf = torch.empty(
        size=(
            len(imgs) - CONFIG.WINDOW_SIZE + 1,
            CONFIG.WINDOW_SIZE,
            np.array(imgs).shape[-2],
            np.array(imgs).shape[-1],
        )
    )

    imgs_arr = np.array(imgs)

    for ind, im in enumerate(imgs):
        if len(imgs_arr[CONFIG.i : CONFIG.i + CONFIG.WINDOW_SIZE]) == CONFIG.WINDOW_SIZE:
            torch_wf[ind] = torch.tensor(imgs_arr[CONFIG.i : CONFIG.i + CONFIG.WINDOW_SIZE])
            CONFIG.i += 1

    return torch_wf


def read_windowed_dff_traces(dff: str) -> Tensor:
    """Load and preprocess mice responses into equitative sliding datapoints to match the windowed-frames time dimension.

    Args:
        dff (str): A string name of the traces/events.

    Returns:
        Tensor: Windowed traces tensor.
    """
    exps_cells = []
    column_exp = []
    column_cell = []
    complete_column_cell_exp_list = []

    dff_df = pd.read_hdf(f"{dff}/dff.h5")

    for column in dff_df.columns[CONFIG.ante_metaparams :]:
        complete_column_cell_exp_list.append(column)
        column_cell.append(column.split("_")[0])
        column_exp.append(column.split("_")[1])

    new_column_exp = np.unique(column_exp)

    print("Average cell count per experiment:")
    print(
        len(np.unique(column_cell)) // len(np.unique(column_exp)),
        "=",
        len(np.unique(column_cell)),
        "//",
        len(np.unique(column_exp)),
    )

    for exp_index in range(CONFIG.exp_start, CONFIG.exp_end + 1):
        for cell_exp in complete_column_cell_exp_list:
            if str(cell_exp).endswith(new_column_exp[exp_index]):
                exps_cells.append(cell_exp)

    dff_exps = dff_df[["frame", "repeat", "stimulus", "session_type", *exps_cells]]
    dff_exps_movie3_list = []

    for exp_cell in exps_cells:
        if dff_exps[dff_exps["stimulus"] == "natural_movie_three"][exp_cell].values.all() is False:
            dff_exp_cell = (
                dff_exps[["stimulus", exp_cell]].where(dff_exps["stimulus"] == "natural_movie_three").dropna()
            )
            dff_exp_rep = torch.tensor(dff_exp_cell[exp_cell].values)
            dff_exp_movie_tensors = torch.split(dff_exp_rep, 3600, 0)

            for ind, dff_exp_sample in enumerate(dff_exp_movie_tensors):
                # ind == 0 is only using the first repetition per cell
                if ind == 0:
                    dff_exps_movie3_list.append(dff_exp_sample)

    dff_exps_movie3 = torch.stack(dff_exps_movie3_list)

    torch_wt = torch.empty(
        size=(len(dff_exps_movie3_list), dff_exps_movie3.shape[1] - CONFIG.WINDOW_SIZE + 1, CONFIG.WINDOW_SIZE)
    )

    for trace_ind, _ in enumerate(dff_exps_movie3):
        torch_wt[trace_ind] = torch.from_numpy(
            np.lib.stride_tricks.sliding_window_view(
                dff_exps_movie3[trace_ind].cpu().numpy(), window_shape=CONFIG.WINDOW_SIZE
            )
        )

    return torch_wt
