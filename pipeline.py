"""Module to streamline data processing, model training, inferences, and visualizations."""
# system modules
import argparse
import os
from pathlib import Path
from typing import Optional, Union

# PyTorch module
import torch

# local module for linked-dictionary LCA
from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error
from lcapt.preproc import make_unit_var, make_zero_mean
from lcapt.read import read_windowed_dff_traces, read_windowed_movie_frames
from lcapt.util import import_module, str2bool
from lcapt.vis import activation_histogram, plot_1input_2recon_3error, spatiotemporal_dictionary


# Progress bar module
# from tqdm import tqdm


PROJECT_DIR: Optional[Union[str, Path]] = os.environ.get("PROJECT_DIR")
if PROJECT_DIR is None:
    PROJECT_DIR = Path.cwd().parent
else:
    PROJECT_DIR = Path(PROJECT_DIR)

DEFUALT_CONFIG_PATH = PROJECT_DIR / "conf" / "default.py"
config_path = os.environ.get("PDYN_ML_CONFIG", DEFUALT_CONFIG_PATH)

if __name__ == "__main__":

    CONFIG = import_module(config_path)

    parser = argparse.ArgumentParser(description="Process arguments for LCA PyTorch.")
    parser.add_argument(
        "--stimuli-name",
        default="natural_movie_three",
        type=str,
        help="Unique experiment name that is appended to save files. Default = natural_movie_three.",
    )
    parser.add_argument("--model-path", default="../models", type=str, help="Path to save models. Default = ../models")
    parser.add_argument(
        "--vis-path", default="../figures", type=str, help="Path to save visuals. Default = ../figures"
    )
    parser.add_argument(
        "--data-dir", default="../data", type=str, help="Base directory to read data. Defalut = ../data"
    )
    parser.add_argument("--trace-type", default="dff", type=str, help="fluorescent traces' data type. Default = dff")
    parser.add_argument("--show", default=False, type=str2bool, help="Show plots during execution? Default = False.")
    parser.add_argument("--save", default=True, type=str2bool, help="Save plots during execution? Default = True.")

    args = parser.parse_args()

    torch_wt = read_windowed_dff_traces(args.data_dir + "/V1 Responses/" + args.trace_type + ".h5")
    torch_wf = read_windowed_movie_frames(args.data_dir + "/V1 Stimuli/" + args.stimuli_name)

    splitted10_wt = torch.split(torch.moveaxis(torch_wt, 1, 0), 10, 0)
    splitted10_wf = torch.split(torch_wf, 10, 0)

    print("incomplete batch", splitted10_wf[-1].shape)
    print("batches_amount(10r/b): ", len(splitted10_wf))
    print("incomplete traces batch", splitted10_wt[-1].shape)
    print("traces_batches_amount(10r/b): ", len(splitted10_wt))

    lca = LCAConv3D(140, 1, "linked_dictionary_tests", 9, 9, 7, 2, 2, 1, 0.5, no_time_pad=True, lca_iters=200)

    with torch.no_grad():
        ckpt = torch.load("../models/lca_imagenet_vid_dict.pth", map_location="cpu")
        lca.assign_weight_values(ckpt.module.weights)

    lca = torch.nn.DataParallel(lca).cuda()
    lca.n_cells = 100  # change to how many cells used

    for ind, (windowed_frames_batch, windowed_traces_batch) in enumerate(zip(splitted10_wf, splitted10_wt)):

        windowed_frames_batch = windowed_frames_batch.cuda()
        windowed_frames_batch = torch.unsqueeze(windowed_frames_batch, 1)
        windowed_frames_batch = make_unit_var(make_zero_mean(windowed_frames_batch))

        windowed_traces_batch = make_unit_var(make_zero_mean(windowed_traces_batch[:, :100].cuda()))

        code = lca((windowed_frames_batch, windowed_traces_batch))

        recon, trace_recon = lca.module.compute_recon(code, lca.module.weights, True)

        lca.module.update_weights(code, windowed_frames_batch - recon, windowed_traces_batch - trace_recon)

        if ind % CONFIG.PRINT_FREQ == 0:
            l1_sparsity = compute_l1_sparsity(code, lca.module.lambda_).item()
            l2_recon_error = compute_l2_error(windowed_frames_batch, recon).item()
            l2_trace_recon_error = compute_l2_error(windowed_traces_batch, trace_recon).item()
            total_energy = l2_recon_error + l1_sparsity + l2_trace_recon_error

            print(
                f"L2 Trace Recon Error: {round(l2_trace_recon_error, 2)}; ",
                f"L2 Recon Error: {round(l2_recon_error, 2)}; ",
                f"L1 Sparsity: {round(l1_sparsity, 2)}; ",
                f"Total Energy: {round(total_energy, 2)}; ",
            )
            print(f"acts_all: {code.shape}, nonzero_counter on activity: {torch.count_nonzero(code)}")

        plot_1input_2recon_3error(windowed_frames_batch, recon, args.save, args.show, args.vis_path, ind)

    torch.save(lca, f"{args.model_path}/default_linked_dictionary.pth")

    spatiotemporal_dictionary(lca, args.save, args.show, args.vis_path)

    activation_histogram(lca, args.save, args.show, args.vis_path)
