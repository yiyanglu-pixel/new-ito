import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from ito import data, utils
from ito.model import ddpm


def main(args):
    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    samples_root = os.path.join(args.root, "samples")
    samples_dir = os.path.join(samples_root, utils.get_timestamp())
    samples_path = os.path.join(samples_dir, "trajectory.npy")
    md_reference_path = os.path.join(samples_dir, "md_reference.npy")
    start_indices_path = os.path.join(samples_dir, "start_indices.npy")

    trajs_list = data.get_ala2_trajs(ala2_path, not args.unscaled)
    trajs_list = data.select_ala2_split(trajs_list, args.split)
    ala2_trajs = np.concatenate(trajs_list)
    ala2_atom_numbers = data.get_ala2_atom_numbers(not args.indistinguishable)

    model = ddpm.TLDDPM.load_from_checkpoint(args.checkpoint)
    model.eval()
    torch.manual_seed(args.seed)

    horizon = args.lag * args.traj_length
    if args.single_start:
        start_idx = np.zeros(args.samples, dtype=int)
    else:
        start_indices = (
            np.load(args.start_indices) if args.start_indices is not None else None
        )
        start_idx = data.sample_start_indices(
            trajs_list,
            horizon=horizon,
            n_samples=args.samples,
            seed=args.seed,
            start_indices=start_indices,
        )
        args.samples = len(start_idx)

    init_positions = ala2_trajs[start_idx]
    md_reference = data.gather_lagged_frames(
        ala2_trajs, start_idx, lag=args.lag, n_steps=args.traj_length
    )

    batch = utils.get_cond_batch(ala2_atom_numbers, init_positions, args.lag)

    trajectory = [utils.batch_to_numpy(batch)]

    utils.reset_peak_cuda_memory()
    sample_start = time.perf_counter()
    for _ in tqdm(range(args.traj_length)):
        batch = model.sample(batch, ode_steps=args.ode_steps)
        trajectory.append(utils.batch_to_numpy(batch))
    runtime = {
        "sample_seconds": time.perf_counter() - sample_start,
        "peak_cuda_memory_bytes": utils.peak_cuda_memory_bytes(),
    }

    trajectory = np.stack(trajectory, axis=1)

    os.makedirs(samples_dir, exist_ok=True)
    json.dump(
        args.__dict__, open(os.path.join(samples_dir, "args.json"), "w"), indent=4
    )
    np.save(samples_path, trajectory)
    np.save(md_reference_path, md_reference)
    np.save(start_indices_path, start_idx)
    json.dump(runtime, open(os.path.join(samples_dir, "runtime.json"), "w"), indent=4)
    utils.replace_symlink(
        src=os.path.abspath(samples_path), dst=os.path.join(samples_root, "latest")
    )

    print(f"samples saved at {samples_path}")



if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("checkpoint",          nargs="?",           default="storage/train/latest/best", help="Path to the model checkpoint file for generating trajectories. Default is the best model from the latest training session.")
    parser.add_argument("--root",              default="storage",   help="Base path for input data and where output samples will be stored.")
    parser.add_argument("--samples",           type=int,            default=10,                          help="The number of initial positions to sample trajectories from, processed in parallel.")
    parser.add_argument("--traj_length",       type=int,            default=100,                         help="The total number of steps (frames) in each generated trajectory.")
    parser.add_argument("--lag",               type=int,            default=100,                         help="Temporal lag between consecutive steps (frames) in the generated trajectory.")
    parser.add_argument("--ode_steps",         type=int,            default=50,                          help="Number of steps for the ODE solver during sampling. Set to 0 for normal denoising.")
    parser.add_argument("--seed",              type=int,            default=0,                           help="RNG seed for selecting test starts and diffusion sampling.")
    parser.add_argument("--split",             default="test",      choices=("train", "test", "all"),    help="ALA2 split for start-frame sampling. Default 'test' (held-out traj 2).")
    parser.add_argument("--grid",              default=None,                                             help="Optional evaluation grid label (e.g. A or B) saved to args.json.")
    parser.add_argument("--start_indices",     default=None,                                             help="Optional .npy file with fixed start indices relative to the selected split.")
    parser.add_argument("--indistinguishable", action="store_true", help="Enable this flag to treat atoms as indistinguishable in the model.")
    parser.add_argument("--unscaled",          action="store_true", help="Enable this flag to use unscaled data. By default, data is scaled to have unit variance.")
    parser.add_argument("--init_from_eq",      action="store_true", help="Deprecated alias retained for old commands; fixed random split starts are now the default.")
    parser.add_argument("--single_start",      action="store_true", help="Legacy diagnostic: repeat the first selected-split frame as every initial condition instead of fixed random test starts.")
    # fmt: on

    main(parser.parse_args())
