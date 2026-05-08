import json
import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch

from ito import data, utils
from ito.model import ddpm


def main(args):
    assert args.tau % (2**args.depth) == 0, (
        f"tau={args.tau} must be divisible by 2**depth={2**args.depth} so every "
        f"recursion midpoint lands on an integer MD frame for evaluation."
    )
    pl.seed_everything(args.seed, workers=True)

    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    samples_root = os.path.join(args.root, "samples_bridge")
    samples_dir = os.path.join(samples_root, utils.get_timestamp())
    samples_path = os.path.join(samples_dir, "trajectory.npy")
    md_midpoints_path = os.path.join(samples_dir, "md_midpoints.npy")
    endpoints_path = os.path.join(samples_dir, "endpoints.npy")

    trajs_list = data.get_ala2_trajs(ala2_path, not args.unscaled)
    trajs_list = data.select_ala2_split(trajs_list, args.split)
    ala2_atom_numbers = data.get_ala2_atom_numbers(not args.indistinguishable)

    valid_starts = []
    offset = 0
    for traj in trajs_list:
        if len(traj) > args.tau:
            valid_starts.extend(range(offset, offset + len(traj) - args.tau))
        offset += len(traj)
    valid_starts = np.array(valid_starts)
    assert len(valid_starts) >= args.n_pairs, (
        f"only {len(valid_starts)} valid starts in split={args.split!r}; "
        f"need n_pairs={args.n_pairs}"
    )

    model = ddpm.BridgeDDPM.load_from_checkpoint(args.checkpoint)
    model.eval()

    rng = np.random.default_rng(args.seed)
    ala2_trajs = np.concatenate(trajs_list)
    start_idx = rng.choice(valid_starts, size=args.n_pairs, replace=False)

    left_positions = ala2_trajs[start_idx]
    right_positions = ala2_trajs[start_idx + args.tau]
    md_midpoint_positions = ala2_trajs[start_idx + args.tau // 2]

    left_batch = utils.get_bridge_batch(ala2_atom_numbers, left_positions)
    right_batch = utils.get_bridge_batch(ala2_atom_numbers, right_positions)

    with torch.no_grad():
        filled = fill(left_batch, right_batch, args.depth, model, args.ode_steps)

    trajectory = np.stack(
        [utils.batch_to_numpy(b) for b in filled], axis=1
    )  # [n_pairs, 2^depth+1, n_atoms, 3]

    os.makedirs(samples_dir, exist_ok=True)
    json.dump(
        args.__dict__, open(os.path.join(samples_dir, "args.json"), "w"), indent=4
    )
    np.save(samples_path, trajectory)
    np.save(md_midpoints_path, md_midpoint_positions)
    np.save(
        endpoints_path,
        np.stack([left_positions, right_positions], axis=1),
    )

    latest_link = os.path.join(samples_root, "latest")
    if os.path.exists(latest_link) or os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(src=os.path.abspath(samples_path), dst=latest_link)

    print(f"samples saved at {samples_path}")
    print(
        f"shape: {trajectory.shape} (n_pairs, 2^depth+1, n_atoms, 3)  depth={args.depth}"
    )


def fill(left_batch, right_batch, depth, model, ode_steps):
    """Recursive midpoint subdivision returning depth+1 batches in temporal order."""
    if depth == 0:
        return [left_batch, right_batch]

    mid_batch = model.sample(left_batch, right_batch, ode_steps=ode_steps)
    mid_batch = mid_batch.detach()

    left_filled = fill(left_batch, mid_batch, depth - 1, model, ode_steps)
    right_filled = fill(mid_batch, right_batch, depth - 1, model, ode_steps)
    return left_filled[:-1] + right_filled


if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("checkpoint",          nargs="?",           default="storage/train_bridge/latest/best", help="Path to the BridgeDDPM checkpoint.")
    parser.add_argument("--root",              default="storage",   help="Base path for input data and where output samples will be stored.")
    parser.add_argument("--n_pairs",           type=int,            default=10,                                 help="Number of (x_t, x_{t+tau}) pairs to draw from MD data and fill in.")
    parser.add_argument("--tau",               type=int,            default=1000,                               help="Endpoint separation in MD frames. Must match the tau used during training.")
    parser.add_argument("--depth",             type=int,            default=3,                                  help="Recursion depth k; output trajectory has 2^k+1 frames per pair. Must satisfy tau %% 2**depth == 0.")
    parser.add_argument("--ode_steps",         type=int,            default=50,                                 help="Number of steps for the DPM-Solver during sampling. Set to 0 for vanilla denoising.")
    parser.add_argument("--seed",              type=int,            default=0,                                  help="RNG seed for selecting endpoint pairs and diffusion noise.")
    parser.add_argument("--split",             default="test",      choices=("train", "test", "all"),           help="ALA2 split for endpoint sampling. Default 'test' (held-out traj 2).")
    parser.add_argument("--grid",              default=None,                                                    help="Free-form tag written to args.json so summarisers can group runs.")
    parser.add_argument("--indistinguishable", action="store_true", help="Treat atoms as indistinguishable; must match training.")
    parser.add_argument("--unscaled",          action="store_true", help="Use unscaled data; must match training.")
    # fmt: on

    main(parser.parse_args())
