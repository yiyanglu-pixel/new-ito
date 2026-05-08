import json
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

from analyse_trajs import compute_dihedral_angles, get_vamp2, plot_marginal

from ito import data, utils


def main(args):
    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    topology = data.get_ala2_top(args.root)
    analysis_dir = os.path.join(args.root, "analysis_bridge", utils.get_timestamp())
    os.makedirs(analysis_dir, exist_ok=True)

    json.dump(
        args.__dict__, open(os.path.join(analysis_dir, "args.json"), "w"), indent=4
    )

    sample_dir = os.path.dirname(os.path.realpath(args.trajs))
    sample_args = json.load(open(os.path.join(sample_dir, "args.json")))
    tau = sample_args["tau"]
    sampling_split = sample_args.get("split", "all")

    trajs = np.load(args.trajs)  # [n_pairs, 2^depth+1, n_atoms, 3]
    endpoints = np.load(os.path.join(sample_dir, "endpoints.npy"))  # [n_pairs, 2, n_atoms, 3]
    md_mid = np.load(os.path.join(sample_dir, "md_midpoints.npy"))  # [n_pairs, n_atoms, 3]

    n_pairs, n_frames, n_atoms, _ = trajs.shape
    depth = int(np.log2(n_frames - 1))
    assert tau % (2**depth) == 0, (
        f"tau={tau} not divisible by 2**depth={2**depth}; "
        f"VAMP2 reference lag would not match the bridge frame spacing exactly."
    )
    ref_lag = tau // (2**depth)
    print(
        f"loaded {n_pairs} pairs, {n_frames} frames each "
        f"(depth={depth}, tau={tau}, ref_lag={ref_lag}, sampling_split={sampling_split!r})"
    )

    closure_check(trajs, endpoints)

    flat = trajs.reshape(-1, n_atoms, 3)
    phi, psi = compute_dihedral_angles(flat, topology)

    ref_split = args.split if args.split != "match_sampling" else sampling_split
    ref_trajs = data.select_ala2_split(data.get_ala2_trajs(ala2_path), ref_split)
    ref_flat = np.concatenate(ref_trajs)
    phi_ref, psi_ref = compute_dihedral_angles(ref_flat, topology)

    # Distribution metrics on full filled trajectory.
    vamp2_score = get_vamp2(trajs=trajs, topology=topology, lag=1)
    ref_vamp2_score = get_vamp2(ref_trajs, topology=topology, lag=ref_lag)
    print(f"VAMP2 (bridge, lag=1): {vamp2_score}")
    print(f"VAMP2 (MD,     lag={ref_lag}): {ref_vamp2_score}")

    # Depth-1-style midpoint metric: bridge midpoint (the centre frame) vs MD midpoint.
    mid_idx = n_frames // 2
    bridge_mid_phi, bridge_mid_psi = compute_dihedral_angles(
        trajs[:, mid_idx], topology
    )
    md_mid_phi, md_mid_psi = compute_dihedral_angles(md_mid, topology)
    midpoint_metrics = {
        "bridge_midpoint_phi_mean": float(bridge_mid_phi.mean()),
        "bridge_midpoint_psi_mean": float(bridge_mid_psi.mean()),
        "md_midpoint_phi_mean": float(md_mid_phi.mean()),
        "md_midpoint_psi_mean": float(md_mid_psi.mean()),
        "phi_circular_rmse": float(circular_rmse(bridge_mid_phi, md_mid_phi)),
        "psi_circular_rmse": float(circular_rmse(bridge_mid_psi, md_mid_psi)),
    }

    json.dump(
        {
            "vamp2": float(vamp2_score),
            "ref_vamp2": float(ref_vamp2_score),
            "ref_lag": ref_lag,
            "tau": tau,
            "depth": depth,
            "n_pairs": n_pairs,
            "sampling_split": sampling_split,
            "ref_split": ref_split,
            "midpoint": midpoint_metrics,
        },
        open(os.path.join(analysis_dir, "metrics.json"), "w"),
        indent=4,
    )

    _, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_marginal(ax[0], phi)
    plot_marginal(ax[0], phi_ref, label="MD")
    ax[0].set_xlabel("Phi")
    ax[0].set_ylabel("Frequency")

    plot_marginal(ax[1], psi, label="Bridge")
    plot_marginal(ax[1], psi_ref, label="MD")
    ax[1].set_xlabel("Psi")
    ax[1].legend()
    plt.savefig(os.path.join(analysis_dir, "marginals.pdf"))

    _, ax = plt.subplots(figsize=(5, 4))
    plt.plot(phi, psi, ".", alpha=0.1, label="Bridge")
    plt.plot(bridge_mid_phi, bridge_mid_psi, "rx", alpha=0.6, label="Bridge midpoint")
    plt.plot(md_mid_phi, md_mid_psi, "g+", alpha=0.6, label="MD midpoint")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.legend()
    plt.savefig(os.path.join(analysis_dir, "ramachandran.pdf"))


def closure_check(trajs, endpoints):
    """Verify the recursion's first/last frame match the input endpoints exactly."""
    left_err = np.max(np.abs(trajs[:, 0] - endpoints[:, 0]))
    right_err = np.max(np.abs(trajs[:, -1] - endpoints[:, 1]))
    print(f"closure check: max left endpoint error  = {left_err:.3e}")
    print(f"closure check: max right endpoint error = {right_err:.3e}")
    assert left_err < 1e-6, "left endpoint drifted during recursion"
    assert right_err < 1e-6, "right endpoint drifted during recursion"


def circular_rmse(a, b):
    d = np.angle(np.exp(1j * (a - b)))
    return np.sqrt(np.mean(d**2))


if __name__ == "__main__":
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("trajs",   nargs="?", default="storage/samples_bridge/latest",  help="Path to the bridge trajectory .npy. tau is read from the sampling args.json next to it.")
    parser.add_argument("--root",  default="storage",                                   help="Base directory used by sampling.")
    parser.add_argument("--split", default="match_sampling", choices=("train", "test", "all", "match_sampling"), help="ALA2 split for the MD reference. Default mirrors the sampling split for an apples-to-apples comparison.")
    # fmt: on

    main(parser.parse_args())
