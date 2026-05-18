import json
import os
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from deeptime import decomposition

from ito import data, utils


def main(args):
    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    sample_dir = os.path.dirname(os.path.realpath(args.trajs))
    sample_args = load_sample_args(sample_dir)
    lag = args.lag if args.lag is not None else sample_args.get("lag", 100)
    sampling_split = sample_args.get("split", "all")
    ref_split = args.split if args.split != "match_sampling" else sampling_split

    topology = data.get_ala2_top(args.root)
    analysis_dir = os.path.join(args.root, "analysis", utils.get_timestamp())
    os.makedirs(analysis_dir, exist_ok=True)

    args_for_json = args.__dict__.copy()
    args_for_json["resolved_lag"] = lag
    args_for_json["sampling_split"] = sampling_split
    args_for_json["ref_split"] = ref_split
    json.dump(
        args_for_json, open(os.path.join(analysis_dir, "args.json"), "w"), indent=4
    )

    trajs = np.load(args.trajs)
    vamp2_score = get_vamp2(trajs=trajs, topology=topology, lag=1)

    ref_trajs = data.select_ala2_split(data.get_ala2_trajs(ala2_path), ref_split)
    ref_vamp2_score = get_vamp2(ref_trajs, topology=topology, lag=lag)

    print(f"VAMP2 score: {vamp2_score}")
    print(f"Reference VAMP2 score: {ref_vamp2_score}")

    phi, psi = compute_dihedral_angles(np.concatenate(trajs), topology)
    phi_ref, psi_ref = compute_dihedral_angles(np.concatenate(ref_trajs), topology)
    metrics = {
        "model": "ito",
        "vamp2": float(vamp2_score),
        "ref_vamp2": float(ref_vamp2_score),
        "ref_lag": int(lag),
        "lag": int(lag),
        "traj_length": int(trajs.shape[1] - 1),
        "n_pairs": int(trajs.shape[0]),
        "sampling_split": sampling_split,
        "ref_split": ref_split,
        "seed": sample_args.get("seed"),
        "grid": sample_args.get("grid"),
    }
    metrics.update(load_checkpoint_metadata(sample_args))
    metrics.update(load_runtime_metrics(sample_dir, prefix="sampling"))
    metrics.update(load_checkpoint_runtime_metrics(sample_args, prefix="training"))
    metrics.update(distribution_metrics(phi, psi, phi_ref, psi_ref, bins=args.bins))
    metrics.update(stability_metrics(trajs, coord_threshold=args.coord_threshold))

    md_reference_path = os.path.join(sample_dir, "md_reference.npy")
    if os.path.exists(md_reference_path):
        md_reference = np.load(md_reference_path)
        md_phi, md_psi = compute_dihedral_angles(
            md_reference.reshape(-1, *md_reference.shape[2:]), topology
        )
        metrics["md_reference_phi_circular_rmse"] = float(
            circular_rmse(phi, md_phi)
        )
        metrics["md_reference_psi_circular_rmse"] = float(
            circular_rmse(psi, md_psi)
        )

    json.dump(
        metrics,
        open(os.path.join(analysis_dir, "metrics.json"), "w"),
        indent=4,
    )
    json.dump(
        {"vamp2": metrics["vamp2"], "ref_vamp2": metrics["ref_vamp2"]},
        open(os.path.join(analysis_dir, "vamp2_scores.json"), "w"),
        indent=4,
    )

    _, ax = plt.subplots(1, 2, figsize=(10, 4))

    plot_marginal(ax[0], phi)
    plot_marginal(ax[0], phi_ref, label="MD")
    ax[0].set_xlabel("Phi")
    ax[0].set_ylabel("Frequency")
    if not args.no_plot_start:
        ax[0].vlines(phi[0], 0, 1, linestyle="--", color="k")

    plot_marginal(ax[1], psi, label="ITO")
    plot_marginal(ax[1], psi_ref, label="MD")
    ax[1].set_xlabel("Psi")
    if not args.no_plot_start:
        ax[1].vlines(psi[0], 0, 1, linestyle="--", color="k", label="Start")

    ax[1].legend()
    plt.savefig(os.path.join(analysis_dir, "marginals.pdf"))

    _, ax = plt.subplots(figsize=(5, 4))
    plt.plot(phi, psi, ".", alpha=0.1, label="ITO")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    if not args.no_plot_start:
        plt.plot(phi[0], psi[0], "k", marker="x", label="Start")

    ax.legend()
    plt.savefig(os.path.join(analysis_dir, "ramachandran.pdf"))
    print(f"analysis saved at {analysis_dir}")


def load_sample_args(sample_dir):
    args_path = os.path.join(sample_dir, "args.json")
    if os.path.exists(args_path):
        return json.load(open(args_path))
    return {}


def load_runtime_metrics(sample_dir, prefix):
    runtime_path = os.path.join(sample_dir, "runtime.json")
    if not os.path.exists(runtime_path):
        return {}
    runtime = json.load(open(runtime_path))
    return {f"{prefix}_{key}": value for key, value in runtime.items()}


def load_checkpoint_runtime_metrics(sample_args, prefix):
    checkpoint = sample_args.get("checkpoint")
    if checkpoint is None:
        return {}

    runtime = load_checkpoint_train_runtime(checkpoint)
    if runtime:
        return {f"{prefix}_{key}": value for key, value in runtime.items()}
    return {}


def load_checkpoint_metadata(sample_args):
    checkpoint = sample_args.get("checkpoint")
    if checkpoint is None:
        return {}

    checkpoint_realpath = os.path.realpath(checkpoint)
    runtime = load_checkpoint_train_runtime(checkpoint)
    metadata = {
        "checkpoint_path": checkpoint,
        "checkpoint_realpath": checkpoint_realpath,
        "checkpoint_epoch": infer_checkpoint_epoch(checkpoint_realpath),
    }
    if runtime:
        train_status = runtime.get("train_status")
        if train_status is not None:
            metadata["train_status"] = train_status
        monitor = runtime.get("checkpoint_monitor")
        if monitor is not None:
            metadata["checkpoint_monitor"] = monitor
        metadata["checkpoint_selection"] = infer_checkpoint_selection(
            checkpoint,
            checkpoint_realpath,
            runtime,
        )
    else:
        metadata["checkpoint_selection"] = infer_checkpoint_selection(
            checkpoint,
            checkpoint_realpath,
            {},
        )
    return metadata


def load_checkpoint_train_runtime(checkpoint):
    checkpoint_path = os.path.realpath(checkpoint)
    candidates = [
        os.path.dirname(os.path.dirname(checkpoint_path)),
        os.path.dirname(checkpoint_path),
    ]
    for train_dir in candidates:
        runtime_path = os.path.join(train_dir, "runtime.json")
        if os.path.exists(runtime_path):
            return json.load(open(runtime_path))
    return {}


def infer_checkpoint_epoch(checkpoint_path):
    basename = os.path.basename(checkpoint_path)
    match = re.search(r"epoch=(\d+)", basename)
    if match:
        return int(match.group(1))
    if basename == "last.ckpt":
        return "last"
    return None


def infer_checkpoint_selection(checkpoint, checkpoint_realpath, runtime):
    best_model_path = runtime.get("best_model_path")
    if best_model_path and os.path.realpath(best_model_path) == checkpoint_realpath:
        return runtime.get("checkpoint_selection") or "best"

    if os.path.basename(checkpoint) == "best":
        return runtime.get("checkpoint_selection") or "best"

    if os.path.basename(checkpoint_realpath) == "last.ckpt":
        return "last"

    if runtime.get("train_status") == "failed_nan":
        return "manual_pre_nan"
    return "manual"


def distribution_metrics(phi, psi, phi_ref, psi_ref, bins=64):
    gen_rama, _, _ = np.histogram2d(
        phi, psi, bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]]
    )
    ref_rama, _, _ = np.histogram2d(
        phi_ref, psi_ref, bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]]
    )
    ref_occupied = ref_rama > 0
    gen_occupied = gen_rama > 0
    md_occupied_bins = max(int(ref_occupied.sum()), 1)
    gen_occupied_bins = max(int(gen_occupied.sum()), 1)

    return {
        "phi_jsd": float(jensen_shannon(circular_hist(phi, bins), circular_hist(phi_ref, bins))),
        "psi_jsd": float(jensen_shannon(circular_hist(psi, bins), circular_hist(psi_ref, bins))),
        "rama_jsd": float(jensen_shannon(normalize_hist(gen_rama), normalize_hist(ref_rama))),
        "rama_md_bin_coverage": float((gen_occupied & ref_occupied).sum() / md_occupied_bins),
        "rama_generated_outside_md": float(
            gen_rama[~ref_occupied].sum() / max(gen_rama.sum(), 1)
        ),
        "rama_generated_bin_count": int(gen_occupied_bins),
        "rama_md_bin_count": int(md_occupied_bins),
    }


def stability_metrics(trajs, coord_threshold=10.0):
    finite = np.isfinite(trajs)
    abs_coords = np.abs(trajs)
    return {
        "nan_count": int(np.isnan(trajs).sum()),
        "inf_count": int(np.isinf(trajs).sum()),
        "nonfinite_fraction": float(1.0 - finite.mean()),
        "max_abs_coord": float(np.nanmax(abs_coords)),
        "mean_abs_coord": float(np.nanmean(abs_coords)),
        "extreme_coord_fraction": float(np.nanmean(abs_coords > coord_threshold)),
        "coord_threshold": float(coord_threshold),
    }


def circular_hist(values, bins=64):
    counts, _ = np.histogram(values, range=(-np.pi, np.pi), bins=bins)
    return normalize_hist(counts)


def normalize_hist(counts):
    counts = np.asarray(counts, dtype=float).ravel()
    total = counts.sum()
    if total == 0:
        return np.ones_like(counts) / len(counts)
    return counts / total


def jensen_shannon(p, q, eps=1e-12):
    p = normalize_hist(p) + eps
    q = normalize_hist(q) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def circular_rmse(a, b):
    d = np.angle(np.exp(1j * (a - b)))
    return np.sqrt(np.mean(d**2))


def plot_marginal(ax, marginal, bins=64, label=None):
    bin_heights, _ = np.histogram(
        marginal, range=(-np.pi, np.pi), bins=bins, density=True
    )
    plot_marginal_dist(ax, bin_heights, label=label, linestyle="-")


def plot_marginal_dist(ax, bin_heights, linestyle="-", c=None, label=None):
    bin_edges = np.linspace(-np.pi, np.pi, len(bin_heights) + 1)
    bin_widths = np.diff(bin_edges)
    bin_heights /= bin_heights.mean() * bin_widths.sum()
    bin_heights = np.append(bin_heights, bin_heights[-1])

    ax.step(
        bin_edges,
        bin_heights,
        where="post",
        linestyle=linestyle,
        c=c,
        label=label,
    )
    ax.semilogy()


def compute_dihedral_angles(traj, topology):
    traj = md.Trajectory(xyz=traj, topology=topology)
    phi_atoms_idx = [4, 6, 8, 14]
    phi = md.compute_dihedrals(traj, indices=[phi_atoms_idx])[:, 0]
    psi_atoms_idx = [6, 8, 14, 16]
    psi = md.compute_dihedrals(traj, indices=[psi_atoms_idx])[:, 0]

    return phi, psi


def featurize_trajs(trajs, topology):
    featurized_trajs = np.stack([featurize_traj(traj, topology) for traj in trajs])
    nan_mask = np.isnan(featurized_trajs).any(axis=(1, 2))

    return featurized_trajs[~nan_mask]


def featurize_traj(traj, topology):
    phi, psi = compute_dihedral_angles(traj, topology)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    features = np.stack([cos_phi, sin_phi, cos_psi, sin_psi]).T
    return features


def get_vamp2(trajs, lag, topology):
    featurized_trajs = featurize_trajs(trajs, topology)
    vamp = decomposition.VAMP(lag).fit_fetch(featurized_trajs)
    vamp2_score = vamp.score(2)

    return vamp2_score


if __name__ == "__main__":
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument( "trajs",           nargs="?",            default="storage/samples/latest", help="Specify the path to the trajectory file containing the trajectories to be analyzed. Default is 'storage/samples/latest'. If the default path is unchanged, ensure that the sampling script has been run prior to analysis.")
    parser.add_argument( "--root",          default="storage",    help="Set the base directory where input data is located and where analysis outputs will be stored. The default directory is 'storage'. Modify this if your data and output directories are different.")
    parser.add_argument( "--lag",           type=int,             default=None,                     help="Reference MD lag in frames. Defaults to the lag recorded by sample_tlddpm.py, falling back to 100.")
    parser.add_argument( "--split",         default="match_sampling", choices=("train", "test", "all", "match_sampling"), help="ALA2 split for the MD reference. Default mirrors the sampling split.")
    parser.add_argument( "--bins",          type=int,             default=64,                       help="Number of bins for circular and Ramachandran distribution metrics.")
    parser.add_argument( "--coord_threshold", type=float,         default=10.0,                     help="Absolute coordinate threshold used for extreme-coordinate stability metrics.")
    parser.add_argument( "--no_plot_start", action="store_true",  help="Do not mark the starting point of generated trajectories in plots.")
    # fmt: on

    main(parser.parse_args())
