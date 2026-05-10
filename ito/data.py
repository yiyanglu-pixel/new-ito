import os

import mdshare
import mdtraj as md
import numpy as np
import torch
from torch.utils import data
from torch_geometric.data import Data as GeometricData

from ito import utils


class StochasticLaggedDataset(data.Dataset):
    def __init__(self, trajs, max_lag, fixed_lag=False):
        self.max_lag = max_lag
        trajs = [traj for traj in trajs if len(traj) > max_lag]
        self.data = np.concatenate(trajs)
        self.data0_idx = np.zeros(len(self.data) - max_lag * len(trajs), dtype=int)
        self.fixed_lag = fixed_lag

        l = 0
        l0 = 0
        for traj in trajs:
            dl = len(traj)
            dl0 = len(traj) - max_lag
            self.data0_idx[l0 : l0 + dl0] = range(l, l + dl0)
            l += dl
            l0 += dl0

    def __len__(self):
        return len(self.data0_idx)

    def __getitem__(self, idx):
        log_lag = np.random.uniform(0, np.log(self.max_lag))
        lag = int(np.floor(np.exp(log_lag)))

        if self.fixed_lag:
            lag = self.max_lag

        data0_idx = self.data0_idx[idx]
        data0 = self.data[data0_idx]
        datat = self.data[data0_idx + lag]

        return self.process(data0, datat, lag)

    def process(self, x0, xt, t):
        raise NotImplementedError


class ALA2Dataset(StochasticLaggedDataset):
    def __init__(
        self,
        max_lag,
        distinguish=False,
        scale=False,
        fixed_lag=False,
        path=None,
        split="all",
    ):
        self.atom_numbers = get_ala2_atom_numbers(distinguish=distinguish)
        trajs = get_ala2_trajs(path, scale)
        trajs = select_ala2_split(trajs, split)

        super().__init__(trajs, max_lag, fixed_lag=fixed_lag)

    def process(self, x0, xt, t):
        batch_0 = utils.get_cond_batch(self.atom_numbers, x0, t)
        batch_t = utils.get_cond_batch(self.atom_numbers, xt, t)
        return {"batch_0": batch_0, "batch_t": batch_t}


class MidpointBridgeDataset(data.Dataset):
    def __init__(self, trajs, tau):
        assert tau % 2 == 0, "tau must be even so that tau/2 lands on a stored frame"
        self.tau = tau
        self.tau_half = tau // 2

        trajs = [traj for traj in trajs if len(traj) > tau]
        self.data = np.concatenate(trajs)
        self.data0_idx = np.zeros(
            len(self.data) - tau * len(trajs), dtype=int
        )

        l = 0
        l0 = 0
        for traj in trajs:
            dl = len(traj)
            dl0 = len(traj) - tau
            self.data0_idx[l0 : l0 + dl0] = range(l, l + dl0)
            l += dl
            l0 += dl0

    def __len__(self):
        return len(self.data0_idx)

    def __getitem__(self, idx):
        i = self.data0_idx[idx]
        x0 = self.data[i]
        xmid = self.data[i + self.tau_half]
        xT = self.data[i + self.tau]
        return self.process(x0, xmid, xT)

    def process(self, x0, xmid, xT):
        raise NotImplementedError


class ALA2BridgeDataset(MidpointBridgeDataset):
    def __init__(self, tau, distinguish=False, scale=False, path=None, split="all"):
        self.atom_numbers = get_ala2_atom_numbers(distinguish=distinguish)
        trajs = get_ala2_trajs(path, scale)
        trajs = select_ala2_split(trajs, split)
        super().__init__(trajs, tau)

    def process(self, x0, xmid, xT):
        batch_0 = utils.get_bridge_batch(self.atom_numbers, x0)
        batch_mid = utils.get_bridge_batch(self.atom_numbers, xmid)
        batch_T = utils.get_bridge_batch(self.atom_numbers, xT)
        return {"batch_0": batch_0, "batch_mid": batch_mid, "batch_T": batch_T}


def select_ala2_split(trajs, split):
    """Train on the first two MD trajectories, hold out the third for evaluation."""
    if split == "all":
        return trajs
    if split == "train":
        return trajs[:2]
    if split == "test":
        return trajs[2:]
    raise ValueError(f"unknown ala2 split: {split!r}")


def get_valid_starts(trajs, horizon):
    """Return start indices that keep a rollout inside one selected trajectory."""
    valid_starts = []
    offset = 0
    for traj in trajs:
        if len(traj) > horizon:
            valid_starts.extend(range(offset, offset + len(traj) - horizon))
        offset += len(traj)
    return np.array(valid_starts, dtype=int)


def validate_start_indices(start_idx, valid_starts):
    invalid = np.setdiff1d(start_idx, valid_starts, assume_unique=False)
    if len(invalid):
        preview = ", ".join(str(i) for i in invalid[:5])
        raise ValueError(
            f"start_indices contains {len(invalid)} invalid starts "
            f"for the selected split/horizon, e.g. {preview}"
        )


def sample_start_indices(trajs, horizon, n_samples, seed, start_indices=None):
    """Choose or validate fixed starts for pair-matched ALA2 evaluation."""
    valid_starts = get_valid_starts(trajs, horizon)
    if start_indices is not None:
        start_idx = np.asarray(start_indices, dtype=int)
        validate_start_indices(start_idx, valid_starts)
        return start_idx

    if len(valid_starts) < n_samples:
        raise ValueError(
            f"only {len(valid_starts)} valid starts for horizon={horizon}; "
            f"need n_samples={n_samples}"
        )

    rng = np.random.default_rng(seed)
    return rng.choice(valid_starts, size=n_samples, replace=False)


def gather_lagged_frames(flat_trajs, start_idx, lag, n_steps):
    offsets = np.arange(n_steps + 1, dtype=int) * lag
    return flat_trajs[start_idx[:, None] + offsets[None, :]]


def get_ala2_trajs(path=None, scale=False):
    filenames = download_ala2_trajs(path)

    topology = mdshare.fetch("alanine-dipeptide-nowater.pdb", path)
    trajs = [md.load_xtc(fn, topology) for fn in filenames]
    trajs = [t.center_coordinates().xyz for t in trajs]

    if scale:
        std = 0.1661689
        trajs = [t / std for t in trajs]

    return trajs


def download_ala2_trajs(path="."):
    if not path:
        path = "data/ala2/"
    if not os.path.exists(path):
        print(f"downloading alanine-dipeptide dataset to {path} ...")

    filenames = [
        os.path.join(f"alanine-dipeptide-{i}-250ns-nowater.xtc") for i in range(3)
    ]

    local_filenames = [
        mdshare.fetch(
            filename,
            working_directory=path,
        )
        for filename in filenames
    ]

    return local_filenames


def get_ala2_atom_numbers(distinguish=False):
    # fmt: off
    ALA2ATOMNUMBERS = [1, 6, 1, 1, 6, 8, 7, 1, 6, 1, 6, 1, 1, 1, 6, 8, 7, 1, 6, 1, 1, 1]
    # fmt: on
    atom_numbers = torch.tensor(  # pylint: disable=not-callable
        list(range(len(ALA2ATOMNUMBERS))) if distinguish else ALA2ATOMNUMBERS,
        dtype=torch.long,
    )
    return atom_numbers


def get_ala2_top(root):
    path = os.path.join(root, "data/ala2")
    topology = mdshare.fetch("alanine-dipeptide-nowater.pdb", path)
    top = md.load(topology).topology
    return top
