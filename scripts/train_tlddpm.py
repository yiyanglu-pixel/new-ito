import json
import os
import time
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from ito import data, utils
from ito.model import cpainn, ddpm


def main(args):
    pl.seed_everything(args.seed, workers=True)

    score_model_class = cpainn.PaiNNTLScore
    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    timestamp = utils.get_timestamp()
    train_dir = os.path.join(args.root, "train", timestamp)
    checkpoint_dir = os.path.join(train_dir, "checkpoints")
    train_dir_link = os.path.join(args.root, "train", "latest")
    best_checkpoint_link = os.path.join(train_dir, "best")

    print(f"saving checkpoints to {checkpoint_dir}")
    os.makedirs(train_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(train_dir, "args.json"), "w"), indent=4)

    score_model_kwargs = {
        "n_features": args.n_features,
        "n_layers": args.n_layers,
        "max_lag": args.max_lag,
        "diff_steps": args.diff_steps,
        "n_neighbors": args.n_neighbors,
        "length_scale": args.length_scale,
        "dist_encoding": args.dist_encoding,
    }

    model = ddpm.TLDDPM(
        score_model_class,
        score_model_kwargs=score_model_kwargs,
        diffusion_steps=args.diff_steps,
        lr=args.lr,
    )

    dataset = data.ALA2Dataset(
        path=ala2_path,
        max_lag=args.max_lag,
        distinguish=not args.indistinguishable,
        fixed_lag=args.fixed_lag,
        scale=not args.unscaled,
        split=args.split,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1, save_last=True, dirpath=checkpoint_dir, filename="{epoch}"
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback],
    )

    utils.reset_peak_cuda_memory()
    train_start = time.perf_counter()
    trainer.fit(model, dataloader)
    runtime = {
        "train_seconds": time.perf_counter() - train_start,
        "peak_cuda_memory_bytes": utils.peak_cuda_memory_bytes(),
    }
    json.dump(runtime, open(os.path.join(train_dir, "runtime.json"), "w"), indent=4)
    best_model_path = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )
    utils.replace_symlink(
        src=os.path.abspath(best_model_path),
        dst=best_checkpoint_link,
    )
    utils.replace_symlink(src=os.path.abspath(train_dir), dst=train_dir_link)
    print(f"best model checkpoint: {best_checkpoint_link}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--root",              type=str,            default="storage", help="Base directory for storing data and training outputs.")
    parser.add_argument("--n_features",        type=int,            default=64,        help="Number of features for the model.")
    parser.add_argument("--n_layers",          type=int,            default=2,         help="Number of layers in the model.")
    parser.add_argument("--n_neighbors",       type=int,            default=100,       help="Number of nearest-neighbor edges in PaiNN message passing.")
    parser.add_argument("--length_scale",      type=float,          default=10.0,      help="PaiNN distance encoding length scale. Use 3.0 for the paper-alignment sensitivity run.")
    parser.add_argument("--dist_encoding",     default="positional_encoding", choices=("positional_encoding", "soft_one_hot"), help="PaiNN distance encoding.")
    parser.add_argument("--epochs",            type=int,            default=50,        help="Number of training epochs.")
    parser.add_argument("--diff_steps",        type=int,            default=1000,      help="Number of diffusion steps in the model.")
    parser.add_argument("--batch_size",        type=int,            default=128,       help="Batch size for training.")
    parser.add_argument("--lr",                type=float,          default=1e-3,      help="Learning rate for the optimizer.")
    parser.add_argument("--max_lag",           type=int,            default=1000,      help="Maximum lag to consider in the ALA2 dataset.")
    parser.add_argument("--split",             default="train",     choices=("train", "test", "all"), help="ALA2 split. 'train' uses trajs 0,1; 'test' uses traj 2.")
    parser.add_argument("--seed",              type=int,            default=0,         help="Random seed for training.")
    parser.add_argument("--devices",           type=int,            default=1,         help="Number of devices for pl.Trainer. Defaults to 1 to avoid silently launching multi-GPU DDP.")
    parser.add_argument("--accelerator",       default="auto",      help="pl.Trainer accelerator (e.g. 'gpu', 'cpu', 'auto').")
    parser.add_argument("--fixed_lag",         action='store_true', help="Enable to use a fixed lag value, disabled by default.")
    parser.add_argument("--indistinguishable", action='store_true', help="Enable this flag to treat atoms as indistinguishable.")
    parser.add_argument("--unscaled",          action='store_true', help="Use unscaled data. When disabled, data is scaled to unit variance.")

    # fmt: on

    main(parser.parse_args())
