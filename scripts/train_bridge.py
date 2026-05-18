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
    assert args.tau % 2 == 0, "tau must be even so that tau/2 lands on a stored frame"
    pl.seed_everything(args.seed, workers=True)

    score_model_class = cpainn.PaiNNBridgeScore
    args.root = os.path.realpath(args.root)
    ala2_path = os.path.join(args.root, "data/ala2")
    timestamp = utils.get_timestamp()
    train_dir = os.path.join(args.root, "train_bridge", timestamp)
    checkpoint_dir = os.path.join(train_dir, "checkpoints")
    train_dir_link = os.path.join(args.root, "train_bridge", "latest")
    best_checkpoint_link = os.path.join(train_dir, "best")

    print(f"saving checkpoints to {checkpoint_dir}")
    os.makedirs(train_dir, exist_ok=True)

    if args.val_fraction < 0 or args.val_fraction >= 1:
        raise ValueError("val_fraction must be in [0, 1)")
    use_validation = not args.no_validation and args.val_fraction > 0
    args.validation_enabled = use_validation
    args.checkpoint_selection = "val_best" if use_validation else "last"

    score_model_kwargs = {
        "n_features": args.n_features,
        "n_layers": args.n_layers,
        "diff_steps": args.diff_steps,
        "n_neighbors": args.n_neighbors,
        "length_scale": args.length_scale,
        "dist_encoding": args.dist_encoding,
    }

    model = ddpm.BridgeDDPM(
        score_model_class,
        score_model_kwargs=score_model_kwargs,
        diffusion_steps=args.diff_steps,
        lr=args.lr,
        scheduler_t_max=args.scheduler_t_max,
        ema_decay=args.ema_decay,
    )

    train_trajs = data.select_ala2_split(
        data.get_ala2_trajs(ala2_path, not args.unscaled), args.split
    )
    val_trajs = None
    if use_validation:
        train_trajs, val_trajs = data.split_train_validation_trajs(
            train_trajs,
            val_fraction=args.val_fraction,
            min_length=args.tau + 1,
        )

    dataset = data.ALA2BridgeDataset(
        path=ala2_path,
        tau=args.tau,
        distinguish=not args.indistinguishable,
        scale=not args.unscaled,
        trajs=train_trajs,
    )
    val_dataset = None
    if val_trajs is not None:
        val_dataset = data.ALA2BridgeDataset(
            path=ala2_path,
            tau=args.tau,
            distinguish=not args.indistinguishable,
            scale=not args.unscaled,
            trajs=val_trajs,
        )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    args.train_dataset_size = len(dataset)
    args.val_dataset_size = len(val_dataset) if val_dataset is not None else 0
    json.dump(args.__dict__, open(os.path.join(train_dir, "args.json"), "w"), indent=4)

    checkpoint_kwargs = {
        "save_last": True,
        "dirpath": checkpoint_dir,
        "filename": "{epoch}",
    }
    if use_validation:
        checkpoint_kwargs.update(
            monitor=args.checkpoint_monitor,
            mode="min",
            save_top_k=args.save_top_k,
        )
    else:
        checkpoint_kwargs["save_top_k"] = -1
    checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback],
    )

    utils.reset_peak_cuda_memory()
    train_start = time.perf_counter()
    train_status = "completed"
    failure = None
    try:
        if val_dataloader is not None:
            trainer.fit(model, dataloader, val_dataloader)
        else:
            trainer.fit(model, dataloader)
    except Exception as exc:
        train_status = "failed_nan" if "loss is nan" in str(exc).lower() else "failed"
        failure = repr(exc)
        raise
    finally:
        best_model_path = (
            checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
        )
        runtime = {
            "train_seconds": time.perf_counter() - train_start,
            "peak_cuda_memory_bytes": utils.peak_cuda_memory_bytes(),
            "train_status": train_status,
            "checkpoint_selection": args.checkpoint_selection,
            "checkpoint_monitor": args.checkpoint_monitor if use_validation else None,
            "best_model_path": best_model_path or None,
            "last_model_path": checkpoint_callback.last_model_path or None,
        }
        if failure is not None:
            runtime["failure"] = failure
        json.dump(runtime, open(os.path.join(train_dir, "runtime.json"), "w"), indent=4)
        if best_model_path:
            utils.replace_symlink(
                src=os.path.abspath(best_model_path),
                dst=best_checkpoint_link,
            )
            print(f"best model checkpoint: {best_checkpoint_link}")
        utils.replace_symlink(src=os.path.abspath(train_dir), dst=train_dir_link)


if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--root",              type=str,            default="storage", help="Base directory for storing data and training outputs.")
    parser.add_argument("--n_features",        type=int,            default=64,        help="Number of features for the model.")
    parser.add_argument("--n_layers",          type=int,            default=2,         help="Number of layers in the endpoint embedding PaiNN.")
    parser.add_argument("--n_neighbors",       type=int,            default=100,       help="Number of nearest-neighbor edges in PaiNN message passing.")
    parser.add_argument("--length_scale",      type=float,          default=10.0,      help="PaiNN distance encoding length scale. Use 3.0 for the paper-alignment sensitivity run.")
    parser.add_argument("--dist_encoding",     default="positional_encoding", choices=("positional_encoding", "soft_one_hot"), help="PaiNN distance encoding.")
    parser.add_argument("--epochs",            type=int,            default=50,        help="Number of training epochs.")
    parser.add_argument("--diff_steps",        type=int,            default=1000,      help="Number of diffusion steps in the model.")
    parser.add_argument("--batch_size",        type=int,            default=128,       help="Batch size for training.")
    parser.add_argument("--lr",                type=float,          default=1e-3,      help="Learning rate for the optimizer.")
    parser.add_argument("--tau",               type=int,            default=1000,      help="Bridge interval length (must be even). The model learns p(x_{t+tau/2} | x_t, x_{t+tau}).")
    parser.add_argument("--split",             default="train",     choices=("train", "test", "all"), help="ALA2 split. 'train' uses trajs 0,1; 'test' uses traj 2 (held-out for evaluation); 'all' uses all three.")
    parser.add_argument("--val_fraction",      type=float,          default=0.05,      help="Fraction of each selected training trajectory tail used for validation checkpoint selection. Set 0 with --no_validation to reproduce last/all-checkpoint behavior.")
    parser.add_argument("--no_validation",     action="store_true", help="Disable train-internal validation and save all epoch checkpoints.")
    parser.add_argument("--checkpoint_monitor", default="val/loss", help="Metric monitored by ModelCheckpoint when validation is enabled.")
    parser.add_argument("--save_top_k",        type=int,            default=3,         help="Number of monitored checkpoints to keep when validation is enabled.")
    parser.add_argument("--gradient_clip_val", type=float,          default=1.0,       help="Gradient clipping value for pl.Trainer.")
    parser.add_argument("--scheduler_t_max",   type=int,            default=20,        help="T_max for CosineAnnealingLR.")
    parser.add_argument("--ema_decay",         type=float,          default=0.99,      help="EMA decay for model weights.")
    parser.add_argument("--seed",              type=int,            default=0,         help="Random seed for training.")
    parser.add_argument("--devices",           type=int,            default=1,         help="Number of devices for pl.Trainer. Defaults to 1 to avoid silently launching multi-GPU DDP.")
    parser.add_argument("--accelerator",       default="auto",      help="pl.Trainer accelerator (e.g. 'gpu', 'cpu', 'auto').")
    parser.add_argument("--indistinguishable", action='store_true', help="Enable this flag to treat atoms as indistinguishable.")
    parser.add_argument("--unscaled",          action='store_true', help="Use unscaled data. When disabled, data is scaled to unit variance.")
    # fmt: on

    main(parser.parse_args())
