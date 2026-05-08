import json
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader

from ito import data, utils
from ito.model import cpainn, ddpm


def main(args):
    assert args.tau % 2 == 0, "tau must be even so that tau/2 lands on a stored frame"

    score_model_class = cpainn.PaiNNBridgeScore
    ala2_path = os.path.join(args.root, "data/ala2")
    timestamp = utils.get_timestamp()
    train_dir = os.path.join(args.root, "train_bridge", timestamp)
    checkpoint_dir = os.path.join(train_dir, "checkpoints")
    train_dir_link = os.path.join(args.root, "train_bridge", "latest")
    best_checkpoint_link = os.path.join(train_dir, "best")

    print(f"saving checkpoints to {checkpoint_dir}")
    os.makedirs(train_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(train_dir, "args.json"), "w"), indent=4)

    score_model_kwargs = {
        "n_features": args.n_features,
        "n_layers": args.n_layers,
        "diff_steps": args.diff_steps,
    }

    model = ddpm.BridgeDDPM(
        score_model_class,
        score_model_kwargs=score_model_kwargs,
        diffusion_steps=args.diff_steps,
        lr=args.lr,
    )

    dataset = data.ALA2BridgeDataset(
        path=ala2_path,
        tau=args.tau,
        distinguish=not args.indistinguishable,
        scale=not args.unscaled,
        split=args.split,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1, dirpath=checkpoint_dir, filename="{epoch}"
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloader)
    os.symlink(
        src=os.path.abspath(checkpoint_callback.best_model_path),
        dst=best_checkpoint_link,
    )
    if os.path.exists(train_dir_link):
        os.unlink(train_dir_link)
    os.symlink(src=os.path.abspath(train_dir), dst=train_dir_link)
    print(f"best model checkpoint: {best_checkpoint_link}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("--root",              type=str,            default="storage", help="Base directory for storing data and training outputs.")
    parser.add_argument("--n_features",        type=int,            default=64,        help="Number of features for the model.")
    parser.add_argument("--n_layers",          type=int,            default=2,         help="Number of layers in the endpoint embedding PaiNN.")
    parser.add_argument("--epochs",            type=int,            default=50,        help="Number of training epochs.")
    parser.add_argument("--diff_steps",        type=int,            default=1000,      help="Number of diffusion steps in the model.")
    parser.add_argument("--batch_size",        type=int,            default=128,       help="Batch size for training.")
    parser.add_argument("--lr",                type=float,          default=1e-3,      help="Learning rate for the optimizer.")
    parser.add_argument("--tau",               type=int,            default=1000,      help="Bridge interval length (must be even). The model learns p(x_{t+tau/2} | x_t, x_{t+tau}).")
    parser.add_argument("--split",             default="train",     choices=("train", "test", "all"), help="ALA2 split. 'train' uses trajs 0,1; 'test' uses traj 2 (held-out for evaluation); 'all' uses all three.")
    parser.add_argument("--devices",           type=int,            default=1,         help="Number of devices for pl.Trainer. Defaults to 1 to avoid silently launching multi-GPU DDP.")
    parser.add_argument("--accelerator",       default="auto",      help="pl.Trainer accelerator (e.g. 'gpu', 'cpu', 'auto').")
    parser.add_argument("--indistinguishable", action='store_true', help="Enable this flag to treat atoms as indistinguishable.")
    parser.add_argument("--unscaled",          action='store_true', help="Use unscaled data. When disabled, data is scaled to unit variance.")
    # fmt: on

    main(parser.parse_args())
