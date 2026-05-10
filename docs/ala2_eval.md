# ALA2 aligned ITO vs Bridge evaluation

This protocol keeps the original ITO repo defaults as the primary baseline and
only changes evaluation hygiene: train/test split, deterministic seeds, one-GPU
training by default, and removal of `overfit_batches=1`.

## Shared settings

- Train split: ALA2 trajectories 0 and 1.
- Test split: ALA2 trajectory 2.
- Seeds: `0 1 2`.
- Training: `--epochs 50 --batch_size 128 --lr 1e-3 --diff_steps 1000`.
- PaiNN: `--n_features 64 --n_layers 2 --n_neighbors 100 --length_scale 10`.
- Sampling: `--ode_steps 50 --n_pairs/--samples 1000`.

Use `--length_scale 3` for the paper-alignment sensitivity run, and apply it to
both models.

## Grid A: original horizon

Bridge:

```bash
python scripts/train_bridge.py --root storage_bridge_A_s0 --tau 1000 --seed 0
python scripts/sample_bridge.py storage_bridge_A_s0/train_bridge/latest/best --root storage_bridge_A_s0 --tau 1000 --depth 3 --n_pairs 1000 --seed 0 --grid A
python scripts/analyse_bridge.py storage_bridge_A_s0/samples_bridge/latest --root storage_bridge_A_s0
```

Original ITO:

```bash
python scripts/train_tlddpm.py --root storage_ito_A_s0 --max_lag 1000 --seed 0
python scripts/sample_tlddpm.py storage_ito_A_s0/train/latest/best --root storage_ito_A_s0 --lag 125 --traj_length 8 --samples 1000 --seed 0 --grid A
python scripts/analyse_trajs.py storage_ito_A_s0/samples/latest --root storage_ito_A_s0
```

## Grid B: original analysis lag

Bridge uses `tau=800, depth=3`, so the effective frame spacing is 100.
Original ITO uses `lag=100, traj_length=8`.

## Summarise

```bash
python scripts/summarise_ala2_metrics.py storage_*/*/*/metrics.json --out_dir storage_eval_summary
```
