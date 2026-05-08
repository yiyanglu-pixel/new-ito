# Midpoint Diffusion Bridge — ITO Surrogate Variant

A two-sided endpoint-conditioned diffusion bridge as an alternative to the
open-loop ITO rollout. The model learns
`p_θ(x_{t+τ/2} | x_t, x_{t+τ})`; recursive binary subdivision then yields
`2^k + 1` frames at uniform spacing `τ / 2^k` between any data-anchored
endpoint pair.

This document records the full implementation and the post-implementation
fixes that landed on `claude/implement-ito-surrogate-uVFwF`.

## 1. Motivation

ITO trains `p_θ(x_{t+Nτ} | x_t, N)` and rolls out by feeding each predicted
frame back as `x_t` for the next step. Two practical issues:

1. **Open-loop drift.** Long rollouts have no anchor, so errors compound
   freely and the trajectory can wander off the equilibrium distribution.
2. **Cross-resolution inconsistency.** `p(x_τ | x_0)` and the composition
   of two `p(x_{τ/2} | x_0)` predictions are not constrained to agree.

The bridge factorization replaces the marginal target with a conditional one
anchored to two known endpoints. Each prediction is intrinsically consistent
with the endpoints it interpolates between; deep recursion still composes
many bridge draws but compounded error stays bounded by the anchored
boundary. The bridge models a different quantity than ITO (conditional vs
marginal) and is therefore complementary, not strictly a replacement.

Phase 1 (this branch) trains the bridge at a single fixed `τ` with no `s`
conditioning. Endpoints come from MD data — that is, the evaluation regime.
Phase 2 (conditional `(τ, s)`) and Phase 3 (ITO-driven endpoint generator
plus self-consistency / closure losses) are sketched in the plan and left
unimplemented.

## 2. Method

### Training objective

Sample a triple `(x_0, x_{τ/2}, x_τ)` from MD data at fixed even `τ`. Sample
diffusion step `σ ~ U{1, ..., T-1}` and Gaussian noise `ε`. Form the noised
midpoint
```
x_{τ/2}^σ = √(α̅_σ) · x_{τ/2} + √(1 - α̅_σ) · ε.
```
Train the score network with the standard ε-prediction MSE
```
L = E[ ‖ ε - ε_θ(x_{τ/2}^σ, σ; x_0, x_τ) ‖² ].
```
Conditioning is on **both** endpoints. Phase 1 has no `t_phys / s` embedding;
`τ` is implicit because the dataset only contains a single fixed `τ`.

### Recursive sampler

Given anchored `(x_0, x_τ)` from data:
```
fill(L, R, k):
    if k == 0: return [L, R]
    M = model.sample(L, R)
    return fill(L, M, k-1)[:-1] + fill(M, R, k-1)
```
Returns `2^k + 1` frames in temporal order. Total `model.sample` calls per
pair: `2^k - 1`.

### Score network

`PaiNNBridgeScore` (in `ito/model/cpainn.py`) mirrors `PaiNNTLScore` with two
endpoint embed towers fused before the diffusion stage:

- `embed_0`, `embed_T`: each `AddEdges → AddEquivariantFeatures →
  NominalEmbedding(atom_number) → PaiNNBase`. Independent weights.
  No `t_phys` block.
- Fusion: invariant features `cat(2n) → MLP(2n→n)`; equivariant features
  `cat along feature axis (2n) → EquivariantLinear(2n→n)`.
- Edges for the noise stage are built once from the midpoint-init coords
  `(x_0 + x_T) / 2` so high-σ diffusion steps don't produce degenerate
  neighbor sets. Distances and directions are still recomputed per
  diffusion step from the current noisy `x` via the existing
  `AddEdges(should_generate_edge_index=False)` layer in `self.net`.
- Diffusion stage: `AddEdges(False) → PositionalEmbedding(t_diff) →
  CombineInvariantFeatures(2n→n) → PaiNNBase`. Same as `PaiNNTLScore.net`.

## 3. Files

### New

| Path | Purpose |
|---|---|
| `scripts/train_bridge.py` | Train `BridgeDDPM` on `ALA2BridgeDataset`. |
| `scripts/sample_bridge.py` | Recursive 2^k bisection from data-sourced endpoint pairs. |
| `scripts/analyse_bridge.py` | VAMP2 + Ramachandran + closure assertion + midpoint circular-RMSE vs MD. |
| `docs/bridge.md` | This report. |
| `.gitignore` | Excludes `__pycache__/`, `storage/`, and other build artifacts. |

### Modified

| Path | Additions |
|---|---|
| `ito/data.py` | `MidpointBridgeDataset`, `ALA2BridgeDataset`, `select_ala2_split`. |
| `ito/utils.py` | `get_bridge_batch` — same as `get_batch_from_atom_numbers_and_position` without the `t_phys` field. |
| `ito/model/cpainn.py` | `PaiNNBridgeScore`. |
| `ito/model/ddpm.py` | `BridgeDDPM`: `get_loss`, `sample(batch_0, batch_T, ode_steps)`. |

The original `TLDDPM`, `PaiNNTLScore`, `ALA2Dataset`, `StochasticLaggedDataset`,
and the corresponding scripts are unchanged.

## 4. Usage

### Train

```
python scripts/train_bridge.py
```

Defaults: `--tau 1000 --split train --devices 1 --accelerator auto
--epochs 50 --batch_size 128 --n_features 64 --n_layers 2 --diff_steps 1000
--lr 1e-3`.

The `train` split uses MD trajectories 0 and 1; trajectory 2 is held out for
evaluation. Pass `--split all` to reproduce upstream ITO's no-split
behavior. `--devices 1` is the default to avoid Lightning silently launching
multi-GPU DDP on a multi-GPU host; raise it explicitly when you want DDP.

Outputs:

```
storage/train_bridge/{timestamp}/checkpoints/{epoch}.ckpt
storage/train_bridge/{timestamp}/best        # symlink to best ckpt
storage/train_bridge/{timestamp}/args.json
storage/train_bridge/latest                  # symlink to {timestamp} dir
```

### Sample

```
python scripts/sample_bridge.py
```

Defaults: `--tau 1000 --depth 3 --n_pairs 10 --ode_steps 50 --split test
--seed 0`. With these defaults the output trajectory is
`(n_pairs, 2^depth + 1, n_atoms, 3) = (10, 9, 22, 3)` covering 9 frames per
pair at lag `τ / 2^depth = 125`.

`--tau` must be divisible by `2^depth`; the script asserts at startup.
`tau=1000` admits any `depth ≤ 3`; for deeper recursion choose e.g.
`tau=1024 --depth 4` (lag 64) and retrain.

Endpoint pairs are sampled per-trajectory so `start + τ` never crosses a
trajectory boundary. The script writes:

```
storage/samples_bridge/{timestamp}/trajectory.npy   # [n_pairs, 2^k+1, n_atoms, 3]
storage/samples_bridge/{timestamp}/endpoints.npy    # [n_pairs, 2, n_atoms, 3]
storage/samples_bridge/{timestamp}/md_midpoints.npy # [n_pairs, n_atoms, 3]
storage/samples_bridge/{timestamp}/args.json
storage/samples_bridge/latest                       # symlink to trajectory.npy
```

### Analyse

```
python scripts/analyse_bridge.py
```

Reads `tau` and `sampling_split` from the sampling `args.json` so the VAMP2
reference lag is exact (`tau // 2^depth`, not truncated) and the MD
reference defaults to the same split as sampling. Override with
`--split {train|test|all|match_sampling}`.

The script emits:

- `metrics.json` — VAMP2(bridge, lag=1) vs VAMP2(MD, lag=ref_lag), midpoint
  circular RMSE in φ and ψ, splits used.
- `marginals.pdf` — φ and ψ marginals, bridge vs MD.
- `ramachandran.pdf` — bridge full trajectory + bridge midpoints + MD
  midpoints overlaid.
- A startup assertion that the recursion's first/last frame match the input
  endpoints to within `1e-6` (closure check).

## 5. Train / test split

`ito/data.select_ala2_split` divides the three published ALA2 trajectories:

| split | trajectories | total frames |
|---|---|---|
| `train` | 0, 1 | 500 000 |
| `test`  | 2    | 250 000 |
| `all`   | 0, 1, 2 | 750 000 |

Defaults: `train_bridge.py --split train`, `sample_bridge.py --split test`,
`analyse_bridge.py --split match_sampling`. This keeps midpoint RMSE and
VAMP2 strictly held-out by default; `--split all` reproduces upstream ITO's
no-split behavior.

## 6. Verification

Numbers reported from a fresh end-to-end run on this branch:

- `python -m compileall ito scripts` passes.
- `select_ala2_split` lengths: `train=[250000, 250000]`, `test=[250000]`,
  `all=[250000, 250000, 250000]`.
- `valid_starts` contains zero cross-trajectory pairs on the previously
  failing case (concatenated ALA2 with `τ=1000`, 10 000 samples; the
  pre-fix code produced ~57 cross-boundary pairs in the same simulation).
- `--tau 1000 --depth 4` fails the divisibility assertion at sample
  startup with a clear message rather than silently truncating to lag 62.
- Real sampling at `--tau 1000 --depth 3 --split test --n_pairs 4`
  produces shape `(4, 9, 22, 3)`.
- `analyse_bridge.py` reads `args.json`, reports `ref_lag = 125`,
  `sampling_split = test`, `ref_split = test`, and a closure error of 0.
- `train_bridge.py` on an 8-GPU host launches with `devices=1` by default
  and does not initialise DDP unless `--devices` is raised explicitly.

## 7. Caveats and knobs

- **`τ` must be divisible by `2^depth`** at sample / analyse time. Without
  this every recursion midpoint would land on a non-integer MD frame and
  the VAMP2 reference lag would be approximate.
- **`scale` and `indistinguishable` must match between train and sample.**
  Both default to scaled / distinguished and the data scaling factor lives
  in `ito.data.get_ala2_trajs`.
- **Edge graph stability.** Edges are built once from
  `(x_0 + x_T) / 2`; very high-noise diffusion steps inherit a sane
  neighbor set instead of scrambled-coordinate edges. Distances and
  directions are still recomputed per step.
- **No `t_phys` / `s` in Phase 1.** Adding them is the Phase 2 extension —
  re-introduce `PositionalEmbedding("t_phys", n_features, max_tau)` per
  endpoint and a parallel `s_phys` embedding to make the same network
  cover arbitrary `τ` and arbitrary midpoint position.
- **Endpoint source.** Phase 1 uses MD data. Phase 3 is the hybrid sampler
  where a pretrained `TLDDPM` provides coarse anchors and the bridge
  fills in between.
- **Off-by-one in `DDPMBase._sample`.** The reverse loop starts at
  `diffusion_steps - 1`; this is inherited unchanged so bridge results
  remain comparable to TLDDPM under the same scheduler.

## 8. Suggested verification plan for new training runs

1. Smoke train: `python scripts/train_bridge.py --epochs 5 --tau 100`.
   Watch `train/loss` decrease.
2. Depth-1 sanity: sample at `--depth 1 --tau 100`. Plot bridge midpoints
   on the Ramachandran density of the held-out trajectory. The midpoints
   should land in physical basins, not in saddle voids.
3. Depth-k visual: `--depth 3` between distant pairs. Inspect `(φ, ψ)` time
   series — they should look MD-like, with continuous fluctuation rather
   than discrete jumps.
4. VAMP2: bridge filled trajectory at lag 1 vs MD held-out at the matched
   `ref_lag`. Match within stochastic noise.
5. Closure: assertion in `analyse_bridge.py` must pass exactly.
6. Compare to ITO baseline: train a `TLDDPM(max_lag=τ)` with the same
   width / depth and compare long-rollout `(φ, ψ)` distribution
   Wasserstein vs the bridge filled trajectory. The bridge should be
   tighter near endpoints and at least competitive elsewhere.

## 9. Phase 2 / Phase 3 roadmap (not implemented)

- **Phase 2 — `(τ, s)` conditioning.** Re-introduce `t_phys` and add an
  `s_phys` embedding. Sample `τ` log-uniformly as ITO does and `s` over
  admissible recursion depths. Single network handles arbitrary `τ` and
  arbitrary midpoint position; recursion at depth-`k` then queries the
  same network with `s = τ / 2^k`.
- **Phase 3a — hybrid sampler.** Pretrained `TLDDPM` rolls out coarse
  anchors at large `τ_coarse`; the bridge fills each interval. Eliminates
  the dependence on MD endpoints at evaluation time.
- **Phase 3b — closure / self-consistency losses.** Add
  `‖ bridge(x_0, x_τ) − bridge(x_0, M) ⊕ bridge(M, x_τ) ‖²` to push
  one-step ≈ two-half-step consistency, plus a marginal-matching term
  `KL(p_bridge(x_{τ/2} | x_0, x_τ) ‖ p_ITO(x_{τ/2} | x_0))` to align the
  bridge's marginal with ITO's.

## 10. Change log

| Commit | Summary |
|---|---|
| `17af6f3` | Add midpoint diffusion bridge ITO surrogate variant — initial Phase 1 implementation: dataset, score net, DDPM module, train / sample / analyse scripts. |
| `441f72c` | `.gitignore` for Python bytecode and local training artifacts. |
| `13e8e56` | Evaluation hygiene fixes — per-trajectory endpoint sampling (no cross-trajectory pairs), `tau % 2^depth` divisibility assertion, MD reference path matches train/sample (`root/data/ala2`), `tau` read from sampling `args.json`, train/test/all split semantics, `train_bridge.py` defaults `--devices 1` and `--accelerator auto`. |
| `0b1731f` | Default `sample_bridge --depth 3` so the no-arg defaults `--tau 1000 --depth 3` run end-to-end without hitting the divisibility assertion. |
