# Next Phase Modification Plan

日期：2026-05-18

本文件基于原始 `plan` 中的 Phase 2/3 路线图、ALA2 对齐评估结果，以及 2026-05-18 对 Bridge 方法边界的修订讨论，整理下一阶段应实施的代码修改。

当前分支已经完成 Phase 1 fixed-midpoint Bridge：

- 固定 `tau`。
- 只学习 `p(x_{t+tau/2} | x_t, x_{t+tau})`。
- 采样通过递归二分得到 `2^depth + 1` 帧。
- 不包含 `tau` / `s` 条件化。

评估结果显示：Bridge 有潜力，但 NaN 和 checkpoint 选择会干扰结论。因此下一阶段不建议直接跳到复杂 Phase 3。本轮已先实现 Phase 1.5 稳定性协议；下一步应先跑 Phase 1.5 formal 对比，再进入 Phase 2a multi-tau midpoint bridge。

## 概念修订：避免把 Phase 1 局限误认为 Bridge 局限

### 固定 `tau` 不是 Bridge 的固有限制

Phase 1 固定 `tau` 只是当前实现选择。Bridge 框架本身可以像原版 ITO 多 lag 训练一样扩展到多 `(tau, s)` 条件化：

```text
p(x_{t+s} | x_t, x_{t+tau}, tau, s)
```

因此“原版 ITO 支持多 lag，而 Bridge 只支持固定 `tau`”不是方法层面的不公平，而是 Phase 1 代码尚未实现 Phase 2 条件化。

### Phase 1 网络没有使用时间信息，但时间信息在流水线中存在

评估/采样流水线明确知道端点间隔 `tau`，也知道递归层级对应的 `s`。问题不是 Bridge 无法获得时间信息，而是 Phase 1 的 `PaiNNBridgeScore.forward` 没有接收或 embedding 这些元数据，只看两个 endpoint 坐标。

Phase 2 要做的修改就是把 `tau_phys` / `s_phys` 显式接进 batch 和 score network。

### MD endpoint 不是方法学错误，但会改变直接对比的输入信息量

Phase 1 评估里 Bridge 输入两个 MD 帧，原版 ITO 输入一个 MD 帧。因此，Phase 1 的 VAMP2 / Ramachandran JSD 不能被解释为“同输入条件下 Bridge 模型本身更强”；它同时包含了第二个端点带来的锚定信息。

正确口径：

- Phase 1 评估的是 endpoint-conditioned interpolator 的质量。
- 它可以证明 Bridge 在给定合理端点时是否能生成物理合理路径。
- 它不能单独证明 Bridge 在只有一个起点的长时间预测任务中优于 ITO。

### Bridge 是不同输入接口的工具，不是减弱版 ITO

Bridge 的接口是 `(x_left, x_right) -> path/refinement`。右端点可以来自：

- MD，用于监督评估和诊断；
- ITO coarse rollout，用于 apples-to-apples hybrid 评估；
- BioEmu / equilibrium sampler / 结构预测模型，用于长 lag endpoint proposal；
- 其他物理或结构先验模型。

因此 Bridge 的关键问题不是“有没有偷用 MD endpoint”，而是“给定端点来源是否和目标任务统计一致”。

### BioEmu / equilibrium endpoint hybrid 是合理方向，但只适合明确场景

BioEmu 等模型更像采样平衡态 `pi(x)`，不是严格采样 `p(x_tau | x_0)`。当 `tau` 远大于体系混合时间时，这两者可以近似一致；中短 lag 时系统仍保留 `x_0` 记忆，平衡态 endpoint 会偏离条件分布。

因此 BioEmu + Bridge 更适合长时间预测或构象覆盖任务。对中短 lag，Phase 3a 的 ITO coarse endpoint generator 更容易保持 apples-to-apples，因为它仍从 `x_0` 条件化生成粗端点。

### 修订后的阶段定位

| Phase | 目标 | 公平性定位 |
|---|---|---|
| Phase 1 | fixed-`tau` MD endpoint midpoint bridge | 评估 endpoint-conditioned interpolation，不作为单起点长 rollout 结论 |
| Phase 1.5 | validation / checkpoint selection / NaN 诊断 | 让原版 ITO 与 Bridge 在同一 checkpoint 规则下比较 |
| Phase 2a | multi-tau midpoint bridge | 让递归每层都落在训练分布内，消除 fixed-`tau` 偶然限制 |
| Phase 2b | arbitrary `s` bridge | 单一网络处理任意 endpoint interval 与目标位置 |
| Phase 3a | hybrid sampler，coarse endpoints + Bridge fill | 输入退回单个 `x_0`，可与纯 ITO long rollout 公平对比 |
| Phase 3a-bio | equilibrium/structure sampler endpoints + Bridge fill | 长 lag 场景的 endpoint-proposal workflow，需声明 `tau >> mixing time` 假设 |

## 下一轮代码修改切片

下一次不要同时改稳定性、multi-tau、hybrid sampler。建议按下面切片推进。

### Cut 1：Phase 1.5 checkpoint protocol（已实现，待 formal rerun）

目的：先让现有原版 ITO 与 Phase 1 Bridge 的训练结果可比、可复现、可从 NaN 前统一选择 checkpoint。

文件范围：

- `ito/data.py`
- `ito/model/ddpm.py`
- `scripts/train_tlddpm.py`
- `scripts/train_bridge.py`
- `scripts/analyse_trajs.py`
- `scripts/analyse_bridge.py`
- `scripts/summarise_ala2_metrics.py`

不改：

- `PaiNNBridgeScore` 结构。
- `BridgeDDPM.sample` 语义。
- 任何 Phase 2 interval metadata。

### Cut 2：Phase 2a multi-tau midpoint bridge（下一轮代码主目标）

目的：让 Bridge 在递归采样的每一层都见过对应 interval，避免 fixed-`tau` Phase 1 的训练分布外调用。

新增文件优先：

- `scripts/train_interval_bridge.py`
- `scripts/sample_interval_bridge.py`

修改文件：

- `ito/data.py`
- `ito/utils.py`
- `ito/model/cpainn.py`
- `ito/model/ddpm.py`
- `scripts/analyse_bridge.py`

第一版只支持：

- `allowed_taus=100,200,400,800`
- `s_sampling=midpoint`
- `tau_phys` / `s_phys` embedding
- `depth=3` with `tau=800`

暂不支持：

- arbitrary `s`
- BioEmu endpoints
- self-consistency loss

### Cut 3：Phase 3a hybrid sampler

目的：在 Phase 2a 稳定后，把输入条件拉回和原版 ITO 一致：只给 `x_0`，由 coarse endpoint generator 产生锚点，Bridge 做细化。

新增文件：

- `scripts/sample_hybrid_bridge.py`

只有当 Phase 2a train/sample/analyse 在 ALA2 上稳定后再进入 Cut 3。

## Phase 1.5：稳定性与公平 checkpoint 选择（代码已实现）

### 目标

把“模型能力”和“训练是否 NaN”拆开，避免用固定 epoch 50 的最后 checkpoint 作为唯一结果。

这一步不是改变模型任务，而是改训练与评估协议：

- 原版 ITO 和 Bridge 都使用同一套 validation / checkpoint-selection 规则。
- 保留 test trajectory 2 只用于最终评估，不用于 checkpoint selection。
- 如果训练后期 NaN，仍可按统一规则评估 NaN 前的 best validation checkpoint，而不是把该 run 直接变成不可比。

### 已实现代码改动

#### `ito/data.py`

新增 train 内部 validation split helper，避免使用 held-out test：

- `select_ala2_split(trajs, "train")` 保持 traj 0,1。
- 新增窗口级 helper，例如：
  - `split_train_validation_trajs(trajs, val_fraction=0.05, val_tail=True)`
  - 或新增 split 名：
    - `train_fit`：traj 0,1 的前 95% frames。
    - `train_val`：traj 0,1 的后 5% frames。

建议优先用每条 train trajectory 的尾部做 validation，理由是：

- 不碰 test traj 2。
- 不减少为单条 train trajectory。
- deterministic，便于 seed 复现。

需要注意：窗口之间仍有时间相关性，所以该 validation 只用于 checkpoint selection，不作为正式泛化指标。

#### `scripts/train_tlddpm.py`

新增参数：

- `--val_fraction`，默认 `0.05`。
- `--no_validation`，用于复现纯 upstream 训练。
- `--checkpoint_monitor`，默认 `val/loss`；无 validation 时回退到 `train/loss` 或 last。
- `--save_top_k`，默认 `3`。
- `--gradient_clip_val`，默认仍为 `1.0`。
- `--scheduler_t_max`，默认仍为 `20`。
- `--ema_decay`，默认仍为 `0.99`。

训练逻辑：

- 构造 train dataloader 和 val dataloader。
- `ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=args.save_top_k, save_last=True)`。
- symlink `best` 指向 validation best checkpoint。
- `runtime.json` 继续记录训练时间和显存。

#### `scripts/train_bridge.py`

和原版 ITO 做完全相同的 validation / checkpoint-selection 修改。

Bridge 额外保留：

- `tau` even assertion。
- `ALA2BridgeDataset`。

#### `ito/model/ddpm.py`

新增 `validation_step` 到 `DDPMBase`：

```python
def validation_step(self, batch, _):
    loss = self.get_loss(batch)
    self.log("val/loss", loss, prog_bar=True, sync_dist=True)
    return loss
```

训练 loss log 建议增加：

- `prog_bar=True`
- `sync_dist=True`

NaN 行为建议保持 fail-fast：

- formal 训练仍然在 NaN 时失败。
- 但由于保存了 validation best checkpoint，失败前 checkpoint 可作为 diagnostic 使用。

#### `scripts/summarise_ala2_metrics.py`

增加 run 状态字段支持：

- `train_status`: `completed` / `failed_nan` / `diagnostic_pre_nan`
- `checkpoint_epoch`
- `checkpoint_selection`: `last` / `val_best` / `manual_pre_nan`

这样 summary 能明确区分 formal 和 diagnostic。

### Phase 1.5 验证

最小验证：

1. 原版 ITO 和 Bridge 都能跑 `--epochs 2 --val_fraction 0.05`。
2. 训练目录里同时有：
   - `checkpoints/epoch=*.ckpt`
   - `best`
   - `last.ckpt`
   - `args.json`
   - `runtime.json`
3. `best` 指向 validation loss 最低 checkpoint。
4. `analyse_*` 输出的 `metrics.json` 记录 checkpoint selection 信息。

建议实验：

- Grid B。
- `length_scale=3`。
- seed 0/1/2。
- epochs 50。
- checkpoint 用 `val_best`，不是 last。

## Phase 2：`(tau, s)` 条件化 Bridge

### 目标

把 Phase 1 的固定 midpoint bridge：

```text
p(x_{t+tau/2} | x_t, x_{t+tau})
```

扩展成可变中间位置 bridge：

```text
p(x_{t+s} | x_t, x_{t+tau}, tau, s)
```

其中：

- `tau` 是左右端点间隔。
- `s` 是左端点到目标中间帧的距离，满足 `0 < s < tau`。

递归二分时，模型会在不同尺度上被调用：

- 第一层：`tau=800, s=400`
- 第二层：`tau=400, s=200`
- 第三层：`tau=200, s=100`

Phase 1 只训练了第一层固定 `tau/2`，深层递归会落到训练分布外。Phase 2 的核心是让训练覆盖这些递归尺度。

### Phase 2 拆分

建议不要一次实现任意 `tau` 与任意 `s`。下一阶段先做可验证的两步：

1. **Phase 2a：multi-tau midpoint bridge**
   - 训练 `tau` 从离散集合中采样，例如 `{100, 200, 400, 800}` 或 `{125, 250, 500, 1000}`。
   - 目标位置固定为 `s = tau / 2`。
   - 网络显式接收 `tau_phys` 和 `s_phys`，但 sampling 仍是二分 midpoint。
   - 目的：让递归 depth=3 的每一层都落在训练分布内。
2. **Phase 2b：arbitrary `s` bridge**
   - 在 Phase 2a 稳定后，让 `s` 从 dyadic positions 或更一般的 admissible positions 采样。
   - 目标：单一网络处理不对称区间和非二分插值。

Phase 2a 是下一次代码修改的主目标；Phase 2b 暂不进入第一轮实现。

### 数据层修改

#### 新增 `IntervalBridgeDataset`

文件：`ito/data.py`

建议新增类：

```python
class IntervalBridgeDataset(data.Dataset):
    def __init__(
        self,
        trajs,
        max_tau,
        min_tau=2,
        allowed_depths=(1, 2, 3),
        tau_sampling="log_uniform",
        s_sampling="dyadic",
    ):
        ...
```

每个 item 返回：

```python
{
    "batch_0": batch at x_t,
    "batch_s": batch at x_{t+s},
    "batch_T": batch at x_{t+tau},
}
```

batch 上需要带 node-wise scalar：

- `tau_phys`
- `s_phys`

Phase 2b 建议支持 dyadic `s`：

```text
s in {tau / 2, tau / 4, 3tau / 4, ...}
```

但 Phase 2a 第一版应更保守：

- `tau` 从 `{100, 200, 400, 800}` 或 `{125, 250, 500, 1000}` 采样。
- `s = tau / 2`。

也就是先做 multi-tau midpoint bridge。等稳定后再做 arbitrary `s`。

#### 新增 `ALA2IntervalBridgeDataset`

文件：`ito/data.py`

类似当前 `ALA2BridgeDataset`：

- 加载 ALA2。
- 应用 `select_ala2_split`。
- 传给 `IntervalBridgeDataset`。

保留 `ALA2BridgeDataset` 不动，便于 Phase 1 复现。

### Utils 修改

文件：`ito/utils.py`

当前：

```python
def get_bridge_batch(atom_numbers, positions):
    return get_batch_from_atom_numbers_and_position(atom_numbers, positions)
```

建议扩展为：

```python
def get_bridge_batch(atom_numbers, positions, tau_phys=None, s_phys=None):
    batch = get_batch_from_atom_numbers_and_position(atom_numbers, positions)
    if tau_phys is not None:
        batch.tau_phys = torch.ones_like(batch.atom_number) * tau_phys
    if s_phys is not None:
        batch.s_phys = torch.ones_like(batch.atom_number) * s_phys
    return batch
```

注意多样本 batch 时，`tau_phys` / `s_phys` 可能是 per-sample array。需要支持：

- scalar：所有 graph 同一个值。
- shape `[n_graphs]`：按 `batch.batch` 展开到每个 atom。

建议新增 helper：

```python
def add_graph_scalar_to_batch(batch, name, value):
    ...
```

### 模型修改

文件：`ito/model/cpainn.py`

有两种实现方式。

#### 方案 A：扩展 `PaiNNBridgeScore`

新增参数：

- `max_tau=1000`
- `condition_on_interval=False`

当 `condition_on_interval=True` 时，在端点融合后、diffusion `t_diff` 前注入 `tau_phys` 和 `s_phys`：

```text
fused endpoint features n
 + tau embedding n
 + s embedding n
 -> CombineInvariantFeatures(3n -> n)
```

然后原来的 diffusion net 保持：

```text
fused condition n
 + t_diff embedding n
 -> CombineInvariantFeatures(2n -> n)
 -> PaiNNBase
```

优点：

- Phase 1 和 Phase 2 共用同一个 class。
- checkpoint 兼容性更容易控制。

缺点：

- class 内分支变多。

#### 方案 B：新增 `PaiNNIntervalBridgeScore`

新增独立 class，保持 `PaiNNBridgeScore` 作为 Phase 1 fixed midpoint。

优点：

- 不影响 Phase 1 结果复现。
- 代码语义更清楚。

建议选择方案 B，原因是当前 Phase 1 还需要反复诊断稳定性，不应让已有 checkpoint 行为被新条件分支影响。

### DDPM 修改

文件：`ito/model/ddpm.py`

新增或扩展 `BridgeDDPM.get_loss`：

- Phase 1 使用 `batch_mid`。
- Phase 2 使用 `batch_s`。

建议新增 class：

```python
class IntervalBridgeDDPM(DDPMBase):
    def get_loss(self, batch):
        batch_0 = batch["batch_0"]
        batch_s = batch["batch_s"]
        batch_T = batch["batch_T"]
        noise_batch, epsilon = self.get_noise_img_and_epsilon(batch_s)
        epsilon_hat = self.forward(noise_batch, batch_0, batch_T)
        ...
```

`sample(batch_0, batch_T, tau, s, ode_steps=0)`：

- 构造 sampling batch 时必须带 `tau_phys` / `s_phys`。
- 或要求调用方传入已经带 metadata 的 `batch_0` / `batch_T`。

建议把 metadata 放在 `batch_0` 和 `batch_T` 上，由 score model 从 `noise_batch` 或 endpoint batch 读取。需要统一约定：

- `tau_phys` 和 `s_phys` 应该存在于 `noise_batch`。
- `BridgeDDPM.sample` 负责在采样模板 batch 上设置它们。

### 训练脚本

新增脚本而不是覆盖 Phase 1：

- `scripts/train_interval_bridge.py`

关键参数：

- `--max_tau`
- `--min_tau`
- `--allowed_taus`，例如 `100,200,400,800`
- `--allowed_depths`，例如 `1,2,3`
- `--s_sampling dyadic|midpoint`
- `--condition_on_interval` 默认 true
- 继承 Phase 1.5 的 validation / checkpoint selection 参数

Phase 2a 第一版只实现 `allowed_taus + midpoint`：

```bash
python scripts/train_interval_bridge.py \
  --root storage_interval_bridge \
  --allowed_taus 100,200,400,800 \
  --s_sampling midpoint \
  --length_scale 3 \
  --seed 0
```

这样先验证 multi-tau 是否解决递归分布外问题，再扩展 arbitrary `s`。

### 采样脚本

新增：

- `scripts/sample_interval_bridge.py`

递归调用时显式传每一段的 interval：

```python
def fill(left, right, tau, depth):
    if depth == 0:
        return [left, right]
    s = tau // 2
    mid = model.sample(left, right, tau=tau, s=s, ode_steps=...)
    return fill(left, mid, s, depth - 1)[:-1] + fill(mid, right, tau - s, depth - 1)
```

第一版要求：

- `tau % 2**depth == 0`
- 每层 `tau_segment` 必须出现在 training `allowed_taus` 中，或至少不小于 `min_tau` 且不大于 `max_tau`。

保存内容沿用 Phase 1：

- `trajectory.npy`
- `md_reference.npy`
- `md_midpoints.npy`
- `endpoints.npy`
- `start_indices.npy`
- `args.json`
- `runtime.json`

`args.json` 需要额外记录：

- `model_type=interval_bridge`
- `allowed_taus`
- `condition_on_interval=True`

### 分析脚本

可以先复用 `scripts/analyse_bridge.py`，只需它能识别：

- `model_type=interval_bridge`
- `tau`
- `depth`
- `ref_lag=tau//2**depth`

如果 arbitrary `s` 后续不再均匀分辨率，则再新增 `analyse_interval_bridge.py`。

第一版 multi-tau midpoint 仍然是均匀 `2^k+1` 帧，所以无需新分析脚本。

## Phase 3：Hybrid endpoint generator + Bridge

Phase 3 不建议现在直接做。应在 Phase 2 multi-tau bridge 稳定后再做。

### Phase 3a：coarse endpoint generator

新增脚本：

- `scripts/sample_hybrid_bridge.py`

流程：

1. 用原版 `TLDDPM` 从 test 初始帧 open-loop 生成 coarse anchors。
2. 对每对 adjacent anchors 调 Bridge/IntervalBridge 填中间帧。
3. 输出完整高分辨率轨迹。

这一路线的关键公平性优点是：输入与纯 ITO rollout 一样，都是单个 MD 起点 `x_0`。Bridge 不再得到额外 MD endpoint，而是作为 coarse trajectory 的分辨率细化器。因此 Phase 3a 才是回答“同样只给一个起点，hybrid 是否优于纯 ITO 长 rollout”的主评估。

关键风险：

- coarse anchors 来自模型而非 MD，可能落在 Bridge 训练分布外。
- 需要 structure outlier 检查。
- 需要和纯 ITO open-loop 做同 lag 的 apples-to-apples 比较。

### Phase 3a-bio：BioEmu / equilibrium sampler endpoint proposal

这是 Phase 3a 的同类 hybrid 架构，只是粗端点来源从 ITO coarse rollout 换成平衡态或结构预测模型。

建议新增脚本时不要和 ITO hybrid 混在一起：

- `scripts/sample_bioemu_bridge.py` 或更通用的 `scripts/sample_endpoint_bridge.py`

建议接口：

```bash
python scripts/sample_endpoint_bridge.py \
  bridge_checkpoint \
  --root storage_endpoint_bridge \
  --left_endpoints x0.npy \
  --right_endpoints xT_proposals.npy \
  --tau 100000 \
  --depth 10 \
  --source bioemu
```

必须在 `args.json` / `metrics.json` 记录：

- `endpoint_source`: `md` / `ito` / `bioemu` / `external`
- `endpoint_conditioning`: `conditional_p_tau_given_x0` / `equilibrium_pi` / `unknown`
- `tau`
- `mixing_time_assumption`

解释约束：

- BioEmu 采样近似 `pi(x)`，不是严格 `p(x_tau | x_0)`。
- 只有当 `tau` 远大于混合时间时，`pi(x)` endpoint 才能近似长 lag 条件端点。
- 如果 `tau` 是中短 lag，BioEmu endpoint 会丢失 `x_0` 记忆，不适合与 ITO 条件 rollout 直接公平比较。

该路线的优势是绕开 ITO coarse rollout 自身可能漂移的问题，适合长时间构象覆盖和轨迹内插。但转变路径是否合理仍取决于 Bridge 训练数据是否覆盖相应 endpoint 间转变，尤其需要 Phase 2 多 `tau` 长间隔训练。

### Phase 3b：self-consistency loss

暂不建议作为下一步实现，原因：

- 会显著增加每个 batch 的 sampling / forward 成本。
- 当前 NaN 尚未解决，加入 consistency loss 会让定位更难。

建议仅在 Phase 2 stable 后做小规模 ablation。

## 建议实施顺序

1. 已完成代码：Phase 1.5 validation / val-best checkpoint selection。
2. 立即下一步实验：用 `length_scale=3` 跑 Grid B seeds 0/1/2，比较 last vs val-best。
3. 如果 NaN 问题显著缓解，进入 Phase 2a：multi-tau midpoint bridge。
4. Phase 2a stable 后，再做 Phase 2b：arbitrary `s`。
5. 最后做 Phase 3a hybrid sampler。
6. 如需长 lag endpoint-proposal workflow，再做 Phase 3a-bio / external endpoint sampler。

## 最小可验收标准

Phase 1.5 完成标准：

- 两个 train 脚本都有相同 validation 参数。
- `best` checkpoint 是 validation best。
- summary 能区分 `last` / `val_best` / `pre_nan`。
- Grid B `length_scale=3` 至少 seed0 完整跑通 train/sample/analyse。

Phase 2a 完成标准：

- `train_interval_bridge.py` 支持 `allowed_taus=100,200,400,800`。
- `sample_interval_bridge.py --tau 800 --depth 3` 每层调用都带正确 `tau_phys/s_phys`。
- analysis closure 通过。
- `metrics.json` 中记录 `model=interval_bridge`、`allowed_taus`、`tau`、`depth`。

Phase 3a 完成标准：

- hybrid sampler 能输出 `[n_traj, n_frames, n_atoms, 3]`。
- 和 pure ITO rollout 使用同一个 initial manifest 和 seed。
- stability metrics 无 NaN/Inf，极端坐标可追踪到具体 trajectory/frame。
- `metrics.json` 明确记录 endpoint source，MD endpoint diagnostic 与 single-start hybrid formal 不混在同一组结论里。
