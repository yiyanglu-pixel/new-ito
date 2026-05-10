# ALA2 原版 ITO vs Bridge ITO 对比进展报告

日期：2026-05-10
远端工作区：`/localhome3/lyy/codex-new-ito-bridge-review/new-ito`
结果目录：`eval_ala2_aligned/`

## 1. 结论摘要

当前结果还不能支持“改版 Bridge ITO 已经严格优于原版 ITO”的结论。更准确的判断是：

- Bridge ITO 在 repo-faithful `length_scale=10` 的部分完成实验中表现很强，尤其 Grid B seed0 的 VAMP2 几乎贴近 held-out MD 参考。
- 但 Bridge ITO 的稳定性没有完全过关：Grid B seed1 在 epoch 39 附近 NaN，`length_scale=3` seed0 在 epoch 41 NaN。
- 原版 ITO 在 repo-faithful `length_scale=10` 的 50 epoch 训练中更不稳定，三个 formal seed 均 NaN；但在 paper-alignment `length_scale=3` 下，原版 ITO seed0 能完整训练 50 epoch，并且 Grid B 指标优于 Bridge 的 `length_scale=3` 预 NaN checkpoint。
- 因此当前最稳妥的汇报口径是：Bridge ITO 有明显潜力，但现阶段主要问题是训练稳定性；在修复 NaN 或定义统一 early-stopping/checkpoint-selection 协议前，不应宣称已经稳健超越原版 ITO。

## 2. 实验目标

本轮实验目标是评估改版 Bridge ITO 是否相对原版 ITO 有真实提升，而不是由 baseline 训练 bug、数据泄漏或参数不公平造成。

严格评估约束：

- 只使用 ALA2。
- train/test split 固定为 train=trajectory 0,1，test=trajectory 2。
- 两个模型使用同一份数据、同一个 conda 环境、同一套采样和分析指标、同一组 seed。
- 原版 ITO 只修公平性问题，不额外调参。
- 原版 `train_tlddpm.py` 的 `overfit_batches=1` 已移除，否则 baseline 只会训练一个 batch。

## 3. 参数含义

### `length_scale` / `ls`

`length_scale` 是 PaiNN 距离特征中的尺度参数。

- `ls10`：repo-faithful 设置。当前源代码 `PaiNNBase` 默认 `length_scale=10`，主评估以此作为“当前 repo 默认实现”的口径。
- `ls3`：paper-alignment 敏感性设置。NeurIPS supplement 对 ALA2 使用 `length_scale=3`，所以用 `ls3` 检查结论是否依赖 repo 默认值。

目前结果显示 `length_scale` 对训练稳定性影响很大：原版 ITO 在 `ls10` 下易 NaN，但在 `ls3` seed0 下能完整训练；Bridge 在 `ls10` 下部分完成，在 `ls3` seed0 下反而 NaN。

### Grid A / Grid B

两个 grid 用于对齐 Bridge 的递归 midpoint 生成和原版 ITO 的 open-loop rollout。

| Grid | Bridge 设置 | Bridge 有效 lag | 原版 ITO 设置 | 目的 |
|---|---:|---:|---:|---|
| A | `tau=1000, depth=3` | `1000 / 2^3 = 125` | `lag=125, traj_length=8` | 保持原版训练 horizon `max_lag=1000` |
| B | `tau=800, depth=3` | `800 / 2^3 = 100` | `lag=100, traj_length=8` | 贴近原版 sample/analyse 默认 lag=100 |

`depth=3` 表示 Bridge 从两个 endpoint 递归二分 3 层，最终每个 endpoint pair 生成 `2^3 + 1 = 9` 帧。原版 ITO 用 `traj_length=8`，同样得到 9 帧轨迹。

### `seed`

`seed` 控制训练随机初始化、数据加载随机性和采样 manifest/噪声。正式计划是 seed 0/1/2。当前完成情况不均衡，所以报告中区分：

- formal run：50 epoch 完成并正常采样分析。
- diagnostic run：训练 NaN 前的 checkpoint sweep 或预 NaN checkpoint 采样，只用于诊断，不作为正式结论。

## 4. 数据与环境

数据目录：

```text
storage/data/ala2/
  alanine-dipeptide-nowater.pdb
  alanine-dipeptide-0-250ns-nowater.xtc
  alanine-dipeptide-1-250ns-nowater.xtc
  alanine-dipeptide-2-250ns-nowater.xtc
```

数据来源为 mdshare ALA2。三条轨迹是独立模拟，每条 250 ns，frame spacing 1 ps，每条约 250000 帧。

本轮 split：

- train：trajectory 0 和 1，共约 500000 frames。
- test：trajectory 2，共约 250000 frames。

训练环境：

- 服务器：`mmgpu11`
- GPU 使用：避免 GPU0。
- conda env：`/localhome3/lyy/miniconda3/envs/codex-ito-bridge-smoke`
- 默认单卡训练：`devices=1`，防止 Lightning 静默拉起多卡 DDP。

## 5. 模型与协议

### 原版 ITO

使用 `TLDDPM` / `PaiNNTLScore`，条件化 `{x_t, lag}` 做 open-loop transition rollout。

主设置：

- epochs：50
- batch size：128
- lr：1e-3
- optimizer/scheduler：Adam + CosineAnnealingLR(T_max=20)
- diffusion steps：1000
- beta schedule：repo 当前 sigmoid schedule
- EMA：0.99
- `n_features=64`
- `n_layers=2`
- `max_lag=1000`
- ODE sampling steps：50
- scale/distinguish：保留 repo 默认

### 改版 Bridge ITO

使用 `BridgeDDPM` / `PaiNNBridgeScore`，条件化 `{x0, xT}` 预测 midpoint，再递归二分生成整段轨迹。

Bridge 只保留任务必要差异：

- 输入为 endpoint pair。
- 输出 midpoint。
- 采样时递归生成 `2^depth+1` 帧。
- 其余训练超参尽量与原版 ITO 对齐。

## 6. 已完成工作

已完成的工程和评估工作：

- 给原版和 Bridge 训练/采样/分析脚本补齐 `--split`、`--seed`、`--devices`、`--accelerator` 等公平评估参数。
- 给两边模型透传 `length_scale`，避免 CLI 参数成为 no-op。
- 移除原版 `train_tlddpm.py` 的 `overfit_batches=1`。
- 给原版 ALA2 数据加载、采样和分析补齐 train/test split，避免 test trajectory 泄漏到训练。
- Bridge 采样按 trajectory 内部生成 valid starts，避免 endpoint 跨 trajectory boundary。
- 统一输出 `metrics.json`，并新增 summariser。
- 完成多轮 formal run、checkpoint sweep、短 epoch smoke、`length_scale=3` pilot 和预 NaN checkpoint 分析。

## 7. 训练完成情况

| 设置 | 模型 | seed | 状态 | 说明 |
|---|---|---:|---|---|
| ls10, Grid A | Bridge | 0,1,2 | 完成 | 三个 seed 均完成 50 epoch |
| ls10, Grid B | Bridge | 0,2 | 完成 | seed0/2 完成 50 epoch |
| ls10, Grid B | Bridge | 1 | 失败 | epoch 39 附近 NaN，仅做预 NaN 诊断 |
| ls10, formal | 原版 ITO | 0,1,2 | 失败 | 分别在约 epoch 41、4、36 NaN |
| ls3, Grid B pilot | 原版 ITO | 0 | 完成 | 完成 50 epoch |
| ls3, Grid B pilot | Bridge | 0 | 失败 | epoch 41 NaN，用 epoch40 checkpoint 做诊断 |

## 8. 主要指标说明

- VAMP2：越接近 held-out MD reference 越好。
- `phi_jsd` / `psi_jsd`：phi/psi marginal histogram JSD，越低越好。
- `rama_jsd`：Ramachandran 2D histogram JSD，越低越好。
- `rama_md_bin_coverage`：生成样本覆盖 MD occupied bins 的比例。较高通常表示覆盖更广，但需结合 `outside_md` 判断是否过分发散。
- `rama_generated_outside_md`：生成样本落在 MD occupied bins 外的比例，越低越好。
- midpoint RMSE：生成中点与 MD ground-truth midpoint 的 circular RMSE，越低越好。Bridge 的 midpoint 是直接任务；原版 ITO 是 open-loop 轨迹中间帧，因此该指标更适合作为对齐诊断。
- closure：Bridge 第一帧/末帧必须等于输入 endpoints。当前已分析 Bridge 样本 closure 均通过。
- stability：NaN/Inf、极端坐标、训练是否 NaN。

## 9. Repo-faithful `length_scale=10` 结果

### Bridge formal results

| Grid | seed | VAMP2 | ref VAMP2 | phi_jsd | psi_jsd | rama_jsd | coverage | outside_md | max_abs_coord |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 1.5969 | 1.6928 | 0.0030 | 0.0051 | 0.0313 | 0.5900 | 0.0032 | 2.87 |
| A | 1 | 1.5041 | 1.6928 | 0.0021 | 0.0039 | 0.0282 | 0.5776 | 0.0018 | 2.78 |
| A | 2 | 1.4034 | 1.6928 | 0.0035 | 0.0045 | 0.0311 | 0.5974 | 0.0048 | 2.91 |
| B | 0 | 1.7413 | 1.7423 | 0.0016 | 0.0033 | 0.0264 | 0.5883 | 0.0030 | 2.87 |
| B | 2 | 1.5242 | 1.7423 | 0.0031 | 0.0067 | 0.0331 | 0.6042 | 0.0029 | 2.91 |

Bridge Grid A 三 seed 均值：

| Metric | Mean ± Std |
|---|---:|
| VAMP2 | 1.5014 ± 0.0968 |
| phi_jsd | 0.00286 ± 0.00071 |
| psi_jsd | 0.00450 ± 0.00058 |
| rama_jsd | 0.03017 ± 0.00174 |
| sample time | 129.8 ± 1.7 s |
| train time | 46035.5 ± 437.5 s |

Bridge Grid B 只有 seed0/2 完成，不能作为完整三 seed formal 结论：

| Metric | Mean ± Std, n=2 |
|---|---:|
| VAMP2 | 1.6328 ± 0.1536 |
| phi_jsd | 0.00237 ± 0.00108 |
| psi_jsd | 0.00502 ± 0.00244 |
| rama_jsd | 0.02973 ± 0.00470 |

### 原版 ITO checkpoint sweep

原版 ITO `ls10` formal 50 epoch 训练全部 NaN，因此补做了 seed0 Grid B 的 pre-NaN checkpoint sweep。

| epoch | VAMP2 | ref VAMP2 | phi_jsd | psi_jsd | rama_jsd | max_abs_coord |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.3835 | 1.7423 | 0.0164 | 0.0165 | 0.0602 | 2.90 |
| 5 | 1.5796 | 1.7423 | 0.0058 | 0.0175 | 0.0468 | 2.87 |
| 10 | 1.6678 | 1.7423 | 0.0072 | 0.0038 | 0.0338 | 2.88 |
| 20 | 1.6405 | 1.7423 | 0.0016 | 0.0058 | 0.0303 | 2.83 |
| 30 | 1.5207 | 1.7423 | 0.0038 | 0.0155 | 0.0429 | 2.84 |
| 40 | 1.4665 | 1.7423 | 0.0037 | 0.0128 | 0.0433 | 2.83 |

解释：

- 原版 ITO 在 NaN 前并非完全无效，epoch 10-20 的指标有竞争力。
- 之后 VAMP2 和 Ramachandran 指标退化，并最终 NaN。
- 这说明原版 `ls10` 问题更像训练稳定性问题，而不是采样或分析链路错误。

### Bridge checkpoint sweep

Bridge `ls10` seed0 Grid B 同样做了 checkpoint sweep。

| epoch | VAMP2 | ref VAMP2 | phi_jsd | psi_jsd | rama_jsd | max_abs_coord |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.2782 | 1.7423 | 0.0155 | 0.0123 | 0.0498 | 2.92 |
| 5 | 1.3700 | 1.7423 | 0.0065 | 0.0062 | 0.0357 | 2.85 |
| 10 | 1.6190 | 1.7423 | 0.0016 | 0.0071 | 0.0293 | 2.88 |
| 20 | 1.6126 | 1.7423 | 0.0017 | 0.0038 | 0.0265 | 2.87 |
| 30 | 1.6424 | 1.7423 | 0.0020 | 0.0058 | 0.0313 | 2.86 |
| 40 | 1.5226 | 1.7423 | 0.0033 | 0.0058 | 0.0331 | 2.84 |
| 50 | 1.7413 | 1.7423 | 0.0016 | 0.0033 | 0.0264 | 2.87 |

解释：

- Bridge seed0 在 epoch 50 达到当前最佳单点结果。
- 但 checkpoint 曲线有波动，不能只凭 seed0 证明整体稳健。
- seed1 后续 NaN，说明 Bridge 也存在训练稳定性问题。

## 10. Paper-alignment `length_scale=3` 结果

`ls3` 是为了检查结论是否依赖 repo 默认 `length_scale=10`。当前只完成 seed0 pilot。

| 设置 | 模型 | 状态 | VAMP2 | ref VAMP2 | phi_jsd | psi_jsd | rama_jsd | coverage | outside_md | max_abs_coord |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grid B, ls3, epoch50 | 原版 ITO | 完成 | 1.6422 | 1.7423 | 0.0038 | 0.0060 | 0.0339 | 0.6217 | 0.0067 | 2.84 |
| Grid A, ls3, epoch50 | 原版 ITO | 完成 | 1.5638 | 1.6928 | 0.0038 | 0.0061 | 0.0352 | 0.6138 | 0.0068 | 2.89 |
| Grid B, ls3, epoch40 | Bridge | 预 NaN | 1.6040 | 1.7423 | 0.0037 | 0.0101 | 0.0363 | 0.5968 | 0.0043 | 2.89 |

解释：

- `ls3` 下，原版 ITO seed0 能完整训练 50 epoch。
- Bridge `ls3` seed0 在 epoch 41 NaN，只能用 epoch40 checkpoint 做诊断。
- 在这个 pilot 中，原版 ITO Grid B 的 VAMP2 和 Ramachandran JSD 均优于 Bridge 预 NaN checkpoint。
- 这削弱了“Bridge 架构天然优于原版”的结论，也提示 length scale 与训练稳定性需要系统排查。

## 11. 稳定性与异常

### 原版 ITO

- `ls10` formal seeds 0/1/2 均训练 NaN。
- seed0 sweep 显示 NaN 前模型可生成合理结构，坐标没有明显崩坏。
- `ls3` seed0 能完整训练，说明原版 ITO 本身不一定有实现级致命错误，`length_scale=10` 可能是不合适或更容易触发优化不稳定。

### Bridge ITO

- `ls10` Grid A 三 seed 完成。
- `ls10` Grid B seed0/2 完成，但 seed1 NaN。
- `ls3` seed0 NaN。
- Bridge 采样 closure check 全部通过。
- Bridge `ls10` seed1 预 NaN checkpoint 出现 `max_abs_coord=23.4`，有少量极端坐标 outlier，应作为稳定性风险记录。

## 12. 当前可支持的判断

可以支持：

- 工程协议已基本打通：训练、采样、分析、summary 都能在统一环境中运行。
- train/test split 和 fixed manifest 已避免主要数据泄漏。
- 原版 ITO 在 `overfit_batches=1` 移除后，`ls10` 长训练不稳定。
- Bridge 在某些 seed 和 grid 上能达到很强指标，尤其 Grid B seed0。
- `length_scale=3` 对原版 ITO 明显更友好，至少 seed0 能完整训练。

不能支持：

- 不能支持 Bridge ITO 已经稳健优于原版 ITO。
- 不能支持 50 epoch 固定终点下 Bridge 比原版更稳定，因为 Bridge 自身也有 NaN。
- 不能用 Bridge Grid B seed0 的单点最好结果代表整体性能。
- 不能把预 NaN checkpoint 结果当作 formal result。

## 13. 建议下一步

优先级最高的是把“模型能力”和“训练崩溃”拆开：

1. 定义统一 checkpoint selection 协议，例如使用固定 validation split 或固定 epoch sweep 中的 best validation loss，而不是强制 50 epoch 最终 checkpoint。
2. 对原版和 Bridge 同时跑 `length_scale=3` 的 seed0/1/2，确认 paper-alignment 下排序是否稳定。
3. 对两边同时测试更保守优化设置，例如 lower lr、gradient clipping 后检查 NaN 是否消失。该步骤应标记为 stability ablation，不混入 repo-faithful 主结论。
4. 继续保留 Grid A/B，但 formal 表格应只纳入完整训练或按统一 checkpoint-selection 规则选出的 checkpoint。
5. 对 Bridge 的 `max_abs_coord=23.4` outlier 做结构定位，确认是否来自个别 pair、采样 ODE、还是模型输出发散。

## 14. 汇报口径建议

建议对外汇报如下：

> 我们已经完成 ALA2 上的严格 train/test 对齐评估框架，并修复了原版/改版对比中的 split、seed、multi-GPU、length_scale 透传和 baseline overfit bug。当前结果显示 Bridge ITO 在部分 repo-default 设置下可以达到非常接近 held-out MD 的 VAMP2，但训练稳定性仍然不足；原版 ITO 在 repo-default length_scale=10 下也不稳定，不过在论文对齐的 length_scale=3 下 seed0 可以完整训练并取得竞争性指标。因此目前结论是 Bridge 有潜力，但还不能宣称稳健超越原版 ITO。下一阶段应先统一 checkpoint selection 或修复 NaN，再进行三 seed formal 对比。
