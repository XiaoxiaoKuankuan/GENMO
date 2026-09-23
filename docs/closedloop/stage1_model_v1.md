# 第 4 步：Stage 1 条件 Actor、监督训练与采样

本实现属于 `feature/bumi-music-closedloop`。外部数据字段完全沿用
[Stage 1 契约](stage1_contract_v1.md) 与第 3 步 Dataset/validator；没有引入第二套
batch、51D 动作或 joint velocity head。原 music-only 入口和 checkpoint 校验不变。

## 1. 数据与模型路径

```text
music_features [B,120,35] ──明确映射 music_embed──原音乐 MLP / 存在标志 MLP──┐
proprio_history [B,H,48] + valid + 相对时间──固定物理尺度──掩码 GRU──零残差─┤
known_qpos30 [B,120,30] + 逐坐标 mask──原 normalize──轻量前缀 MLP──零残差──┤
                                                                      ↓
未知坐标 xt + 干净已知坐标 ───────────原 NetworkEncoderRoPE / RoPE Transformer
                                                                      ↓
                               normalized qpos30 + 独立 contact2 logits
                                                                      ↓
                            原 denormalize / qpos28 解码 / BUMI FK
```

- [`gem/closedloop/actor.py`](../../gem/closedloop/actor.py) 中 `Stage1Actor` 是普通
  `nn.Module`，不持有优化器、loss 或训练步数。后续阶段可以直接复用同一 Actor。
- 默认主干仍是 1024 维、16 层、8 heads，120 帧、30 Hz；历史不拼入动作时间轴。
- `adapt_conditions()` 只选择 `STAGE1_CONDITION_KEYS`，直接调用既有条件 validator。
  不访问 `target_qpos30`、`target_contact` 或其 valid 字段，也不复制 contact 作前缀条件。
- `known_qpos30` 与 `target_qpos30` 都是 physical 值，分别显式调用现有
  `BumiEndecoder.normalize()`。前缀未知占位在标准化前后均屏蔽，不能把 physical 零
  变换成的非零数值解释为条件。target 只进入训练加噪及损失。
- 保留原有共同 anchor、MuJoCo qpos30 顺序、GMT proprio48 顺序和时间轴；不重新
  分别编码 prefix/future。Dataset provenance/meta 不变，设备迁移只移动张量。

## 2. 历史和前缀条件

历史用 `GRUCell(48+1, hidden_dim)` 按原时间顺序处理。额外一维是
`proprio_history_times - decision_time`，单位秒。无效槽跳过状态更新；全空历史的输出
严格为零，即使残差输出投影已经训练也成立。`history_steps=H` 可配置为任意正整数。

Proprio 独立归一化为逐字段 `value / scale`，四个尺度依次对应单位重力方向、角速度、
相对关节位置、相对关节速度。首版默认 `[1,1,1,1]`，即明确保留原物理单位；它是固定
物理尺度配置，不是经验 mean/std，没有拟合 train/val/test 数据，也不使用或裁剪 GMT
69D normalizer。第 3 步的 train-only stats 工具保持独立，当前 Actor 不自动加载其产物。

前缀编码每帧接收 `[masked normalized qpos30, bool mask as float]` 共 60 个输入量。
因此同一 normalized 数值为零时，已知／未知仍可区分。这只是条件编码输入宽度，不改变
30D 动作表示。无已知坐标的帧输出严格为零；支持 `P=0` 和各坐标不同的已知长度。

两条新分支的最后输出投影初始化为零，并加到原音乐条件上。初始化时原有音乐条件与
有效帧 denoiser 输出保持一致。padding、无效历史、缺失音乐和未知前缀占位都在参与
对应模块计算前屏蔽；推理即使整条 `future_valid` 全空也通过零 sentinel 避免全 mask
softmax，再屏蔽全部输出。

## 3. 一致的条件扩散

训练使用原 1000 步 Gaussian diffusion 日程和 x0 prediction：

```text
unknown_valid = target_qpos30_valid & ~known_qpos30_mask
known_x = mask(normalize(known_qpos30))
xt = where(known_mask, known_x, q_sample(normalize(target), timestep, noise))
```

无效 target 在 normalize 前后屏蔽；缺少 halo 的未知末端 XY 没有 x0 标签，其输入仅用
独立噪声，padding 则为零。训练与推理都把已知部分当干净条件，denoiser 接收前再次
合成已知值，预测 x0 后也合成已知值。

采样复用 `SpacedDiffusion.ddim_sample()`、原噪声日程和 timestep respacing，`eta=0`。
每个 DDIM 更新后再次执行逐坐标约束；前缀末端未知 XY 继续生成，不会误锁整帧。
最终 physical qpos30 用原输入 known 值精确回写，避免标准化往返舍入改变承诺坐标。

CFG 仅切换音乐可见性；历史和前缀残差只计算一次，共用于条件／无音乐分支。
动作与独立 contact head 的 CFG 插值沿用原行为。训练音乐 dropout 默认为 0.1，
从不移除历史、前缀、mask 或标签。

```python
# actor 已由配置构造并加载匹配权重；conditions 是原契约的十个条件字段。
actor.eval()
generated = actor.sample(conditions, steps=20, guidance_scale=2.5)
physical_qpos30 = generated["qpos30"]       # [B,120,30]
contact_probability = generated["contact"] # [B,120,2]
canonical_qpos28 = generated["qpos"]       # [B,120,28]，原 canonical anchor
```

`sample()` 不需要 target，即使传入带 target 的整个 batch，也只选择条件键。
`normalized` 返回网络 x0 域，`contact_logits` 供独立监督；`return_trace=True` 可返回
每步保持已知坐标后的 normalized 样本，用于约束诊断。

## 4. 损失适配与旧损失边界

旧 `BumiRobotLosses.forward()` 的主要路径使用整帧 valid，不能直接表达本任务的逐坐标
已知／未知和无 halo XY。新增
[`Stage1BumiLosses`](../../gem/closedloop/losses.py) 继承其严格 v5 配置检查，复用原
FK、SO(3)、有限差分、限位、物理尺度与 top-k/max 数学，但独立适配归约和支持点。

| 项目 | Stage 1 实际处理 |
|---|---|
| 主动作重建 | 仅 `target_qpos30_valid & ~known_qpos30_mask`；原三个表示组各除以有效未知元素数，另记录全 30D `raw_reconstruction_loss` |
| contact BCE | 只用 `target_contact` 和逐脚 `target_contact_valid`，有效元素归约 |
| FK／姿态／限位 | 合成已知前缀与生成预测后，在统一时间轴上 decode、积分和 FK |
| root XY | 第 t 帧积分需要此前有效 delta 和 heading，显式传播支持点有效性 |
| 速度／加速度／jerk | 一阶至三阶分别要求全部 2／3／4 个有效支持点，保留跨前缀边界项 |
| 无 halo 的末帧 XY | 无效值不进入监督或 FK 支持点；有真实 halo 时仅恢复该位移的重建 |
| 脚滑／接触高度 | 使用独立 GT contact 与逐脚有效 mask，不能靠预测低接触逃避物理损失 |

当前全部 v5 权重、长尾及 warmup 均支持；未知权重会报错。小规模配置明确把 warmup
设为零，使第 4 步验证实际覆盖全部损失，不从旧 global_step 推导权重进度。

此路径要求 floor-zero 数据。第 3 步 batch 没有提供 legacy 数据所需的全序列地面估计，
因此加载器明确拒绝 `legacy_body_origin_min_zero`，不会从裁剪验证窗口伪造地面。
Dataset 每个实际 entry 的运动学 SHA 必须与 Actor 相同，随后 reader 继续核验数据资产
和有序关节，避免同为 30D 却使用不同 BUMI 资产。

## 5. checkpoint 与独立入口

[`checkpoint.py`](../../gem/closedloop/checkpoint.py) 先调用原
`BumiMusicGEM._validate_representation_checkpoint()`，再按明确映射迁移三组权重：

| 原 checkpoint 前缀 | Stage 1 Actor 前缀 |
|---|---|
| `pipeline.denoiser3d.denoiser.*` | `denoiser.*` |
| `music_embedder.*` | `music_embedder.*` |
| `cond_exists_embedder.encoded_music.*` | `cond_exists_embedder.*` |

报告逐项列出 `loaded`、`new`、`missing`、`shape_conflicts`、`unexpected`。新增分支
保留初始化，缺失、冲突或意外参数拒绝加载，最后用完整 state 执行 `strict=True`。
不恢复旧 optimizer、scheduler、epoch/global_step；新 AdamW 从 step 0 开始。

有些原 checkpoint 只内嵌表示版本，未记录 stats／运动学 SHA 或 heads／dropout 等
无参数配置。未提供额外证据时，报告分别标明 `not_embedded_in_legacy_checkpoint`，
参数迁移通过不等于这些身份已经核验。可显式提供 `warm_start_source_assets` 和
`warm_start_source_model_config`；一旦提供，内容不匹配就拒绝迁移。不得通过切换到
另一份 stats 后仍宣称原模型数值完全一致。

新 checkpoint 写入：条件接口 `genmo.bumi_closedloop.actor.v1`、已有 batch 契约、
历史长度与尺度、前缀／CFG 策略、主干及条件模块无参数配置、完整运行配置、stats/FK
SHA、有序关节、权重和 warm-start 来源报告。新 checkpoint 同样严格核验这些身份。
当前保存的是 Actor 权重，明确不提供 optimizer resume。

独立文件：

- [`configs/closedloop/stage1_train.yaml`](../../configs/closedloop/stage1_train.yaml)：
  默认 **8 个小规模优化步**、batch 2、H=50、可变 P∈[0,24]，四库来源加权采样。
- [`configs/closedloop/stage1_validate.yaml`](../../configs/closedloop/stage1_validate.yaml)：
  默认 2 个验证 batch、DDIM 10、CFG 2.5；采样调用只传条件，生成后才计算标签损失。
- [`tools/train_closedloop_stage1.py`](../../tools/train_closedloop_stage1.py)：独立 CLI；
  默认选择 validate，所有默认产物写系统临时目录并自动删除。

四库训练来源权重位于既有 Dataset 配置的 `train_sampling_reference`，由 Stage1
`WeightedRandomSampler(replacement=True)` 实际读取。2026-09-23 按用户选择提高
高质量小库的覆盖率，当前权重和为 1，实际抽样概率为：

| 来源 | 训练抽样概率 |
|---|---:|
| AIST++ | 20% |
| AIOZ-GDANCE | 35% |
| FineDance | 25% |
| Mine | 20% |

FineDance 与 Mine 合计由此前约 11.92% 提高到 45%；固定训练步数下，两库预期抽样次数
分别变为此前约 3.71 倍和 3.86 倍。比例描述长期随机抽样期望，单个 batch 不保证固定
配额；库内继续使用既有 duration-aware 索引和随机 decision frame，验证/测试不使用
这组训练权重。此调整体现用户对数据质量的取舍，生成质量收益仍需后续正式对照验证。

默认数据配置现指向筛选后重新 90/5/5 划分的独立发布版本，详见
[划分与评估边界](stage1_resplit_90505_v1.md)。新 train 的 stats 不能自动替换旧权重的 stats。
原始划分配置保留；以下历史 warm-start 示例显式指定原数据配置，避免混用新统计量。
Mine 的旧地面语义仍会被训练器拒绝，必须在后续独立地面适配完成后才能运行四库训练。
在数据、stats 与匹配权重实际可用且地面准入满足的机器上，显式设置路径后执行：

```bash
export BUMI_CLOSEDLOOP_FOURSET_ROOT=/实际路径/bumi_music_umr70_mine_pass_v1
export BUMI_MUSIC_QPOS30_STATS_PATH=/与所选权重匹配的已有qpos30_stats.json
python tools/train_closedloop_stage1.py \
  --config configs/closedloop/stage1_validate.yaml \
  --set dataset_config=configs/closedloop/stage1_dataset_server1_fourset_v1.yaml \
  --set warm_start_checkpoint=/匹配的原qpos30_contact2.ckpt

python tools/train_closedloop_stage1.py \
  --config configs/closedloop/stage1_train.yaml \
  --set dataset_config=configs/closedloop/stage1_dataset_server1_fourset_v1.yaml \
  --set warm_start_checkpoint=/匹配的原qpos30_contact2.ckpt \
  --set train.max_steps=8
```

只在明确希望保留小规模产物时传 `--output-dir /独立空目录`；会保存参数审计、配置、
结果以及训练模式的新 Actor checkpoint，不覆盖已有内容。以上配置不构成第 5 步正式
训练配置，也没有自动启动服务器任务。

## 6. 第 4 步实际验证

已执行原第 3 步 Dataset、实际 fe934 FK、小型真实 Transformer 的条件/梯度回归；
覆盖 H=1/7/50/61，P=0/1/6/119，全空历史、全 padding、逐坐标 mask、占位不变性、
target 隔离、相对时间与历史顺序、CFG 共用条件、完整 v5 反传和 checkpoint 拒绝路径。
合并运行 `tests/closedloop` 与原 `tests/bumi`：**260 passed、4 skipped**，10.71 秒；
其中新增 Actor 20 项、损失 10 项、训练/checkpoint/CLI 12 项。已有条件跳过项不变。
独立入口完成短训练→保存新 checkpoint→严格重载→纯条件验证；Ruff lint/format 通过。

真实权重验证使用本机以下捕获包，不修改原文件：

```text
/home/weili/GENMO/inputs/checkpoints/
  bumi_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000_20260914/
```

核验 checkpoint、stats、kinematics 三份文件 SHA 与捕获清单一致：

- checkpoint：`fdf3bd67910b76b252d77932b485445258262fa24b5286850812fbe33aca51cc`
- stats：`3bf7251551d07b4e97ec6bd9cc196e3abc3803cf64681cdc035039ba27550dba`
- kinematics：`c08731704dccece11351b6fa877e30bac5ca2a8d363de30af7ca2ea1398f4029`

完整 1024 维／16 层／8 heads 模型严格加载原 312 项参数，新增 10 项，缺失／冲突／
意外项均为 0。另从捕获包 `training/config.yaml` 和原构造函数明确提取 backbone、
condition_modules、diffusion_steps、noise_schedule 四组无参数配置，严格迁移再次核验通过，
资产报告为 `verified_exact_content`，配置报告为 `verified_explicit_fields`。
GPU 使用 RTX 4090，临时合成配对数据经原 Dataset/collate 构造
120 帧布局，batch 2 包含 P=0 与 P=4、有效帧 120 与 5、有效历史 0 与 50。
两个 AdamW 小步的 loss 为 `10.1970701 → 9.4821854`，裁剪前梯度范数
`770.7177734 / 437.4634705`，均有限。4 步 DDIM、CFG 2.5 的每步约束通过，最终
已知 physical 前缀误差为 0；固定噪声时篡改 target 不改变生成。峰值 CUDA allocated
约 4112 MiB。临时数据与该次更新权重已释放，没有保留 smoke checkpoint。

这些是接口、严格迁移、可微训练和采样约束证据；合成数据上的两次随机噪声更新不能
证明收敛、动作质量或 proprio 控制有效性。尚未执行服务器四库全量读取／正式统计、
第 5 步正式训练、GMT rollout、Critic／DPPO、动力学、部署或实机验证。
