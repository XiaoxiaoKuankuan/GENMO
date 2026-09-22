> 历史文档，属于整理前提交 `23fd802` 的功能与路径；不作为当前文本分支运行说明。
> 当前入口见 [文本分支 README](../../README.md)，旧源码请在对应功能分支或 Git 历史查看。

# MotionMillion 路线 A0：完整文本与完整动作训练

本文对应 `feature/smpl-text-only`，实现起点是
`9c6d40f55d3a5f94baa10324ace9db470a587eae`。新入口：
`exp=gem_smpl_motionmillion_text_fullseq`。

本次完成代码、CPU 功能测试、配置检查和命令说明。没有启动正式训练、GPU smoke、
多卡任务、全量推理、全量评测或服务器操作，也没有停止既有训练。
这解决训练读取时人为造成的文本/动作时间范围不一致，不证明原 caption 全部准确，
不证明动作质量已经改善。A0 固定补齐到 300；长度分桶 A1 不在本次实现中。

## 1. 数据和训练契约

| 概念 | 旧实验 | A0 |
|---|---|---|
| 源动作真实长度 F | 60—300 | 60—300，30 FPS |
| release 的 `motion_frames` | 120 | 仍为 120，作为旧 release 元数据核对 |
| Dataset 取样 | 随机/中心裁剪至 120 | `sequence_mode=full`，保留 `[0,F)` |
| batch 时间维 L | 120 | `pad_to_frames=300` |
| `length` / `valid_length` | 裁剪后有效长度 | 原始 F |
| 有效帧 mask | 裁剪后有效帧 | 前 F 为 True，其余 False |
| 文本 | T5-3B，150×1024 | 不变，文本 mask 为 150 |
| 动作表示 | SMPL/GVHMR 151D | 不变 |
| denoiser 局部范围 | 120 | 显式 `max_len=120` |
| caption 选择 | 默认与 random_crop 联动 | 独立 `caption_sampling` |
| loss reduction | mask 乘残差后整体 mean | 每样本有效元素均值，再 batch 均值 |

训练 caption 随机选择，验证固定选择第一条。动作、caption、text_index、紧凑 T5
offset 及 attention mask 一起读取，不进行重新标注、时间缩放、循环或动作重复增强。
尾部重复末帧只为存储和组批，mask 始终为 False，不能解释为真实动作延长。

新增配置是独立实验，没有覆盖 `gem_smpl_motionmillion_text_only.yaml`。
网络层数、隐藏宽度、151D 表示、原始各 loss 权重、AdamW 和学习率方案均沿用原配置。
默认随机初始化，不自动加载 s210000 或其他 checkpoint。

## 2. 为什么能复用旧数据

现有转换代码 `tools/data/motionmillion/common.py:recover_smpl_from_272` 按输入 F
返回完整 `pose[F,66]` / `trans[F,3]`；构建器保存完整 record，旧 120 帧裁剪在
`MotionMillionDataset._load_data` 中发生。T5 特征按完整 caption 编码，和是否裁剪
动作无关。因此代码支持复用原始数据、`genmo_smpl_v1`、`t5_3b_v1_fp16`、原 split、
官方 evaluator 权重及其 mean/std，不需要重新转换全量动作或重新编码全量文本。

本机缺少正式 MotionMillion 动作和 T5 分片；**源码支持复用，实际制品待核验**。
测试只对临时构造的分片及其指纹进行了读写和不变性验证，没有检查服务器上的真实分片。

full 模式没有把旧 manifest 的 120 修改为 300，也没有改变数据 fingerprint。
Dataset 的 `motion_frames=120` 专门核对旧 release 值，`pad_to_frames=300` 才控制新
张量长度。两项不能混用。继续校验：schema、split、build fingerprint、shard 顺序、
记录数、motion/T5 的 source SHA 绑定、motion_id、caption offset/数量、索引 F 与
实际 pose/trans 长度、60—300 范围、所有所需 shape 和有限性。
full 索引还必须每个 record 恰好一条、window_index=0；不按帧数重复采样。

初始化只对 manifest / sample index 等小型身份文件计算 SHA；实际分片沿用 LRU。
不在每次 `__getitem__` 哈希大型分片。真实文件 SHA 检查由显式 preflight 完成，
Dataset 校验绑定关系不能替代对实物文件的 SHA 预检。
checkpoint 分别保存 `genmo_data_identity` 和 `genmo_sequence_contract`。
移动数据目录不改变身份；修改 manifest 内容、索引或 release 身份会阻断完整恢复。

## 3. 完整动作、padding 和空间增强

`_load_data` 保留原始有效帧及 crop_start=0。`_process_data` 在调用 BaseDataset 前
取回真实 F 帧，在 F 上完成原有旋转/平移等空间增强、身体模型 FK、相机增强和派生量，
之后递归尾部补齐到 300，再接入 150-token 文本。

SMPL 姿态、平移、betas、相机矩阵等使用末帧延拓；bool 逐帧 mask 的 padding 为 False。
没有构造全零旋转矩阵或全零 6D 旋转。所有嵌套逐帧字段以及图像、音乐、音频占位字段
均保持 300，`f_cond/f_uncond/f_empty` 也使用 L=300。text-only 中不使用音频，新的
audio_array 占位按帧分配零值，不继承旧的每动作帧 600 个音频采样占位。

BaseDataset 相机路径原先每 10 帧抽取后 repeat，非整十长度可能多出帧数；现在 repeat
后截回原 F，覆盖 97、183、299 等长度。既有空间增强及强度保留。读取、切片和增强
不会原地改动 LRU 中的动作或文本。

metadata 包含 motion_id、source_frames、valid_length、crop_start、sequence_mode、
pad_to_frames 和 text_index。F>300、F<60、shape/索引错配会报错，不静默裁剪。

## 4. 注意力、时间边界和 loss

新模式按每条真实 F 构造局部 mask：F≤120 时所有有效帧可见；F>120 时在 `[0,F)` 内
沿用旧的非自回归窗口和两端扩展规则。padding key 对有效 query 始终不可见。
局部窗口与 key padding 共同造成的全遮蔽行，在 softmax 前仅对该行建立有限输入，
之后将该行概率和投影输出归零。没有开放 padding key，也没有全局 nan_to_num。
非空 attention 行保持原计算；旧 120 帧路径有回归测试。

实际 motion RoPE 使用 `rotary_embedding.py` 的 4096 长度缓存；CPU 测试走真实 RoPE、
文本交叉注意力和小型 Transformer，验证 F 输入与 pad300 输入的有效输出一致。
比较显式复制同一段有效噪声，而不是假定相同 seed 在不同 shape 下采出相同噪声。

局部 mask **不是稀疏 attention 实现**。score 仍为稠密 `[B,heads,300,300]`，其面积
相对 120 帧约为 6.25 倍，其他逐帧计算也增加；不能说显存或训练时间保持不变。

`loss_reduction=valid_per_sample` 对每项残差执行：

```text
ell_b = sum(where(broadcast(valid_mask), residual, 0))
        / max(number_of_broadcast_valid_elements, 1)
loss = mean_b(timestep_importance_weight_b * ell_b)
```

覆盖 simple 151D、相机空间关节、平移、2D 投影、顶点、顶点投影、世界平移和 static
分类；可选 shape 监督也使用同一归一化。关节/接触/投影有效性进入广播后的计数，原
关节权重保留在分子。空 mask 返回可求导的有限零。按 FP32 累加，避免半精度大张量
求和溢出。原配置默认均匀扩散采样的 importance weight 为 1；新策略支持非均匀权重
在样本均值后生效，旧策略保持历史数值行为。

151D 末三维是 t→t+1 局部根速度，最后真实帧没有下一帧观测，不监督该帧的速度。
static/contact 标签同样要求两帧都有效；不把最后真实帧与 padding 组成的差分当作
接触观测。通用差分 mask 也覆盖二阶边界测试。原 static 分类不是新增动力学约束；
A0 不启用额外 physics_losses。动作归一化统计量没有更换。

有效元素归一化会改变原实现的有效样本权重，不能宣称数值目标完全相同。后续严格
对照应使用第 10 节的 crop120 数值修复对照组，区分取消裁剪和归一化变化的影响。

## 5. 训练预算、采样和 checkpoint

候选默认每卡 batch=64、8 卡、梯度累积 4，完整累积的 global batch=2048，**待显存验证**。
`training_budget.max_steps=215000`、`warmup_steps=5000` 是继承原方案的起点；修改预算
必须同时通过这两个入口设置，Trainer max_steps 与 scheduler total_steps 自动一致。
warmup 必须小于总步数，max_epochs=-1，避免旧 epoch 上限先结束。没有继承旧墙钟估计。

继续使用 shard-aware sampler，`use_distributed_sampler=false`，sampler 和 DataLoader
均 drop_last，不增加分桶、动态 batch 或采样权重。日志额外报告：每 rank 样本数、
micro-batch 数、累积倍数、每 epoch 优化器步数及最后一次累积的 micro-batch 数。
若每 rank 有 N 条，则 micro-batch 数 M=floor(N/64)，优化器更新数 ceil(M/4)。
epoch 尾部不足 4 个 micro-batch 时仍会更新，该次有效 global batch 小于 2048；
这沿用 Lightning 行为，没有通过重复样本补齐。实际 N、各 rank 日志及 epoch 变化
必须在授权后的多卡 smoke 核验。CPU rank 模拟和 Lightning 装配测试不是 DDP 验收。

新 checkpoint 显式写入：序列模式、真实 min/max F、padding、FPS、attention 模式/范围、
loss reduction、训练/验证 caption 规则，同时保留既有文本契约和独立数据身份。
完整 resume 会比较序列契约，并在 fit 开始时比较 release 身份。
旧 checkpoint 缺少新字段时按旧约定解释；不能自动当成 fullseq。旧→fullseq 或
attention/loss 策略不同的完整恢复会明确拒绝，提示改用 weights-only 初始化。

`pretrain_ckpt=...` 是只载权重的新实验，global_step/优化器/scheduler 重新开始；
`resume_mode=...` 是相同 fullseq 实验的完整恢复。这两种参数不允许同时设置。

## 6. 环境及 CPU 验证命令

复用原 GENMO 训练 `.venv`，没有新增 pip 依赖。需要现有 PyTorch、Lightning、Hydra、
NumPy/SciPy、pytest 等。真实 SMPL CPU 集成测试需要仓库人体资产，缺少会明确 skip；
训练只读已编码的 T5 特征，不启动在线 T5。单条推理另需本地 T5-3B 快照和 GPU。

以下 CPU 命令已在本机运行；pytest 临时目录自动清理，不在生产 outputs 下写测试：

```bash
cd /home/weili/GENMO
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
import os, subprocess, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='genmo-a0-cpu-') as work:
    env = dict(os.environ, MPLCONFIGDIR=work + '/mpl')
    result = subprocess.run([
        '.venv/bin/python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--basetemp', work + '/pytest',
        'tests/test_motionmillion_fullseq.py', 'tests/test_motionmillion_text_only.py',
        'tests/test_demo_smpl_text.py', 'tests/test_resident_text_motion.py',
        'tests/test_training_continuation.py', 'tests/test_text_motion_server_protocol.py', '--tb=short',
    ], env=env)
assert not Path(work).exists()
raise SystemExit(result.returncode)
PY

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tools/data/motionmillion/preflight_motionmillion.py --config-only

# 同时检查短预算，仍然不读取真实数据、不建立模型、不访问GPU：
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tools/data/motionmillion/preflight_motionmillion.py --config-only \
  --config-override training_budget.max_steps=8 \
  --config-override training_budget.warmup_steps=2
```

`--config-only --report /指定位置/config.yaml` 可保存解析后的完整 YAML；非 config-only
模式的 `--report` 则输出数据预检 JSON。CONFIG_PASS 明确标注
`actual_release_verified=false` 和 `gpu_memory_verified=false`。

## 7. 真实数据预检与单卡/多卡 smoke（未执行）

以下命令是运行说明，需要真实数据/资产和相应资源授权；本任务没有执行。先在要运行
的机器进入对应分支仓库。`MM_ROOT` 必须设成已有 MotionMillion 根目录，不要创建新
release，不要把不存在的路径当成已准备好：

```bash
cd /home/weili/GENMO
: "${MM_ROOT:?请先 export MM_ROOT 为已有 MotionMillion 绝对路径}"
test -f "$MM_ROOT/genmo_smpl_v1/manifests/train.json" || exit 1
test -f "$MM_ROOT/t5_3b_v1_fp16/manifests/train.json" || exit 1
A0_DATA=(
  "train_datasets.motionmillion_text_fullseq_train.root=$MM_ROOT"
  "train_datasets.motionmillion_text_fullseq_train.motion_manifest_path=$MM_ROOT/genmo_smpl_v1/manifests/train.json"
  "train_datasets.motionmillion_text_fullseq_train.embedding_manifest_path=$MM_ROOT/t5_3b_v1_fp16/manifests/train.json"
  "test_datasets.motionmillion_text_fullseq_val.root=$MM_ROOT"
  "test_datasets.motionmillion_text_fullseq_val.motion_manifest_path=$MM_ROOT/genmo_smpl_v1/manifests/val.json"
  "test_datasets.motionmillion_text_fullseq_val.embedding_manifest_path=$MM_ROOT/t5_3b_v1_fp16/manifests/val.json"
)
```

下面采用独立 subshell + mktemp；trap 只清理本命令确认位于 `/tmp` 的精确目录。
预检每个 split 最多 1 个 shard，报告已读动作的真实长度分布、padding 比例、有效帧
总数和裁剪计数；full 模式裁剪计数非零即失败。可选 151D 检查用 8 条跨索引样本，
可能读取额外少量 shard。此处不是全量 SHA 或完整 split 泄漏验收。

```bash
(
  set -e
  A0_TMP="$(mktemp -d /tmp/genmo-a0-preflight.XXXXXX)"
  [[ "$A0_TMP" == /tmp/genmo-a0-preflight.* && -d "$A0_TMP" && ! -L "$A0_TMP" ]]
  trap 'rm -rf -- "$A0_TMP"; test ! -e "$A0_TMP"' EXIT
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
    .venv/bin/python tools/data/motionmillion/preflight_motionmillion.py \
    --motion-root "$MM_ROOT/genmo_smpl_v1" \
    --embedding-root "$MM_ROOT/t5_3b_v1_fp16" \
    --sequence-mode full --pad-to-frames 300 --max-shards 1 \
    --normalized-stats-samples 8 --verify-sha256 --report "$A0_TMP/preflight.json"
  cat "$A0_TMP/preflight.json"
)
```

单卡 smoke：真实完整网络，64 条以内人工限量选择，8 个优化器步数，warmup 2；每卡
batch=1，关闭验证/渲染与保存 checkpoint，只检查 forward/backward、数值和峰值显存。
这里的 64 是样本数上限，不是逐条动作截断。

```bash
(
  set -e
  A0_TMP="$(mktemp -d /tmp/genmo-a0-gpu1.XXXXXX)"
  [[ "$A0_TMP" == /tmp/genmo-a0-gpu1.* && -d "$A0_TMP" && ! -L "$A0_TMP" ]]
  trap 'rm -rf -- "$A0_TMP"; test ! -e "$A0_TMP"' EXIT
  PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$A0_TMP/mpl" CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python scripts/train.py exp=gem_smpl_motionmillion_text_fullseq \
    "${A0_DATA[@]}" "output_dir=$A0_TMP/run" \
    training_budget.max_steps=8 training_budget.warmup_steps=2 \
    pl_trainer.devices=1 pl_trainer.accumulate_grad_batches=1 \
    data.loader_opts.train.batch_size=1 data.loader_opts.train.num_workers=0 \
    data.loader_opts.val.num_workers=0 \
    +train_datasets.motionmillion_text_fullseq_train.limit_size=64 \
    +pl_trainer.limit_val_batches=0 +pl_trainer.enable_checkpointing=false callbacks.vis=null
)
```

多卡 smoke：使用空闲且获授权的 8 张 GPU；仍然每卡 batch=1，不验证候选 batch64。
重点核对各 rank 样本数和 batch 数相同、无二次分片、无 NaN/OOM/NCCL 错误。

```bash
(
  set -e
  A0_TMP="$(mktemp -d /tmp/genmo-a0-gpu8.XXXXXX)"
  [[ "$A0_TMP" == /tmp/genmo-a0-gpu8.* && -d "$A0_TMP" && ! -L "$A0_TMP" ]]
  trap 'rm -rf -- "$A0_TMP"; test ! -e "$A0_TMP"' EXIT
  PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$A0_TMP/mpl" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train.py exp=gem_smpl_motionmillion_text_fullseq \
    "${A0_DATA[@]}" "output_dir=$A0_TMP/run" \
    training_budget.max_steps=8 training_budget.warmup_steps=2 \
    pl_trainer.devices=8 pl_trainer.accumulate_grad_batches=1 \
    data.loader_opts.train.batch_size=1 data.loader_opts.train.num_workers=0 \
    data.loader_opts.val.num_workers=0 \
    +train_datasets.motionmillion_text_fullseq_train.limit_size=64 \
    +pl_trainer.limit_val_batches=0 +pl_trainer.enable_checkpointing=false callbacks.vis=null
)
```

服务器需要的 NCCL 环境应按当时环境核验后设置；本文不把历史通信参数视为已验证的
当前环境。候选 batch64×累积4 必须再做独立显存/吞吐验证，不能由上述 batch1 推断。
正式训练前还需真实模型完整恢复 smoke 和真实数据后处理/视频核验。

## 8. 独立正式训练与恢复（未执行）

以下命令会真正启动训练，只有完成真实数据、GPU 和授权检查后才执行。输出单独放在
`outputs/motionmillion_text_fullseq_a0/run01`，不覆盖旧实验；已有同名实验时另选新名字。
保持同一 shell 中第 7 节的 A0_DATA。

```bash
A0_RUN="$PWD/outputs/motionmillion_text_fullseq_a0/run01"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py \
  exp=gem_smpl_motionmillion_text_fullseq "${A0_DATA[@]}" "output_dir=$A0_RUN" \
  training_budget.max_steps=215000 training_budget.warmup_steps=5000 \
  pretrain_ckpt=null ckpt_path=null resume_mode=null
```

真正的训练配置快照保存在该次 Hydra 输出目录 `.hydra/config.yaml`；后续评测使用
该文件，而不是重新解析当前 YAML 假定等同。checkpoint 按原 10000 步周期保存，
新配置额外在训练正常结束时保存非周期最后一步。

相同 fullseq 实验完整恢复（`FULLSEQ_CKPT` 必须指向实际存在且保存完整训练状态的文件；
本任务没有产生训练 checkpoint）：

```bash
: "${FULLSEQ_CKPT:?请设置真实 fullseq checkpoint 的绝对路径}"
test -f "$FULLSEQ_CKPT" || exit 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py \
  exp=gem_smpl_motionmillion_text_fullseq "${A0_DATA[@]}" "output_dir=$A0_RUN" \
  training_budget.max_steps=215000 training_budget.warmup_steps=5000 \
  "resume_mode=$FULLSEQ_CKPT" pretrain_ckpt=null ckpt_path=null
```

若原实验改过预算、batch 或其他参数，恢复时必须复用原实际配置。resume 会恢复 optimizer、
scheduler、global_step；序列契约不同会拒绝。现有 DataLoader 不是可保存游标的数据流，
不能承诺中途恢复的逐样本 RNG/顺序与完全不中断运行逐位相同。

可选 weights-only 迁移只需在**新的输出目录**使用 `pretrain_ckpt="$OLD_CKPT"`、
`resume_mode=null ckpt_path=null`；需要先 `test -f "$OLD_CKPT"`。这不是 A0 默认起点，
不能和随机初始化实验混为一组。

## 9. 推理和固定 128 条按 GT 长度评测（未执行）

demo/resident 从 checkpoint 读取 sequence contract，覆盖推理模型的 attention/range/
reduction 设置，不只依赖当前目录 YAML。新 fullseq 接受 60—300 帧、30 FPS，长度仍
由用户指定，没有长度预测/自动终止模型。请求 240 帧就生成、处理、导出 240 帧。
常规单条推理直接建立 F 帧输入；带 padding 的验证 batch 先拆出有效 F，再执行生成、
轨迹恢复、平滑、足锁和 IK，避免尾部 padding 进入后处理。

```bash
: "${FULLSEQ_CKPT:?请设置实际 fullseq checkpoint}"
: "${T5_SNAPSHOT:?请设置训练 embedding 对应的本地 T5-3B 快照}"
test -f "$FULLSEQ_CKPT" && test -d "$T5_SNAPSHOT" || exit 1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/demo/demo_smpl_text.py \
  --ckpt_path "$FULLSEQ_CKPT" --t5_model "$T5_SNAPSHOT" --local_files_only \
  --prompt 'A person walks forward, then turns left.' \
  --num_frames 240 --fps 30 --seed 42 --ddim_steps 50 --guidance_scale 2.5 \
  --shape_mode zero --output_root outputs/motionmillion_text_fullseq_a0/user_demo
```

该命令保存 SMPL/NPZ/metadata，并按既有 Open3D 路径渲染视频；`--no_render` 可只输出动作。
视频后端可用性、240 帧真实视频/解码尚未验收。旧 checkpoint 无新字段时继续使用旧路径，
50/150-token 根据既有文本契约解析，不静默解释成 fullseq。

小批评测优先复用已有 **固定 128 条** eligibility、GT272D 和 evaluator identity。
用户设置下列实际文件变量，测试产物统一写入临时目录并自动清理；若需要正式保留结果，
另行明确保存目录和保留原因，不覆盖历史 s210000 fixed120 报告。

```bash
: "${ELIGIBILITY:?已有128条 eligibility.json 的绝对路径}"
: "${EVALUATOR_IDENTITY:?绑定该资格集合的 evaluator_identity.json}"
: "${TRAIN_CONFIG:?该fullseq训练保存的 .hydra/config.yaml 绝对路径}"
.venv/bin/python - "$ELIGIBILITY" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))['records']
assert len(rows) == 128, '必须使用已经固定的128条集合，不在这里重新随机抽样'
assert all(60 <= int(r['frames']) <= 200 for r in rows)
assert len({r['motion_id'] for r in rows}) == 128
PY
(
  set -e
  A0_TMP="$(mktemp -d /tmp/genmo-a0-eval128.XXXXXX)"
  [[ "$A0_TMP" == /tmp/genmo-a0-eval128.* && -d "$A0_TMP" && ! -L "$A0_TMP" ]]
  trap 'rm -rf -- "$A0_TMP"; test ! -e "$A0_TMP"' EXIT
  export PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$A0_TMP/mpl"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/eval/generate_motionmillion_val_predictions.py \
    --checkpoint "$FULLSEQ_CKPT" --eligibility "$ELIGIBILITY" \
    --t5-model "$T5_SNAPSHOT" --t5-release "$MM_ROOT/t5_3b_v1_fp16/embedding_release.json" \
    --output-root "$A0_TMP/generation" --seed 42 --length-mode gt --ddim-steps 50 --cfg-scale 2.5
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/eval/run_motionmillion_official_metrics.py \
    --predictions "$A0_TMP/generation/predictions_272.jsonl" --eligibility "$ELIGIBILITY" \
    --evaluator-identity "$EVALUATOR_IDENTITY" --checkpoint "$FULLSEQ_CKPT" \
    --experiment-config "$TRAIN_CONFIG" --dataset-release "$MM_ROOT/genmo_smpl_v1/dataset_release.json" \
    --seed 42 --length-mode gt --ddim-steps 50 --cfg-scale 2.5 --output "$A0_TMP/metrics.json"
  cat "$A0_TMP/metrics.json"
)
```

评分命令的 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 只作用于该进程：当前官方旧 wrapper
没有显式指定 weights_only，PyTorch 2.6+ 需要此兼容开关。前提是使用已经核验官方
来源/SHA 的 evaluator，评分脚本会先复核身份文件；不把该变量设置为全局默认。

如果没有已经物化的 128 条 GT，可在后续获授权的数据准备任务中用既有工具：
`tools/eval/prepare_motionmillion_official_eval.py --raw-root "$MM_ROOT/raw_hf"
--motion-root "$MM_ROOT/genmo_smpl_v1" --output-root "$EVAL_ROOT"
--max-samples 128 --selection-seed 42`。它会读取 val 元数据和相关原始归档，不是本次
已执行的轻量测试。新 evaluator 工作目录还必须按旧文档准备固定权重、代码及身份；
不能只生成 eligibility 就假定 evaluator 完备。已有资格集合不重建、不改写。

`gt` 命名为 `full_sequence_matched_length_v1`：每条 GT 原始 F 对应生成 F，97/183
这样的尾部也保留；不会截到 4 帧整数倍。默认 `fixed` 保留固定 120 帧生成。
这两者均没有实施官方 4 帧单位随机裁剪，不能直接叫论文同协议复现。
官方资格仍为 60—200 帧；201—300 帧只可另做明确标记的长动作诊断，不能混入本指标。
官方 evaluator、mean/std、272D 定义、batch32/drop_last 均不改。

每轮 JSON 输出 FID、Diversity、R@1/2/3、Matching Score、真实数据参考值以及 checkpoint/
数据/evaluator/资格集合/长度协议/seed/DDIM/CFG 身份。生成进度也记录协议，恢复时拒绝
fixed/gt 切换。汇总及候选对比拒绝混合长度协议或资格集合。128 条一 seed 只是流程和
小样本诊断，不是正式 20-seed 评测，也不能和历史 fixed120 得分宣称严格改进。

## 10. crop120 同数值修复对照

旧实验默认仍是旧 reduction。要控制归一化变化的混杂，可在新配置上独立覆盖：

```text
train_datasets.motionmillion_text_fullseq_train.sequence_mode=crop
train_datasets.motionmillion_text_fullseq_train.pad_to_frames=120
test_datasets.motionmillion_text_fullseq_val.sequence_mode=crop
test_datasets.motionmillion_text_fullseq_val.pad_to_frames=120
data.dataset_opts.max_motion_frames=120
```

其余保持 `attention_mode=valid_length`、`loss_reduction=valid_per_sample`，并使用新的
output_dir / 实验名称，从零初始化。序列契约会保存为 crop，无法与 fullseq 完整 resume
混用。此组合已做 CPU 配置解析，未运行真实训练。

## 11. 实际验证与尚未验证

- CPU 测试包含 F=60/97/120/183/299/300 的完整原帧保留、缓存不变性、随机 caption
  对齐、batch/占位、真实 RoPE attention 有限性和 padding 不变性、反向梯度、有效元素
  loss、差分边界、checkpoint 契约、旧模式回归、fixed/gt 身份和实际 Lightning loader
  装配；测试 rank 是模拟上下文，不是实际 DDP。
- 本机有 SMPL-X neutral 资产，实际身体模型、相机增强、EnDecoder151D、扩散及原辅助
  loss 的 CPU 集成已运行。Transformer 缩为 32 维两层，不能据此宣称完整网络显存通过。
- resident 的精确 F 导出使用明确的模型/T5 替身，检查真实保存代码写出的 NPZ/JSON；
  不是 GPU 动作质量或视频验收。评测控制流中的推理器和 272D 转换也明确标为替身。
- 本机正式分片缺失的测试明确 skip。没有重新转换、重新编码、修改旧数据/统计量或
  既有 checkpoint；历史 `motionmillion_s210000_eval128` 报告保持原样。
- 待授权：真实分片完整性、真实模型 BF16/显存/吞吐、8 卡采样和通信、真实完整恢复、
  60—300 帧真实推理/后处理/视频、固定小批 gt 指标，以及之后独立的质量对照。

最终测试数量、清理结果和提交说明见根目录 `记录文本.md` 的本次 A0 条目。
