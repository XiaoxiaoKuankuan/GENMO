# BUMI-native Music-only GENMO

## 目标与边界

这条路径让 GENMO 直接学习 BUMI 机器人状态，而不是先生成 SMPL 人体动作、再在在线推理阶段调用 GMR/IK。模型输入仍是 EDGE baseline35 音乐特征，生成主干仍是带 RoPE 的 16 层扩散 Transformer、DDIM 和 CFG；动作输出改为 BUMI 93D 表示，并确定性解码成 qpos28。

训练和推理代码不 import OMG、`robot_retarget`、GMR-CPP 或 MuJoCo。OMG 只提供过数学定义参考；GMR-CPP 只负责数据到达前的离线人体到机器人重定向。MuJoCo 只出现在 MJCF 资产导出、FK parity 检查和离线渲染工具中。

仓库不版本化真实 BUMI 数据和 train-split 93D stats。与当前 6,610 条 PASS 数据严格
同源的 `482138…` MJCF 导出结果和真实 ankle-roll mesh 足底代理配置分别固定为
`configs/bumi/bumi_kinematics_482138_v1.json` 与
`configs/bumi/sole_proxies_482138_v1.json`。正式数据由
`tools/data/bumi/build_bumi_music_dataset.py` 从只读 legacy pickle 原子转换发布。

## qpos28 契约

模型内外的权威机器人状态是 MuJoCo-native 顺序：

| Slice | 维度 | 含义 |
|---|---:|---|
| `[0:3]` | 3 | 世界或 canonical root XYZ，Z-up，米 |
| `[3:7]` | 4 | root quaternion，固定 `wxyz` |
| `[7:28]` | 21 | 真实 MJCF qpos address 7..27 对应的关节角，弧度 |

21 个关节名及顺序只从真实 `BUMI_KINEMATICS_PATH` 读取。代码不会猜测关节名，也不会自动在 MuJoCo/GMT 顺序之间重排。发布给 GMT 时必须显式调用 `reorder_mujoco_joints_to_gmt()` 并同时提供两套完整、同集合的关节名；默认输出始终保持 MuJoCo-native。

所有输入 root quaternion 都先检查 finite/范数，再归一化，并沿时间轴用相邻点积修复 `q`/`-q` 符号连续性。输出 quaternion 同样归一化。

## 93D 表示

所有 slice 只在 `gem/robots/bumi/feature_codec.py` 定义：

| 字段 | Slice | 维度 |
|---|---|---:|
| `root_pos_local` | `[0:3]` | 3 |
| `root_rot_local`（rot6d） | `[3:9]` | 6 |
| `joint_dof` | `[9:30]` | 21 |
| `body_link_pos_local` | `[30:93]` | 21 × 3 |

总维度为 93。后 63 维是辅助几何监督：它们是 21 个驱动关节对应 feature child-body 的位置，不是 GMR 直接输出，也不是 MJCF 的全部 body。推理时只使用前 30 维组成 qpos28，再通过 Torch FK 重新计算权威 body positions；不会对预测的 63D 位置执行 IK。

## First-frame canonicalization

每个 120 帧训练 crop 独立建立 anchor，不使用 OMG 的 10 帧历史。设 crop 第一帧 root position 为 `p0`，root quaternion 为 `q0`；从 `q0` 提取绕世界 Z 轴的 yaw，并记对应 heading rotation 为 `H0`。`z_default` 来自真实 kinematics 资产中的 `default_qpos[2]`：

```text
p_anchor = [p0.x, p0.y, z_default]

root_pos_local[t] = H0^-1 (p[t] - p_anchor)
root_rot_local[t] = H0^-1 q[t]
body_link_pos_local[t,j] = H0^-1 (body_pos_w[t,j] - p_anchor)
```

因此第一帧水平位置和 yaw 为零，root 的 `z - z_default`、roll 和 pitch 仍保留。canonical qpos 的 root position 就是 `root_pos_local`；其局部地面高度为 `-z_default`。放回世界时，使用期望的 root XY/yaw（以及可选 anchor Z）左乘 heading 并加回 `[x, y, z_default]`。

## 可微 BUMI FK 与资产契约

`BumiKinematics` 加载 `genmo.bumi_kinematics.v1` JSON，至少严格保存并校验：

- root body、21 个 feature body 的顺序；
- 21 个关节顺序和 qpos address 7..27；
- feature-body parent/child 拓扑；
- joint axis、local origin XYZ/quaternion、joint anchor；
- joint range 和 default qpos；
- 明确的左右足 sole proxy、半径和 foot ID；
- `wxyz` 与 `mujoco_native` 标记。

Torch FK 对任意 `[...,28]` 输出 `body_pos_w [...,22,3]` 和 `body_quat_w [...,22,4]`：一个 root body 加 21 个 feature bodies。固定 body 变换由 exporter 折叠进最近的上游 feature body，不把 mesh/fixed bodies 塞进 93D 表示。

真实 sole proxy 必须由版本化 proxy config 明确指定真实 body/geom 或 feature-body local point；exporter 不含任何 BUMI 名称猜测。当前 proxy 使用左右 ankle-roll STL 最低 1 mm 支撑簇中的真实 mesh vertex。正式 v1 保留 legacy root 高度，地面语义不是鞋底地面，因此这些 proxy 仅用于 FK/未来版本，正式随机初始化实验硬性关闭 penetration/contact/sliding。

当前 mesh-vertex proxy config 使用已经转换到 ankle-roll feature-body 局部系的
真实支撑点；名称、坐标和半径都必须来自绑定资产：

```json
{
  "contract_version": "genmo.bumi_proxy_config.v1",
  "sole_proxies": [
    {
      "name": "...",
      "foot": "left",
      "feature_body_name": "l_ankle_roll_link",
      "local_position": [0.0, 0.0, 0.0],
      "radius": 0.0
    },
    {
      "name": "...",
      "foot": "right",
      "feature_body_name": "r_ankle_roll_link",
      "local_position": [0.0, 0.0, 0.0],
      "radius": 0.0
    }
  ],
  "evaluation_proxies": []
}
```

## 数据契约 `genmo.bumi_music.v1`

GMR 生产版 SMPL-X→BUMI3 legacy pickle 在进入本契约前，必须先执行独立的
source-asset 绑定、
动力学和贴地风格预筛选；规则、命令和报告字段见
[BUMI3 重定向动作预筛选](bumi_motion_quality_filter.md)。正式转换只能接收报告中
`quality_accepted=true` 的 PASS 样本，不能把 REVIEW 静默提升为接受。

```text
<dataset_root>/
├── manifests/{train,val,test}.jsonl
├── motions/*.pt
├── musicfeat_v2/*.pt
├── audio/*.wav
├── reports/conversion_report.json
└── meta/dataset_info.json
```

`dataset_info.json` 必须包含：

```json
{
  "contract_version": "genmo.bumi_music.v1",
  "robot_name": "bumi",
  "qpos_dim": 28,
  "joint_dim": 21,
  "joint_names": ["21 names from the real kinematics asset"],
  "quaternion_convention": "wxyz",
  "qpos_order": "mujoco_native",
  "fps": 30,
  "quality_filter_applied": true,
  "mjcf_sha256": "64 lowercase hex chars",
  "source_mjcf_sha256": "same as mjcf_sha256",
  "kinematics_sha256": "64 lowercase hex chars",
  "retarget_config_sha256": "64 lowercase hex chars",
  "quality_config_sha256": "64 lowercase hex chars",
  "ground_semantics": "legacy_body_origin_min_zero",
  "root_z_adjusted": false
}
```

Motion `.pt` 是字典，必需字段为 `qpos [T,28]`、`fps=30`、`robot_name=bumi`、完整 `joint_names`、`quaternion_convention=wxyz`、`qpos_order=mujoco_native`；可包含 `foot_contact [T,2]`、`quality_accepted`、source/retarget metadata。正式数据的 manifest 和 dataset info 必须已经表明 quality accepted/applied。Music `.pt` 明确定义为原始 finite `Tensor[T,35]`，通道顺序与 EDGE baseline35 一致。

Manifest 每行必须含 `sample_id`、`sequence_id`、`music_group_id`、`audio_key`、`dataset`、motion/EDGE35/audio 三个相对路径、`fps`、`num_frames`、`split`、`quality_accepted`，以及 source motion/EDGE35/audio 三个 SHA256。路径必须留在 dataset root 内。严格验证工具会扫描 payload 和三个来源 SHA；训练只读取 qpos/EDGE35，WAV 用于审计、demo 和特征重算。

训练 crop 对 qpos 和 music 使用同一个 `[start:end]`。短序列 qpos 用最后一帧补到 120，music 用零补齐；`length` 保存真实长度，`valid`/`has_music_mask` 只标记真实帧。正式 v1 设置 `enable_contact_targets=false`，既不伪造标签，也不派生或训练 contact head。

正式采样不是 `ConcatDataset` 的多人总时长加权，而是确定性的四层采样：先按每数据集去重音乐时长平方根（投影到 5%～50%）选择数据集，再按组最大有效时长选择音乐组，组内均匀选舞者/编舞/角色，最后均匀选 120 帧窗口。当前目标为 AIST++ 11.3993%、AIOZ 50%、FineDance 33.6007%、CoMPAS3D 5%。每个 epoch 全局 52,224 条，8 卡按全局 draw index 无重复分片，每卡 batch 128 时恰好 51 step。

磁盘解析集中在 `BumiMusicDatasetReader`。未来真实转换格式变化时，应只调整 reader/转换器，不能让 Pipeline 或 loss 依赖目录布局。

## Stats 契约

`BUMI_MUSIC_STATS_PATH` 必须指向 `genmo.bumi_stats.v1` JSON，包括 93 维 mean/std、固定 slices、真实 joint names、anchor mode、kinematics SHA 和各数据集 fingerprint。统计只使用 train split；每条序列按 stride 120 枚举 120 帧窗口并包含最后一个合法窗口，每个窗口独立以第一帧 canonicalize。短序列只让真实有效帧进入统计。

实现使用流式 Welford，不拼接全量数据。最终只执行 `std.clamp_min(1e-6)`，不会采用 SMPL 的 `std < 1 -> 1`。`is_placeholder=true` 默认被正式 Endecoder 拒绝；仓库不附带可被训练静默使用的 identity stats。
正式 DataModule 还会在启动时重新计算四个 `dataset_info.json` 和 train manifest 的
SHA256，与 stats 中的 fingerprint 逐项比对，并要求 train 序列总数严格为 5,537；
数据或统计量任一方变化都会在创建训练 loader 前失败。

## 模型与训练路径

```text
BumiMusicDanceDataset
  → BumiEndecoder (qpos/FK → normalized 93D)
  → diffusion_lg_bumi93 (RoPE Transformer + music CFG)
  → BumiMusicPipeline
  → canonical/world qpos28 + Torch FK
```

`BumiEndecoder` 提供：

- `encode()` 和 `encode_with_aux()`；后者返回 normalized/physical 93D、canonical qpos28、GT body-link positions、anchor metadata、contact target/mask；
- `normalize()` / `denormalize()`；
- `decode()`，返回 local root XYZ、root rot6d/quaternion、21 DOF 和 raw 21×3 body positions；
- `compose_qpos()`，从前三组组成 canonical qpos，可选放回 world anchor；
- `authoritative_body_link_positions()`、obs slice 查询和 contact 派生。

`BumiMusicGEM` 继承通用优化器、scheduler、日志、扩散和 CFG 生命周期，但覆盖了 SMPL `prepare_batch`、condition mask、validation 和 predict。它不实例化 SMPL body model，不建立 camera/2D/image/audio/text 条件，不计算 vertices/COCO17/bbox/projection，也不进入 SMPL flip/postprocess/IK。唯一训练模式是 diffusion，唯一条件是 `encoded_music`。CFG 训练只执行一次 sample-level 全序列 music dropout，不再执行 per-frame random null。

## 损失

正式实验使用独立的 `physical_v1` 合约，旧 `music_only_bumi.yaml` 骨架路径保留不变。所有物理项在 FP32 和 `valid` mask 下计算；一、二、三阶差分分别要求连续 2、3、4 帧全部有效。四组 normalized representation MSE 权重均为 1.0，其余辅助项在前 10k step 线性 warmup，并同时记录 raw/normalized/weighted：

- Root XYZ `SmoothL1(/1m)` 0.1，Root SO(3) geodesic `/π` 0.1；
- Joint DOF `SmoothL1(/1rad)` 0.1；
- qpos Torch FK 对 GT FK `SmoothL1(/1m)` 0.5；
- raw 63D/FK 一致性 `SmoothL1(/1m)` 0.1；
- joint velocity `SmoothL1(/6rad/s)` 0.01；
- joint acceleration `SmoothL1(/180rad/s²)` 0.002；jerk 首轮为 0；
- soft joint limit `SmoothL1(violation/0.1rad)` 0.01；
- root height GT `SmoothL1(/1m)` 0.05；
- contact BCE、foot sliding、sole penetration 均为 0，且 ground contract 会拒绝误开启。

推理结果始终以 qpos FK 为准，raw 63D 不进入最终机器人状态。

## SMPL music checkpoint 迁移

设置：

```bash
pretrain_ckpt=/path/to/gem_smpl_music_only_4set.ckpt \
model.model_cfg.checkpoint_adapter=smpl_music_to_bumi
```

显式 adapter 只精确加载 music embedder、`encoded_music` condition-exists embedder、diffusion timestep embedder、RoPE Transformer blocks 及其 attention/LayerNorm/MLP。它跳过 SMPL Endecoder/body model、151D final layer、pred-cam head、6D static head、SMPL stats/body-pose/betas 模块。

`add_cond_linear.weight` 的输入拼接是 `[f_cond, xt]`：只复制前 `latent_dim` 个 condition columns；新的 93 个 motion columns 保持 BUMI 模型初始化。Bias 全量复制。任何未分类 shape mismatch 都立即报错。报告分类为 `loaded_exact`、`loaded_partial`、`reinitialized`、`skipped_expected`、`missing_expected`、`unexpected`、`unclassified_shape_mismatch`，并在 `on_fit_start` 写到实际 run 目录的 `checkpoint_adaptation_report.json`。

`checkpoint_adapter=null` 表示普通 BUMI weights-only 初始化。Lightning 的 `resume_mode` 仍负责完整训练状态 resume，两者语义没有混用。

## Hydra 配置

骨架入口仍为 `configs/exp/gem_bumi_music_only_4set.yaml`；正式随机初始化入口是 `configs/exp/gem_bumi_music_only_4set_random_v1.yaml`，覆盖：

```text
data       = music_robot/trainX_testY
model      = bumi_music_gem
network    = diffusion_lg_bumi93_no_contact
pipeline   = music_only_bumi_physical_v1
endecoder  = bumi_93d_no_contact
```

四个训练 root 必须通过 `AISTPP_BUMI_ROOT`、`AIOZ_GDANCE_BUMI_ROOT`、`FINEDANCE_BUMI_ROOT` 和 `COMPAS3D_BUMI_ROOT` 提供；资产通过 `BUMI_KINEMATICS_PATH` 和 `BUMI_MUSIC_STATS_PATH` 提供。正式配置完全随机初始化，AdamW `2e-4`，300k/450k 各衰减一半，固定 500k step，每 10k 保存和验证四数据集。

原 `gem_smpl_music_only_4set` 未被替换。公共默认仍是 `motion_backend=smpl`、SMPL EnDecoder 151D、`diffusion_lg` 151D、原 SMPL Pipeline/数据集/checkpoint 行为；公共改动只有动态 motion dimension 和显式 backend 扩展点。

## 推理输出

`scripts/demo/demo_music_bumi.py` 接收 WAV 或预计算 EDGE35、BUMI checkpoint、CFG scale、DDIM steps、可选帧数及 world root XY/yaw。输出 `.pt` 至少包含：

```text
qpos                    [T,28]
qpos_canonical          [T,28]
fps                     30
robot_name              bumi
joint_names             exact 21-name MuJoCo order
quaternion_convention   wxyz
qpos_order              mujoco_native
feature_dim             93
anchor_mode             first_frame_xy_yaw_default_height
music_path              source path
```

推理不调用 SMPL、GMR、在线 IK 或 pred_cam。未提供 world anchor 时，`qpos` 与 canonical qpos 相同；提供 anchor 时，`qpos` 是放回世界的轨迹，而 `qpos_canonical` 始终保留。

## 运动学评估与动力学验证

`eval_bumi_music.py` 实现 joint angle MAE、root trajectory/FK error、joint limits/margin、sole penetration/sliding、root height/tilt、joint velocity/acceleration/jerk P95、root linear/angular velocity、contact accuracy、beat alignment 和 batch diversity。它们都是运动学质量指标。

`render_bumi_motion.py` 对每帧设置 qpos 后只调用 `mujoco.mj_forward` 渲染。FK parity 同样只比较几何。它们不证明 GMT 可跟踪、机器人不会跌倒或扭矩可行。下一阶段必须执行“生成轨迹 → GMT → MuJoCo `mj_step` dynamics tracking”才能评估这些声明。

## 服务器 2 构建、验收与训练

服务器 2 的固定资产路径如下；MJCF SHA 必须是 `482138…`，不能替换为 OMG/GMR-CPP 的其他版本：

```bash
cd /home/user/liwei/GENMO
source .venv/bin/activate

export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_v1
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export AISTPP_BUMI_ROOT=$BUMI_BASE/AIST++
export AIOZ_GDANCE_BUMI_ROOT=$BUMI_BASE/AIOZ-GDANCE
export FINEDANCE_BUMI_ROOT=$BUMI_BASE/FineDance
export COMPAS3D_BUMI_ROOT=$BUMI_BASE/CoMPAS3D
export BUMI_MUSIC_STATS_PATH=$BUMI_BASE/meta/bumi_93d_stats_train_v1.json
```

先生成固定传输清单，再在四套 WAV 到齐后执行一次全有或全无的转换：

```bash
python tools/data/bumi/build_bumi_transfer_filelists.py \
  --selected-root /data0/user/liwei/datasets/bumi_motions_quality_v1 \
  --human-root aistpp=/data0/user/liwei/datasets/music_dance_genmo/AIST++ \
  --human-root aioz_gdance=/data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE \
  --human-root finedance=/data0/user/liwei/datasets/music_dance_genmo/FineDance \
  --human-root compas3d=/data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D \
  --output /data0/user/liwei/datasets/bumi_transfer_plan_v1

python tools/data/bumi/build_bumi_music_dataset.py \
  --selected-root /data0/user/liwei/datasets/bumi_motions_quality_v1 \
  --human-root aistpp=/data0/user/liwei/datasets/music_dance_genmo/AIST++ \
  --human-root aioz_gdance=/data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE \
  --human-root finedance=/data0/user/liwei/datasets/music_dance_genmo/FineDance \
  --human-root compas3d=/data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D \
  --audio-root aistpp=/data0/user/liwei/datasets/bumi_music_audio_selected_v1/aistpp \
  --audio-root aioz_gdance=/data0/user/liwei/datasets/bumi_music_audio_selected_v1/aioz_gdance \
  --audio-root finedance=/data0/user/liwei/datasets/bumi_music_audio_selected_v1/finedance \
  --audio-root compas3d=/data0/user/liwei/datasets/bumi_music_audio_selected_v1/compas3d \
  --source-mjcf /data0/user/liwei/datasets/bumi_assets_482138_v1/mjcf/bumi3.xml \
  --ik-config /data0/user/liwei/datasets/bumi_assets_482138_v1/ik/smplx_to_bumi3_auto.json \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --output-root "$BUMI_BASE"
```

对四套数据逐一执行带 source SHA 的严格扫描，然后只用 train split 计算统计量：

```bash
for ITEM in \
  "AIST++ aistpp_bumi" \
  "AIOZ-GDANCE aioz_gdance_bumi" \
  "FineDance finedance_bumi" \
  "CoMPAS3D compas3d_bumi"; do
  set -- $ITEM
  python tools/data/bumi/validate_bumi_music_dataset.py \
    --root "$BUMI_BASE/$1" --dataset-name "$2" \
    --kinematics "$BUMI_KINEMATICS_PATH" --splits train val test
done

python tools/data/bumi/compute_bumi_93d_stats.py \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --dataset "aistpp_bumi=$AISTPP_BUMI_ROOT" \
  --dataset "aioz_gdance_bumi=$AIOZ_GDANCE_BUMI_ROOT" \
  --dataset "finedance_bumi=$FINEDANCE_BUMI_ROOT" \
  --dataset "compas3d_bumi=$COMPAS3D_BUMI_ROOT" \
  --output "$BUMI_MUSIC_STATS_PATH"
```

单 batch、显存和 100-step smoke 通过后，正式训练使用同一入口并去掉
`max_steps` 覆盖。随机初始化实验不得设置 `pretrain_ckpt`：

```bash
NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_BLOCKING_WAIT=1 python -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1 \
  pl_trainer.max_steps=1 use_wandb=false

NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_BLOCKING_WAIT=1 python -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1 \
  pl_trainer.max_steps=100 use_wandb=false

NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_BLOCKING_WAIT=1 python -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1
```

这些检查只验证数据、运动学、生成训练和运动学指标，不声明 GMT 动力学可跟踪、平衡或扭矩可行。
