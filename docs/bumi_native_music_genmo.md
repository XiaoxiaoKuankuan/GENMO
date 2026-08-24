# BUMI-native Music-only GENMO

> 当前代码的数据表示已升级到 `genmo.bumi_motion_features.v2`。下方 2026-08-19 的训练
> 状态和 `s430000` 属于旧 `root_pos_local` v1 历史，不能由当前代码加载；v2 必须重算
> stats 并重新训练，尚不能借用旧视频宣称模型质量已经验证。

> 本文是数据契约、工程实现、训练和验证说明。若要从方法角度理解模型为什么采用
> qpos28、93D、canonicalization、音乐条件扩散和 Torch FK，请先读
> [BUMI-GENMO 方法与框架讲解](BUMI_GENMO_METHOD_CN.md)。

## 2026-08-19 v1 本地与服务器 2 历史验收结论

结论：**正式数据、代码、CUDA 环境和训练配置均已具备训练条件，而且服务器 2
上的正式随机初始化训练已经启动并稳定运行。当前不要再启动第二个同配置作业。**

本次对 `/home/weili/GENMO` 和服务器 2 的
`/home/user/liwei/GENMO` 做了逐项只读核对：

| 检查项 | 结果 |
|---|---|
| Git 分支 | 本地和服务器 2 均为 `feature/bumi-music-only` |
| Git commit | 两端均为 `3bcc030d2e9911c604035e478515e92dbaab8ea5` |
| 工作区 | 核查时两端均干净；本次只在本地修改本说明文档，训练代码未变 |
| 自动化测试 | 指定真实 kinematics 后 BUMI `42 passed`；全仓 `504 passed, 1 skipped, 3 subtests passed` |
| 服务器 Python | `/data0/user/liwei/envs/GENMO-cu128/bin/python`，PyTorch `2.7.1+cu128` |
| GPU 架构 | 8 × RTX 6000D；PyTorch wheel 已包含 `sm_120` kernel |
| 正式动作数 | 6,610，严格 split 为 `5537 / 547 / 526` |
| EDGE35 | 2,548 个唯一文件；动作与特征逐条等长 |
| 资产绑定 | MJCF `482138…`，kinematics `226306…`，均与数据 metadata 一致 |
| 93D stats | 只由 5,537 条 train 计算，非 placeholder，dataset fingerprint 已通过启动校验 |
| 正式作业 | 8 卡 DDP 正在 `tmux: bumi-train` 中运行，输出为 `version_0` |

2026-08-19 11:51 CST 的服务器快照已完成 Epoch 1896，约 96.7k/500k step；
每 epoch 51 step，速度约 2.25～2.34 step/s，`loss_epoch` 近期约 0.24～0.27，
日志未出现 NaN/Inf，已正常生成 `s010000.ckpt` 至 `s090000.ckpt`。
进程列表中的大量同名 Python 进程是 8 个 DDP rank 及其 DataLoader worker，
不是重复启动了多份训练。

训练总 loss 的代表点为 Epoch 0 `3.02`、100 `0.403`、500 `0.339`、
1000 `0.309`、1500 `0.262`、1897 `0.256`；优化过程在稳定下降。
这只能证明训练数值健康，是否收敛仍需结合四数据集 validation 指标和固定音乐生成结果，
不能仅凭训练 loss 判定。

这里的“可以训练”只表示数据契约、可微 FK、扩散训练和运动学损失链路可用；
它不等于模型已收敛，也不证明 GMT 动力学可跟踪、实物平衡或扭矩安全。

## 为 BUMI-native 训练新增和修改了什么

### 新增的独立 BUMI 路径

这次实现没有把原 SMPL 151D 路径改造成机器人专用代码，而是在公共扩散主干旁新增
一条显式 `motion_backend=bumi` 的完整路径：

| 层次 | 主要文件 | 职责 |
|---|---|---|
| 正式实验 | `configs/exp/gem_bumi_music_only_4set_random_v1.yaml` | 随机初始化、四数据集、500k step、8 卡 DDP |
| 数据配置 | `configs/data/music_robot/trainX_testY.yaml` 和 `configs/{train,test}_datasets/*bumi*` | 四数据集 root、split 和严格 reader 参数 |
| Dataset | `gem/datasets/music_dance/music_dance_bumi.py` | 读取 qpos28/EDGE35、同步 crop/pad、校验 manifest/SHA/资产契约 |
| DataModule | `gem/datamodule/music_robot_trainX_testY.py` | 组合四个 reader、校验 stats fingerprint、安装自定义 sampler |
| Sampler | `gem/datasets/music_dance/bumi_sampler.py` | 去重音乐分层采样、确定性 DDP 分片和 `set_epoch` |
| 93D codec | `gem/robots/bumi/feature_codec.py` | qpos28、first-frame canonicalization、rot6d 和 93D 编解码 |
| Torch FK | `gem/robots/bumi/kinematics.py` | 由版本化真实 MJCF 导出参数执行可微前向运动学 |
| EnDecoder | `gem/robots/bumi/endecoder.py` | 归一化、反归一化、权威 qpos 组合和 FK 输出 |
| GEM 模型 | `gem/bumi_gem.py` | 绕开 SMPL/camera/mesh，建立纯 music→BUMI 扩散训练/验证/预测 |
| Pipeline/loss | `gem/pipeline/bumi_music_pipeline.py`、`gem/robots/bumi/losses.py` | 93D diffusion 目标、物理辅助损失及 raw/normalized/weighted 日志 |
| Checkpoint adapter | `gem/utils/bumi_checkpoint_adapter.py` | 可选迁移 SMPL music Transformer；正式 v1 不使用 |
| 数据工具 | `tools/data/bumi/*.py` | 质量筛选、转换、传输清单、严格校验和 train-only stats |
| 资产工具 | `tools/robots/*.py` | MJCF→Torch FK 导出及 MuJoCo parity 检查 |
| 推理/评估 | `scripts/demo/demo_music_bumi.py`、`tools/eval/*bumi*` | music→qpos28、运动学指标和离线渲染 |
| 测试 | `tests/bumi/` | 数据契约、93D/FK、loss、sampler、adapter 和转换失败路径 |

### 对公共 GENMO 框架的最小改造

公共代码只增加后端扩展点，没有改变旧 SMPL 实验的默认语义：

- `gem/network/gem_diffusion.py` 和 `gem/network/gem_cfg_sampler.py` 不再把动作维度
  写死为 151，可由网络配置显式使用 93D；CFG 仍复用原 diffusion 逻辑。
- `gem/gem.py`、`gem/pipeline/gem_pipeline.py` 和 `configs/model/gem.yaml`
  增加显式 motion backend/动态动作维度接口；默认仍是 SMPL。
- `gem/utils/smpl_augment.py` 把通用旋转辅助函数扩展为可被 BUMI codec 复用，
  原 SMPL 调用保持兼容。
- 进度条修正 DDP sampler 下的 epoch 总步数显示；不会再把每卡/全局样本数混淆。
- 正式 BUMI 网络使用原 16 层 RoPE diffusion Transformer 和 EDGE35 music CFG，
  只替换动作输入/输出层及机器人数据、解码和损失路径。

从 BUMI 功能开始的提交到当前 HEAD 共新增/修改 77 个文件，约新增 11.5k 行；
原 `gem_smpl_music_only_4set` 配置、151D 表示和 SMPL 推理路径仍然保留。

## 数据集做了哪些修改

原始人体数据不被覆盖。数据处理采用“原人体 ID → 离线 GMR BUMI pickle → 自动质量筛选
→ 严格正式格式”的可追溯链路：

1. 对 7,286 条 legacy BUMI 动作自动计算 root/joint 速度、加速度、jerk、
   root 高度/倾斜及地面风格等指标。
2. 自动判定得到 PASS 6,610、REJECT 287、REVIEW 389；正式数据只接收 PASS，
   `include_review=false`，不需要人工挑选。
3. 筛选目录以硬链接保留 PASS，原始动作文件不改写；每条记录保存源 SHA、质量配置 SHA、
   MJCF SHA、判定和原因。
4. 正式转换把 legacy root quaternion 从 `xyzw` 变为 `wxyz`，归一化并修复
   `q/-q` 时间符号；21 个关节按完整名称映射到 `482138…` MJCF 的 qpos 顺序。
5. 保持原 root XYZ 和原人体 sample basename，不重新贴鞋底地面；因此 v1 明确关闭
   contact、sliding 和 sole penetration 训练损失。
6. 继承原人体 train/val/test split，把动作与正式 EDGE35 逐帧配对；WAV 用于审计、
   demo 和重算特征，不进入训练 batch。
7. 转换先写 staging，所有文件、重名、帧数和 SHA 均通过后再原子发布；manifest
   记录 motion/music/audio 路径及三者 SHA。

正式数据结果为：

| 数据集 | 原始条数 | PASS | REJECT | REVIEW | train / val / test | 唯一 EDGE35 | WAV |
|---|---:|---:|---:|---:|---:|---:|---:|
| AIST++ | 1,020 | 824 | 121 | 75 | 790 / 15 / 19 | 824 | 60 |
| AIOZ-GDANCE | 6,011 | 5,608 | 111 | 292 | 4614 / 513 / 481 | 1,578 | 1,578 |
| FineDance | 183 | 111 | 54 | 18 | 99 / 1 / 11 | 111 | 111 |
| CoMPAS3D | 72 | 67 | 1 | 4 | 34 / 18 / 15 | 35 | 35 |
| 合计 | 7,286 | 6,610 | 287 | 389 | 5537 / 547 / 526 | 2,548 | 1,784 |

拒绝/复核原因由代码自动记录，主要包括 joint velocity/acceleration/jerk 异常、
root linear/angular velocity 异常、持续或碎片化贴地动作，以及低 root 姿态复核。
完整阈值、优先级和统计字段见 [BUMI3 重定向动作预筛选](bumi_motion_quality_filter.md)。

服务器 2 的正式目录为：

```text
/data0/user/liwei/datasets/
├── bumi_motions_quality_v1/   # 3.6 GiB，自动筛选结果和 6,610 条 PASS
├── bumi_assets_482138_v1/     # 绑定的 MJCF、mesh、IK 和 kinematics
└── bumi_music_genmo_v1/       # 15 GiB，四套正式 motion/EDGE35/WAV/manifest/stats
```

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
| `root_delta_xy_heading` | `[0:2]` | 2 |
| `root_height_offset` | `[2:3]` | 1 |
| `root_rot_local`（rot6d） | `[3:9]` | 6 |
| `joint_dof` | `[9:30]` | 21 |
| `body_link_pos_root` | `[30:93]` | 21 × 3 |

总维度为 93，合约版本为 `genmo.bumi_motion_features.v2`。后 63 维是逐帧 root
坐标系下的辅助几何监督：它们是 21 个驱动关节对应 feature child-body 的位置，不是 GMR
直接输出，也不是 MJCF 的全部 body。推理时只使用前 30 维组成 qpos28，再通过 Torch FK
重新计算权威 body positions；不会对预测的 63D 位置执行 IK。

## First-frame canonicalization

每个 120 帧训练 crop 独立建立 anchor，不使用 OMG 的 10 帧历史。设 crop 第一帧 root position 为 `p0`，root quaternion 为 `q0`；从 `q0` 提取绕世界 Z 轴的 yaw，并记对应 heading rotation 为 `H0`。`z_default` 来自真实 kinematics 资产中的 `default_qpos[2]`：

```text
p_anchor = [p0.x, p0.y, z_default]
root_rot_local[t] = H0^-1 q[t]
root_height_offset[t] = p[t].z - z_default
root_delta_xy_heading[t] = Heading(q[t])^-1 (p[t+1] - p[t]).xy
body_link_pos_root[t,j] = q[t]^-1 (body_pos_w[t,j] - p[t])
```

最后一帧没有下一帧可求差分，重复最后一个真实水平增量；单帧序列使用零增量。解码时令
canonical `p[0].xy=[0,0]`，按当前帧 heading 把 `root_delta_xy_heading[t]` 转回轨迹坐标并
递推 `p[t+1].xy`；Z 不积分，始终直接取 `root_height_offset[t]`。因此第一帧水平位置和
yaw 为零，root 的 `z-z_default`、roll 和 pitch 仍保留；水平运动没有窗口绝对位置歧义，
竖直速度偏置也不会累积成持续下沉。放回世界时再使用期望的 root XY/yaw 和默认根高，
不要求 GMT 知道地图意义上的全局 XY。

## 长序列重叠融合

固定 120 帧 ONNX 的长序列运行时使用 `genmo.bumi_sliding_motion_overlap_add.v3`。相邻窗口
仍为 30 帧重叠、90 帧步长，每个窗口按训练时的独立完整 crop 分布生成。候选根旋转先以
重叠首帧对齐到已有轨迹 yaw；heading-local 水平增量对全局平移和 yaw 不变，无需猜测或
对齐世界 XY。随后在完整 30 帧双侧预测上执行：

- 把两侧水平增量转到共同轨迹坐标后线性 overlap-add，再转回融合后 heading；
- `z-z_default` 逐帧线性交叉淡化，不积分 Z；
- 根四元数先处理 `q/-q`，再执行最短弧 SLERP；
- 21 个有界关节角线性交叉淡化；
- 融合完整状态后统一修复四元数时间符号，全局只积分一次水平增量，再按绝对帧重建
  qpos chunks。

不能直接平均 normalized 93D：各通道统计量不同，rot6d 也不能保证单位旋转和最短路径。
运行时先反归一化，只融合物理量；最终 qpos 和其 Torch FK 是权威结果，raw 63D 辅助位置
仍不参与控制。单一水平积分链从结构上保证根位置没有分块重置；旋转和关节仍必须
overlap-add，因为平移积分无法约束这些自由度的窗口边界。

旧的无版本硬覆盖结果不会被批量脚本复用：selection、artifact、sample report 和 GMT demo
都保存 `sliding_qpos_contract_version`。改变滑窗算法时必须生成新 artifact/video，不能只按
checkpoint、ONNX、seed 和 DDIM 参数命中旧缓存。

## 为什么不是完整 XYZ 速度

完整 XYZ 局部速度也能保持 93D，但 Z 的微小预测偏置会在长音乐中逐帧累积，而且模型不再
直接表达首帧蹲伏、跳起等非默认根高。v2 因此只积分水平 2D，把第三维保留为每帧绝对
`z-z_default`。它不需要 94D，也不需要 GMT 提供世界 XY，同时保留机器人浮动基座的物理
高度语义。代价是水平速度仍可能产生长期里程漂移；这是相对运动生成的固有限制，应由
训练损失、运行时速度门和未来的接触/里程约束控制，不能伪装成地图定位。

v2 与旧 s430000 不兼容。代码会检查 stats、checkpoint、ONNX 和 TensorRT 合约，拒绝把
旧 93D 权重静默解释成新语义；必须重算 train statistics、重新训练、重新导出并重做
parity、视频质量和 GMT 安全阈值验证。

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

`BUMI_MUSIC_STATS_PATH` 必须指向 `genmo.bumi_stats.v2` JSON，并显式声明
`genmo.bumi_motion_features.v2`；其余仍包括 93 维 mean/std、固定 slices、真实 joint
names、anchor mode、kinematics SHA 和各数据集 fingerprint。统计只使用 train split；
每条序列按 stride 120 枚举 120 帧窗口并包含最后一个合法窗口。短序列只让真实有效帧
进入统计。

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

`scripts/demo/demo_music_bumi.py` 接收 WAV 或预计算 EDGE35、BUMI checkpoint、CFG scale、DDIM steps、可选帧数及 world root XY/yaw。它默认组合正式的 `gem_bumi_music_only_4set_random_v1` 无 contact-head 架构；只有加载其他架构的 checkpoint 时才显式传 `--exp`，避免把正式 checkpoint 误载入带未训练 contact head 的骨架配置。脚本固定 Python/NumPy/PyTorch/CUDA seed，校验 checkpoint 的 music/93D 架构，保存 checkpoint、kinematics 和 stats SHA，并输出速度、加速度、jerk、关节限位、root 和 beat 等运动学报告。提供配对 target 时还会报告 joint/root/FK GT error；提供 MJCF 时可直接渲染并选择是否混入原音乐。输出 `.pt` 至少包含：

```text
qpos                    [T,28]
qpos_canonical          [T,28]
fps                     30
robot_name              bumi
joint_names             exact 21-name MuJoCo order
quaternion_convention   wxyz
qpos_order              mujoco_native
feature_dim             93
anchor_mode             first_frame_xy_yaw_heading_delta_absolute_height
representation_contract_version  genmo.bumi_motion_features.v2
music_path              source path
normalized_motion_93d   [T,93]
music_features          [T,35]
checkpoint/stats/asset  path + SHA256
```

推理不调用 SMPL、GMR、在线 IK 或 pred_cam。未提供 world anchor 时，`qpos` 与 canonical qpos 相同；提供 anchor 时，`qpos` 是放回世界的轨迹，而 `qpos_canonical` 始终保留。

## Checkpoint Demo、ONNX 导出和完整验证链路

### 边界设计

BUMI 不能复用旧 `music_only_onnx.py` 的 SMPL 图：旧图把动作、camera 和 static head
固定为 `151 + 3 + 6`，而正式 BUMI 是 93D 且两个附加 head 均关闭。新增链路为：

```text
WAV / EDGE35
  → PyTorch checkpoint demo
  → normalized motion93
  → BUMI EnDecoder
  → canonical/world qpos28
  → Torch FK / kinematic metrics / MuJoCo render

checkpoint
  → export_bumi_music_onnx.py
  → fixed [1,120,93] ONNX denoiser step
  → validate_bumi_music_onnx.py
  → same noise + same repository DDIM scheduler
  → compare 93D + qpos28 + FK
  → ONNX BUMI motion artifact
```

ONNX 内包含 EDGE35 embedding、conditional/unconditional 两个 CFG 分支和一次
Transformer x-start 预测；两个 CFG 分支在图内合成 batch=2，只调用一次 Transformer。
图只输出 `pred_motion[1,120,93]`。DDIM 循环、统计量反归一化、rot6d→quaternion、
qpos28 组合和 FK 不重复导出，继续使用仓库权威实现。这样不会在图中展开 20/50 份
Transformer，也不会产生另一套扩散公式。

相关入口：

| 入口 | 用途 |
|---|---|
| `scripts/demo/demo_music_bumi.py` | checkpoint 固定 seed 生成、指标、可选 GT 和带音乐渲染 |
| `tools/export/export_bumi_music_onnx.py` | 导出 batch=1、T=120、图内 CFG 的 93D 单步网络 |
| `tools/eval/validate_bumi_music_onnx.py` | PyTorch/ONNX 单步及完整 DDIM、qpos、FK parity |
| `tools/eval/render_bumi_motion.py` | 渲染 PyTorch 或 ONNX 产生的 qpos28 artifact |

### 1. 验证一个训练 checkpoint

下面以服务器 2 的 `s090000.ckpt` 为例。训练尚未结束时不要与正式作业争抢 GPU；
可以等 checkpoint 保存完成后在另一台机器验证，或明确选择有余量的 GPU。

```bash
cd /home/user/liwei/GENMO
export GENMO_PYTHON=/data0/user/liwei/envs/GENMO-cu128/bin/python
export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_v1
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export BUMI_MUSIC_STATS_PATH=$BUMI_BASE/meta/bumi_93d_stats_train_v1.json
export BUMI_MJCF=/data0/user/liwei/datasets/bumi_assets_482138_v1/mjcf/bumi3.xml
export CKPT=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1/version_0/checkpoints/s090000.ckpt

# 替换成要验证的真实音乐；公平比较不同 checkpoint 时保持音乐、区间和 seed 不变。
export AUDIO=/absolute/path/to/music.wav
mkdir -p outputs/bumi_checkpoint_validation/s090000

CUDA_VISIBLE_DEVICES=0 $GENMO_PYTHON scripts/demo/demo_music_bumi.py \
  --wav "$AUDIO" \
  --checkpoint "$CKPT" \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --stats "$BUMI_MUSIC_STATS_PATH" \
  --start-sec 0 --duration-sec 4 --num-frames 120 \
  --seed 42 --cfg-scale 2.5 --ddim-steps 50 \
  --world-root-x 0 --world-root-y 0 --world-root-yaw 0 \
  --output outputs/bumi_checkpoint_validation/s090000/motion.pt \
  --report outputs/bumi_checkpoint_validation/s090000/report.json \
  --render-mjcf "$BUMI_MJCF" \
  --video outputs/bumi_checkpoint_validation/s090000/motion_with_audio.mp4 \
  --mux-audio
```

如果音乐、动作是同一正式样本，再添加以下参数可计算配对 GT 指标。目标 qpos 会先用
同一个 BUMI codec canonicalize，避免把 legacy 世界坐标直接与生成 canonical 坐标比较：

```bash
  --target-motion "$BUMI_BASE/AIST++/motions/<sample_id>.pt" \
  --target-start-frame 0
```

### 2. 导出 BUMI ONNX

导出环境必须安装 `onnx`；下面的检查不会安装或修改环境：

2026-08-19 实查服务器 2 的 `GENMO-cu128` 环境尚未安装 `onnx`、`onnxruntime`
或 `mujoco`。因此该环境当前可以运行不渲染的 PyTorch checkpoint demo，但执行导出、
ONNX parity 或 MuJoCo render 前必须显式补齐对应依赖；脚本不会静默降级或假装成功。

```bash
$GENMO_PYTHON -c "import onnx; print(onnx.__version__)"

export ONNX_DIR=outputs/onnx/bumi_s090000_t120
mkdir -p "$ONNX_DIR"
CUDA_VISIBLE_DEVICES=0 $GENMO_PYTHON tools/export/export_bumi_music_onnx.py \
  --ckpt "$CKPT" \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --stats "$BUMI_MUSIC_STATS_PATH" \
  --seq-len 120 --opset 18 --device cuda \
  --output "$ONNX_DIR/bumi_music_denoiser_t120.onnx" \
  --overwrite
```

导出同时生成 `.onnx.json`，记录 checkpoint/ONNX 外部权重/kinematics/stats SHA、
固定输入输出 shape、opset、checkpoint step/epoch 和 PyTorch reference 统计。

### 3. 单步和完整 DDIM parity

CUDA 验证需要带 `CUDAExecutionProvider` 的 ONNX Runtime；没有时可用 `--provider cpu`
验证正确性，但 50-step 大模型会明显更慢：

```bash
$GENMO_PYTHON -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"

CUDA_VISIBLE_DEVICES=0 $GENMO_PYTHON tools/eval/validate_bumi_music_onnx.py \
  --audio "$AUDIO" --audio-start-sec 0 --audio-duration-sec 4 \
  --ckpt "$CKPT" \
  --onnx "$ONNX_DIR/bumi_music_denoiser_t120.onnx" \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --stats "$BUMI_MUSIC_STATS_PATH" \
  --seq-len 120 --seed 42 --cfg-scale 2.5 \
  --provider cuda --device cuda \
  --full-ddim-steps 50 \
  --output-dir "$ONNX_DIR/validation"
```

校验器首先核对 checkpoint/asset/stats SHA，随后比较同一 noisy motion 和 timestep 的
单步输出；完整模式让两个后端使用相同初始噪声和同一 DDIM scheduler，并依次检查
normalized 93D、canonical qpos28 和 FK positions。任一输出非 finite、shape 不一致或
超出容差都会非零退出。成功时输出：

```text
validation_report.json
onnx_pred_motion_step_93d.pt
onnx_bumi_motion.pt
```

最后一个 artifact 已将 canonical 轨迹放置到世界原点和默认 root 高度，可直接渲染：

```bash
$GENMO_PYTHON tools/eval/render_bumi_motion.py \
  --motion "$ONNX_DIR/validation/onnx_bumi_motion.pt" \
  --mjcf "$BUMI_MJCF" \
  --output "$ONNX_DIR/validation/onnx_bumi_motion.mp4"
```

这条 ONNX parity 链路验证神经网络、DDIM、93D 解码和 FK 数值一致性；MuJoCo 渲染仍是
逐帧 `mj_forward`，不代表 GMT 控制器动力学跟踪已经验证。

## smooth_q1 + auto025 下一次训练数据版本（2026-08-20）

手工 q1 的 3,163 条真实 50 Hz SONIC NPZ 使用独立的
`quality_filter_sonic_npz_50hz_auto025_v1.yaml` 复筛。关节最大允许越界由 strict
配置的 0.0001 rad 放宽为 0.25 rad，其余动力学、根姿态和贴地规则保持不变；结果为
PASS 2,986、REVIEW 59、REJECT 118。AIST++ 有一对基础版/`_armfix` 的 qpos 完全相同，
正式发布显式选择 `_armfix` 并在 conversion report 中记录替代关系，因此最终是 2,985
条唯一动作，而不是把同一人体/音乐样本重复计权。

服务器2的新版本均为新目录，不覆盖旧 `bumi_music_genmo_v1`：

```text
/data0/user/liwei/datasets/bumi_motions_smooth_q1_50hz_auto025_v1   # 2,986 条 PASS 源 NPZ、质量报告和源资产快照
/data0/user/liwei/datasets/bumi_music_audio_smooth_q1_auto025_v1   # 新旧音频配对合集；旧 WAV 用硬链接，新补 38 个 WAV
/data0/user/liwei/datasets/bumi_music_genmo_smooth_q1_auto025_v1   # 30 Hz qpos28/EDGE35/WAV/manifest/stats 正式训练版本
```

50→30 Hz 使用源导出的右端点不包含时间网格，位置/关节线性插值、连续 wxyz 根四元数
最短弧 SLERP，末帧只保持而不外推。源 publish-order 21 关节按完整名称重排到 GENMO
`482138…` MuJoCo 顺序。每条输出再用 GENMO kinematics 做 FK，并施加一个常量 root-Z
偏移，使 `legacy_body_origin_min_zero` 地面规范成立；metadata 明确记录
`root_z_adjusted=true`。motion `.pt` 的权威轨迹仍是 qpos28，93D 由固定 codec 在线编码，
不是另一份互相独立的轨迹。

| 数据集 | 唯一动作 | train / val / test | 总小时 | train 去重音乐组 |
|---|---:|---:|---:|---:|
| AIST++ | 853 | 818 / 17 / 18 | 2.8881 | 50 |
| AIOZ-GDance | 1,947 | 1,667 / 136 / 144 | 20.6022 | 555 |
| FineDance | 118 | 105 / 0 / 13 | 4.2951 | 98 |
| CoMPAS3D | 67 | 33 / 18 / 16 | 2.8066 | 2 |
| 合计 | 2,985 | 2,623 / 171 / 191 | 30.5919 | — |

四库严格 payload、qpos/EDGE35/manifest 等长及 source motion/EDGE35/WAV SHA 扫描均
通过；FineDance 因 val 为空而严格扫描 train/test，下一次实验也用 test 做 FineDance
周期评估，不改写原 split。train-only stats 共 24,274 个窗口、2,912,765 个统计帧，
SHA256 为 `3c9aa73b172cbe0e687f24686c1b912289166a150d6ad11eefccd2a69c279ea2`；
DataModule 已验证四库 fingerprint，并实取到 finite 的 qpos `[128,120,28]` 与
EDGE35 `[128,120,35]` batch。最终训练配置快照和上述 conversion/stats/validation SHA
汇总在 `meta/data_version_release.json`，该 release manifest 的 SHA256 为
`283e27150632f6fa7f27db1b7e36a26f12c6ba4d4822372e61725d890a975b23`。

下一次训练入口为 `gem_bumi_music_only_4set_smooth_q1_auto025_v1`。顶层采样不再随
manifest 时长隐式漂移，而固定为 AIST++ 30%、AIOZ 50%、FineDance 17%、CoMPAS3D
3%。CoMPAS3D train 只有两个音乐组，因此低于旧版 5%，避免小库被过度重复；AIOZ
具有 555 个组并承担 50%，其余两库保留跨舞种覆盖。启动前设置：

```bash
export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_smooth_q1_auto025_v1
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export AISTPP_BUMI_ROOT=$BUMI_BASE/AIST++
export AIOZ_GDANCE_BUMI_ROOT=$BUMI_BASE/AIOZ-GDANCE
export FINEDANCE_BUMI_ROOT=$BUMI_BASE/FineDance
export COMPAS3D_BUMI_ROOT=$BUMI_BASE/CoMPAS3D
export BUMI_MUSIC_STATS_PATH=$BUMI_BASE/meta/bumi_93d_stats_train_auto025_v1.json

$GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_4set_smooth_q1_auto025_v1 \
  output_dir=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_smooth_q1_auto025_v1
```

这里没有自动停止或覆盖当前服务器2训练，也没有自动启动下一次正式作业；新入口用于
当前作业结束后的独立随机初始化训练。

## 人工 q1 GMR temporal-bounded v3 五库训练

该版本把人工评分为 1 的 3,162 条动作作为收录集合，再逐条要求 GMR 嵌入式
`safety_overall=true`。硬安全门覆盖 finite、XML/URDF 关节限位一致、关节位置/速度/
加速度约束、上半身最大帧差、Root Z 足底穿透及 Root Z 加速度；`fidelity_overall`
只衡量轨迹优化后末端相对 raw IK 的偏移，作为诊断字段保留，不删除用户已选的动作。
AIST++ 评审 NPZ 做过明确的坐标转换，因此其旧源索引哈希不同是预期行为；完整性必须核对
评审包自身的 `motion_sha256.txt`，不能错误要求 `source_index_hash_match=true`。

适配入口为 `tools/data/bumi/prepare_gmr_manual_q1_selected_root.py`。它同时核对人工选择
索引、评审 NPZ SHA、GMR release audit、输出 PKL SHA、每条 legacy payload 与嵌入式质量
报告，并原子生成下游 converter 使用的 selected root。正式本地结果为 3,162 条、
3,589,146 帧、33.232833 小时；四库数量为 AIST++ 963、AIOZ-GDance 1,978、
FineDance 149、CoMPAS3D 72，train/val/test 为 2,792/174/196。全部安全门通过；
2,614 条 fidelity 通过、548 条仅 fidelity 诊断未通过。

Root Z 不再套用旧 `legacy_body_origin_min_zero` 二次平移。四个 GMR 库使用
`gmr_foot_sole_ground_zero_v1`，保留真实足底网格、有界 QP、最大 2.1 mm 穿透门和原始
加速度审计；自建库仍是历史 body-origin 地面。联合训练入口把这种组合声明为
`mixed_floor_zero_no_contact_v1`，并硬保持 contact BCE、foot slide、penetration 权重为
零，避免从两种地面定义伪造统一接触标签。根高度、表示、FK、关节时序和限位监督继续有效。

正式实验入口是 `gem_bumi_music_only_5set_manual_q1_v3`。train 序列为
2,792+99=2,891，固定采样概率 AIST/AIOZ/FineDance/CoMPAS3D/Mine 为
29%/47%/16%/3%/5%。每卡 batch 192、8 卡全局 batch 1,536；每 epoch 52,224 draw
恰好是 34 个全局 step。训练从随机初始化开始，350k step 共 5.376 亿 draw，210k/315k
各衰减一半，略高于旧 128×8×500k 的 5.12 亿 draw，目标墙钟时间约三天。五库 stats
必须用新四库与自建库的 train manifests 联合重算，旧四库或旧五库 fingerprint 会被拒绝。

350k 训练完成后的低学习率物理微调入口为
`gem_bumi_music_only_5set_manual_q1_v3_finetune_50k`。它必须通过
`pretrain_ckpt=/absolute/path/to/s350000.ckpt` 以 weights-only 方式加载已完成模型，在新
输出目录重新初始化 optimizer、scheduler 和 global step；不要用 `resume_mode`，否则
`max_steps=50000` 会与旧 checkpoint 的 350k global step 冲突。微调保持网络、数据版本、
五库采样比例和混合地面契约不变，AdamW 学习率为 2e-5，在 30k/45k 各减半；每 5k step
验证并保存 checkpoint。根据 350k 曲线，关节速度、加速度、jerk、限位以及 FK/根监督被
适度提高，但 contact/slide/penetration 仍为零，不会从两类地面定义伪造接触监督。

服务器 2 的正式启动方式如下，环境变量路径必须与人工 q1 v3 发布包一致：

```bash
cd /home/user/liwei/GENMO
export GENMO_PYTHON=/data0/user/liwei/envs/GENMO-cu128/bin/python
export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_manual_q1_gmr_v3
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export AISTPP_BUMI_ROOT=$BUMI_BASE/AIST++
export AIOZ_GDANCE_BUMI_ROOT=$BUMI_BASE/AIOZ-GDANCE
export FINEDANCE_BUMI_ROOT=$BUMI_BASE/FineDance
export COMPAS3D_BUMI_ROOT=$BUMI_BASE/CoMPAS3D
export MINE_BUMI_ROOT=/data0/user/liwei/datasets/bumi_music_genmo_mine_v1
export BUMI_MUSIC_STATS_PATH=/data0/user/liwei/datasets/bumi_music_genmo_5set_manual_q1_v3/meta/bumi_93d_stats_train_manual_q1_v3.json
export BUMI_FINETUNE_CKPT=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_5set_manual_q1_v3_repr_v2_b192_s350k/version_0/checkpoints/last.ckpt
export BUMI_FINETUNE_OUTPUT=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_5set_manual_q1_v3_finetune_physical_s350k_s50k
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 服务器 2 的 8 卡 DDP 必须固定使用这四项。尤其不能把
# NCCL_CUMEM_HOST_ENABLE 误写成语义不同的 NCCL_CUMEM_ENABLE；当前驱动/NCCL 组合只有
# 在禁用 cuMem host allocation、IB，并把 socket bootstrap 固定到回环接口后通过了 8 卡
# 完整 forward/backward/optimizer smoke。
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1

$GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_5set_manual_q1_v3_finetune_50k \
  pretrain_ckpt="$BUMI_FINETUNE_CKPT" \
  output_dir="$BUMI_FINETUNE_OUTPUT" \
  pl_trainer.devices=8 \
  pl_trainer.strategy=ddp
```

## 运动学评估与动力学验证

`eval_bumi_music.py` 实现 joint angle MAE、root trajectory/FK error、joint limits/margin、sole penetration/sliding、root height/tilt、joint velocity/acceleration/jerk P95、root linear/angular velocity、contact accuracy、beat alignment 和 batch diversity。它们都是运动学质量指标。

`render_bumi_motion.py` 对每帧设置 qpos 后只调用 `mujoco.mj_forward` 渲染。FK parity 同样只比较几何。它们不证明 GMT 可跟踪、机器人不会跌倒或扭矩可行。下一阶段必须执行“生成轨迹 → GMT → MuJoCo `mj_step` dynamics tracking”才能评估这些声明。

## 服务器 2 路径、验收与训练命令

服务器 2 的固定资产路径如下；MJCF SHA 必须是 `482138…`，不能替换为 OMG/GMR-CPP 的其他版本：

```bash
cd /home/user/liwei/GENMO

# RTX 6000D (sm_120) 必须使用包含 sm_120 kernel 的 CUDA 12.8 环境；仓库原
# `.venv` 是 torch 2.6.0+cu124，不能用于这台机器。该解释器已在服务器 2
# 实测完成 CUDA tensor、单 batch forward/backward。
export GENMO_PYTHON=/data0/user/liwei/envs/GENMO-cu128/bin/python
$GENMO_PYTHON -c "import torch; assert 'sm_120' in torch.cuda.get_arch_list(); print(torch.__version__, torch.cuda.get_device_name(0))"

export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_v1
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export AISTPP_BUMI_ROOT=$BUMI_BASE/AIST++
export AIOZ_GDANCE_BUMI_ROOT=$BUMI_BASE/AIOZ-GDANCE
export FINEDANCE_BUMI_ROOT=$BUMI_BASE/FineDance
export COMPAS3D_BUMI_ROOT=$BUMI_BASE/CoMPAS3D
export BUMI_MUSIC_STATS_PATH=$BUMI_BASE/meta/bumi_93d_stats_train_v1.json
```

### 当前作业：只监控，不要重复启动

服务器 2 当前已有一个正式 8 卡作业。查看状态：

```bash
ssh -p 50031 user@112.65.216.193
tmux ls
tmux attach -t bumi-train

# 在 tmux 内按 Ctrl-b，再按 d，只分离终端，不会停止训练。
# 不进入 tmux 也可以只读日志：
tail -f /data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1/formal_train.log
```

核对主作业、GPU 和 checkpoint：

```bash
pgrep -af 'scripts/train.py.*gem_bumi_music_only_4set_random_v1' | head
nvidia-smi
ls -lh /data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1/version_0/checkpoints
```

### 中断后的完整状态续训

只有当前训练进程确实退出后才执行。必须保持相同的 `output_dir`，并使用
`resume_mode=last`；它会恢复模型、optimizer、scheduler、epoch、global step 和 sampler
的 epoch 语义。**不要用 `ckpt_path` 代替 `resume_mode`**：`ckpt_path` 只做
weights-only 初始化，会重置训练状态。

```bash
tmux new -s bumi-train-resume

cd /home/user/liwei/GENMO
export GENMO_PYTHON=/data0/user/liwei/envs/GENMO-cu128/bin/python
export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_v1
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export AISTPP_BUMI_ROOT=$BUMI_BASE/AIST++
export AIOZ_GDANCE_BUMI_ROOT=$BUMI_BASE/AIOZ-GDANCE
export FINEDANCE_BUMI_ROOT=$BUMI_BASE/FineDance
export COMPAS3D_BUMI_ROOT=$BUMI_BASE/CoMPAS3D
export BUMI_MUSIC_STATS_PATH=$BUMI_BASE/meta/bumi_93d_stats_train_v1.json
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1

RUN_ROOT=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1
$GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1 \
  output_dir="$RUN_ROOT" \
  resume_mode=last \
  2>&1 | tee -a "$RUN_ROOT/formal_train.log"
```

要固定从某个 checkpoint 完整续训，可把上一条命令的 `resume_mode=last` 换成：

```bash
resume_mode=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1/version_0/checkpoints/s090000.ckpt
```

### 从随机初始化重新训练

同样只能在 8 张卡空闲时执行。使用新的 output root，避免覆盖或混淆当前
`version_0`；正式配置已经固定 `pretrain_ckpt=null` 和 `checkpoint_adapter=null`。

```bash
tmux new -s bumi-train-fresh

cd /home/user/liwei/GENMO
export GENMO_PYTHON=/data0/user/liwei/envs/GENMO-cu128/bin/python
export BUMI_BASE=/data0/user/liwei/datasets/bumi_music_genmo_v1
export BUMI_KINEMATICS_PATH=/data0/user/liwei/datasets/bumi_assets_482138_v1/kinematics/bumi_kinematics_482138_v1.json
export AISTPP_BUMI_ROOT=$BUMI_BASE/AIST++
export AIOZ_GDANCE_BUMI_ROOT=$BUMI_BASE/AIOZ-GDANCE
export FINEDANCE_BUMI_ROOT=$BUMI_BASE/FineDance
export COMPAS3D_BUMI_ROOT=$BUMI_BASE/CoMPAS3D
export BUMI_MUSIC_STATS_PATH=$BUMI_BASE/meta/bumi_93d_stats_train_v1.json
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1

RUN_ROOT=/data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1_rerun
mkdir -p "$RUN_ROOT"
$GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1 \
  output_dir="$RUN_ROOT" \
  use_wandb=false \
  2>&1 | tee "$RUN_ROOT/formal_train.log"
```

上述命令使用配置内的 8 GPU、每卡 batch 128、global batch 1024、500k step、
AdamW `2e-4`、300k/450k 学习率减半和每 10k step 保存。`samples_per_epoch=52224`
是全局样本数，因此每个 rank 每 epoch 6,528 条，`6528 / 128 = 51 step/epoch`。

### 数据重新构建命令（当前服务器不需要重跑）

以下命令仅用于数据或资产发生版本变化时重建。当前服务器 2 已完成构建、严格扫描和
stats 计算；训练期间不要重复执行全量 SHA 扫描或重建，以免争用 I/O。

先生成固定传输清单，再在四套 WAV 到齐后执行一次全有或全无的转换：

```bash
$GENMO_PYTHON tools/data/bumi/build_bumi_transfer_filelists.py \
  --selected-root /data0/user/liwei/datasets/bumi_motions_quality_v1 \
  --human-root aistpp=/data0/user/liwei/datasets/music_dance_genmo/AIST++ \
  --human-root aioz_gdance=/data0/user/liwei/datasets/music_dance_genmo/AIOZ-GDANCE \
  --human-root finedance=/data0/user/liwei/datasets/music_dance_genmo/FineDance \
  --human-root compas3d=/data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D \
  --output /data0/user/liwei/datasets/bumi_transfer_plan_v1

$GENMO_PYTHON tools/data/bumi/build_bumi_music_dataset.py \
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
  $GENMO_PYTHON tools/data/bumi/validate_bumi_music_dataset.py \
    --root "$BUMI_BASE/$1" --dataset-name "$2" \
    --kinematics "$BUMI_KINEMATICS_PATH" --splits train val test
done

$GENMO_PYTHON tools/data/bumi/compute_bumi_93d_stats.py \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --dataset "aistpp_bumi=$AISTPP_BUMI_ROOT" \
  --dataset "aioz_gdance_bumi=$AIOZ_GDANCE_BUMI_ROOT" \
  --dataset "finedance_bumi=$FINEDANCE_BUMI_ROOT" \
  --dataset "compas3d_bumi=$COMPAS3D_BUMI_ROOT" \
  --output "$BUMI_MUSIC_STATS_PATH"
```

如果将来重建了数据，必须重新跑单 batch 和 100-step smoke，再启动正式训练。
随机初始化实验不得设置 `pretrain_ckpt`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_BLOCKING_WAIT=1 $GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1 \
  pl_trainer.max_steps=1 use_wandb=false

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_BLOCKING_WAIT=1 $GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_4set_random_v1 \
  pl_trainer.max_steps=100 use_wandb=false
```

这些检查只验证数据、运动学、生成训练和运动学指标，不声明 GMT 动力学可跟踪、平衡或扭矩可行。
