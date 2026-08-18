# GMR 生产版 BUMI3 动作预筛选与 GENMO 接入方案（修订版）

## 1. 结论

当前 `data/motions` 应按用户确认的 **GMR 生产版 BUMI3** 解释。文档和代码中的
`legacy` 只表示参考脚本保存的 pickle 中间格式，不表示机器人是旧版。

参考脚本 [smplx_to_robot_dataset.py](</home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/scripts/smplx_to_robot_dataset.py>)
在 `--robot bumi3` 时经过 `ROBOT_XML_DICT` 解析到：

```text
/home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml
```

当前文件的 SHA256 是：

```text
482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c
```

这就是本方案的唯一 source/target 机器人资产。正式 BUMI kinematics、质量检查、
视频渲染和 GENMO 转换都必须绑定这个 SHA，不能混入名称相近但 body tree、qpos
顺序或关节限位不同的 MJCF。

特别说明：

```text
/home/weili/robot_retarget/asset/robot/noetix_bumi_v1_3/mjcf/bumi3.xml
```

是另一份不同契约的资产，SHA 为 `406c1bdf...`，具有 31 个非 world body、
腰/手臂优先的 qpos 顺序和不同 arm-roll 限位。它不用于解释当前 pickle，也不应传给
本预筛选工具。上一版中把 GMR 生产资产称为“旧版”的表述已经纠正。

## 2. 资产与生成链证据

### 2.1 当前参考链的固定指纹

| 项目 | 当前值 |
|---|---|
| Robot key | `bumi3` |
| GMR MJCF | `GMR-master/assets/bumi3/bumi3.xml` |
| MJCF SHA256 | `482138b437db...` |
| 数据生成脚本 SHA256 | `496cfcdb842d...` |
| GMR `params.py` SHA256 | `acedae4f58f3...` |
| SMPL-X→BUMI IK config | `smplx_to_bumi3_auto.json` |
| IK config SHA256 | `20c5305ef3f5...` |
| FPS | 30 |
| qpos 维度 | 28 = root 7 + joint 21 |
| MuJoCo root quaternion | `wxyz` |
| pickle `root_rot` | `xyzw` |
| pickle `dof_pos` | 生产 MJCF qpos address 7..27 顺序 |
| pickle `local_body_pos` | 25 个生产 body origin，root=0、identity 下的 FK |

### 2.2 21 关节顺序

生产 MJCF 和 pickle `dof_pos` 的顺序为：

```text
l_leg_pitch_joint
l_leg_roll_joint
l_leg_yaw_joint
l_knee_pitch_joint
l_ankle_pitch_joint
l_ankle_roll_joint
r_leg_pitch_joint
r_leg_roll_joint
r_leg_yaw_joint
r_knee_pitch_joint
r_ankle_pitch_joint
r_ankle_roll_joint
waist_yaw_joint
l_arm_pitch_joint
l_arm_roll_joint
l_arm_yaw_joint
l_elbow_pitch_joint
r_arm_pitch_joint
r_arm_roll_joint
r_arm_yaw_joint
r_elbow_pitch_joint
```

不能因为另一份 BUMI 资产采用腰/手臂优先顺序，就直接按数组位置解释当前
`dof_pos`。未来即使目标 kinematics 顺序改变，也必须按完整关节名映射。

### 2.3 25 body 顺序

当前所有 pickle 的 `link_body_list` 均等于生产 MJCF 深度优先 body 顺序，包括：

- `base_link`；
- 左右腿各 6 个驱动 link；
- `waist_yaw_link` 和 `torso_link_virtual`；
- 左右手臂各 4 个驱动 link和一个 `*_hand_link_virtual`。

这也是当前数据来自该 GMR 生产 asset tree 的直接证据。另一份 31-body MJCF 会输出
`hips_sphere`、`neck_sphere`、`left_hand`、toe/foot-end 等不同列表，不符合现有文件。

### 2.4 FK 对照结果

四个数据集各均匀抽取 10 条、每条抽取 5 帧，用生产 MJCF 设置：

```text
root_xyz = 0
root_quaternion_wxyz = [1, 0, 0, 0]
joints = pickle.dof_pos
```

MuJoCo `data.xpos` 与 pickle `local_body_pos` 的最大绝对误差为：

| 数据集 | 最大误差 |
|---|---:|
| AIOZ-GDANCE | `3.63e-7 m` |
| AIST++ | `2.69e-7 m` |
| CoMPAS3D | `2.95e-7 m` |
| FineDance | `2.69e-7 m` |

因此当前 source 资产、关节顺序、body 顺序和 local FK 语义可以视为已验证。

## 3. 参考生成脚本的数据语义

参考脚本每条动作的关键步骤如下：

1. 加载 SMPL-X 并重采样到 30 FPS；
2. `GeneralMotionRetargeting(src_human="smplx", tgt_robot="bumi3")`；
3. GMR 使用 `mink.ConfigurationLimit`，关节值受生产 MJCF range 约束；
4. GMR 输出 MuJoCo qpos28，其中 root quaternion 是 wxyz；
5. 脚本把 `qpos[:,3:7]` 显式排列成 xyzw 后保存为 `root_rot`；
6. `dof_pos=qpos[:,7:]`，不做关节重排；
7. 用同一 `retargeter.xml_file` 构造 `KinematicsModel`，生成 `local_body_pos`；
8. `HEIGHT_ADJUST` 把整条序列所有 25 个 **body origin** 的最低 Z 平移到 0；
9. `ROOT_ORIGIN_OFFSET` 把第一帧 Root XY 平移到原点；
10. 保存 pickle，但没有保存 MJCF、脚本、IK config 的 SHA。

这带来三个重要约束：

- pickle 里的地面 0 是“最低 body origin”，不是鞋底 mesh 的真实最低点；
- 不能在现有数据上用足底穿地深度作为硬筛选，除非先重新定义地面和 sole proxy；
- 当前数据资产来源可以通过用户确认、body order 和 FK 对照证明，但未来重定向必须
  在文件中直接写入资产指纹，不能继续依靠目录和记忆。

## 4. 筛选目标和决策模型

预筛选解决两类问题：

1. **数据质量问题**：损坏、错契约、关节越界、跳变、速度/加速度/jerk 异常；
2. **训练风格问题**：不希望 BUMI GENMO 学习的持续躺地、滚地、上身贴地动作。

输出采用三态：

- `PASS`：允许进入正式 GENMO 数据转换；
- `REVIEW`：边界动作，默认不进入训练集，人工确认后才能升级；
- `REJECT`：硬契约失败、严重异常或持续贴地风格。

物化工具默认只选 `PASS`。`REVIEW` 不会因为被复制到人工复核目录而变成
`quality_accepted=true`。

## 5. 质量指标与阈值

### 5.1 P0：资产和结构硬门禁

以下任一失败直接 `REJECT`：

- source MJCF SHA 不等于 `482138b...`；
- `fps != 30`；
- 缺少 `root_pos/root_rot/dof_pos/local_body_pos/link_body_list`；
- shape 不是 `[T,3] / [T,4] / [T,21] / [T,25,3]`；
- 任意 NaN/Inf；
- 25 body 顺序不完全一致；
- 四元数范数最大误差大于 `1e-3`；
- 生产 MJCF 关节限位违反超过 `1e-4 rad`；
- local FK 与生产 MuJoCo FK 抽样最大误差超过 `1e-5 m`；
- 生成脚本定义的全序列 body-origin 地面误差超过 `2 cm`；
- Root Z 超出仅用于识别坐标损坏的 `[-0.02, 2.1] m`。

四元数 q/-q 表示跳变只记录并 canonicalize，不直接拒绝，因为二者表示同一旋转。
距离关节限位小于 `0.01 rad` 记录 warning，不单独改变状态。

### 5.2 P1：动力学统计

所有导数按 30 FPS 转为物理单位。关节信号先在每帧对 21 维取 L2：

| 信号 | 初始阈值 | 单位 |
|---|---:|---|
| 关节速度 L2 | 30 | rad/s |
| 关节加速度 L2 | 1200 | rad/s² |
| 关节 jerk L2 | 60000 | rad/s³ |
| Root 线速度 | 3 | m/s |
| Root SO(3) 角速度 | 12 | rad/s |

参考 OMG v2 的统计策略：

- 最大值超过 `3 × threshold`：`REJECT/*_SEVERE`；
- P95 超阈值且超限帧比例大于 8%：`REVIEW/*_SOFT`；
- 连续超限至少 6 帧：`REVIEW/*_SOFT`。

Root 角速度通过相邻四元数绝对点积计算，不受 q/-q 影响。v1 不使用硬件速度上限，
因为当前生产 MJCF/GMR 没有提供可作为数据清洗真值的完整 per-joint 动态规格；如果
后续拿到部署控制器规格，应新增“可跟踪性评估”，不要混进数据连续性阈值。

### 5.3 P1：躺地和滚地风格

每帧用保存的 local FK 和 Root 世界变换恢复 25 个 body origin：

```text
world_body[t,j] = R_xyzw(root_rot[t]) @ local_body_pos[t,j] + root_pos[t]
```

地面证据：

```text
root_low_tilt = root_z < 0.30m 且 root_tilt > 45°
torso_ground  = torso_link_virtual_z < 0.30m
upper_ground  = 任一非手部上臂/肘部 link_z < 0.08m
```

再要求以下 gate 至少一个成立，减少正常脚部接地误触发：

```text
root_z < 0.34m
或 root_tilt > 40°
或左右 ankle body origin 均高于 0.08m
```

决策：

- 组合条件连续至少 15 帧（0.5 秒）：`REJECT/FLOOR_STYLE_SUSTAINED`；
- 累计至少 15 帧、占比超过 2%，但不连续：`REVIEW/FLOOR_STYLE_FRAGMENTED`；
- 仅 Root `<0.22m` 连续 15 帧：`REVIEW/LOW_ROOT_REVIEW`；
- 仅手部接近地面：只记录，不改变状态。

单独按 Root `<0.28m` 持续 15 帧会命中 741 条，容易误伤深蹲、跪姿和坐姿；组合规则
只命中 91 条，并已抽样覆盖明确躺地、滚地和坐地动作，因此不能退化成 Root 单阈值。

### 5.4 P2：只记录、不作为 v1 硬拒绝

- 鞋底穿地：当前地面由 body origin 生成，不是真实鞋底地面；
- 脚部滑动：没有可靠 contact 标签；
- 自碰撞：生产 MJCF 的碰撞 geom/ignore pair 尚未校准为数据清洗标准；
- COM/支撑多边形：高速舞蹈中的瞬时越界不等于不可用；
- 手撑地：可能是正常编舞；
- 低 Root：可能是深蹲或跪姿。

这些指标可写入报告或用于人工排序，但第一版不能自动删除。

## 6. 当前全量 dry-run 结果

当前共有 7,286 条、8,069,444 帧，约 74.72 小时：

| 状态 | 数量 |
|---|---:|
| PASS | 6,610 |
| REVIEW | 389 |
| REJECT | 287 |

持续贴地规则命中 91 条：

| 数据集 | 持续贴地 |
|---|---:|
| AIOZ-GDANCE | 57 |
| AIST++ | 11 |
| FineDance | 23 |
| CoMPAS3D | 0 |

另有 30 条碎片贴地候选。全量没有 shape、finite、FPS、body-order、生产关节限位或
source-ground 契约错误。287 条 REJECT 还包含 3 倍动力学峰值，应用删除前必须按
reason code 查看视频，不能把总数全部解释为滚地动作。

报告位于：

```text
/home/weili/GENMO/outputs/bumi_quality_v1
```

由于本次 dry-run 使用的正是 GMR 生产 MJCF SHA `482138b...`，在用户确认“新版”指
该 GMR `bumi3` 后，报告仍然有效，不需要因为之前的命名误解重算。

## 7. 代码架构与逐文件规划

### 7.1 当前已经具备

| 文件 | 职责 |
|---|---|
| `configs/bumi/quality_filter_v1.yaml` | 固定生产 MJCF SHA、顺序、限位和阈值 |
| `gem/robots/bumi/legacy_motion.py` | NumPy pickle 兼容、严格字段、xyzw→wxyz、按名重排、世界 body 重建 |
| `gem/robots/bumi/quality_filter.py` | OMG 风格软指标、三态决策、贴地 mask 和安全区间 |
| `tools/data/bumi/filter_legacy_bumi_motions.py` | 并行 dry-run、JSONL/CSV、SHA 报告、PASS-only 安全物化 |
| `tools/eval/render_legacy_bumi_motion.py` | 生产 MJCF 完整长度渲染和质量 sidecar |
| `tests/bumi/test_bumi_quality_filter.py` | 序列化、四元数、贴地、防误杀、动力学和物化测试 |

### 7.2 下一阶段 P0：生产资产契约强化

新增 `gem/robots/bumi/source_contract.py`：

- 直接从传入 MJCF 提取 `nq`、joint qpos address、range、body tree；
- 校验提取结果与 YAML 完全一致；
- 保存 MJCF、生成脚本、`params.py`、IK config 的 SHA；
- 明确 `robot_key=bumi3`、GMR root quaternion=wxyz、pickle quaternion=xyzw；
- 拒绝同名但不同 SHA 的 BUMI3。

修改 `filter_legacy_bumi_motions.py`：

- 增加 `--gmr-params`、`--retarget-script`、`--ik-config`；
- 报告 source contract 的四个 SHA；
- 启动时先完成 MJCF 语义检查，再创建 worker；
- 每条动作确定性抽取首/中/尾等 5 帧做 MuJoCo FK parity；
- parity 超 `1e-5m` 直接 `MOTION_FK_CONTRACT_ERROR`；
- `--limit` 报告必须标记为 partial，禁止用于正式 `--apply`。

### 7.3 下一阶段 P0：修订重定向产物契约

参考 GMR 脚本，但未来不再只保存无元数据 pickle。建议在 GMR 侧修改
`smplx_to_robot_dataset.py`，或在 GENMO 增加受控 wrapper：

```text
tools/data/bumi/retarget_smplx_to_gmr_bumi3.py
```

每条 motion 至少保存：

```json
{
  "contract_version": "genmo.gmr_bumi3_motion.v2",
  "robot_key": "bumi3",
  "fps": 30,
  "qpos": "Tensor[T,28], wxyz",
  "joint_names": ["21 production names"],
  "body_names": ["25 production names"],
  "local_body_pos": "Tensor[T,25,3]",
  "source_smplx_sha256": "...",
  "source_mjcf_sha256": "482138...",
  "retarget_script_sha256": "...",
  "gmr_params_sha256": "...",
  "ik_config_sha256": "...",
  "height_adjust_policy": "global_min_body_origin_to_zero",
  "root_xy_policy": "first_frame_to_zero"
}
```

v2 建议直接保存完整 `qpos[T,28]` 的 wxyz，不再把 root 拆成 xyzw；旧 pickle reader
继续保留，只用于兼容当前数据。写文件采用临时文件加原子 rename，并在输出根目录
另存 `dataset_info.json` 和生成失败清单。

### 7.4 下一阶段 P1：批量视觉复核

新增：

```text
tools/eval/render_bumi_quality_review.py
```

行为：

- 读取 `quality_report.jsonl`；
- 按 status、reason、数据集、指标排序选择样本；
- 默认渲染全部 `FLOOR_STYLE_SUSTAINED`、全部严重动力学峰值、全部 FineDance
  REJECT、每数据集随机 20 条 PASS；
- 完整视频保留 30 FPS，页面支持原始 source ID、指标和 FLOOR 时间段跳转；
- 输出 `index.html`、视频和人工 `decision.csv` 模板；
- 人工结论只允许 `keep/reject/unsure`，并记录 reviewer、notes 和配置 SHA。

新增 `apply_bumi_quality_review.py`，将自动规则和人工决策合并成不可变的 final
selection manifest；不能直接修改原 JSONL。

### 7.5 下一阶段 P1：正式 GENMO 数据转换

新增：

```text
tools/data/bumi/build_bumi_music_dataset.py
```

转换器只接收 final selection 中 `quality_accepted=true` 的样本：

1. 读取旧 pickle，并把 root xyzw 显式转换为 wxyz；
2. 从同一个 SHA `482138b...` 导出的 `genmo.bumi_kinematics.v1` 获取 joint order；
3. 即使当前顺序相同，也按关节名映射并断言集合完全一致；
4. 保存正式 qpos28、joint names、source SHA、quality report/config SHA；
5. 根据各音乐数据集的 sample/song mapping 对齐 EDGE35 音乐特征；
6. 动作与音乐帧数不一致时 fail closed，不做静默截断；
7. 生成 `genmo.bumi_music.v1` manifest 和 dataset_info；
8. 只在正式 train split 上计算 93D mean/std。

用于 GENMO 的 kinematics JSON 必须从同一生产 MJCF 导出：

```bash
python tools/robots/export_bumi_kinematics.py \
  --mjcf /path/to/GMR-master/assets/bumi3/bumi3.xml \
  --proxy-config /path/to/versioned_bumi_proxy_config.json \
  --output /path/to/bumi_kinematics.json
```

sole proxy 必须显式来自生产 mesh/geom，不允许根据 body 名猜测。完成导出后先运行
`validate_bumi_fk_parity.py`，再构建正式数据。

## 8. 报告契约规划

下一版每行 JSONL 至少包括：

```json
{
  "report_contract_version": "genmo.bumi_quality_report.v2",
  "sample_id": "aistpp/...",
  "dataset": "aistpp",
  "source_relative_path": "aistpp/....pkl",
  "source_sha256": "...",
  "robot_key": "bumi3",
  "source_mjcf_sha256": "482138...",
  "retarget_script_sha256": "496cfc...",
  "gmr_params_sha256": "acedae...",
  "ik_config_sha256": "20c530...",
  "provenance": "user_attested_and_fk_verified",
  "status": "PASS|REVIEW|REJECT",
  "quality_accepted": false,
  "reason_codes": [],
  "metrics": {},
  "fk_parity_max_error_m": 0.0,
  "floor_intervals": [[120, 193]],
  "valid_intervals": [[0, 105], [208, 900]]
}
```

汇总必须包含按数据集的数量/时长/status/reason、关键指标分位数、配置快照和 Git
commit。所有输入 SHA、配置 SHA 和 selection SHA 必须进入最终 dataset_info。

## 9. 测试规划

单元测试：

- NumPy 2 pickle 在 NumPy 1 环境加载；
- xyzw↔wxyz 和 q/-q 连续性；
- 21 关节按名重排与错集合拒绝；
- MJCF SHA、joint order、body order、range 漂移拒绝；
- MuJoCo/local FK parity 正常和失败路径；
- 一至三阶导数短序列；
- 正常直立、深蹲、跪姿、手撑地不被直接当成滚地；
- 躯干贴地、低 Root+大倾角、非手部上身贴地持续 15 帧被拒绝；
- 碎片贴地进入 REVIEW；
- 并行 worker 数变化不改变报告排序和结论；
- materialize 目标已存在、源 SHA 变化和路径逃逸时失败。

集成验证：

- 四数据集各 10 条、每条 5 帧 source MuJoCo FK；
- 7,286 条全量 dry-run；
- 全部持续贴地候选视频；
- PASS/REVIEW/REJECT 各数据集分层抽样；
- PASS-only 物化后重新逐文件 SHA 校验；
- 正式 dataset reader 全 payload 扫描；
- 93D encode/decode、Torch/MuJoCo FK parity；
- 单 batch forward/backward 和 100-step smoke。

## 10. 实施顺序与验收门槛

1. 冻结 GMR 生产资产与四个生成链 SHA；
2. 增加 source-contract introspection 和每条动作 FK 抽样；
3. 生成 v2 dry-run 报告；
4. 批量渲染并人工复核全部自动 REJECT 和边界 REVIEW；
5. 冻结 quality config SHA 和 final selection SHA；
6. PASS-only 物化，不改 `data/motions`；
7. 从同一生产 MJCF 导出 GENMO kinematics；
8. 构建四数据集音乐对齐的 `genmo.bumi_music.v1`；
9. 校验、统计 93D、单 batch 和 smoke；
10. 才允许正式训练。

进入正式转换前必须满足：

- source MJCF SHA、生成链 SHA 和 report config SHA 均真实有效；
- 全量契约错误为 0；
- FK parity 最大误差 `<1e-5m`；
- 全部 `FLOOR_STYLE_SUSTAINED` 已人工确认；
- REVIEW 未经人工确认不得进入训练；
- 原始 7,286 个 pickle 数量和内容 SHA 保持不变；
- 最终 dataset_info 明确记录 quality 与 kinematics 指纹。

## 11. 当前可运行命令

Dry-run：

```bash
cd /home/weili/GENMO
source .venv/bin/activate

python tools/data/bumi/filter_legacy_bumi_motions.py \
  --input-root data/motions \
  --source-mjcf /home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml \
  --output-dir outputs/bumi_quality_v1 \
  --workers 8 \
  --overwrite
```

单条完整视频：

```bash
MUJOCO_GL=egl python tools/eval/render_legacy_bumi_motion.py \
  --motion data/motions/finedance/114.pkl \
  --source-mjcf /home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml \
  --output outputs/bumi_quality_v1/videos/finedance_114.mp4
```

人工确认后只物化 PASS：

```bash
python tools/data/bumi/filter_legacy_bumi_motions.py \
  --input-root data/motions \
  --source-mjcf /home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml \
  --output-dir outputs/bumi_quality_v1 \
  --workers 8 \
  --overwrite \
  --apply \
  --materialize-root data/bumi_motions_quality_v1
```

当前阶段建议先完成 v2 provenance/FK 强化和批量视频人工复核，再执行最后一条正式
物化命令。

