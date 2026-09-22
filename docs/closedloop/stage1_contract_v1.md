# BUMI closed-loop Stage 1 条件与监督数据契约 v1

本文定义 `genmo.bumi_closedloop.stage1.v1`。第 2 步固定接口、物理语义、时间轴和因果
边界；第 3 步已经新增独立的音乐—BUMI监督样本构造，但仍不改网络，也不实现
GENMO↔GMT 通信、Stage 2 或 DPPO。

核验基线：

- GENMO：`feature/bumi-music-closedloop`，继承自
  `feature/bumi-music-only` 的 qpos30/contact2 正式实现；
- GMT：`feature/bumi-frozen-gmt-backend@be05067c`；
- 匹配任务：`mimic_noetix_bumi3_mha_sonic`，Gym ID
  `gmt-bumi3-MHA-him-v0`；
- Isaac Lab builtin observation 实现来自当前 `/home/weili/IsaacLab` checkout。

配置的机器可读副本在 `configs/closedloop/stage1_contract_v1.yaml`，Python 类型和
validator 在 `gem/closedloop/contracts.py`，第 3 步 Dataset 在
`gem/closedloop/stage1_dataset.py`。服务器1四库实例配置在
`configs/closedloop/stage1_dataset_server1_fourset_v1.yaml`。

## 1. 两阶段方法和本轮边界

Stage 1 仍是监督式 diffusion training：

```text
音乐 + 机器人最近实际可获得的状态历史 + 已承诺的短期参考
    -> GENMO
    -> qpos30 + 独立 contact2 head
```

Stage 2 的目标链路是：

```text
Stage 1 GENMO
    -> Frozen GMT
    -> BUMI dynamics
    -> reward / next observation
    -> upper Critic + DPPO
```

Stage 2 只更新 GENMO 和上层 Critic，GMT 权重始终冻结。这个说明只固定方法边界；本轮
没有实现 Stage 2、Critic、DPPO、仿真 rollout 或梯度链路。

## 2. 第一版继续输出 qpos30 + contact2

第一版不创建 51D 动作表示，不增加 21 维 joint velocity prediction，不修改
`BUMI_FEATURE_DIM=30`，不重算所谓 51D stats，也不改变旧 checkpoint、ONNX 或 TensorRT
契约。

现行表示版本是 `genmo.bumi_motion_features.qpos30.v3`：

| qpos30 切片 | 字段 | 现行精确语义 |
|---|---|---|
| `[0:2]` | `root_delta_xy_heading` | 世界 XY 的 `p[t+1]-p[t]`，表达在第 `t` 帧水平 heading 坐标系 |
| `[2:3]` | `root_height_offset` | 逐帧绝对根高相对权威 default root height 的偏移 |
| `[3:9]` | `root_rotation_6d`（代码名 `root_rot_local`） | 移除 crop 第一帧 yaw anchor 后的局部根旋转 rot6d |
| `[9:30]` | `joint_dof` | MuJoCo-native BUMI 21 关节角，单位 rad |

代码证据是 `gem/robots/bumi/feature_codec.py`。contact 继续由独立 2D head 输出，顺序为
`[left, right]`；它不属于 30D 动作，也不拼入 `known_qpos30`。证据是
`gem/robots/bumi/contacts.py`、`configs/network/diffusion_lg_bumi30_contact.yaml` 和
`gem/runtime/bumi_music_contract.py`。

### 为什么本版不预测 dq

GMT 最终需要的 joint velocity 已由最终 50 Hz qpos 时间线确定性派生。若 GENMO 同时预测
q 和 dq，会产生两份可能互相矛盾的运动描述，并改变输入投影、输出 head、stats、loss、
checkpoint 和部署图的形状。本版保留已训练、已导出、已有 FK/loss/runtime 保护的
qpos30；速度连续性仍可由现有 q/dq/加速度损失监督，执行侧只从最终重采样后的 qpos 求
一次速度。

### physical qpos30 与训练标准化

本契约中的 `known_qpos30` 是 **stats 标准化前的 physical qpos30**。后续第 4 步接入网络
时，才通过现有 `BumiEndecoder.normalize()` 显式映射到 denoiser 的 normalized x0 域。
这样上层条件协议不绑定某一份 stats，也不修改旧 checkpoint 的 qpos30 标准化契约。

## 3. `genmo.bumi_proprio48.v1`

GENMO 第一版输入 GMT 可观测状态的 48 维共同子集，不输入 `last_action21`：

```text
[0:3]   projected_gravity   3
[3:6]   base_ang_vel        3
[6:27]  joint_pos_rel      21
[27:48] joint_vel_rel      21
                              --
                              48
```

GMT 自己的 policy 单帧仍是这 48 维再拼 `last_action21`，共 69 维。排除 last action 的原因
不是“真机取不到 action”，而是第一阶段音乐动作数据没有真实 GMT action，同时上层首版
希望使用“可从示范动作构造、也可从真机直接读取”的一致子集。

### 3.1 `projected_gravity`

Isaac Lab 的实际执行路径是：

```text
simulation gravity_w = [0, 0, -9.81]
    -> normalize(gravity_w)
    -> inverse(root_link_quat_w)
    -> projected_gravity_b
```

所以它是世界“向下”重力的**单位方向**在 BUMI `base_link`/root actor frame 中的表达：

- 坐标系：`base_link`/root link actor frame；
- 顺序：`[x, y, z]`；
- 方向与符号：直立、单位根姿态时为 `[0, 0, -1]`；
- 幅值：原始物理值模长 1，是无量纲方向，不是 `[0,0,-9.81] m/s²`。

Isaac builtin descriptor 虽保留了 `m/s²` 标签，但 `articulation_data.py` 的实际代码先把
gravity 归一化；本契约以执行代码为准。证据位于：

- `IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py` 的
  `projected_gravity()`；
- `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py` 的
  `GRAVITY_VEC_W` 初始化和 `projected_gravity_b`；
- `legged_lab_gmt/.../assets/robots/bumi3/urdf/bumi.urdf` 的根 `base_link`。

### 3.2 `base_ang_vel`

匹配任务调用 Isaac Lab builtin `base_ang_vel()`，返回 `root_ang_vel_b`。当前实现中它是
`root_com_ang_vel_b` 的别名：世界系 root COM angular velocity 由
`inverse(root_link_quat_w)` 旋转到 `base_link`/root actor frame。

- 坐标系：`base_link`/root link actor frame；
- 顺序：`[ωx, ωy, ωz]`；
- 单位：rad/s；
- observation term 本身没有 scale、clip 或 modifier。

### 3.3 `joint_pos_rel`

精确公式为：

```text
joint_pos_rel = joint_pos - active_default_joint_pos
```

单位是 rad，顺序是后文列出的 GMT/PhysX native DoF order。这里的 `default` 不能简单写成
固定名义姿态：`mimic_noetix_bumi3_mha_sonic` 在 startup 时对每个环境、每个关节独立执行：

```text
active_default_joint_pos
    = Bumi_CFG nominal default_joint_pos + U[-0.02, 0.02] rad
```

事件同时更新 joint-position action offset。因此，仿真 rollout producer 必须使用其当前
环境里的 active default；真机 producer 必须绑定部署控制器实际采用的 default，并将它写入
后续数据 provenance。不能静默地把 active default 与名义表混用。

名义非零值为左右 leg pitch `-0.1495`、左右 knee pitch `+0.3215`、左右 ankle pitch
`-0.1720`、左 arm roll `+0.3`、右 arm roll `-0.3` rad，其余为 0。

### 3.4 `joint_vel_rel`

精确公式为：

```text
joint_vel_rel = joint_vel - default_joint_vel
```

单位是 rad/s，使用同一 GMT/PhysX native DoF order。当前 `Bumi_CFG` 的全部
`default_joint_vel=0`，匹配任务没有事件修改它，所以本任务中数值上
`joint_vel_rel == joint_vel`。契约仍保留 `rel` 名称，以反映实际 builtin term 的定义。

### 3.5 21 维关节顺序

`joint_pos_rel` 与 `joint_vel_rel` 没传自定义 `SceneEntityCfg`，因此使用 PhysX articulation
native DoF order。当前资产记录的 expected order 是：

```text
 0 l_leg_pitch_joint        11 l_knee_pitch_joint
 1 r_leg_pitch_joint        12 r_knee_pitch_joint
 2 waist_yaw_joint          13 l_arm_yaw_joint
 3 l_leg_roll_joint         14 r_arm_yaw_joint
 4 r_leg_roll_joint         15 l_ankle_pitch_joint
 5 l_arm_pitch_joint        16 r_ankle_pitch_joint
 6 r_arm_pitch_joint        17 l_elbow_pitch_joint
 7 l_leg_yaw_joint          18 r_elbow_pitch_joint
 8 r_leg_yaw_joint          19 l_ankle_roll_joint
 9 l_arm_roll_joint         20 r_ankle_roll_joint
10 r_arm_roll_joint
```

这与 qpos30 尾部的 MuJoCo 顺序（waist、左臂、右臂、左腿、右腿）不同。后续从示范 qpos
构造 proprio 时必须按名字显式置换，不能按位置复制。

静态列表本身不是运行时最终证明；PhysX 的 `shared_metatype.dof_names` 才是权威来源。
匹配任务启动会打印 `robot.joint_names`。后续首次 Isaac producer 集成测试必须执行：

```python
assert tuple(robot.joint_names) == GMT_EXPECTED_JOINT_ORDER
```

若不一致应立即失败，不得隐式重排后继续写数据。

## 4. 原始物理语义、训练 corruption 和 normalizer 必须分层

`genmo.bumi_proprio48.v1` 定义的是：

```text
物理值
    -> GMT observation noise 之前
    -> GMT actor empirical normalizer 之前
```

匹配 GMT policy group 当前仅在训练时逐元素加 additive uniform noise：

| 字段 | 原始物理语义 | GMT policy 训练 corruption |
|---|---|---|
| `projected_gravity` | 机体系单位重力方向 | `+ U[-0.05, 0.05]` |
| `base_ang_vel` | 机体系角速度，rad/s | `+ U[-0.2, 0.2]` |
| `joint_pos_rel` | 相对 active default，rad | `+ U[-0.01, 0.01]` |
| `joint_vel_rel` | 相对零 default velocity，rad/s | `+ U[-0.5, 0.5]` |

`HistoryObsCfg` 的 `enable_corruption=False`。GMT actor 的 empirical normalizer 只处理当前
`policy[69]`，公式是 `(x-mean)/(std+1e-2)`；`history_obs[690]` 和
`command_window[1092]` 不通过这一个 69D normalizer。

GENMO 后续可以定义自己的 48D condition normalization，但不能切用、裁剪或绑定 GMT 的
69D normalizer。训练 augmentation 也必须另立配置，不能改变本物理协议的字段含义。

## 5. 50 Hz 实际状态历史

默认：

```text
proprio_fps = 50
history_steps = 50
proprio_history.shape = [B, 50, 48]
```

若最新有效控制 tick 为 `t_k`，50 个采样点严格是：

```text
t_k - 49/50, ..., t_k - 1/50, t_k
```

共有 50 个采样点，首尾时间差是 `49/50 = 0.98 s`；工程上称“约 1 秒历史”，但不能把
最后一个点误写成首点之后恰好 1.0 秒。

伴随字段：

```text
proprio_history_valid  [B,50] bool
proprio_history_times  [B,50]
decision_time          [B]
```

时间单位为秒，三组 time 使用同一个每样本时钟坐标；推荐 `float64`。约束是：

- 时间严格递增，标称步长 `1/50 s`；
- 有效历史样本不得晚于 `decision_time`；
- 缺历史可以 padding，但必须置 `proprio_history_valid=false`；
- padding 数值可以为零，也可以为其他有限值；只有 mask 表示有效性，消费者不得把“值为
  零”自动解释成真实静止状态或 padding；
- 后续 Stage 1 数据构造只能使用 decision cutoff 之前实际可获得的因果信息。

需要特别区分：第 3 步从目标/示范 qpos 构造的是
`causal_demo_kinematic_proxy_not_actual_gmt_rollout`，不等同于经过 frozen GMT 和 BUMI
dynamics 后的实际机器人状态。训练样本 metadata 明确记录该 provenance；部署时
`proprio_history` 仍必须来自实际机器人/控制器观测。validator 可以检查时间和 mask，却不能
仅凭张量数值证明外部 producer 的来源。

### 5.1 第 3 步的严格因果 demo 30→50 Hz 规则

令 decision frame 为 `d`，示范源帧 `n` 的时刻为 `n/30`。历史槽 `j=0...H-1` 使用
150 Hz 整数公共时基：

```text
history_tick[j] = 5*d - 3*(H-1-j)
latest_source[j] = floor(history_tick[j] / 5)
```

负 tick 是左侧 padding。非负 tick 只读取时间不晚于该槽的最新 30 Hz 源帧，并将该源帧
proprio 保持到 50 Hz 槽；不做需要右侧括点的线性插值、SLERP、中心差分或滤波。源速度先在
30 Hz 上用 `n-1 -> n` 后向差分得到，再随当前观测一起保持：

- `projected_gravity[n] = inverse(q[n]) * [0,0,-1]`；
- `base_ang_vel[n]` 使用 `R[n] R[n-1]^T` 的世界系最短旋转向量乘 30，再由当前
  `inverse(q[n])` 转入当前机体系；
- `joint_pos_rel[n]` 先按名字从 MuJoCo-native 顺序置换到 GMT 顺序，再减 GMT 名义 default；
- `joint_vel_rel[n] = (joint[n]-joint[n-1])*30`，同样按名字置换，default velocity 为零。

第 0 个源帧没有过去样本。由于当前只有整帧 `[H]` mask，它的 48 维槽整体
`proprio_history_valid=false`，有限零速度只作占位，不能冒充真实静止；也不会用第 1 帧做
前向差分。示范数据没有 GMT startup 时每环境的随机 default offset，因此这里明确使用 GMT
名义 default，不随机伪造 active default。将来实际 rollout producer 必须改用其真实 active
default。实现虽然要求调用方只交出 `0..d` 的完整 causal prefix 来封闭未来边界，但实际计算
只截取 H 个历史槽所需的最小过去范围，并额外向左取一个速度前驱，不会随 decision frame
线性重算全部久远历史。

## 6. 120 点、30 Hz 的共享未来时间轴

保持现有正式设置：

```text
music_fps = motion_fps = 30
motion_window_frames = 120
future_motion_qpos30 [B,120,30]
future_contact       [B,120,2]
music_features       [B,120,35]
```

若第 0 点为 `t0`：

```text
future_times[i] = t0 + i / 30,  i=0...119
future_times[119] = t0 + 119/30 = t0 + 3.966666666... s
```

“120 frames @ 30 Hz = 4 秒窗口”描述的是 120 个采样周期对应的常用窗口长度；第一与最后
采样点的严格跨度只有 `119/30 s`。不能写成最后采样点位于 `t0+4.0`。

音乐和动作只共享一份 `future_times[B,120]`，避免两套时间轴漂移。首点是否恰好等于
`decision_time` 不在 v1 中强制，但首点不得早于 `decision_time`：生成延迟和发布调度可以
引入非负 offset，后续 planner 以显式 timestamp 为准。

现有实际音乐条件已从代码核验为 EDGE35，而不是仅照抄设计值：

```text
onset strength  1
MFCC           20
chroma CENS    12
onset peak      1
beat peak       1
               --
               35
```

当前训练内部键名是 `music_embed`；新外部契约使用更明确的 `music_features`。第 4 步网络
改造时应通过显式 adapter 映射，不能在本轮悄悄改现有模型输入。

## 7. committed prefix 与 actual proprio 的区别

两类条件不得混用：

```text
proprio_history:
    机器人实际上怎么动了（actual state）

known_qpos30:
    上一轮 GENMO 已经发布、接下来短期内仍承诺怎么动（reference plan）
```

prefix 直接使用现有 physical qpos30：

```text
known_qpos30       [B,120,30]
known_qpos30_mask  [B,120,30] bool
```

prefix 长度可以为 0，即整个 known mask 为 false。contact 若以后确实需要作为条件，必须用
独立字段和独立可见性规则，不能拼进 qpos30。

`nominal_commit_duration_seconds` 在 v1 配置中为 `null`。本契约不把它固定为 0.4 秒；真正
commit horizon 应由 GENMO 生成延迟、GMT 前瞻和发布/调度余量共同决定。

### 为什么 known mask 必须逐坐标

`root_delta_xy_heading[t]` 依赖 `root_position[t+1]-root_position[t]`。因此当 frame `t` 仍在
承诺 prefix、但 `t+1` 已进入可重写 future 时：

```text
known_qpos30_mask[t, 0:2] = false
```

同一个 `t` 的 height、rotation 和 joints 只依赖当前帧，仍可为 true。这就是 mask 必须是
`[B,120,30]` 而不能退化成 `[B,120]` 的原因。

现有 codec 对完整序列最后一个 root delta 使用“复制最后一个真实 transition”的 terminal
padding，并且 decode 只积分 `delta[:-1]`。数据构造绝不能把 prefix 单独截断后重新编码，
再把该 terminal padding 当成 prefix 边界的已知真实 delta；也不能用本轮未来 GT 计算它后
作为 condition。窗口最后一个 delta 若没有显式的窗口外 `t+1` provenance，同样不得标 known。

若未来发现其他派生字段依赖未知后续，也必须单独缩短对应坐标的 known 区域。contact 标签
继续复用版本化 payload；只有 payload 缺失时才复用现有完整序列 FK、高度、forward foot
speed 与迟滞规则派生。contact 始终只是监督 target，不创建 contact prefix，因此其未来依赖
不会进入 GENMO 条件。

## 8. Stage 1 条件与监督 batch

第 3 步返回以下输入；`H` 可配置，默认 50：

| 字段 | shape | 语义 |
|---|---:|---|
| `music_features` | `[B,120,35]` | EDGE35，沿共享 future 时间轴 |
| `music_valid` | `[B,120]` | 音乐采样有效 mask |
| `proprio_history` | `[B,H,48]` | 原始物理 proprio48 历史，默认 `H=50` |
| `proprio_history_valid` | `[B,H]` | 历史有效 mask |
| `proprio_history_times` | `[B,H]` | 50 Hz 历史时间戳 |
| `known_qpos30` | `[B,120,30]` | 上一轮已承诺的 physical qpos30 |
| `known_qpos30_mask` | `[B,120,30]` | 逐坐标 known mask |
| `future_valid` | `[B,120]` | 未来动作/监督时间点有效 mask |
| `future_times` | `[B,120]` | 音乐与动作共享的 30 Hz 时间轴 |
| `decision_time` | `[B]` | 本次决策可用信息截止时间 |

`gem.closedloop.validate_stage1_condition_batch()` 接受显式 `history_steps`，fail-closed
检查：必需字段集合、shape、bool
mask、有限值、50/30 Hz 时间步长、历史因果边界、invalid future 上不可 known、逐坐标 prefix
单调性、future 首点不早于 decision time，以及无窗口外 provenance 时末帧 root delta 不可
known。它不填数据、不自动修 mask，也不证明中间帧 root delta 的 `p[t+1]` 来自上一轮计划；
qpos30 不保存绝对 root XY，不能拿其他 28 个 known 坐标伪装这项证明。样本 ID、provenance
或后续独立 target 可以作为附加键存在，但必须由各自契约另行验证，不能替代这里的必需
条件键。第 3 步同时构造：

```text
target_qpos30          [B,120,30]
target_qpos30_valid    [B,120,30]
target_contact         [B,120,2]
target_contact_valid   [B,120,2]
```

`target_qpos30_valid` 必须逐坐标：若真实序列最后一帧存在，它的 height、rotation 和 joints
仍有效；只有缺少 `root_position[t+1]` 的 `[0:2]` 无效。padding 的全部坐标无效。
`target_contact_valid` 固定为 `[B,120,2]`，与现行左右足独立 contact loss 兼容。训练 validator
还要求窗口内部 `root_delta_xy_heading[t]` 的有效性严格等于 `future_valid[t+1]`，不允许内部
挖洞；第 119 点则由窗口外第 121 帧 halo 是否存在决定。除此之外，音乐与 future mask 必须
对齐、known mask 不得暴露无效 label，并且每条样本至少保留一个有效 unknown qpos30 坐标。

### 8.1 qpos30 halo、prefix 与尾部处理

target 编码以 decision frame 为 crop anchor，但最多读取 `qpos[t:t+121]`：前 120 帧是窗口，
第 121 帧只提供窗口末帧 root XY delta 的真实下一点。若 halo 不存在，codec 产生的 terminal
repeat 会立即被清零并由逐坐标 mask 标无效。

P 通过 `prefix_min_frames/prefix_max_frames` 配置：二者相等为固定 P，不等为闭区间内的稳定
可复现可变 P，二者都可以为 0；没有把 P 固定为 0.4 秒。对尾部样本，effective P 会裁到
`future_valid_frames-1`，从而始终留下真实 unknown target。teacher-forced 数据 prefix 的
metadata 值是 `teacher_forced_demo_reference_v1`；部署语义仍是“上一轮已发布计划”，两者不
得混写成实际 robot state。

known tensor 先全零，再只向 mask=true 的坐标复制 target：状态字段 `[2:30]` 可覆盖前 P 帧，
root delta `[0:2]` 最多覆盖前 `P-1` 帧。未知区域的有限零只是占位，绝不是由完整未来标签先
复制后再隐藏。

## 9. GMT 69/690/1092 下游契约

这些是冻结 GMT consumer 的事实记录，不是 Stage 1 要重新实现的输入或 GENMO 输出：

```text
policy          [B,69]    = proprio48 + last_action21
history_obs     [B,690]   = 10 × 69
command_window  [B,1092]  = 21 × 52
```

command window 是 past 10 + current 1 + future 10。单帧 52D 为：

```text
target root height       1
target gravity_b         3
target root linear vel   3
target root angular vel  3
target joint position   21
target joint velocity   21
                         --
                         52
```

证据位于 GMT 任务的 `tracking_env_cfg.py`、`mdp/commands.py::get_command_window()` 和
`agents/rsl_rl_mha_him_ppo_cfg.py`。

Stage 1 不重写 command window，不把 1092D 作为 GENMO 输出，也不把 GMT 自己的 10×69
history 当成 GENMO 的 50×48 actual history。

## 10. 30→50 Hz 和速度派生只复用现有权威实现

执行适配顺序固定为：

```text
physical qpos30
    -> qpos28
    -> 最终、连续的 30 Hz qpos 时间线
    -> 30 Hz -> 50 Hz qpos
    -> 从最终 50 Hz qpos 派生：
         root linear velocity
         root angular velocity
         joint dq
    -> 构造 GMT reference
```

权威实现：

- qpos30 反标准化/解码：`gem.robots.bumi.endecoder.BumiEndecoder`；
- 连续增量重采样：`gem.runtime.qpos_timeline.IncrementalQposTimeline`；
- 离线重采样：`gem.runtime.gmt_trajectory.resample_qpos_timeline`；
- 50 Hz 速度和 GMT frame：
  `gem.runtime.gmt_trajectory.qpos_timeline_to_gmt_frames`。

根平移与 joints 用线性插值，根四元数先做符号连续化再做 shortest-arc SLERP；速度只在最终
50 Hz 时间线上求。Stage 1 不新建 dq 算法、root velocity 算法或第二套 command-window
逻辑，也不从 30 Hz 预测 dq 后再冒充 50 Hz consumer reference。

这里的“执行适配”与第 3 步 demo history builder 是两个不同用途：执行适配拥有完整参考轨迹，
必须继续复用上述权威插值/中心差分；训练 history 必须模拟 decision cutoff，因而只能使用
后向差分和 latest-available hold。后者不生成 GMT command，不应替代前者。

按现有端点计数规则，一个独立 120 点、30 Hz 序列的首尾跨度为 `119/30 s`，离线重采样会
得到 `floor((119/30)*50)+1 = 199` 个 50 Hz 点，而不是简单按 `4*50` 写成 200。在线连续
链路应使用全局增量时间栅格，避免逐窗舍入和重复端点。

## 11. 系统频率与尚未实测的重规划能力

当前事实：

```text
GENMO motion       30 Hz
GENMO sequence     120 samples
GMT policy         50 Hz，0.02 s/step
GMT history        10 steps，690D
GMT command        21 reference frames，1092D
physics            200 Hz，0.005 s/step
policy/physics     1:4
```

上层 GENMO 重规划频率尚未最终锁定。首版计划后续依次测试 1 Hz、2 Hz；这只是待验证配置，
不是当前已实现、已计时或已在仿真/真机达成的能力。commit horizon 也必须在新模型的端到端
P50/P95/P99 延迟和 GMT 前瞻需求已知后再确定。

## 12. 第 3 步已完成、尚未完成的第 4 步

第 3 步已经实现并由合成配对数据测试覆盖：

- 独立 `BumiClosedLoopStage1Dataset`，不改变旧 `BumiMusicDanceDataset`；
- 每个 split 各自打开 `manifests/<split>.jsonl`，不会跨集合切片；
- configurable H、固定/可变/P=0 prefix、序列头尾 padding 和共享时间戳；
- 因果 demo proprio48、按名字的关节置换和 GMT 名义 default provenance；
- qpos30 右侧 halo、逐坐标 target/known mask 与有限占位；
- 版本化 contact payload 优先，缺失时复用现有全序列 FK contact；
- train-only proprio48 stats 手动入口，拒绝覆盖已有输出；
- 服务器1 UMR70+Mine 四来源实例配置，继续只读引用正式 qpos30 stats。

仍未完成的是实际 frozen-GMT/BUMI dynamics rollout 历史。当前监督数据里的 history 和
prefix 分别是 demo-derived causal proxy 与 teacher-forced reference；它们已显式标 provenance，
但还不能替代部署分布或证明闭环恢复能力。

尚未完成的第 4 步网络改造至少要解决：

- proprio-history encoder、prefix encoder 和显式 mask；
- `music_features -> music_embed` 适配；
- physical qpos30 -> 现有 normalized x0 的显式适配；
- 旧 music-only checkpoint 的可审计 weights-only warm start；
- 新 condition normalization 与 CFG/dropout 语义；
- mask 全关时对现有 music-only baseline 的回归一致性。

本文的类型、Dataset、配置和合成单元测试只证明代码构造与契约一致，不证明服务器1全量
数据已在本轮重跑、网络能够消费新字段、训练收敛、GMT dynamics 稳定、sim2sim、实时
2 Hz 或实机安全。
