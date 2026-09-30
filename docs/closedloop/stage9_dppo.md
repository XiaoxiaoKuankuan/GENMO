# 第九步：实际执行数据、独立 Critic 与有限 DPPO

本步沿用 Stage1 的 120 帧、30 Hz、qpos30/contact2 输出及十字段条件，GMT 始终冻结。
训练进程拥有 Actor、Critic、Buffer 和优化器；另一个解释器中的 GMT worker 只负责
69/690/1092 推理链、50 Hz 控制和 200 Hz PhysX。默认执行模式为 latency，默认入口
模式为 collect；这不构成实机实时性或训练效果改善的证明。
当前默认奖励为 `stage9.execution_reward.v2`，按照用户新的活动门控与物理代价公式。
其中 Tracking 已改为 `gmt.motion_tracking.v1`，对齐 Frozen GMT **当前任务**的六项
motion-tracking 定义及已启用项的权重比例；其余奖励和学习算法保持不变。
此前 2026-09-30 的第一轮 DPPO/恢复验收使用奖励 v1，其报告保持不变；后续奖励v2
也已完成真实训练、同批学习率校准及恢复，见[奖励v2训练验收](stage9_reward_v2_training.md)。
不同奖励和源码身份的checkpoint不能无条件完整续训。

## 实施顺序与执行数据

先验收 ACK 执行协议、异常记录、缓存与参考裁剪，再验收随机概率、冻结采集、Critic，
最后允许一次 Actor 更新。上层一次转移从实际发出请求到下一个无 pending 的合法决策
边界，可能超过 25 个控制步；单个 advance RPC 仍最多 25 步。生成等待期间消耗旧参考，
错过的决策不补发。真实音乐结束为 terminated；30 秒行政上限及批次末尾为 truncated，
在可继续且可信的下一条件上 bootstrap。任意时刻发生物理或诊断故障，不能伪造终态。

每个有副作用请求携带 backend_session_id、单调 mutation_seq、episode 和操作参数。
worker 对未确认结果仅保留一个槽；客户端先向 SQLite FULL 事务日志保存完整回复，
再 ACK 回收。ACK 水位之前的旧请求明确过期，不能再次执行；断线只重试原编号。
物理步完成后立即建立最小 trace，再补充历史、误差和诊断。正常转移要求 trace 条数、
控制步数、全局 tick 和 4 倍物理步数一致；未知的物理推进计数不补零。

Buffer 在 CPU 中保存条件、已承诺前缀、raw 去噪链、旧概率、均值/标准差、生成结果、
发布/拒绝结果、分项奖励、下一条件、真实步数和所有策略/episode/plan 身份。
完整源动作与实际发布参考不可替换进链中重算概率。每轮 Actor 版本固定，更新前截断，
更新后清空 Buffer 并 reset，不在不同版本之间继续 pending 计划。

## 随机采样和 DPPO

`DPPODiffusionPolicy` 复用同一个 Stage1Actor。原 1000 步 cosine/x0 日程、SpacedDiffusion
时间映射和只对音乐做 CFG 的语义不变，首版 20 步、CFG=2.5。采样与概率重算复用同一
`transition_parameters`，使用 stochastic DDIM、eta=0.1、normalized 标准差下限 0.001。
均值使用未加下限的基础 sigma，下限只影响实际高斯方差；末步也实际加入该噪声。

自由坐标为 `future_valid[...,None] & ~known_qpos30_mask`。概率针对完整 120 帧 normalized
原始变量，逐自由坐标高斯密度相加得到 `[N,K]` joint_sum；不是逐维平均或经过裁剪的
伪联合概率。contact2 是辅助确定性 head，只通过原监督损失保持。初始纯噪声分布不含
Actor 参数，不作为可学习策略转移。

收集和概率重算均为 eval，前者 no_grad，后者允许梯度。网络和采样链 FP32，概率相减
及联合归约 FP64，首轮关闭 AMP/TF32。重算把相邻链状态 detach，只对当前条件编码和
均值反传，不穿过 GMT、PhysX 或历史采样链。零更新门槛为 log_probability 最大差
不超过 1e-4、ratio 与 1 的最大差不超过 1e-3。

每个内部去噪步使用独立 PPO ratio 和 clipped objective，clip=0.01。内部折扣为
`0.99**(19-step)`，最后一步为 1；不把 20 步当成 20 个环境步。64 条上层转移的全部
1280 个去噪转移以 microbatch=1 累积一次梯度，Actor起始lr=1e-9、梯度上限1。
当前可选同批候选[1e-9,3e-9,1e-8]分别恢复同一模型/Adam/梯度起点，每次试探计入
预算；只保留原KL门槛内最大合格候选的完整单步状态，不累计三步。全部候选不合格
或异常则恢复初始状态，保存诊断。v2实测只有1e-9通过，checkpoint及恢复核对实际lr。
最初计划的 lr=1e-6 在真实首轮产生平均联合 KL=2964.81，被 0.02 门槛拒绝；当前值是
依据该证据进行的一次保守短测修正，不改变奖励、噪声、概率定义或 KL 门槛。
更新前、加入 BC 之前单独核验 PPO 梯度；BC 权重 0.1、batch=2，只每次优化计算一次，
复用原 Stage1 loss、train 配对数据和独立 bc_update_steps。独立解析 Gaussian KL 的
平均联合量阈值为 0.02；超过时停止并保存失败候选，不提高阈值继续验收。

## 奖励、价值与时间

奖励基于实际执行轨迹。每个 20 ms 区间只积分一次：

`r = .02 * (2.5*gate*track + 2*gate*music + stable + .5*alive - .15*cmd - .10*torque - .20*contact - .50*joint_limit)`。

跟踪采用下文六项 GMT motion-tracking 指数分数。稳定性只评价根高与 non-yaw，不抑制正常舞蹈角速度
或关节加速度。音乐节拍复用原函数；强度改为实际与配对示范同时间窗活动度的对数
比值匹配，完全移除旧 onset 强度代理。门控同时作用于跟踪和音乐，活动度采用最近
0.5 秒关节速度 RMS。配对动作仅作 reward 监督，不进入 Actor/Critic 条件；缺失、
身份不符或窗口不一致使转移无效。初始不足半秒时两者取相同的已有部分窗口并明确
标记窗口未满，不伪造历史。节拍不足一秒时该子项无效为 0，强度仍单独计算。

命令代价使用实际提交 PhysX 的关节位置目标，相邻 50Hz 目标差除以 .02 与实际限速。
力矩采用 200Hz 未裁剪 implicit PD estimate，80% 限力以下不罚，近饱和按用户公式
计算并在四子步平均。接触沿用原任务脚/肘允许集合和接触阈值，支撑球附近切向速度
明确是几何速度代理。关节限位同时检查实际位置与当前消费参考，优先实际 soft limits。
机械功率和冲击只做诊断；功率使用步前力矩估计乘步后速度，时间戳明确记录非同步，
不称真实电功率或能耗。参考位置—速度一致性以原算子和 1e-4 容差作为有效性门禁，
超限不是策略负奖。

全部权重、混合比例、尺度及容差写入配置与 resolved_config；逐步保存 raw、normalized、
score、gate、weighted_rate、分项积分及最终奖励。真正执行失败一次扣 5，有限非法
候选一次扣 .5，均不乘 dt；正常结束/截断不罚。late_plan保留为到达时序事件，不当作
非法生成参考；RPC/程序故障保存 invalid 转移，未知执行计数为 null，不送入训练。
完整字段、公式与本轮验证见[奖励v2说明](stage9_reward_v2.md)。

### Tracking 与 Frozen GMT 当前任务对齐

原定义来自 GMT 的 `mimic_noetix_bumi4340_mha_sonic/mdp/rewards.py`；std、选择集合和
源权重从当前任务 `tracking_env_cfg.py::RewardsCfg` 在冻结关闭奖励前读取。每项为
`R_i = exp(-E_i / std_i²)`，这里 `E_i` **已经是平方误差**，不会再次平方。

| 配置键 | 原 GMT 函数 | 平方误差 E 的定义 | std | Stage9 内部权重 |
| --- | --- | --- | ---: | ---: |
| `anchor_pos` | `motion_global_anchor_position_error_exp` | 世界 anchor 位置差的三轴平方和 | 0.30 m | 1/7 |
| `anchor_ori` | `motion_global_anchor_orientation_error_exp` | 世界 anchor 完整四元数最短旋转角的平方 | 0.40 rad | 1/7 |
| `body_pos` | `motion_relative_body_position_error_exp` | 原 MotionCommand 对齐后的 body 位置差，先三轴平方和，再对 body 平均 | 0.30 m | 2/7 |
| `body_ori` | `motion_relative_body_orientation_error_exp` | 同一对齐后的 body 完整最短旋转角平方，对 body 平均 | 0.40 rad | 2/7 |
| `joint_pos` | `motion_joint_position_error_exp` | 原任务选中的 12 个腿关节位置平方差的均值 | 0.25 rad | 1/7 |
| `joint_vel` | `motion_joint_velocity_error_exp` | 原函数默认全部 21 个原生关节的速度平方差均值 | 1.40 rad/s | 0 |

原任务对应五个启用项的权重为 `0.5:0.5:1:1:0.5`，这里除以总和 3.5，保持原比例，
同时让 `R_track` 落在 [0,1]。外层仍为 `2.5 * activity_gate * R_track`，统一乘 .02。
`joint_vel` 原配置未启用，1.40 是 Stage9 的预留 std，不冒称原训练参数；误差及分数
始终记录。需要启用时改配置并重新分配六项权重，使其非负且总和为 1。

body 保持当前 `MotionCommand.cfg.body_names` 的 22 个 body 顺序，实际状态严格按
`body_indexes` 取出；anchor 为 `base_link`。关节数组保持实际机器人/ONNX 的原生
21 关节顺序。位置奖励沿用原 `motion_leg_joint_pos` 的左腿六关节、右腿六关节选择，
在原生数组中的索引为 `[0,3,7,11,15,19,1,4,8,12,16,20]`，不会误把腰或手臂加进去。
每步同时记录完整名称、所选名称/索引、参考值、实际值和源函数配置，供复核对应关系。

body 的对齐严格复用原 MotionCommand 公式：参考 anchor 的 XY 平移到实际 anchor，
Z 保持参考高度；用实际 anchor 与参考 anchor 相对旋转的 yaw 旋转参考 body。
global anchor 两项仍使用未经对齐的世界误差。姿态直接调用 IsaacLab 原
`quat_error_magnitude`，跨 ±π 使用最短旋转，四元数 q 与 -q 等价，不将 yaw 欧拉角
相减当作完整姿态误差。既有稳定性中的 non-yaw 和根高定义不变。

StreamingMotionCommand 的 inherited relative 缓存没有随流式参考刷新；新增只读
`TrackingDiagnostics` 在完整 50 Hz 区间末端、同一实际消费 tick 上构造局部参考视图，
逐项调用原 GMT 奖励函数核验 `source_score`。它不调用带采样副作用的 `_update_command`，
也不写回旧缓存或改变 GMT 的观察、策略、归一化、PD、控制时钟。诊断失败沿用最小
trace 记录，已执行步数不会丢失，转移标记 invalid。

参数均在 `reward.tracking.{objective,std,weights}`，已同步两份 `stage9_dppo*.yaml`
及共用奖励实现的两份 Stage10 配置。六项 raw 平方误差、error/std²、分数和权重均
随奖励保存。新增诊断文件纳入跨仓库源码指纹；缺少新字段或名称/时刻/坐标契约错误
会拒绝该转移，没有旧五项回退。

这里对齐的是**当前可核验的任务定义及这六项子目标**，不是复制 GMT 全部训练奖励。
`model_135000_stage2.json` 明确历史 `source_training_config_available=false`，因此不
声称完整还原这个 ONNX 检查点当年的训练配置。此前 v2/Stage10 的真实训练与全 val
报告使用旧五项 Tracking，保留为历史结果；不能与新定义的累计奖励直接比较。
奖励身份绑定完整配置，旧 checkpoint 不能完整续训到新版；需新 run、新执行采集和
新基线。原 Stage1 的 weights-only Actor 初始化仍可使用，Critic/优化器重新建立。

2026-09-30 本次验证：GENMO 相关 CPU 回归 561 项通过，GMT 相关回归 155 项通过，
另有 1 项旧 IsaacLab 包导入测试跳过；新增 Tracking 测试全部执行。服务器 1 使用
GENMO `81de6e5` / GMT `59ab0a7`，在真实冻结 GMT/PhysX 上跨两次 reset 执行了
60 个控制区间、240 个物理步；六项与原函数分数的最大差为 `3.21e-8`，逐步参考与
实际关节值、trace、journal SHA/序号/ACK 全部一致，GMT 模型与运行参数/归一化
指纹保持不变，worker 正常退出。此短测使用静止 bootstrap，只验证诊断与评分接口；
没有加载 Actor/Critic、没有更新网络，也不代表新 Tracking 的音乐训练效果或全量
质量评估。临时日志、执行 journal、socket 和本轮 USD 均已清理，摘要保存在根日志。

独立 Critic 输入同一音乐、50×48 实际历史、相同已承诺前缀及真实音乐剩余秒数。
历史 GRU 128，音乐和前缀各 MLP 128，拼接 385→256→128→1。没有 Actor/GMT 参数
共享，不输入本轮新动作或行政采集额度；只估计可见条件下的价值，不声称完全 Markov。

gamma_upper=.99、lambda_upper=.95 均对应 0.5 秒。执行 m 步时 gamma_low=.99**(1/25)，
`R=sum(gamma_low**i*r_i)`、`Gamma=gamma_low**m`，lambda 同理。bootstrap 与 continuation
使用两个独立 mask，真终止不 bootstrap，可信采集截断 bootstrap 但不跨 reset 递推。
return target 由未归一化优势和 old value 一次生成；整批有效上层优势只归一化一次。
Critic 默认 lr=1e-4、20 步、batch=32，值目标不随 Critic 更新而改变。

## 入口、预算与恢复

入口为 `tools/train_closedloop_dppo.py`，支持 preflight、collect、critic、train、
resume-check、eval。必须显式传独立 `--output-dir`，默认模式 collect。本机配置为
`configs/closedloop/stage9_dppo_smoke.yaml`，服务器配置为 `stage9_dppo_server1.yaml`。
正式执行命令及实际验证结果见[服务器1有限验收报告](stage9_dppo_acceptance_20260930.md)。
该报告保留首轮 KL 超限、第二轮通过及重启恢复的完整边界；以下仅展示 CLI 结构：

```bash
python tools/train_closedloop_dppo.py --config CONFIG --mode preflight --output-dir NEW_PREFLIGHT_DIR
python tools/train_closedloop_dppo.py --config CONFIG --mode collect --output-dir NEW_COLLECT_DIR
python tools/train_closedloop_dppo.py --config CONFIG --mode train --output-dir NEW_RUN_DIR
python tools/train_closedloop_dppo.py --config CONFIG --mode resume-check --output-dir NEW_RUN_DIR --resume NEW_RUN_DIR/checkpoints/stage9_000001.pt
python tools/train_closedloop_dppo.py --config CONFIG --mode eval --output-dir NEW_EVAL_DIR
```

全轮最多 256 次真实生成、10000 个控制步、40000 个物理步尝试和 3 轮更新，默认只更新
1 轮。校准 4+12 为有限预检，不冒充原 10+100 校准；预算包含 warmup、等待、拒绝、
故障和 resume，副作用前写盘预占，未知消耗不退款。校准、main、resume、A/B/C 和诊断
分别有控制步分配，不通过换输出目录在同一验收轮中重置预算。

Stage1 s350000 只作 weights-only 初始化，不恢复旧优化器或 global_step。Stage9
checkpoint 单独保存 Actor、Critic、两优化器、随机状态、音乐/BC 采样器、策略版本、
预算和身份。resume 在对象构造后恢复 RNG，创建新 worker session 并 reset；不声称
恢复 PhysX 内部状态。通过报告同时保存模型资产 SHA、实际源码/解释器和 GMT 冻结证据。

已有产物可通过 `tools/eval/audit_closedloop_dppo.py` 只读审计：

```bash
python -B tools/eval/audit_closedloop_dppo.py \
  --run-dir RUN_DIR --budget-file CAMPAIGN_BUDGET_JSON --output NEW_AUDIT_JSON
```

该工具不启动网络或仿真：从 SQLite 回复独立核对执行计数、trace、四个子步及身份，
关联 rollout 奖励，独立重算固定价值目标与变长 GAE，并交叉检查概率、梯度、KL、
冻结报告和完整 checkpoint 元数据。checkpoint 使用 CPU mmap，不扫描大权重；
概率与网络梯度依赖真实运行保存的测量报告，不声称再次运行网络验证。
审计输出排他创建，不能覆盖原报告。默认必须具备完整 64+16 条及 checkpoint；
`--allow-incomplete` 仅允许缺失阶段标记为 not_run，已有失败或损坏仍然失败。
SQLite 在临时快照中读取，避免在源目录创建 shm；临时文件退出回收。

## 验收边界

CPU 单测覆盖协议故障、缓存/参考有界、概率/mask/CFG、奖励分段不变、GAE 和恢复。
随后在真实 Stage1、真实 train 音乐、真实 Isaac 执行数据上检查零更新概率、64 条采集、
Critic、一次 DPPO+BC、完整 checkpoint 和重启后 16 条新转移。A/B/C 分别是原确定性、
同权重新随机、有限更新后的策略；三组结果不能混淆。GMT 参数、normalizer、PD、动作
缩放、资产和严格终止阈值均不能为验收而改变。

成功标准是正确数据上的正确网络更新，不要求奖励立即上升。任何未实际执行的层级
须标记未运行；概率、奖励、执行记录或 GMT 冻结性出错时不得判定第九步通过。
