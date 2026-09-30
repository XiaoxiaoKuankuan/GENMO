# 第九步：实际执行数据、独立 Critic 与有限 DPPO

本步沿用 Stage1 的 120 帧、30 Hz、qpos30/contact2 输出及十字段条件，GMT 始终冻结。
训练进程拥有 Actor、Critic、Buffer 和优化器；另一个解释器中的 GMT worker 只负责
69/690/1092 推理链、50 Hz 控制和 200 Hz PhysX。默认执行模式为 latency，默认入口
模式为 collect；这不构成实机实时性或训练效果改善的证明。

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
1280 个去噪转移以 microbatch=1 累积后只做一次优化，Actor lr=1e-9、梯度上限 1。
最初计划的 lr=1e-6 在真实首轮产生平均联合 KL=2964.81，被 0.02 门槛拒绝；当前值是
依据该证据进行的一次保守短测修正，不改变奖励、噪声、概率定义或 KL 门槛。
更新前、加入 BC 之前单独核验 PPO 梯度；BC 权重 0.1、batch=2，只每次优化计算一次，
复用原 Stage1 loss、train 配对数据和独立 bc_update_steps。独立解析 Gaussian KL 的
平均联合量阈值为 0.02；超过时停止并保存失败候选，不提高阈值继续验收。

## 奖励、价值与时间

奖励基于实际执行轨迹。每个 20 ms 区间只积分一次：

`r = .02 * (2*music + 2*track + stable - .2*actuator - .5*contact - .5*consistency)`。

音乐使用 EDGE35 节拍与强度代理、因果 1 秒实际动作窗口和活动门控；静止不能获得
完整音乐分。跟踪包含 q/dq、根 XY、yaw 和末端相对高度；稳定性检查根高、non-yaw、
角速度和关节加速度。执行器使用 200 Hz 的 PD 估计力矩/功率与目标变化代理，不能称为
真实能耗；接触用几何、足速和净力证据，不用 contact head 免除惩罚，不将净力称为
纯地面力。位置与速度的一致性复用参考时间线差分支持。完整权重、尺度和有效性写入
resolved_config 与逐步报告。有限非法候选一次扣 1，任务失败一次扣 5。

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
正式执行命令及实际验证结果以本轮验收报告为准，以下是 CLI 结构，不表示已验收：

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

## 验收边界

CPU 单测覆盖协议故障、缓存/参考有界、概率/mask/CFG、奖励分段不变、GAE 和恢复。
随后在真实 Stage1、真实 train 音乐、真实 Isaac 执行数据上检查零更新概率、64 条采集、
Critic、一次 DPPO+BC、完整 checkpoint 和重启后 16 条新转移。A/B/C 分别是原确定性、
同权重新随机、有限更新后的策略；三组结果不能混淆。GMT 参数、normalizer、PD、动作
缩放、资产和严格终止阈值均不能为验收而改变。

成功标准是正确数据上的正确网络更新，不要求奖励立即上升。任何未实际执行的层级
须标记未运行；概率、奖励、执行记录或 GMT 冻结性出错时不得判定第九步通过。
