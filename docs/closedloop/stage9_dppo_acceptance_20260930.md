# 第九步真实执行训练验收：2026-09-30

已在服务器1完成单环境、真实 Stage1 模型与 Isaac PhysX 下的有限验收：先冻结采集，
再训练独立 Critic，最后执行一次通过门槛的 DPPO＋原配对数据监督更新；保存完整
checkpoint 后重启 worker，恢复训练状态并采集 16 条新数据。总体结果见
`acceptance_20260930_run02/stage9_acceptance.json`，其中 `stage9_passed=true`。

这证明执行数据进入了正确网络的更新路径，不证明奖励改善、收敛、未见音乐泛化或
实机可用。首轮更新曾被 KL 门槛拒绝，失败数据与候选权重仍保留。

## 实现与原问题的处理

| 部分 | 实际修改与理由 |
| --- | --- |
| 执行记录 | GMT 在第四个物理子步正常返回后立即建立最小控制记录，再补历史和诊断。部分推进、未知物理计数和诊断异常明确保留；GENMO 拒绝 `len(trace) != executed_control_steps` 的转移。 |
| 幂等与内存 | 新 ACK v2 协议使用 session、跨 episode 的单调序号及单个未确认回复槽。客户端先将完整回复写入 SQLite FULL 事务，再 ACK；重复请求回放，已确认旧请求过期拒绝。旧接口 reset 释放缓存，并设置 256 项/64 MiB 上限。 |
| 参考准备 | `ReferenceTimeline` 沿共同时间网格裁剪历史，保留 GMT 回看及差分支持；已承诺的六组参考数组保持原值。转换长度不再随 episode 累积。 |
| 训练采样 | 复用 Stage1Actor，增加 stochastic DDIM、完整去噪链、旧概率、均值和标准差。只对自由 qpos30 坐标计算联合概率，前缀和 padding 不作为当前自由动作。 |
| 执行奖励 | 音乐、跟踪、稳定性、执行器、接触和位置—速度一致性来自实际执行；四个 200 Hz 子步支持执行器与接触诊断，50 Hz 奖励只积分一次。 |
| 价值与回报 | 独立 Critic 输入音乐、50×48 实际观测、承诺前缀及真实音乐剩余时间；按实际控制步数计算折扣，区分 bootstrap 与跨转移递推。目标在更新前固定。 |
| DPPO 与恢复 | 内部去噪转移与外部上层执行区间分别组织；梯度只进入 GENMO，不穿过 GMT 或 PhysX。保留原 Stage1 配对监督；保存两网络、两优化器、RNG、采样器、身份及预算。 |

代码入口为 `tools/train_closedloop_dppo.py`；模块位于 `gem/closedloop/dppo/`。
实现细节见 [第九步接口与算法说明](stage9_dppo.md)。本轮沿用已确定的 latency 模式，
用实测生成准备时间驱动旧参考下的仿真等待；尚未实现真正异步生成线程。

## 固定基线与版本

- GENMO 通过训练及恢复时提交：`4f7eb45`；GMT 执行提交：`631e6e6`。
- 首轮失败时 GENMO 为 `3958189`；后续修改仅降低短测 Actor 学习率，并补齐 run_id
  噪声种子隔离与旧价值审计保存。完整执行源码清单包含 112 个文件，通过轮训练与
  恢复均核验不变，manifest SHA256 为
  `d1399840e2032af438ab3a25cf5b411136d384d83fdbda4ea9f26ab9db12c067`。
- Stage1 从 `bumi_closedloop_stage1_90505_p6to18_z15_b256_s350k_8gpu_v1/checkpoints/s350000.pt`
  加载权重，Stage9 使用新优化器与新计数。这是 weights-only 初始化；后面的
  resume-check 才是完整 Stage9 训练状态恢复。
- qpos30/contact2、120 帧@30 Hz、实际历史 50×48；GMT 输入 69/690/1092，控制
  50 Hz，CPU PhysX 200 Hz。GENMO 使用单张 RTX 6000D GPU0。
- 严格终止阈值保持根高误差 0.20 m、非 yaw 姿态误差 0.60 rad、yaw 误差
  1.50 rad、脚/肘相对高度误差 0.15 m，严格大于触发。
- GMT ONNX SHA256：`d2e176657d72d1b0efcb04406abd17145fc45a3a5755b62aae7b866e0a6e3d1b`；
  运行参数指纹：`f8e5c7f9e8414a84708f48ff5ff1609d163ef19a26a2c1b0e60ddeaa33e4a9d4`。
  训练、失败诊断及恢复退出时均核验冻结，原模型、统计量与资产 SHA 未变。

## 两次更新尝试与验收值

| 检查项 | 首轮 run01 | 通过轮 run02 |
| --- | --- | --- |
| 真实上层训练转移 | 64 | 64 |
| 零更新最大 log-prob 差 | 0 | 0 |
| 零更新最大 ratio−1 绝对值 | 0 | 0 |
| 独立 Gaussian 密度核对最大差 | 1.82e-12 | 1.82e-12 |
| Critic 更新 | 20 步，参数改变，Actor 未变 | 20 步，参数改变，Actor 未变 |
| PPO-only 裁剪前梯度范数 | 25384.0098 | 25241.9531 |
| Actor 学习率 | 1e-6 | 1e-9 |
| Actor 更新 | 1 次，参数改变，Critic 未变 | 1 次，参数改变，Critic 未变 |
| 平均联合 KL / 门槛 | 2964.8141 / 0.02，失败 | 0.00193760 / 0.02，通过 |
| 后续流程 | 保留失败候选，停止 C 和恢复 | C 对照、完整 checkpoint、重启恢复均完成 |

没有提高 KL 门槛、修改噪声或奖励来使首轮通过。run02 从原 Stage1 权重重新开始，
采集新数据，且预算继续包含 run01 的失败消耗。1e-9 是这次链路短测的保守设置，
不能据此认定为后续长训练的合适学习率。

通过轮 KL 指标先对每个内部去噪转移的全部自由坐标求和，再对 64×20 个转移取平均。
其 P95 为 0.01531698，最大单项为 0.04588867；整条 20 步链的平均和为 0.03875199。
门槛针对前述平均内部联合 KL，不能表述为每个样本或整条链都低于 0.02。

通过轮 BC loss 为 0.38322850，权重 0.1，batch=2，以两个 microbatch 执行一次
监督累积；原 Stage1 loss、配对数据、独立采样器及监督更新计数保留。PPO 梯度是在
加入 BC 前独立核验的，因此 Actor 更新不能仅由监督项解释。总梯度范数
25242.1621 在优化前裁剪至上限 1。Critic MSE 为 344.8203、explained variance
为 0.08714，只代表这次短拟合的诊断值。

恢复验证读取 `checkpoints/stage9_000001.pt`，恢复 Actor、Critic、优化器、随机状态
及采样器，创建新 backend session `c522d24d-4de3-4b82-96cc-f7807d0b5281`。
旧 Buffer 丢弃，reset 后采集 16 条新转移，概率比误差再次为 0。恢复阶段未再次
优化 Actor，也不声称恢复了 PhysX 的内部状态。

## 数据覆盖、对照与已知局限

训练任务来自固定修正后的 200 首 train 音乐选择，保持四库来源选择概率
20%/35%/25%/20%。为在有限预算中覆盖四库并保留一个长 episode，主采集前三个
episode 各在一个上层转移后截断，第四个 Mine episode 连续 30 秒，最后补足到
64 条。这个批次不是四库平衡训练集。采集截断与真正任务结束分别处理。

通过轮 4+12 次有限校准的 12 个测量值为 317.869～326.687 ms，预算映射为
420 ms；这不是原 Stage8 的 10+100 次校准，也不是部署硬实时性证明。
本批训练与恢复的实际前缀均为 P=21，`prefix_over_18_fraction=1.0`，长于 Stage1 原训练的 P≤18 范围，
故本轮还不能证明这种条件分布下的生成质量。前缀掩码、概率自由维度和执行记录
的正确性与生成效果必须分开判断。

A/B/C 固定同一 AIST++ 样本 `gMH_sBM_cAll_d23_mMH5_ch03` 与 seed=1729；分别为
原确定性采样、同权重新随机采样、更新后的随机采样。每组 4 次决策、100 个实际
控制步、2 秒音乐执行，全部完成，无拒绝或控制终止：

| 对照 | 实际累计奖励 |
| --- | ---: |
| A：原确定性 | 6.67097623 |
| B：更新前随机 | 6.66696314 |
| C：更新后随机 | 6.55918707 |

C 的奖励低于 B。单曲、单 seed、两秒结果只作为更新前后执行可用性的短检查，
不能说明策略改善。音乐奖励前 49 个控制步因不足因果历史而明确无效并计 0；
后 51 步有效，但此样本的 onset 强度代理饱和为 1，未验证音乐强弱变化的跟随。
执行器项使用 implicit PD 估计，接触项使用净接触力代理，不等同真实能耗或纯地面力。
本轮没有视频或实机验收。

## 测试与预算

- GENMO 相关 CPU 集成测试 215 passed；学习率、种子与旧价值保存修改后的定向
  回归 59 passed。仅有原 rotary autocast 弃用 warning。
- GMT 故障注入及参考/配置/诊断回归 119 passed、1 skipped；跳过项因本地 GENMO
  解释器不包含 IsaacLab，之后真实服务器执行单独验证动力学路径。
- 新增只读审计器 29 项故障测试通过。服务器上读取全量执行数据和完整 checkpoint
  运行 `audit_closedloop_dppo.py`，独立验收 **32 项全部通过**，结果为
  `independent_audit.json`。主轮 journal 为 455 次 mutation、170 次 advance、
  2350 个控制步；恢复为 92 次 mutation、40 次 advance、600 个控制步。
  训练/恢复 rollout 的 1600/400 个音乐控制步全部关联到原始执行记录；固定 GAE
  与 return 独立重算最大差 3.3751e-13。审计包括完整恢复元数据与新物理 session，
  不重新运行网络或仿真；概率与梯度交叉核对原真实运行报告。
- 参考时间线 60 秒、120 次合成移动参考重规划逐轮核验六数组和来源；准备时最多
  129 个源姿态/214 个参考点，裁剪后最多 114/189。这个合成长度测试不冒充真实
  长期训练或物理稳定性测试。
- 两轮与恢复累计 196/256 次生成、5150/10000 个控制步、20600/40000 个物理步、
  2/3 次 Actor 更新尝试；其中仅一次更新通过验收。预算包括校准、站立预热、
  A/B/C、失败轮及恢复，不只统计进入训练 Buffer 的音乐步。
- 两次训练及恢复 worker 均正常退出；结束后的服务器检查 8 张 GPU 显存均为
  0 MiB，未留本轮训练或模型执行任务。

## 产物与复现

服务器完整产物根：

```text
/data0/user/liwei/GENMO_outputs/closedloop_stage9/
  campaign_20260930_budget.json
  acceptance_20260930_run01/                 # 失败候选与原始数据保留
  acceptance_20260930_run02/
    acceptance/                            # 64条链、执行SQLite、奖励、固定targets、A/B/C
    checkpoints/stage9_000001.pt            # 完整Actor/Critic/优化器/RNG等，约2.5GB
    resume_check/                          # 新session的16条执行数据
    stage9_acceptance.json
```

本地证据根为 `/home/weili/bumi-closedloop-worktrees/GENMO/outputs/closedloop_stage9/`。
通过轮执行数据与报告回传本地；完整 checkpoint 留在服务器，不用仅权重文件替代。
首轮本地保留 JSON/YAML 报告，全量失败数据仍位于服务器。
通过轮 134 个执行证据文件共 430786172 字节已逐 SHA256 核验与服务器一致，结果为
`local_transfer_verification.json`；服务器完整文件清单为 `artifact_manifest.json`。
完整 checkpoint 为 2602826754 字节，SHA256 为
`088d9bb23ec3cfce66edfe66450c9e382b6e8384980c0981238884314eb08e11`。
独立审计 JSON 已回传，SHA256 为
`38bfb4551c99dd78aef4383f62197b0e16e12ac08da9eeb34a90dc3ca9ee9305`。
审计完成后重新哈希服务器原有 135 个文件、3033612926 字节，全部与审计前清单一致，
结果为 `audit_input_integrity.json`。审计工具提交为 `4ec8e1f`，独立于执行训练的
`4f7eb45`；后续只增文档/日志，未改变已验收训练源码。

真实执行时在 `/home/user/liwei/GENMO-bumi-closedloop` 使用以下入口；两个命令按
顺序执行，恢复命令沿用同一个输出目录与共享预算。复现新实验需使用新的目录和
明确的新验收预算，不能覆盖下列已完成产物，也不能换目录绕过同轮预算：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_dppo.py \
  --config configs/closedloop/stage9_dppo_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage9/acceptance_20260930_run02 \
  --budget-file /data0/user/liwei/GENMO_outputs/closedloop_stage9/campaign_20260930_budget.json

env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_dppo.py \
  --config configs/closedloop/stage9_dppo_server1.yaml --mode resume-check \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage9/acceptance_20260930_run02 \
  --resume /data0/user/liwei/GENMO_outputs/closedloop_stage9/acceptance_20260930_run02/checkpoints/stage9_000001.pt \
  --budget-file /data0/user/liwei/GENMO_outputs/closedloop_stage9/campaign_20260930_budget.json
```

默认入口模式仍为 collect，便于单独验证冻结采集。完整恢复绑定资产、配置与执行
源码身份；不能未经核对把服务器 checkpoint 当成在任意本地路径都可无条件续训。

复核服务器已有产物的命令如下，输出路径必须是尚不存在的新文件，不会占用 GPU：

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/eval/audit_closedloop_dppo.py \
  --run-dir /data0/user/liwei/GENMO_outputs/closedloop_stage9/acceptance_20260930_run02 \
  --budget-file /data0/user/liwei/GENMO_outputs/closedloop_stage9/campaign_20260930_budget.json \
  --output /tmp/stage9_audit_$(date +%Y%m%d_%H%M%S).json
```
