# Stage2 连续滚动 RoboFFT-D

本实现位于 `feature/bumi-continuous-rolling`，开发工作树为
`/home/weili/bumi-closedloop-worktrees/rolling/GENMO`，起始 HEAD 为
`40f8067750be39e51827ba10aa236d0318ab4b25`。入口是
`tools/train_closedloop_stage10.py`，包路径是 `gem/closedloop/`。
Python、八卡 Bash 包装和 supervisor 均要求显式 `--config`；漏传会在创建
CUDA/Isaac 上下文前失败。RoboFFT 源配置为 `configs/closedloop/stage2_continuous_rolling_robofft.yaml`，
从 `stage2_continuous_rolling_train10.yaml` 复制并仅更换 Actor 优化相关设置。

**首次代码验收：CPU 数学、真实小型网络、双进程 Gloo、fake world 接线及恢复通过；
候选参数不代表最优。实际部署进展另见文末记录，物理收益须独立验证。**
首次代码验收没有停止训练；用户随后授权八卡正式训练，最终更正目标为服务器1，
并明确要求服务器1若有训练则停止旧任务后启动 RoboFFT。
整个过程保留共享工作树分支及旧实验目录。

## 官方依据与适配边界

2026-10-23 阅读官方 `release` 分支以下文件。下列 SHA 是文件 blob 的追溯信息，
不参与本项目恢复门禁：

| 官方文件 | blob SHA |
| --- | --- |
| [diffusion_fpo.py](https://github.com/shshsunny/RoboFFT/blob/release/robofft-d/model/diffusion/diffusion_fpo.py) | `24d6d241a38710eeb4e58474edc61edb7c3d75f4` |
| [train_fpo_diffusion_agent.py](https://github.com/shshsunny/RoboFFT/blob/release/robofft-d/agent/finetune/train_fpo_diffusion_agent.py) | `bb39996350fc2b9968932f65d78901235b7c654e` |
| [robofft_diffusion_mlp.yaml](https://github.com/shshsunny/RoboFFT/blob/release/robofft-d/cfg/robomimic/finetune/square/robofft_diffusion_mlp.yaml) | `b63f1f4710f98bac018c0b6ffe88d676e123859b` |

采用官方 `use_per_sample_ratio=false` 的聚合次序、Delta log-SNR 权重、
logratio clamp 和负优势折扣。GENMO 是 START_X 预测器，因此使用同 CFG 的 x0
预测转换成 epsilon；没有改成 Flow Matching，没有学生约束。
官方单个 logp 的 `[-200, 2]` 范围在该路径中用于诊断，并不实际裁剪单个 logp；
实际数值裁剪作用在每个 MC 的新旧 logp 差 `[-10, 10]`。

## 保持的生成、物理和价值合同

- 60 帧、30 Hz、normalized qpos30；contact2 独立，不进入代理概率。
- Stage1 结构、统计量、音乐、历史和已知前缀保留；CFG=2.5。
- 20 步采样：索引 0..8 使用冻结网络，9..18 使用当前 Actor，19 使用冻结网络且确定性。
  原 `std_schedule`、eta 和探索噪声不变，末步网络没有解冻。
- GMT 冻结，控制 50 Hz、物理 200 Hz；连续滚动生成、真实延迟合同及未来参考替换不变。
- `returns.py` 保留按实际控制步数计算折扣和 GAE；每个控制步的已积分奖励只计一次。
  真实终止不自举，时间上限自举，跨 episode/行为版本切断。
- `reference_credit.py` 只审计首次前瞻可见、中心消费、奖励参考的时刻，不重新分配奖励。
  优化不推进或重置连续世界，不给旧计划重标行为版本。
- 源配置的 `stage2.gmt8_reward_std.v4`、Critic 结构、预热、优化参数及质量门禁不变。
  BC 和 best-policy anchor 关闭；当前实现明确拒绝将它们重新接入 RoboFFT。

`test_configuration_preserves_physical_critic_rewards_and_removes_gaussian_controls`
直接比较两个解析后配置的物理、时序、奖励、Critic、资产路径、采样噪声和 GAE 设置。

## 动作与固定 MC 的数据流

`DPPODiffusionPolicy.sample_rollout(store_chain=False)` 在最后一步结束、解码之前
固定 `normalized`。`RoboFFTDiffusionPolicy` 将其复制为 `action_x0`。
此路径从采样开始就不分配完整链，不计算旧 Gaussian 概率，不先存链再删。
它与保存链模式在相同条件、RNG 下的输出及 RNG 终态完全相同。

`UpperTransition` 显式保存 `algorithm='robofft_d.v1'`、`action_x0` 和 `mc_record`；
`chain/old_log_prob` 为 `None`，trace 不含旧 mean/std/logprob。专用校验拒绝假零链。
`DeviceRolloutBuffer`、`RolloutTensorCache`、持久化及 Actor validator 按算法分派。
collector 通过原批量生成和转移构造传递最终动作，不从 GMT 状态或后续参考缓存重建动作。

每轮学习前、任何 optimizer 更新前，为所有动作固定 16 组 eps、采样索引、original
timestep 和旧代理损失。三类索引严格分开：

- MC 编号：0..15，只标识抽样副本。
- 采样索引：从 9..18 均匀抽取。
- original timestep：由真实 `timestep_map[sampling_index]` 获取，属于原 1000 步日程。

MC 使用独立 CPU Generator，种子为训练 seed + `mc_seed_offset` + rank 偏移；
逐动作抽样顺序不受 action microbatch 大小影响。完整 checkpoint 保存其状态。
整轮 PPO 包括所有 epoch、重排和微批都复用同一记录，旧损失不随 Adam 更新。
记录使用独立 detach/CPU 副本并检查张量版本；再次生成或原地污染会拒绝。

## 数学目标和归一化

有效未知区域固定为整个生成窗口：

```text
free = future_valid[..., None] & ~known_qpos30_mask
x_t = sqrt(alpha_bar[t]) * action_x0 + sqrt(1-alpha_bar[t]) * eps
```

前向腐蚀后，已知坐标恢复干净 normalized 前缀，padding 置零。
这里的 alpha 来自原 1000 步扩散日程，绝不使用 rollout 的 `std_schedule`。
没有附加已消费中心帧 mask。

新旧代理都以 eval 模式、关闭随机 dropout 的同一 CFG 前向预测 `pred_x0`：

```text
eps_hat = (x_t - sqrt(alpha_bar[t]) * pred_x0) / sqrt(1-alpha_bar[t])
L[i,m] = 0.5 * Delta_logSNR[t] * sum_frames(mean_30(free * (eps_hat-eps)^2))
```

Delta log-SNR 使用当前采样时刻和相邻较干净采样时刻的原始 alpha。
权重内方差与对数分子按官方 `1e-3` 下限计算；此裁剪不改变实际 `q_sample`。
已知/padding 坐标的误差为零，特征维分母固定为 30，时间帧求和。
当前 Actor、音乐/历史/前缀编码器都保留梯度，冻结采样网络不接收梯度。

```text
d[i,m] = clamp((-L_new[i,m] + L_old[i,m]) * logprob_normalizer, -10, 10)
logratio[i] = mean_MC(d[i,m] * logratio_alpha)
ratio[i] = exp(logratio[i])
loss[i] = max(-A[i]*ratio[i], -A[i]*clip(ratio[i], 1-clip, 1+clip))
loss[i] *= beta_neg if A[i] < 0 else 1
```

不先 exp 再平均，不对每个 MC 或每个反向转移分别 PPO clip。优势直接复用原运行时
已全局标准化一次的值。每个 minibatch 按实际全局有效且有未知坐标的动作数除一次，
各 rank 使用 SUM 梯度，不再除 MC、去噪步骤或 world_size。尾批使用其实际动作数；
没有样本的 rank 参与相同 collective 和 optimizer 更新。

两遍流式反传先无梯度求全部新损失，再对动作目标求每个 MC 的系数，随后逐 MC
微批重算并反传。它与完整单遍图的链式法则一致，并限制同时存活的激活。

逐 optimizer.step 的诊断覆盖**当前全局 minibatch 与固定哨兵的去重并集**，
不再重算整个 rollout。哨兵是本轮全局有效且有 free 坐标的 manifest 顺序中等距选取
的最多 64 个动作，跨 epoch 不变、不消耗环境/MC/重排 RNG。诊断记录 scope、实际
覆盖动作数、哨兵与当前小批的重叠数及耗时；这些样本不额外进入目标或梯度分母。
初始化和整轮结束各做一次完整有效动作诊断，最终指标仍覆盖全量。
可选 surrogate 早停使用逐步并集诊断的全 rank 同一归约值，不能将该子集指标冒称
全量或真实 KL；默认仍为 null。空 rank 和尾批参与原有 SUM 与一致错误传播。


`surrogate_kl = mean((ratio-1)-logratio)` 仅是前向代理的诊断量，不是真实策略 KL。
同时记录每 MC logratio clamp 命中率、动作 clip 率及单个 logp 范围命中率。
默认 `surrogate_kl_stop=null`；可显式设置正门限，同步停止剩余 minibatch。
不使用旧坐标 KL 门限、整链 Gaussian KL 或 Gaussian 回溯控制，不自动缩小 LR。

## 候选配置与模块分工

| 设置 | 初始候选 |
| --- | --- |
| MC / MC microbatch | 16 / 1 |
| PPO clip / beta_neg | 0.01 / 0.125；beta_neg=1.0 可显式消融 |
| logprob_normalizer / logratio_alpha | 1.0 / 0.1 |
| Actor base LR | 1e-5 |
| 输出头 / Transformer / 其他倍率 | 1 / 0.1 / 0.1 |
| epoch / 全局动作 minibatch / action microbatch | 1 / 8192 / 8 |
| 梯度裁剪 | 1 |
| BC / anchor | 关闭 |

名义容量从原 8 rank × 128 环境和连续滚动时钟推导；真实更新使用实际 manifest，
不会用名义容量当分母或填充假动作。MC 缓存放 CPU，cache 报告额外 MC 字节数。

| 代码 | 修改目的 |
| --- | --- |
| `gem/closedloop/robofft/surrogate.py` | 可独立核验的腐蚀、x0 转 epsilon、权重和动作 PPO |
| `robofft/policy.py`、`dppo/policy.py` | compact 采样及共享的确定性 CFG 前向 |
| `robofft/records.py`、`dppo/buffer.py`、`memory_execution.py`、`tensor_cache.py` | 最终动作和不可变 MC 数据合同，保留中立 Critic 输入 |
| `robofft/updater.py` | 固定旧损失、两遍流式反传、SUM 梯度及一致早停 |
| `robofft/configuration.py`、`dppo/training_config.py`、`run_control.py`、`training.py` | 显式配置到正式更新分派、独立 RNG、回滚和阶段资源记录 |
| `robofft/checkpoint.py`、`dppo/training_checkpoint.py`、入口 CLI | 新算法恢复 schema、旧 DPPO 仅显式 weights-only |
| `robofft/evaluation.py`、`dppo/evaluation_runtime.py`、`tensorboard_metrics.py` | 固定 compact 动作/MC 漂移和参考变化，指标明确命名 surrogate_kl |
| `dppo/diagnostic_rollout.py`、`rollout_storage.py`、`trainer.py`、`env_adapter.py` | 持久化、collector 转移和验证接线；无完整链强制依赖 |

`dense_policy.py`、`frozen_prefix.py` 的冻结路由以及 `runtime.py`、`world_collector.py`、
`vector_environment.py`、`returns.py`、`reference_credit.py`、`rolling_schedule.py` 的
中立运行逻辑已沿调用链核对并复用，没有重写 Critic 或物理世界。

## 恢复和数值失败

主对比直接从配置绑定的相同 Stage1 权重开始，在新运行目录初始化 Actor/Critic/Adam。
旧 DPPO checkpoint 只接受 `--initialize-actor-weights-only`，只导入兼容 Actor 参数；
不导入旧 rollout、Gaussian 概率、Adam、Critic 或 RNG。
该模式必须新建输出目录，不能与 `--resume`、旧完整初始化或评估参数混用。

RoboFFT 完整恢复要求 `genmo.closedloop.stage2.robofft_d.checkpoint.v1` 和所有 rank 的
独立 MC RNG。模型形状、优化器状态、资产、采样核和实际执行配置仍校验。
初始化来源保留为血缘记录，不阻塞正常完整恢复；源码 SHA 仅追溯。
恢复重新创建物理 episode、重新采样，不复用中途 rollout 或旧 MC 损失。

数值阶段沿用跨 rank 错误传播。任一 rank 失败后，外层整轮事务恢复 Actor、Critic、
两个 Adam、控制器及 RNG；已消耗的物理交互与预算不回退。发布前仍检查所有 rank 的
Actor/Critic/Adam 一致性，沿用原子 checkpoint 与 SHA 指针发布。

## 显式启动命令

以下命令用于具有配置中已绑定资产的服务器独立工作树。本地开发机不具备这些条件。
入口要求八张空闲 GPU、完整资产 SHA、同一 Python 3.11 Isaac/GENMO 环境和新输出目录；
不满足时主动失败，不停止其他作业。将代码同步到独立目标分支并确认没有使用该源码的
正式训练后，再执行。先执行当前 compact 核的实际标定，再启动最多两轮 Actor 试验。
源配置中的 deployment_profile 是历史实测样本；部署前必须重新测量，不能仅修改其
kernel SHA 冒充新标定。正式入口保留行为核/数值执行/资产的实际兼容检查。

第一步只标定，不训练 Actor，输出独立 `deployment_calibration.json`：

```bash
cd /home/user/liwei/continuous-rolling/GENMO
bash scripts/train_stage10_8gpu_server1.sh \
  --config configs/closedloop/stage2_continuous_rolling_robofft.yaml \
  --genmo-repo /home/user/liwei/continuous-rolling/GENMO \
  --gmt-repo /home/user/liwei/continuous-rolling/legged_lab_gmt \
  --output-dir /data1/user/liwei/robofft_runs/stage2_robofft_20261023_calibration01 \
  --calibrate-only
```

第二步复制独立运行配置，只写入实际标定结果；原仓库候选和旧实验保持原样。
现有更新工具检查八个 rank 的核、执行和资产一致、64 个延迟样本均在前缀预算内，
并验证其它配置逐项不变。若超预算，应明确记录失败并分析时序，不伪造样本。

```bash
mkdir -p /data1/user/liwei/robofft_runs/setup_20261023_trial01
cp --no-clobber configs/closedloop/stage2_continuous_rolling_robofft.yaml \
  /data1/user/liwei/robofft_runs/setup_20261023_trial01/stage2_continuous_rolling_robofft.yaml
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python \
  tools/update_deployment_profile.py \
  --config /data1/user/liwei/robofft_runs/setup_20261023_trial01/stage2_continuous_rolling_robofft.yaml \
  --calibration /data1/user/liwei/robofft_runs/stage2_robofft_20261023_calibration01/deployment_calibration.json
```

第三步显式使用该配置进行最多两轮训练；参数仍为本候选，不自动延长：

```bash
cd /home/user/liwei/continuous-rolling/GENMO
bash scripts/train_stage10_8gpu_server1.sh \
  --config /data1/user/liwei/robofft_runs/setup_20261023_trial01/stage2_continuous_rolling_robofft.yaml \
  --genmo-repo /home/user/liwei/continuous-rolling/GENMO \
  --gmt-repo /home/user/liwei/continuous-rolling/legged_lab_gmt \
  --output-dir /data1/user/liwei/robofft_runs/stage2_robofft_20261023_trial01 \
  --stop-after-iteration 2 \
  --capture-validation-iteration 1
```

脚本名称保留历史 `server1`，实际上是两台服务器共用的八卡入口；它在 Python 启动前
设置 Isaac 动态库、八卡可见顺序、NCCL 和线程数，避免直接调用 Python 时缺少动态库。
Critic 原有完整 MC 预热和质量门禁仍执行，原固定物理初始/结束评估仍执行。
`--stop-after-iteration` 限制累计 Actor 轮次，不缩减预热或评估任务；初始试验可能耗时较长。
需要先采一轮真实固定数据时，在同一命令上增加 `--collect-only` 并使用另一个新目录；
此模式固定 MC/旧损失并保存学习输入，不更新 Actor/Critic。这里和下文所指训练命令
均为第三步使用已标定运行配置的命令。

恢复同一试验时使用同一 config/output-dir，增加 `--resume latest`，保留
`--stop-after-iteration 2`，去掉 `--capture-validation-iteration 1`。
累计已到两轮时不追加训练。保存路径由 `latest.json` 和 session summary 给出。

显式 DPPO 权重消融使用另一个新 output-dir，在第三步训练命令上增加
`--initialize-actor-weights-only /实际绝对路径/旧DPPO编号checkpoint.pt`。
这不是主对比，也不是旧实验的完整恢复。评估指定 RoboFFT checkpoint 时可用新目录、
相同 config，增加 `--evaluate-only --evaluation-actor /实际绝对路径/RoboFFT编号checkpoint.pt`。
不要把可变的 `latest` 字符串当作该参数的 checkpoint 文件。

## 实际验证记录（2026-10-23）

本机使用 `/home/weili/GENMO/.venv/bin/python`（torch 2.6.0+cu124），显式隐藏 CUDA。
没有向项目环境安装依赖；测试仅写本次专用 `/tmp` 目录，结束后精确清理。

```bash
cd /home/weili/bumi-closedloop-worktrees/rolling/GENMO
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/weili/GENMO/.venv/bin/python -m pytest -q -ra -p no:cacheprovider \
  --basetemp /tmp/genmo-robofft-acceptance \
  tests/closedloop/dppo
```

复现时先确认 basetemp 是专用测试路径，结束后按 AGENTS.md 精确清理。
实际全量结果：**796 passed、7 skipped，83.82 秒**。
跳过项是 2 个 CUDA 回滚、1 个容量、1 个真实 GPU 时钟、2 个 host-copy 和 1 个
CPU/GPU 反向边界测试。保留一个既有 autocast 弃用警告。
RoboFFT 三个新测试文件最终复验：**11 passed，13.50 秒**，属于全量集合的子集。
另完成 29 个改动 Python 文件 AST 检查、CLI `--help` 及 `git diff --check`。

| 用户要求 | 实际证据 |
| --- | --- |
| old=new 时 ratio=1、surrogate KL=0，梯度存在 | 纯数学精确相等；真实网络微批重排误差按浮点容差检查 |
| epsilon 与 SNR 加权 x0 误差等价 | float64 值和梯度同时比较；另验证权重裁剪不改 q_sample |
| 官方 MC/clamp/negdisc/PPO 次序 | 独立逐项公式比较值和梯度，并拒绝先 exp 再均值的替代结果 |
| 已知/padding、CFG、条件梯度 | 修改屏蔽位置噪声不影响 loss；手工 CFG 对照；三个条件编码器均有梯度 |
| MC 跨 epoch/重排/微批、旧数据不可变 | 两个 epoch、逆序反传、逐动作 RNG 重放、序列化和原地污染拒绝 |
| 单遍/流式、分布式/拼接 | 真实小网络梯度比较；真实双进程 Gloo，含空 rank、尾批、完全一致参数 |
| compact/full 采样等价 | normalized/action_x0/qpos/contact 和 RNG 终态逐元素完全相等 |
| fake world 完整接线和恢复 | 显式配置→真实 collector/环境适配→buffer/GAE→正式 Actor 分派→Critic→原子发布→恢复 |
| 旧 Gaussian fail-on-call | compact 采样、训练分派、更新和固定评估均禁用旧 Gaussian 概率/PPO/KL/回溯 |
| 连续时钟、奖励、版本 | 2 动作共 30 个控制步；优化不推进物理；旧行为版本不变；前瞻/中心/奖励时刻分别检查 |

双进程还验证同一 surrogate 门限的一致早停，以及 rank0 注入数值故障后两个 rank
都得到一致失败。恢复测试验证 MC RNG 连续、Actor/Adam 恢复、源码变化可接受、
缺失 MC RNG 拒绝、旧 DPPO 完整恢复拒绝、显式 weights-only 接受。
旧连续滚动测试中过时的 v3 reward 断言改为当前源配置已有的 v4；未更改奖励代码或数值。
TensorBoard 指标上限从 70 调整为 75，新增五个明确命名的 RoboFFT 指标。

最终 fake world 两动作 CPU 小步实测如下。Actor 时间含 MC 准备、更新和输出检查；
整轮时间还含 fixture 后续持久化、发布、恢复检查，不能当作生产训练吞吐：

| 阶段 | CPU 墙钟时间 | 峰值 CUDA 显存 |
| --- | ---: | --- |
| rollout | 0.100426 s | 不适用，未使用 CUDA |
| Actor | 0.564469 s | 不适用，未使用 CUDA |
| Critic | 0.010340 s | 不适用，未使用 CUDA |
| 完整测试流程 | 0.787611 s | 不适用，未使用 CUDA |

同条件同独立噪声的生成参考变化：normalized RMS `3.757566e-5`，最大绝对变化
`1.458973e-4`；关节参考变化 RMS `3.657779e-5 rad`，最大 `1.123548e-4 rad`。
生成关节参考相对零点的 RMS 幅度从 `0.233999277 rad` 到 `0.233995372 rad`。
这些是完整有效未知窗口的生成参考统计，不是已消费中心、真实关节运动或跟踪误差。
使用候选 LR=1e-5，没有回溯或调低温度来掩盖问题；数值小不构成训练有效的结论。

生产日志已接入 `phase_resources.rollout/actor/critic/round` 的跨 rank 最大墙钟时间和
逐 rank PyTorch allocated/reserved 峰值；Actor 包含 MC 准备。峰值不包含 PhysX 外部分配，
整轮前台时间不包含之后的 checkpoint/固定物理评估，日志明确标注该范围。

## 未验证事项与下一步验收

首次本地代码验收时只检测到一张 RTX 4090，已有 python 作业 PID 1133978 使用 GPU；配置中的
`/data0/.../env_isaac_genmo`、Stage1 权重和 IsaacLab 目录不存在。当时未连接或修改训练
服务器，当时空闲资源及资产状态尚未核验。后续服务器部署证据另见下文。
完整 Actor 训练轮次的 GPU 耗时/峰值显存、生产权重更新的数值稳定性和
8192 动作实际吞吐，仍须等待新训练记录，不能用标定前向替代。

资源满足后，使用上述最多两轮命令，从相同 Stage1 开始，与固定音乐/种子的原基线
比较原物理报告中的真实跟踪误差、失败率、关节活动/幅度、奖励各项及完成时长；
并结合新固定噪声参考变化，区分参考更活跃与机器人实际跟得更好。
本次未获得真实跟踪误差或失败率，不用 fake world 的零误差或代理 KL 替代这些结果。

## 服务器2部署历史（2026-10-23，任务随后按用户更正取消）

用户在代码交付后明确授权正式八卡训练，并要求停止服务器2已有训练。新代码同步到
`/home/user/liwei/continuous-rolling/GENMO`，配套 GMT 位于同级 `legged_lab_gmt`；
两者均为 `feature/bumi-continuous-rolling`。服务器旧工作树没有切分支或覆盖源码。
Stage1/GMT/kinematics/stats 五项资产 SHA 全部通过，服务器 CPU 专项为
11 passed、1 项既有弃用 warning，耗时 13.52 秒。

旧任务 `stage2_ddim10_last5_motion1s_formal_20261023` 的自动恢复器先暂停，8 个 worker
收到合法整轮边界退出请求。旧代码随后在参考变化诊断触发 CUDA 索引越界，该轮未发布；
17:35:24 全部 worker、launcher、恢复器均确认退出。保留原日志、模型与最新完整
iteration 0 checkpoint，大小 931772507 bytes，SHA256
`ba933b156ebbaeadd54d9f2b003171977f4a8e3af465cb7d5a7cc2402dd750ac`。
这次停止包含故障退出，不宣称旧任务的未完成更新已正常保存。

部署证据集中保存在服务器
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_20261023_setup`。
初始候选配置只复制到该目录，未导入旧 DPPO 的 Adam、Critic、rollout 或概率缓存。
两次原样八卡标定各 64 个样本，分别有 2 个超 0.32 秒预算，最大值为 0.682396 和
0.720339 秒；正常会话退出不等于延迟预算通过，profile 更新工具拒绝首轮数据。
没有筛除慢样本，也没有提高预算或降低学习率。

只读 GC 观察器进一步确认：rank 2/6 在冻结模型身份核验处触发第 2 代垃圾回收，
各回收 39208 个对象，耗时 0.438554/0.451320 秒，对应请求为 0.677289/0.699498 秒。
观察器没有关闭 GC 或修改随机数，这组带诊断开销的测量不进入正式 profile。
`training.py` 因此在模型、Isaac、配置全部初始化完成后、首个物理请求前执行一次完整
回收；标定和训练共用此边界，session 报告记录各 rank 的对象数、耗时及自动 GC 状态。
自动 GC、真实请求计时、物理时钟、奖励、冻结模型校验和训练超参保持原样。

修正后无观察器标定通过：64 个样本全部在原 0.32 秒预算内，最小 0.225769 秒、
中位数 0.236210 秒、平均 0.236199 秒、最大 0.253676 秒，8 rank 正常结束。
一次性启动回收为 0.426954–0.714729 秒，各 rank 自动 GC 均保持开启。
报告位于 `stage2_robofft_calibration_20261023_startupfix03/deployment_calibration.json`；
运行 YAML 相对候选仅实测 `timing.deployment_profile` 不同。

17:53:49 曾启动服务器2的新任务 `stage2_robofft_formal_20261023`。用户随后更正为
服务器1，17:55:37 先暂停自动恢复器并请求停止；由于仍处于 Critic 预热的初始物理采集、没有任何
Actor 更新，校验完整初始 checkpoint 后终止本次启动及评估子进程。
17:58:37 确认全部 worker/评估子进程退出、服务器2 GPU 进程为空，自动恢复器随后退出。
初始 checkpoint 保留（iteration 0、Actor/Critic updates 均为 0），大小 914459819 bytes，
SHA256 `607259039e6b86c13e44d2492ea0ac8e89ffc33279b32f0b9596658e3bbca6f9`。
该任务的取消不是数学或 Actor 更新失败，不将部分预热采集视为已完成固定物理验证。

## 服务器1正式部署（2026-10-23，最终目标）

服务器1为 ZP-NC578，SSH 公网端口为 50030。用户已明确授权停止其旧训练后启动
RoboFFT；使用已有独立 `continuous-rolling/GENMO` 和 `continuous-rolling/legged_lab_gmt`，
不切换共享主工作树分支。服务器1的五项模型/统计/运动学资产 SHA 全部与候选匹配。
等待旧训练整轮退出期间，使用 GitHub 已提交 `125833f` 的隔离临时源码执行 CPU 专项：
11 passed、1 项既有 warning、13.79 秒；原活动源码 HEAD 仍为 `e95562e`，临时目录已删除。

新配置与操作证据集中在服务器1
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_20261023_setup`。
新实验从原 Stage1 s595000 初始化，保持 MC16、clip .01、beta_neg .125、Actor base LR
1e-5、倍率 1/.1/.1、全局动作 minibatch 8192、action microbatch 8；不继承旧 DPPO
学习状态。服务器2的延迟样本不会冒充服务器1实测，需要在服务器1重新标定。

服务器1旧训练于 18:03:11 确认正常退出（退出码 0），第 76 轮完整保存，累计
Actor updates 303、Critic updates 32672。最终完整 checkpoint 为
`stage2_rolling_formal_20261022_182720/checkpoints/stage10_r000076_a000303_01792749730840219957_87b4f561e8e3_periodic.pt`，
2633464025 bytes，SHA256
`ff4e835d82f9540b2e85916c3eaa92869a1e9f286f62a3e37cdce0d5974a463a`，已逐字节核验。
旧日志和模型保留。确认八卡空闲后才把独立 GENMO 工作树从 `e95562e` 快进到
`125833f`，配套 GMT 仍为 `12c5a080`；18:03:14 启动服务器1实际八卡标定。

服务器1标定报告正常结束，64/64 样本满足 0.32 秒预算：最小 0.227651 秒、
中位数 0.232755 秒、平均 0.232853 秒、最大 0.237989 秒。八个 rank 的核、数值执行和
资产一致，profile 更新工具 exit 0；配置逐项比对确认仅部署 profile 变化。
实际运行配置 SHA256 为
`122941a9c85f921dec576173812e0256c894de4ba5592e7a6ce83c115c81325a`。

18:05:18 在**服务器1**启动正式任务：监督器 PID 3171991，torchrun PID 3172071，
8 个 worker PID 3174681–3174688，累计上限 200 轮。运行目录为
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_formal_20261023`，
监督状态和主日志位于同名加 `_supervision` 的目录。首次启动没有 `--resume` 或旧模型
初始化参数；之后监督器仅从这个新实验自身的完整 `latest.json` 恢复。

本次正式监督器实际命令如下（任务已经启动）：

```bash
cd /home/user/liwei/continuous-rolling/GENMO
task_run_root=/data1/user/liwei/GENMO_outputs/closedloop_stage10
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/run_stage2_training_supervisor.py \
  --config "$task_run_root/stage2_robofft_server1_20261023_setup/stage2_continuous_rolling_robofft_server1.yaml" \
  --output-dir "$task_run_root/stage2_robofft_server1_formal_20261023" \
  --status-dir "$task_run_root/stage2_robofft_server1_formal_20261023_supervision" \
  --stop-after-iteration 200
```

监督器内部使用前述八卡 shell 入口，保留 Critic 完整 MC 预热、质量门禁和固定初始
物理评估。启动时的 worker/GPU 存活只证明任务已运行，不能替代完整 Actor 更新验证。
首轮完成后，从 `metrics/` 中读取 `phase_resources.rollout/actor/critic/round` 的耗时和
各 rank PyTorch 峰值显存；真实跟踪误差、失败率、活动幅度及参考变化由后续固定物理
评估报告提供。截至本次启动，尚未生成完整 RoboFFT Actor 训练轮次和物理收益对照。

18:08:34 启动核验：正式 `training_identity.json` 的目标确认为
`robofft_d_action_clipped_ppo`，初始化为 `source_stage1/global_step=595000`，MC16、
BC/anchor 关闭，8 个 worker 均存活，监督器 attempt 1，日志无 traceback/OOM。
初始 checkpoint 已发布（iteration 0、Actor/Critic updates 0），正在
`mc_calibration/preheat_val` 的真实物理采集阶段。八卡当时使用约 13.6–14.1 GiB，
这是 `nvidia-smi` 当前占用，不能当作 Actor 训练峰值。完整训练轮次统计仍待生成。

启动内存修正 `125833f` 的 CPU 验收分两批完成：725 passed、7 CUDA skipped 后，
历史 fork 用例等待，独立新进程执行该用例及剩余 70 项得到 71 passed（15.53 秒）；
完整覆盖 796 passed、7 skipped，不将它写成一次无中断全量通过。此前首次数学接线
提交的 796 passed/7 skipped/83.82 秒是另一轮已完成验证，二者不累加。

## 2026-10-23 审查修正与 MC 微批对照

`updater.py` 已按前述策略移除每小批后的整轮重复前向；有界诊断与旧全量诊断
在相同 MC、动作、优势、参数、Adam 和 epoch 重排起点上的所有最终参数/Adam 状态
逐项完全相等。专项21项通过（20.09秒）；扩大集合812项通过（91.23秒），既有fork
单用例在CUDA不可见的独立进程中1项通过（3.14秒），总覆盖813项。该fork用例此前
在CUDA可见时等待，已精确清理本次测试进程与临时目录，不计为一次无中断全量通过。

真实GPU微批对照使用下面的有限工具。在目标GPU空闲时执行；输出指定临时目录，
记录结果后按仓库约定清理：

```bash
cd /home/user/liwei/continuous-rolling/GENMO
task_bench_dir=$(mktemp -d /tmp/robofft-mc-benchmark.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/benchmark_robofft_mc.py \
  --config /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_20261023_setup/stage2_continuous_rolling_robofft_server1.yaml \
  --gmt-repo /home/user/liwei/continuous-rolling/legged_lab_gmt \
  --device cuda:0 --actions 8 --repeats 3 --output "$task_bench_dir/result.json"
```

每种形状先预热一次再测三次；共享真实 s595000、同一组真实四来源条件、最终
compact动作、16个MC、旧损失和人工非零优势。测量范围是生产两遍目标反传，不含
Adam、MC准备、rollout、Critic或物理；分配/保留峰值也不包括训练中的Critic/PhysX。
工具检查完整可训练参数梯度与代理数值一致，以及旧数据、MC RNG、网络参数未变。
这项计算对照不证明真实跟踪质量，正式训练的四阶段资源与固定物理评估另行报告。

### 服务器1真实 GPU 对比结果

2026-10-23，在服务器1原任务完整退出且八卡空闲后，使用 RTX 6000D、Torch
2.7.1+cu128、native_fp32/sdpa_math/stable_tf32x3，action microbatch=8、M=16，
每个 MC 微批预热一次、实测三次。源权重为 s595000，与尚未进行 Actor 更新的正式
任务相同。服务器1另行复验21项专项全部通过（19.96秒）。

| MC microbatch | 三次耗时（秒） | 中位数（秒） | 峰值 allocated / reserved（GiB） | 相对微批1的梯度 L2 误差 | 最大梯度绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 1.384169 / 1.390278 / 1.377987 | 1.384169 | 3.652101 / 3.763672 | 0 | 0 |
| 2 | 1.172417 / 1.172774 / 1.170688 | 1.172417 | 4.792527 / 5.039063 | 2.210441e-7 | 2.123415e-7 |
| 4 | 1.083532 / 1.084375 / 1.083379 | 1.083532 | 7.168184 / 7.441406 | 3.429036e-7 | 3.650784e-7 |

三组逐MC损失最大差为0；同一旧/新Actor得到ratio=1、surrogate_kl=0，动作目标
均为-0.38188130791298674，参考梯度L2范数26.7283690484，确有非零梯度。
参数、MC RNG、旧损失均未改变。微批4的这段反传比微批1耗时低约21.72%，同时
增加约3.52GiB激活等分配；这不是完整Actor轮或八卡物理训练吞吐。
本次恢复保持原实验的mc_microbatch=1及完整优化配置，只改变诊断调度；微批4
保留为通过数值检查的吞吐候选，尚未在驻留Critic/PhysX的完整生产轮测量峰值。
临时目录 /tmp/robofft-review-gpu-_t603nz0（含pytest/JIT/CUDA缓存/结果）已精确删除。

### 安全退出与本次恢复基线

服务器1原本次RoboFFT任务正常exit0，18:53:33核验八卡空闲，自动恢复器为
explicitly_stopped。最终checkpoint为
`checkpoints/stage10_r000000_a000000_01792752707218537419_b3bd450d49f6_evaluated_initial.pt`，
931833163bytes，SHA256 `3dd260e3341af64561c1db2fa3bd69be70146059c6d2dcca3e53a69cf90ad85a`，
已逐字节核验。Actor0、Critic960，保留完整Critic预热、Adam、独立MC RNG和初始固定
评估；恢复仍使用这个RoboFFT实验的latest，不导入旧DPPO，也不重置已完成预热。

预热独立MC验证MSE 369.964944→3.925446、EV -0.001820→0.963265，质量门禁
通过；预热总墙钟1782.991963秒，属于启动校准，不当作常规Critic优化轮耗时。
固定评估32任务全部纳入：29音乐完成、2物理失败（6.25%）、1行政截断、0基础设施
失败，执行2075.38机器人秒/103769控制步/415076物理步。全部控制时间加权跟踪
RMS：根位置0.548527m、根姿态0.842045rad、局部身体位置0.033514m、腿关节
0.110774rad；实际/参考关节活动速度均值1.207329/1.488472rad/s。这些是活动度，
不是关节位置幅度。固定噪声参考变化RMS为0、surrogate KL为0，符合Actor尚未更新。
该基线有失败和行政截断，不能称为全部完成或训练收益；后续须同任务配对比较。
评估墙钟850.184417秒。正式四阶段耗时和峰值仍以每轮phase_resources为准。

本次修复前监督目录保留；恢复使用新的监督目录
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_formal_20261023_reviewfix_supervision`，
以免覆盖旧attempt退出记录。原运行目录、已标定config及累计200轮上限均保留。

### 审查修正后的正式恢复（18:58:30）

服务器1 ZP-NC578 已于2026-10-23 18:58:30（Asia/Shanghai）启动八卡正式恢复，
训练代码提交c0e93a9，启动时HEAD 5623716（后者只追加实测文档）。监督器335404、
torchrun335469、八worker335475–335482；新session为
`4befe6bb-67cc-4d22-be40-8ca0eeec0658`。恢复前实际载入checkpoint核验RoboFFT
schema、mc_preheat_complete=true、8份MC RNG，Actor Adam为空符合Actor0步，
Critic Adam有48个状态条目。配置SHA仍为122941a9c85f921dec576173812e0256c894de4ba5592e7a6ce83c115c81325a。

以下是已执行的实际命令；当前任务已运行，不能重复开启第二个监督器：

```bash
cd /home/user/liwei/continuous-rolling/GENMO
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/run_stage2_training_supervisor.py \
  --config /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_20261023_setup/stage2_continuous_rolling_robofft_server1.yaml \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_formal_20261023 \
  --status-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_formal_20261023_reviewfix_supervision \
  --stop-after-iteration 200
```

监督器生成的八卡命令已带`--resume latest`，没有DPPO初始化参数。19:02:44核验
attempt1、八worker运行，已进入`collections/train_000001/rank00..07`，参考执行
日志持续更新；未重新做Critic预热，未见Traceback、OOM或异常退出。尚无新的Actor
小批记录或已完成训练轮，不把进程存活和首次采集等同于Actor已完成更新。

实时查看（在本地主机执行）：

```bash
ssh -p 50030 user@112.65.216.193 \
  'tail -n 50 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_server1_formal_20261023_reviewfix_supervision/formal.log'
```

首轮完成后，四阶段耗时和逐rank峰值位于原运行目录
`metrics/4befe6bb-67cc-4d22-be40-8ca0eeec0658.jsonl`的`phase_resources`，逐小批
记录的`post_step_surrogate.scope`应为`current_minibatch_union_fixed_sentinel`，
最终`final_surrogate.scope`为`full_rollout`。当前仍未完成更新后的固定物理效果比较。

## 2026-10-24 执行相关时间权重与三轮优化变体

这是 **execution-weighted RoboFFT surrogate**，不是未经修改的原始 RoboFFT。
60帧输入、Stage1/GMT、20步采样核、已知前缀、奖励、GAE和Critic目标都保持原值。
`execution_weights.py`按环境/episode合并来源账本，沿父计划解析复制前缀；已在实际
GMT窗口出现的自由生成帧取权重1，随后0.5秒取0.3，更远取0.1。跨采集边界尚未有
充分观测时，用本计划前缀、提交/到达时间、决策间隔及前瞻预测，明确标为预测。
窗口含前瞻和回看；源帧插值支持不等于完整动力学因果导数。未观察不等于永不消费。
审计分别报告中心生成帧、只在窗口出现的生成帧、被已提交后继新生成区覆盖的未见自由帧。

令`m[t,j]`为有效自由坐标掩码，`u[t]`为1/0.3/0.1原始时间权重：

```text
Z = sum(t,j) m[t,j]
W = sum(t,j) u[t] * m[t,j]
w[t] = u[t] * Z/W                  # 非自由帧权重为0
L_k = 0.5 * DeltaLogSNR_k / 30
      * sum(t,j) w[t] * m[t,j] * (epsilon_theta - epsilon_k)^2
logratio = mean_k(alpha * clamp(-L_new,k + L_old,k, -10, 10))
ratio = exp(logratio)
```

全一时间权重精确退化为原整窗目标；任意整体缩放原始权重不改变归一后的结果。
同一条动作的时间权重、MC噪声、t和旧损失在首个optimizer前固定，所有epoch和新旧
代理共用；时间权重进入MCRecord的不可变检查和固定评估记录。上述ratio和KL仍为
代理量，不是最终动作真实密度比/真实KL，也不保证梯度方差或裁剪率与原目标相同。

候选配置`configs/closedloop/stage2_robofft_execution_weighted_server1.yaml`采用10epoch、
global minibatch2048、beta_neg0.5、MC微批4、每卡16环境、每轮8秒，MC样本数仍16。
主干倍率仍0.1（实际1e-6），输出头1e-5。提高到0.3需要先看到近期参考变化小且代理
裁剪不饱和；不能仅凭代理loss下降决定。减小并行环境是本次三小时时限内的有限实验，
三轮数据量与原128环境/卡不同，不将“三轮”解释为同等采样预算的正式消融。

`--initialize-robofft-variant`在新目录严格核对行为、价值、资产、物理契约，保留两模型
与Adam动量，新计轮、新RNG、新rollout、新MC并重新建立固定32任务物理基线。继承的
Critic预热结论仅限同一Actor/价值契约的warm-start，不冒称新独立MC评估。监督器只在
首次启动导入源checkpoint，重启使用新实验自己的完整checkpoint。执行命令为：

```bash
# Server-1 主机，rolling GENMO仓库内；PARENT须为已核验完整RoboFFT checkpoint。
bash scripts/train_stage10_8gpu_server1.sh \
  --config configs/closedloop/stage2_robofft_execution_weighted_server1.yaml \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_execweighted_20261024_3round \
  --initialize-robofft-variant "$PARENT" --stop-after-iteration 3
```

物理效果以新基线和每轮相同32任务/seed的完成率、失败、每秒奖励、跟踪误差和活动幅度
判断；旧原训练在第28轮正常退出，保留原模型/日志。实际运行结果在记录文本继续追加。

### 三轮真实训练与物理评估结果（2026-10-24）

**结论：执行权重实现和真实更新均通过验证；第二轮出现根部跟踪改善，第三轮整体回报
退化，不建议按当前小采集批加10epochs的配置直接长期续训。** 这不证明执行加权方向
无效，也不能将联合修改的收益或退化单独归因于时间权重、beta或epochs。

本任务11:05:42开始，三小时时限为北京时间14:05:42。旧实验在第28轮合法保存并退出；
新实验11:28:01启动、13:19:06完成，监督器仅一次启动、三轮、exit0。Actor/Critic与两套
Adam继承源第28轮；新实验计数为Actor60、Critic192，实际Actor Adam步数111→171、
Critic12704→12896。三轮分别3062/3325/3180条有效动作，每轮20次Adam、10epochs；
训练前台耗时836.08/911.82/870.44秒（合计43.64分钟，不含随后评估和checkpoint）。
每卡16环境相对原128环境减少了新采样量，本实验不是相同数据预算的严格消融。

每轮均与新实验自己的起点配对，该起点是**旧RoboFFT第28轮，不是Stage1**。任务计划
SHA为`f536995f08b7e9052cc722a4903d48821f4eb5b7aaa576c0fa8521b37ee6f3dc`，包含16首
训练音乐×seed42/1729，共32任务；AIOZ/AIST/FineDance/Mine分别4/2/5/5首。
完整控制证据、任务、seed、资产、核与聚合量经过现有严格比较器复核。每秒回报对
四来源等权，在来源内按音乐聚合两seed做10000次bootstrap；不是留出音乐验收。

| 模型 | 来源等权每执行秒回报 | 相对起点 | 95%区间（相对起点） |
| --- | ---: | ---: | --- |
| 起点：旧第28轮 | 2.435375019 | — | — |
| 新第1轮 | 2.438774445 | +0.1396% | [-0.1101%, +0.4298%] |
| 新第2轮 | 2.435411073 | +0.0015% | [-0.2512%, +0.2489%] |
| 新第3轮 | 2.424663403 | -0.4398% | [-0.6659%, -0.2223%] |

四次评估均为29项music_end、2项物理失败、1项reference_horizon_truncated，所有任务
执行时长逐项相同，总计2075.38秒。两个FineDance115任务仍在144.64秒因非偏航姿态
条件失败；Mine `dance_3__红莲华` seed1729仍为参考时域截断。没有新增失败，也没有
修复原失败。实际关节活动幅度1.206665→1.207992/1.210441/1.209923 rad/s，未靠静止
换取改善。回报区间第三轮整体低于零，是该固定训练集合上的退化信号。

| 实际跟踪RMS（按执行时长汇总） | 起点 | 第1轮 | 第2轮 | 第3轮 |
| --- | ---: | ---: | ---: | ---: |
| 根部位置，m | 0.61245251 | 0.58014077 | 0.54056743 | 0.56774621 |
| 根部整体姿态，rad | 0.84002836 | 0.81255096 | 0.81825953 | 0.76423447 |
| 身体位置，m | 0.03354062 | 0.03350790 | 0.03355616 | 0.03364858 |
| 身体姿态，rad | 0.18320119 | 0.18302806 | 0.18338595 | 0.18363892 |
| 身体线速度，m/s | 0.31318879 | 0.30480064 | 0.31239631 | 0.30780280 |
| 身体角速度，rad/s | 1.95981792 | 1.89241118 | 1.95402539 | 1.92162333 |
| 足高，m | 0.01688669 | 0.01698433 | 0.01688952 | 0.01707631 |
| 腿部关节，rad | 0.11096791 | 0.11075792 | 0.11111175 | 0.11117886 |

第二轮根部位置-11.74%、姿态-2.59%，按共同真实执行前缀做音乐聚类，其95%区间
分别[-14.31%,-3.28%]和[-3.47%,-1.33%]；这是未作多指标校正的探索统计，且各有
18/32和16/32任务改善。第三轮根部点估计-7.30%/-9.02%，但相应区间跨零；身体位置、
身体姿态、腿与足高点估计变差。根部整体姿态并非独立偏航指标，不宣称严重偏航已修复。

第三轮各来源每秒回报增量：AIOZ -0.0347873、AIST -0.0052659、Mine -0.0034116、
FineDance +0.0006184。根部时长加权RMS受到长FineDance片段影响，不能替代来源等权
任务质量。奖励中根部位置与姿态各占跟踪项约7.94%，其余身体/脚/腿和惩罚项也参与；
根部误差下降不意味着总回报同比例增加。三轮采集仅发生1/0/0次物理失败，尚未核验
原失败邻域的训练覆盖，不能认为增大负优势系数已足以学会修复难例；负优势也不只
来自失败动作。

### 真实消费与时间权重验收

三轮合计9567动作的来源统计：当前中心消费79229个原生成帧、只进入GMT窗口37210帧；
每条平均8.28和3.89帧。窗口同时包括前瞻和回看，统计的是源帧支持，不是动力学导数。
三轮未观察到且已被提交后继的新生成区覆盖的自由帧共315280个；预测边界1260条
（13.17%），来源链缺失0条。未观察不等于永不消费，预测边界没有冒充真实消费。

三轮结束后，从同一个旧第28轮源模型另外做了一轮有限collect-only，3062条真实动作、
Actor/Critic/optimizer更新均0，exit0。工具对case、每rank manifest/rollout SHA及所有
MCRecord进行验证，逐计划重算的时间权重与缓存**精确相等**，来源审计也完全一致。
汇总消费统计与训练第1轮相同，但旧训练原始学习张量已按原生命周期释放，不能据此
宣称所有历史输入逐字节重放。

该复采的有效自由坐标共4591444个；近期/中期/远期自由坐标质量1191030/1377900/2022514。
归一后近期损失系数质量占比从25.9402%增至65.7818%，中期22.9543%、远期11.2639%。
这是损失系数质量，不是实际梯度能量或效果占比；不是简单按自由帧数乘30，部分帧
只有少量自由坐标。归一后的总质量仍为4591444。

真实计划示例（帧从0编号）：0～10帧已知，第11帧只有2个自由坐标；12～18帧进入当前
中心，19～22帧只进窗口，12～22取原始权重1；23～37取0.3，其后远期及未观察的第11帧
自由坐标取0.1。归一后的三档约2.71461/0.81438/0.27146，已知坐标始终屏蔽。
这展示了为什么既不能固定取前15帧，也不能只按帧数进行损失尺度归一。

| 训练轮次 | 全量代理比值越界比例 | 近/中/远参考变化RMS，rad |
| --- | ---: | --- |
| 1 | 71.75% | 0.007929 / 0.007629 / 0.008507 |
| 2 | 76.06% | 0.010133 / 0.012771 / 0.009482 |
| 3 | 78.24% | 0.008314 / 0.013073 / 0.012486 |

参考变化是每rank2条、合计16条固定条件及固定独立噪声探针的rank均值，不是机器人
实际跟踪误差，也不代表全体动作。远期仍明显变化，时间权重不保证共享Transformer
的输出变化只集中在近期。越界比例也不等于按优势方向实际被裁剪为零梯度的比例。
三轮全量代理loss均下降，第三轮-0.18964523→-0.19494914，而物理每秒回报下降0.44%；
本次直接证实代理目标改善不足以证明物理收益。由于越界比例已高，没有执行条件性
主干LR1e-6→3e-6，也没有增加到20epochs或更改20步行为采样核。

### 验证、模型和后续选择

- 加权数学覆盖全一退化、整体缩放不变、部分自由坐标质量、已知/padding零梯度、
  固定新旧权重、固定MC拆分、缓存不可变与来源父链；原/新目标双进程Gloo覆盖SUM
  归约、空rank、尾批、多epoch及协作异常，专项10项通过。
- 扩大原回归首次820通过、1项因父进程CUDA初始化后fork失败；该项及相关权重测试
  在隔离进程9项通过，不冒称一次全量无失败。初始化/监督专项19、算法身份/数学
  专项20、逐计划工具11、最终统计工具4项均通过；详细过程保存在`记录文本.md`。
- 真实生产权重MC微批1/2/4的逐MC损失与ratio最大差0，微批4梯度相对L2差3.999514e-7；
  中位反传时间1.381476→1.080531秒。MC样本数仍16；这是计算等价/吞吐验证。
- 三轮及零更新复采均正常退出、源码不变、GMT冻结且世界关闭、无pending；最终GPU
  计算进程为空。三个编号checkpoint的SHA、CPU反序列化、模型和两个Adam有限值、
  八rank MC RNG完整性、评估Actor指纹绑定全部核验；没有宣称实际重启续训或部署测试。

| 编号 | 保留理由 | 完整文件SHA256 |
| --- | --- | --- |
| 新第1轮 | 默认奖励best；整体增益区间仍跨零 | `d9aa6ff5eb64bd62dee9817cb06d98180f1228e142924e5a379fada4c3c50ee7` |
| 新第2轮 | 根部跟踪研究候选，整体回报基本持平 | `558559121243b8c4cced269c0753242dd96c0c813ed4907c7c8ecdc4a8e5522c` |
| 新第3轮 | latest/完整结束点，整体回报已退化 | `44981514dcf59651e78c3ef2f6d700771a01d5c75dd0e740661a740afe5a8f22` |

服务器主机结果根为
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_robofft_execweighted_20261024_3round`。
编号文件的完整路径、大小、模型指纹和Adam步数在`analysis/checkpoint_validation.json`。
三轮训练核心为027aef5，运行中只补了7cadafe的只读汇总；结束后ff-only同步059f97a，
包含独立算法标记、逐计划审计与区间统计。独立标记不改变本次损失、采样及超参数；
历史checkpoint内仍有明确execution_weighting身份，不将其当作原始等权RoboFFT。

结果已镜像到本地`outputs/stage2_robofft_execweighted_20261024_review/`，包括严格配对
JSON、跟踪分布、60帧逐计划JSONL、权重质量、checkpoint核验和PNG/SVG图。零更新
复采的临时模型/学习张量及本次测试临时目录已精确清理；保留导出审计、case/manifest
哈希与零更新报告，不声称保留了完整可重放学习张量。旧正式实验与本次全部编号模型保留。

下一步建议保留执行加权与beta0.5，以第2轮作为跟踪研究候选，先做每卡32环境、5epochs、
global minibatch2048的近似相同Actor计算预算对照，让更多新数据替代反复复用；主干
LR仍1e-6。随后单独对照等权与执行加权，补留出音乐和FineDance115失败邻域覆盖，
并记录按优势方向真正生效的裁剪比例。上述属于后续方案，**本次没有自动启动**，也
没有用代理下降或本训练子集结果替代sim2sim、实时部署或真机验证。

本地审阅包为`outputs/stage2_robofft_execweighted_20261024_review.zip`（1512843 bytes），
SHA256 `882d3c25e3d1dd9997d637e10437529911f31a3719071a62af02b40637f7c208`；包内含
23份与服务器逐项SHA一致的结果/图表及额外artifact_manifest，不含2.6GB级完整模型。
服务器对应归档为上述run根的`analysis_review.zip`。ZIP完整性、3062条60帧记录和三轮
32任务结果结构均已检查；图表已目视检查，无本任务临时目录残留。


## 2026-10-24 决定性诊断：代理方向、延迟信用与可执行参考空间

本节根据用户随后提出的两小时诊断要求更新判断。前节的“每卡32环境、5epochs”仅是当时
未执行的建议；现在不把学习率、epoch或长训作为优先解释。普通训练保持停止，诊断使用
独立入口，不更新Adam、Critic或GMT；所有方向评估后恢复父Actor。

### 实验身份与统计边界

- 父模型为旧正式第28轮，SHA256 `3f40d8ef46229c25e99ad77da312bb8c8060bb6b5398bdc62f2fbbf2d487df57`。
  A使用同一批3062条真实物理行为中的512条、每条固定16个MC噪声；完整20步真实采样链仅在
  `--decisive-diagnostic` 中保留，生产RoboFFT紧凑采样不变。
- A分别计算真实GAE优势、全局随机置换优势、随机参数方向和中间十步真实Gaussian联合score。
  所有方向逐模块匹配参数L2，再使用共同半径；分别测试正向和反向。方向是模块尺度匹配的
  梯度诊断，不冒充正常多epoch Adam轨迹。
- 物理队列为4个来源、8首完整音乐、2种子，共16任务，包含FineDance115的晚期失败。
  每卡使用完全相同任务、初态与创新噪声；8份父模型重复基线的逐任务每秒回报差为0。
  来源等权、来源内音乐等权、同曲两种子整体bootstrap，95%区间是小训练队列的探索区间，
  没有多重比较校正，也没有留出音乐泛化或真机有效声明。
- C将原Stage1权重加载到中间十步，使20步全部使用Stage1；保存其世界参考银行。
  对相同计划的未提交自由部分施加高度±1cm、yaw±0.02rad、腿幅±2%、局部平滑10%/20%。
  已知前缀、已执行参考和原prepare/commit/插值均保留。无基线后续参考时显式计数，
  不伪造配对反事实。限位/FK/速度/加速度检查外，另核查50Hz实际消费序列的拼接。
- B独立读取此前父模型完整32任务的SHA绑定控制档案及同次冻结Critic预测；只用31条
  真正终止的完整轨迹计算MC，排除一条行政参考时域截断。它是已实现轨迹的MC诊断，
  不是已积分动作随机性的真实条件期望V。

### 为什么最终动作去噪代理不是当前混合采样策略的精确score

设中间十个随机转移为 pθ(z[k+1]|z[k],c)，末步冻结映射为 x0=F(z[19],c)。增广链的
score为 `s_chain = Σ(k=9..18) ∇θ log N(z[k+1]; μθ(z[k],c), σ[k]²)`，对所有自由坐标
和十步求和。前九步与末步的参数冻结，因而没有显式θ项；但末步对最终动作和奖励的
影响仍通过真实链样本与回报的相关性进入REINFORCE。这里不需要把末步解冻，也不需要
对机器人或冻结GMT反向传播。D使用这个真实联合score；已有坐标均值式DPPO代理未被
拿来冒充它。D仍使用当前GAE，因此score正确不保证有偏或高方差优势能给出正确物理方向。

RoboFFT则从最终动作重新前向加噪，使用 `s_proxy = -0.1 ∇θ mean_MC Lθ(x0,t,ε,w)`。
这个前向腐蚀状态不是那次9/10/1反向采样的中间状态，损失还经过时间加权、MC归一化，
不具备等于混合采样策略边缘score的一般恒等式。末步冻结映射可能放大或消除中间变化，
最终动作去噪误差本身不能证明捕获了这项影响。官方RoboFFT也将其称为surrogate likelihood：
[官方实现](https://github.com/shshsunny/RoboFFT/blob/release/robofft-d/model/diffusion/diffusion_fpo.py)。
真实链比较依据[DPPO的扩散MDP观点](https://diffusion-ppo.github.io/)，本次联合score只用于诊断。

在θ=θold、固定样本且未裁剪处，正的全局alpha/normalizer只缩放梯度，不单独旋转方向；
MC平均降低当前代理估计的噪声，不能把它变成真实策略score。进入exp与PPO clip后，
尺度会改变哪些样本还在贡献梯度。执行时间权重确实改变方向；按总权重质量归一只保留
损失尺度，不提供概率归一化或真实KL信任域。固定新旧时间权重避免目标漂移，但不修复
这一概率解释缺口。

负优势折扣令 `w(A)=A (A>=0), beta*A (A<0)`。有
`E[w*s] = Cov(w,s) + E[w]E[s]`。真实归一化策略score的条件均值为0；当前代理没有该保证，
所以Critic基线误差可能不再抵消，负优势折扣还会令优势权重均值为正。
本次beta=.5，有效优势均值0.178623；全局均值score项的范数约占A梯度17.90%。这是一项
已量化偏置通道，不代表它单独解释全部物理退化。取消折扣后与原A的余弦仍为0.965791；
取消执行加权后的余弦为0.822771。

拟合半区得到的代理score均值方向，在严格排除拟合音乐的56首留出音乐、228个状态上，
投影均值0.0181612，音乐聚类95%区间[0.0118259,0.0247986]。该有限数据检验支持存在
非零均值方向；投影的绝对数值依赖本次模块扰动尺度，不能当作无量纲效应量。

### A：首次正反物理方向结果（跨phase复位混杂，不能作为最终因果证据）

随后发现同Stage1权重跨rank在新参考生效前已有差异，单曲完整回报最大相差7.0293。
下表保留原始计算以供审计，现降级为探索记录；新的每候选独立PhysX世界复测结果另附。
首次八份父模型基线相同，只证明该首次phase可重复，不能证明复用世界跨phase无残留。

父模型来源等权每秒回报2.458002045，16任务14完成、2次姿态保护失败。
下面是相对父模型的完整音乐每秒回报变化，百分比均除以同一父模型回报：

|方向|正向变化|正向音乐95%区间|反向变化|反向音乐95%区间|
|---|---:|---:|---:|---:|
|A 真实RoboFFT优势|−0.2464%|[−0.4468,−0.0461]%|−0.2223%|[−0.5064,+0.0686]%|
|B 打乱优势|−0.3409%|[−0.6282,−0.0536]%|−0.2291%|[−0.7188,+0.2606]%|
|C 随机参数方向|−0.5570%|[−0.7958,−0.3182]%|−0.3981%|[−0.6378,−0.1583]%|
|D 真实Gaussian链score|−0.0937%|[−0.5283,+0.3409]%|−0.0960%|[−0.3555,+0.1635]%|

用 `(R_plus−R_minus)/2` 隔离正反的一阶方向响应，A/B/C/D分别为
−0.00029644/−0.00137395/−0.00195326/+0.00002795 reward/s，四个音乐区间均跨0。
A−B为0.00107751，区间[−0.00215942,+0.00431444]；A−C为0.00165682，区间
[−0.00445566,+0.00776930]；D−A为0.00032439，区间[−0.00843093,+0.00907971]。
因此本次未检出真实RoboFFT方向优于打乱/随机方向，也未检出D稳定优于A。

所有候选仍为14完成/2保护失败。A+活动度点估计+0.2905%，并非通过明显减小动作活动度
获得变化。实际中心参考A+的根位置/姿态/腿关节差分别为0.033346m/0.098425rad/
0.016863rad；D+为0.020064m/0.054658rad/0.011262rad。这是长闭环中累计的消费参考
差，不能等同于固定条件下单次生成敏感性。随机参数方向也能经长闭环放大，因此逐模块
参数范数匹配不意味着动作空间变化相等；固定共同输入的补充检查单列报告。

原始A与D梯度全局余弦0.005694，分模块大致−0.0014至0.0550；A与打乱B为0.334632。
是否能把近正交归因于代理自身，还必须结合分半梯度可靠性与取消beta/权重后的比较，
不能仅凭这一数字宣布真实期望梯度方向相反。

历史72%–78%日志是ratio越界比例，不是零梯度比例。真正被clip置零的是
`A>0且ratio>1.01` 或 `A<0且ratio<0.99`；另两个越界方向仍有梯度。
在本次512条新父模型行为和新固定MC上核对三个历史模型：

|历史模型|ratio越界|实际clip零导数|
|---|---:|---:|
|第1轮|67.1875%|65.2344%|
|第2轮|65.8203%|58.7891%|
|第3轮|67.7734%|53.9063%|

Autograd逐样本导数核对与上述符号规则一致。此表不是历史原rollout/原MC重放，不能用它
倒推原72%–78%中的确切零梯度数量。代理改善、代理clip率以及真实物理信任域仍是三件事。

### B：延迟、Critic与长期失败覆盖

6927条计划的时间线及后续真实中心参考奖励已导出；6813条获得中心奖励消费。
典型生成到达/提交/首次生成自由帧进入GMT前瞻均为0.24s，首次成为中心参考为0.32s。
逐控制来源守恒核对后，新计划在自己的转移内中心奖励消费为0，后续转移为2063.476秒；
这说明需要后续回报信用，不说明本转移完全没有因果影响：新参考在0.24s进入前瞻，
仍能影响0.30s转移结束前的GMT控制。没有把奖励整体平移0.3s。

当前returns.py按实际50Hz步数换算γ/λ；终止无bootstrap，时间截断有合法next_context
才bootstrap，GAE续接还检查同环境/episode/policy/tick。未发现明显mask或单位实现错误。
Critic有已知前缀、1秒跟踪历史和剩余音乐信息，但没有完整已提交参考缓存、pending计划
与来源时序。已知前缀不是完整未来缓存，存在不同承诺未来被映射为相似价值输入的可能。

独立完整MC比较如下。这里误差是预测减去已实现MC，完整31任务6745状态：

|状态组|状态数|Critic偏差|Critic RMSE|8秒GAE回报RMSE|
|---|---:|---:|---:|---:|
|全部|6745|+0.0458|1.3997|0.8882|
|根位置误差最大10%|675|+1.3148|1.7518|1.2982|
|采集切分前1秒|702|−0.0232|1.2787|1.2426|
|失败前5秒|34|+6.7847|8.0067|0.5303|
|音乐100秒以后|1268|+0.6924|2.3073|1.2848|

最差根误差组全部来自FineDance102两种子，误差阈值1.1532m；失败组来自FineDance115
两种子。前者不能被表述为多首音乐普遍结论。8秒切分附近GAE优势与完整MC减同一V的
相关系数0.2414、符号分歧42.17%；全状态为0.7792和23.83%。失败已被完整采到时，
GAE优势相关0.9977、符号分歧0，lambda=1的终止片段回报与MC达到数值一致。
这说明不能笼统声称GAE丢失所有失败信用，主要问题是bootstrap边界、价值输入和覆盖。

γ_upper=.95是每0.5秒折扣，半衰期6.7567秒；5/10/20/100秒的折扣质量分别
0.5987/0.3585/0.1285/0.0000351。当前训练episode上限20秒；本次真实采集512个状态中
48个处于音乐100秒以后，但最大连续物理历史只有7秒。随机晚音乐相位重置不能重现
连续执行100秒积累的漂移，不能将“采过晚相位”视为“覆盖长期失败状态”。


### 补充验证：梯度噪声、真实链KL与score均值的阴性对照

真实Gaussian旧核均值重算最大差0，20步零更新最终动作重放最大差0；D确实对应本次
实际混合链，而不是仅有可反传的任意代理。不过512条数据分为两半后，A两半余弦
只有0.010375，D为0.008299。取消beta折扣后的代理与D余弦0.007811，同时取消执行
加权为0.023058。由此不能把近正交直接解释为两个真实期望梯度的差别，估计噪声本身很大。
该结论限定于512条诊断数据，不冒充已经测量过完整生产minibatch的梯度信噪比。

全部候选的逐模块匹配更新L2约0.00089226034。相同八条条件和全部创新噪声下，真正
消费的腿关节RMS变化：A+0.00035030rad、B+0.00039139rad、随机C+0.00000188rad、
D+0.00032384rad。随机参数方向的动作变化小约两个数量级，参数范数相等不等于功能
变化相等；随机方向比较只能按原设计解释。父模型相对Stage1的同条件腿关节RMS差
为0.00391645rad，A+为0.00380763rad；保真变化和物理收益仍不能互相替代。

在A+所在rank的64条旧行为上，代理logratio均值0.00086960、标准差0.00404207，
而十步真实链的条件KL均值4.54361，链联合logratio均值−4.03763，二者相关仅0.11133。
这直接说明代理clip不是该真实链的概率信任域。这里的KL是沿旧链状态估计的条件
Gaussian KL总和，不宣称等于最终动作边缘KL；不同rank的KL各自使用本地64条，
不能用其差值做严格同状态方向排名。

为排除有限差分高阶项，又逐动作计算了精确Autograd score投影。56首严格未见于拟合
半区音乐的228个留出状态上，代理投影均值0.00181640，音乐95%区间
[0.00116846,0.00246980]；该方向使用最终模块半径，和前面有限差分报告的单位不同。
**但真实Gaussian阴性对照同样给出非零投影**：均值−0.374427，区间
[−0.664152,−0.081621]。因此，这批单次有限行为数据及音乐bootstrap不足以作为
“RoboFFT独有的总体非零score均值”的决定性统计证明。保留原始逐状态投影，明确区分
数学上没有零均值保证、样本分解中的17.9%均值项，以及尚未确证的总体系统偏置。
下一步应固定条件独立重采多条旧策略链，以独立噪声重复校准该检验；不能忽略对照失败。

### 严重错误处的奖励信号

原v4根位置/姿态std分别为0.3m/0.4rad，两项各占跟踪权重的7.9365%。独立逐控制
重算发现：全轨迹根姿态评分低于1e-4的时间占比15.84%，100秒以后54.40%，失败前
5秒100%；失败前该项均值仅2.38e-7。根位置在100秒以后低于1e-4的时间占比30.96%。
严重错误稍有改善时，这两项常仍接近0，而其他相对身体、腿部及存活项仍有正常量级。
这不是“所有优势都为0”，但解释了为何总回报容易忽略最关心的严重根部误差。

FineDance115两种子在144.64秒触发non_yaw_anchor_ori保护。最后根高误差仅约
0.00161/0.00319m，原始根姿态整体误差约2.014/1.761rad；不能把这两个保护终止直接
描述为已经倒地。根姿态评分分别约9.67e-12/3.79e-9。长期状态覆盖、非饱和根部
纠错信号和可用的优势估计，比单纯多跑epoch更直接对应这类失败。


### 最终复核：A 独立世界重测及早期观测一致子集

首次诊断为节省启动时间显式启用 persistent_world=True，导致跨阶段重置混杂；这是本次诊断设置，原正式训练 persistent_world=False，不将它归因于原训练。随后每个候选独立创建世界，完整跑完八方向各16个音乐/种子任务，保存 report 与 causal_references。该进程在 A 全部保存后行政停止，未运行任何训练更新，不能将整个 fresh_retry 会话标记为正常完成。

重测仍有部分任务在参考生效前十二个控制步出现微小差异，因此全16任务的均值和区间保留为探索结果。共同筛选八方向奖励和八项误差在该窗口完全相同的任务，获得10任务、6首音乐、4数据源；筛选不依据候选收益。它保证记录的早期观测一致，并非全部 PhysX 隐状态的完整快照证明。部分来源只有一首音乐，音乐bootstrap对这些来源不产生音乐间变异，区间存在小队列局限。

定义方向响应 G=(J(theta+delta)-J(theta-delta))/2，单位为每秒奖励差，未除以参数半径。结果：

|方向|G|按音乐聚类95%区间|
|---|---:|---:|
|A 真实优势 RoboFFT|+0.0007983|[-0.0004930,+0.0020896]|
|B 打乱优势|-0.0036885|[-0.0044399,-0.0029370]|
|C 匹配参数范数随机方向|-0.0034221|[-0.0042150,-0.0026292]|
|D 十步真实 Gaussian score|-0.0019391|[-0.0030277,-0.0008505]|

同一筛选子集 A-B=+0.0044868，区间[+0.0029870,+0.0059865]；A-C=+0.0042204，区间[+0.0031339,+0.0053069]。D-A=-0.0027374，区间[-0.0051173,-0.0003575]。A 正向相对原模型仍为 -0.0023677 奖励/秒，区间[-0.0033772,-0.0013581]，失败数未改善。因此该局部实验支持优势中有信息，不支持当前正向步长已经获得绝对收益，更不支持直接换成 D 就能改善。全16任务的 A-B 区间跨零；结果对配对质量/任务范围敏感，不能以子集结论宣称普遍有效。

可复算命令：`python -B tools/summarize_decisive_strict_subset.py outputs/stage2_decisive_diagnostic_20261024_review/physics_A_fresh`。完整JSON同时保留八项跟踪误差、完整奖励、每秒奖励、失败、活动度、音乐对齐和真正消费的根位置/姿态/腿关节变化。正式继续训练门槛应为独立完整音乐上的绝对物理收益及可靠零扰动复测，不能仅要求优于随机方向。

### 算法结论与后续优先顺序

RoboFFT 的前向加噪最终动作去噪评分没有归一化策略密度的 score 恒等式；它也不是实际中间十步反向转移的联合 log 概率。冻结首九步和末步不妨碍对中间随机转移用 likelihood-ratio 估计梯度，但最终动作代理没有自动补上该链式分布关系。D 的似然比针对扩展的潜变量链，仍非显式最终动作边缘密度；使用有误差的优势时 D 同样可能失效。

alpha=0.1 和正的统一MC质量归一化在未clip处主要改变梯度尺度，不能单独据此宣称方向错误。执行时间权重改变目标及方向；clip按优势符号截断；beta<1改变正负优势相对权重。三者合用会改变样本贡献，但本次取消beta/时间权重并未恢复与D的稳定一致性。不能继续以“代理loss下降+ratio在范围内”充当执行质量与真实信任域的证明。

建议顺序：先让同一原Actor零扰动的独立物理重放通过，再完善Critic的已提交未来参考及待提交状态，增加真实连续长音乐和失败前状态覆盖，用完整MC局部校准切分边界；随后为严重根误差增加仍有分辨率的奖励项，并单独版本化，旧八项误差继续原样报告。完成合法参考扰动的零对照后，才用消费参考功能变化或真实链KL校准扰动半径，再决定保留代理还是使用Gaussian链score。当前证据不支持再直接扩大epoch、学习率或盲目续训。


### 两小时时限内的 C 最终状态及交付边界

C 首轮完成了 Stage1 与八种参考扰动各16任务，但跨阶段初始状态复现未通过，且在真实50Hz中心参考上发现额外拼接速度/加速度，不能作为“合法扰动改善GMT”的证明。新增六帧Hermite前缀连续过渡、组合扰动FK/关节边界/速度检查、八音乐单种子及零扰动对照；一次因train评估purpose不符合合同而被入口拒绝，已修复并通过9项数学测试。最终复测在用户两小时时限前行政停止，尚未完成基线/候选配对，不能把代码实现或启动成功说成实验成功。

因此三个决定性问题没有全部解决：B完成了来源账本、完整MC和奖励信号审计；A完成真实链、固定MC、模块范数匹配及完整音乐物理采集，但强结论受复现和小队列限制；C仍未可靠证明是否存在更优合法参考。没有以执行后状态监督Actor，没有移动奖励时间戳，没有修改正常紧凑RoboFFT采样模式，没有启动续训，也没有更换GMT。

产物位于 `outputs/stage2_decisive_diagnostic_20261024_review`，含原始分析JSON/JSONL、被降级结果、完整来源路径和SHA记录。服务器的大体积行为与完整链张量继续保留在原诊断目录，归档仅包含复核摘要和分析逐状态数据。新工具、边界解释及实际验证记录已提交到独立rolling分支并同步服务器。


## 2026-10-24 P0修复与因果未来参考Critic（本次两小时任务）

本节继续使用独立rolling工作目录。固定Stage1为s595000，GMT为model_140500，Actor和GMT权重不更新。新增方法没有改变RoboFFT代理、扩散采样核、奖励定义或50Hz半马尔可夫GAE，也没有把执行后物理状态用作动作目标。

### 重复性门槛及审核修正

每个候选独立新建PhysX世界，每个slot只承担一条完整任务，避免reset后的求解器残留与不同slot历史。8音乐×2种子在8个独立进程重复。后端导出初始物理张量、GMT动作/历史、完整权威参考和源缓存的形状/类型/SHA；评估记录音乐SHA、随机种子、初始噪声索引和部署时序。

原P0报告的128个任务在初态、逐步奖励、八项误差、任务终止及步数上全部精确一致，但审核误把latency_seconds（GPU墙钟耗时）作为部署时钟而报失败。modeled_deployment.v3实际仿真由标定profile决定：首动作两个阶段的GPU墙钟分别1.3886和0.9059秒，真实到达均为744tick。已修正为比较实际到达、提交、消费窗口和事件，墙钟仍保留作为性能证据。原失败报告未覆盖，独立重审未剔除样本，额外逐元素检查37,632份当前/下一状态上下文和18,880条参考/接触/噪声身份，全部通过。此为当前软件与固定队列的仿真重复性，不是实机确定性或实时部署结论。

### 新Critic具体方法

- 原Stage1音乐与已知前缀编码、50帧跟踪历史GRU继续保留；Actor只接收原Stage1十项条件。
- 从当前GMT权威50Hz参考数组采样未来0.2–1.0秒共41帧，只读取已经提交的有效区域。每帧38维：相对当前根部的位置3、旋转6D共6、线/角速度6、21个关节、来源生成/提交年龄2。无效参考严格置零并mask。
- 10维元信息及mask记录已提交参考剩余时长、保护/源缓存剩余时长、已有pending数量/年龄/deadline、来源切换数及参考版本；到达时间及其置信度预留两维，未知时始终mask，不回填未来实际提交。
- 逐帧MLP加入相对时间，经过有效mask注意力汇聚；元信息单独MLP，结果进入原Transformer时序token和value token。最终投影零初始化，显式继承旧共享权重后初始预测逐元素相等。
- 冻结条件可缓存，可训练历史和未来编码每次重新建立计算图。当前及bootstrap上下文各自校验真实tick和行为policy_version，禁止复用当前缓存冒充下一状态。
- 新context为genmo.critic.causal_tracking90_committed38.v3，完整训练checkpoint为genmo.stage2.future_committed_critic.checkpoint.v1。旧Critic不能隐式恢复成新结构；显式共享权重迁移不继承旧Adam。预热与Actor-only读取同步区分身份，完整RoboFFT恢复仍要求独立MC RNG。

### 同完整MC的公平比较结果

P0基线共有16条完整物理轨迹、2,360个决策状态，无行政截断MC。seed42的1,180状态训练、seed1729的1,180状态测试；音乐相同，因此只是新噪声验证，不是未见音乐泛化。旧网络和新网络从同一第28轮旧Critic权重出发，各384次更新，采样索引和优化设置完全一致，50%均匀、50%困难状态采样。初始旧权重来自Stage2行为，而本次MC属于固定Stage1；因此其原始误差还包含行为分布变化，不能把拟合前后差距全部归因于训练轨迹不足，也不能把本次结果直接当作第28轮Actor下的价值验收。测试集不参与选epoch或早停。独立逐50Hz MC实现与训练实现逐项相等；Actor指纹不变。

| 测试组 | 状态数 | 原旧权重RMSE | 同预算旧Critic RMSE | 新Critic RMSE | 旧GAE/MC符号一致 | 新GAE/MC符号一致 |
|---|---:|---:|---:|---:|---:|---:|
| 全部状态 | 1180 | 1.7966 | 0.3106 | 0.2611 | 81.9% | 76.0% |
| 8秒切分前1秒 | 117 | 1.4183 | 0.2789 | 0.2380 | 65.8% | 58.1% |
| 大根位置误差 | 135 | 3.7235 | 0.2669 | 0.1631 | 87.4% | 84.4% |
| 大根姿态误差 | 144 | 3.6363 | 0.3224 | 0.1554 | 84.7% | 84.0% |
| 失败前5秒 | 17 | 8.1354 | 0.1416 | 0.2407 | 100.0% | 100.0% |
| 执行超过100秒 | 149 | 4.1481 | 0.3161 | 0.1607 | 85.9% | 83.2% |

新网络对大根误差与100秒后状态的预测有改善；清空未来缓存会使这些组的RMSE明显升高，说明新增信息确实被使用。但失败前5秒RMSE从同预算旧网络的0.1416升至0.2407，整体及边界的优势符号一致性也变差，独立困难状态门槛为false。不能宣布Critic已修复，更不能将旧权重到新网络的巨大下降全部归因于结构：旧网络仅补完整MC训练也从1.7966降至0.3106。测试失败前只有17个状态，来自有限失败轨迹，不能以逐状态数冒充独立失败样本量。

保存的新/旧比较权重属于genmo.stage2.future_committed_critic_mc_comparison.v1，仅用于诊断，不是完整训练恢复或自动部署预热文件。本次未恢复Actor训练。

### 合法参考C的实现与验收边界

固定P0逐计划Stage1世界参考及contact，只修改未提交自由区域。候选包括根高度±5mm、yaw±0.01rad、pitch/roll各0.01rad、腿部幅度缩小1%、轻微关节平滑。全音乐绝对时间0.5–2.5秒用五次平滑函数起效，相邻计划根据已提交参考偏移及一二阶导数做12帧五次桥接，绝不从机器人实际执行状态反推动作目标。

真实银行预检发现，直接把Stage1原有轻微关节超限逐点裁回会产生速度折角：916条前20秒计划中38条未通过。因此改为整段自由关节凸二次投影，保持所有已提交前缀，联合限制限位及一二阶差分，不放宽阈值。修改后916条单计划预检全部通过，但它仍不是跨计划物理验证。各干预包含共同的平滑限位修正，收益不能简单归因于单个根姿态方向。

提交前同时用原生产30→50Hz插值检查自由区限位、关节/根姿态增量导数和全身FK（包含足部）。之后还要重建实际50Hz中心参考，检查真正发生的跨计划拼接。消费侧增量上限：关节速度0.6rad/s、加速度20rad/s²；全身位置速度0.25m/s、加速度12m/s²；根姿态速度0.25rad/s、加速度3rad/s²。它们是本诊断的工程验收阈值，不构成实机安全认证。

20秒零回放通过后才运行干预；完整音乐准入要求全部零配对、实际消费合法性和干预前一致性通过；八个原定方向全部进入长期诊断，不按20秒局部收益筛选赢家。完整音乐超过基线失败点的后续参考没有固定银行反事实，须单独计数，跟踪比较仅在共同物理前缀上进行。物理结果待本节后续追加，当前不能据代码实现声称存在稳定收益空间。

服务器产物：

- P0：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_p0_futurecritic_20261024_0841（保留p0/summary.json及p0/revalidated_summary.json）。
- Critic：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_future_critic_matched_mc_20261024。
- C20：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_reference_c20_after_p0_20261024。
- 本地摘要：outputs/diagnostics/server1_p0_futurecritic_20261024。


### 长期失败覆盖与奖励区分度补查

新未来Critic配置的完整MC预热明确从音乐开头连续执行到真实终止（start_distribution: beginning），保留独立训练/验证种子，避免把“从音乐第100秒重新初始化”当作“连续执行100秒后”的状态。Actor原20秒采集与8秒GAE设置未改；此配置尚未用于Actor长训。

P0固定Stage1原奖励检查覆盖35,318个实际50Hz控制步。音乐连续执行超过100秒的4,466步，根姿态RMS为1.992rad，姿态奖励均值1.854e-10，100%低于1e-4；真实失败前5秒共502步，根姿态RMS为1.903rad，姿态奖励均值9.941e-10，同样100%低于1e-4。相同失败区间的腿关节分数仍为0.778。说明严重偏航区域中根姿态项几乎失去区分能力，但不能据此单独证明代理优化失效的全部原因。

本次C保持原奖励不变，先判定合法参考能否改善原始物理误差。后续奖励候选应在大误差处保留可分辨的分数变化，再以独立新世界配对验证；不在当前对照中途改评分，也不因总体EV或代理Loss下降自动放行Actor更新。原始八项分项数据在reward_resolution.json。
