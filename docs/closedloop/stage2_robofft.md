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
