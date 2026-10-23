# Stage2 连续滚动 RoboFFT-D

本实现位于 `feature/bumi-continuous-rolling`，开发工作树为
`/home/weili/bumi-closedloop-worktrees/rolling/GENMO`，起始 HEAD 为
`40f8067750be39e51827ba10aa236d0318ab4b25`。入口是
`tools/train_closedloop_stage10.py`，包路径是 `gem/closedloop/`。
显式配置为 `configs/closedloop/stage2_continuous_rolling_robofft.yaml`，
从 `stage2_continuous_rolling_train10.yaml` 复制并仅更换 Actor 优化相关设置。

**交付状态：CPU 数学、真实小型网络、双进程 Gloo、fake world 接线及恢复通过；
生产权重的 CUDA/Isaac 训练与物理收益尚未验证。候选参数不代表最优。**
本次未停止训练、未切换共享工作树分支、未覆盖旧实验。

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

以下命令用于具有配置中已绑定资产的服务器独立工作树。当前本机不具备这些条件。
入口要求八张空闲 GPU、完整资产 SHA、同一 Python 3.11 Isaac/GENMO 环境和新输出目录；
不满足时主动失败，不停止其他作业。将代码同步到独立目标分支并确认没有使用该源码的
正式训练后，再执行。先执行当前 compact 核的实际标定，再启动最多两轮 Actor 试验。
源配置中的 deployment_profile 是历史实测样本；本次未重新测量，不能仅修改其
kernel SHA 冒充新标定。正式入口保留行为核/数值执行/资产的实际兼容检查。

第一步只标定，不训练 Actor，输出独立 `deployment_calibration.json`：

```bash
cd /home/user/liwei/continuous-rolling/GENMO
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/train_closedloop_stage10.py \
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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python \
  -m torch.distributed.run --standalone --nproc-per-node=8 \
  tools/train_closedloop_stage10.py \
  --config /data1/user/liwei/robofft_runs/setup_20261023_trial01/stage2_continuous_rolling_robofft.yaml \
  --genmo-repo /home/user/liwei/continuous-rolling/GENMO \
  --gmt-repo /home/user/liwei/continuous-rolling/legged_lab_gmt \
  --output-dir /data1/user/liwei/robofft_runs/stage2_robofft_20261023_trial01 \
  --stop-after-iteration 2 \
  --capture-validation-iteration 1
```

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

本机只检测到一张 RTX 4090，已有 python 作业 PID 1133978 使用 GPU；配置中的
`/data0/.../env_isaac_genmo`、Stage1 权重和 IsaacLab 目录不存在。未连接或修改训练
服务器，服务器当前空闲资源及资产状态为待核验。因此真实两轮训练和固定物理评估
均未运行，GPU 耗时/峰值显存、生产权重的数值稳定性、8192 动作实际吞吐为待核验。

资源满足后，使用上述最多两轮命令，从相同 Stage1 开始，与固定音乐/种子的原基线
比较原物理报告中的真实跟踪误差、失败率、关节活动/幅度、奖励各项及完成时长；
并结合新固定噪声参考变化，区分参考更活跃与机器人实际跟得更好。
本次未获得真实跟踪误差或失败率，不用 fake world 的零误差或代理 KL 替代这些结果。
