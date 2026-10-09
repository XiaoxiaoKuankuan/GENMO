# GENMO + 冻结 GMT 的 GPU 多环境第二阶段

本文对应独立分支 `feat/stage10-gpu-vectorized`。开发目录为
`/home/weili/bumi-stage10-gpu-vectorized/GENMO` 和
`/home/weili/bumi-stage10-gpu-vectorized/legged_lab_gmt`。
原工作区、训练模型、日志保持不变。所有数值测试与真实性能测试只在服务器 1 执行；
实际 GPU 测试使用全部八张卡。本文件随验收结果更新，不把未完成项目标为通过。

## 架构和不变项

每个训练 rank 持有一个 GENMO Actor/Critic 和一个独立 Isaac 进程。后者在该 rank
对应 GPU 上建立一个 PhysX 场景，默认场景内有 8 个机器人，并执行冻结 GMT 批量推理。
八个 rank 共 64 个物理环境；每轮每 rank 仍然只交付 20 条真实上层转移，全局 160 条。

各环境有独立音乐采样器、噪声 Generator、状态历史、前缀、参考时间线、episode、
决策时间和执行 journal。GENMO ready queue 合并不同真实环境的请求，实际批量通常为
8、8、4；没有复制环境填充有效样本。每条链仍是 20 步，CFG 沿 batch 合并为两倍。
提前完成的环境继续执行原参考并记录真实反馈，直至共同片段边界；尾段奖励和实际时长
合并到该环境最后一条上层转移，不额外生成动作，不扩充训练 batch。

GPU 场景使用原资产、关节顺序、执行器、观测、动作缩放和终止定义，控制 50Hz、物理
200Hz。关闭原冻结配置禁止的随机化、噪声、课程学习和渲染。通用 `step()` 被显式
禁止；只有闭环 `step_control()` 可以推进，不会偷偷自动 reset。终止快照保留用于
真实反馈；恢复完整训练断点时显式开始新物理 episode，不冒充序列化了 PhysX 内部状态。

| 项目 | 当前正式配置 |
|---|---|
| Actor | 固定 `5e-9`；PPO 2 epochs，最多 4 次实际更新 |
| 全局 optimizer minibatch | 80 条完整链、1600 个内部去噪转移 |
| 学习 microbatch | 32；B=64、128 的数值已通过，但实测没有更快 |
| Critic | 80 次，全局 batch 32，学习率 `1e-4` |
| BC | 每次实际 Actor 更新全局 2 条、权重 0.1 |
| 随机核 | 20 步、eta 0.1、std_floor 0.001、CFG 2.5、joint_sum |
| KL | 软停止 0.015，整轮全量硬限制 0.03 |
| 完整保存 | 每 500 个外层接受轮；初始化、正常结束可额外保存 |
| 独立评估 | 初始化、每 100 轮、正常结束；四来源各 4 条、两个 seed |

所有 old log-prob、old/next value、return、advantage 在本轮更新前固定。GAE 按
env_id、episode_id、连续执行区间和实际执行时间计算，各环境不串接。PPO 后续微批
读取更新后的参数。BC 全局归一化及 SUM 梯度同步不额外乘/除八。最终全量 KL 失败时
恢复 Actor、Critic、两个 Adam、BC 和 RNG；失败消耗的预算不退款。

## 关键实现

- GMT 的 `closedloop/vector_env.py`、`vector_reference.py`、`vector_backend.py`：共享
  GPU 世界、独立状态和历史、GPU 参考查询、显式局部 reset、四个物理子步的完整诊断。
- GMT 的 `vector_service.py`、`vector_journal.py`：共享世界的因果调度和完整执行确认。
  非控制网格上的局部 reset 等待其他活动环境到达正确边界；单环境不会等待自身而死锁。
- `scripts/rsl_rl/bumi4340_frozen_torch_policy.py`：从已核验 ONNX 的原始 initializer
  恢复等价 PyTorch 策略。所有参数/缓冲按原规则使用、全部冻结，没有 GMT optimizer；
  保留原 normalization、历史重排、关节顺序和动作裁剪。此方式无需猜测训练 checkpoint
  对应的策略结构，权重绑定原 ONNX SHA。
- GENMO 的 `vector_collector.py`、`vector_boundary.py`、`vector_runtime.py`：真实多环境
  批量采集、片段尾部、全局预算租约、块级 rollout、原管理器/归档器/完整恢复集成。
- `policy.py`、`batch_execution.py`、`tensor_cache.py`：解除旧两槽位限制，支持相同去噪步
  来自多个不同链；条件每条链编码一次并在当前参数版本内保留梯度。固定数据一次搬到
  GPU，Linear 和门控梯度使用样本独立归约，避免 batch 改变带来的高维概率/梯度误差。
- `vector_metrics.py`：保留各奖励分项、拒绝/超时和真实批量计时。共享批量生成耗时
  只记一次；多线程 host 时间可能重叠，不能相加当成墙钟。额外记录 thread CPU 时间。
- `vector_evaluation.py`：独立 GPU 评估世界，不重置训练中的 8 个环境。评估每 rank
  使用 1 个环境，并明确记录在评估身份中；真实训练仍是每 rank 8 个环境。

## 数值和物理边界

概率门槛维持 log-prob `1e-4`、ratio `1e-3`、独立 Gaussian `1e-8`。
固定真实旧 rollout 在 B=1/8/16/32/64/128 和 CFG 合批下通过零更新检查。
B=128 曾因单个 `gate_msa` 梯度不通过而被拒绝；修复门控反向归约后，全部未裁剪
梯度、参数和 Adam 状态通过原容限。源归档 SHA 在测试前后不变，没有重采样旧链。

冻结 GMT GPU 批量与原 CPU ONNX 在 B=1/2/4/8/16/32、不同输入尺度和置换下通过
`atol=rtol=1e-4`，最大绝对误差约 `2.10e-5`。这证明输入输出对照，不能推出
CPU/GPU PhysX 轨迹逐位相同。

CPU/GPU 固定参考重放的根位置 RMS 约 0.4mm、关节 RMS 约 0.0025rad，参考轨迹
逐元素相同，历史时间/掩码和终止一致。但是 Mine 的奖励 RMS `0.00625` 超过该测试
预先设定的 `0.005`，原报告仍保留失败状态。进一步检查确认同一物理输入的奖励重算
逐字节一致；差异主要来自 CPU/GPU 微小速度差改变离散动作节拍局部极小值。奖励公式
和权重没有修改。必须在新 GPU 物理与计时身份下建立新基线，不能将其收益当成学习收益。

新参考到达时间排除训练专用审计持久化，但保留必要推理、传输和参考转换。两层 journal
均先持久化后 ACK；世界 journal 耗时通过真实区间传递并只扣重叠部分。新协议为
`nested_world_journal_excluded.v1`。有限参考覆盖不足时使用显式
`available_reference_deadline_cap.v1`，实际生成时延和迟到判定不隐藏。这些身份禁止
把旧 CPU checkpoint 当作新 GPU run 的透明完整恢复；旧模型仍可读取和独立评估。

## 已有有限证据

结果根目录：
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/gpu_vectorized_validation_20261009`。
这些完整成功/失败证据按本任务交付要求保留，不进入 Git 或正式训练目录。

| 测试 | 结果及限制 |
|---|---|
| `gmt_torch_exact_history.json` | 八卡冻结 GMT B1..32 对照通过 |
| `worlds_n1`、`worlds_n8` | GPU 状态、局部 reset、物理诊断对照通过 |
| `saved_learning_large_v2.json` | B32/64/128 全部梯度、Adam、参数和原概率通过 |
| `vector_train8_v14_first` / `vector_train8_v14_resume` | 真实 160 条，停止进程后完整恢复第 2 轮通过 |
| `production8_v3_short` | 原正式管理接口的首轮/独立进程恢复/短评估/保存/归档均通过 |
| `condition_graph_v1.json` | 无梯度条件图逐元素一致；收益很小，默认关闭 |
| `cpu_gpu_replay150_v1` | 状态对照通过，Mine 奖励 RMS 门槛未通过，保留原结论 |

固定同一批旧数据、4 次 Actor 更新、不含 BC/物理的实测：B32 为 19.32/19.53 秒，
B64 为 19.54/19.52 秒，B128 为 19.82/19.82 秒；最终完整 KL 约 1.01/1.03/1.06 秒。
峰值显存约 17.5/30.5/54.9 GB（十进制）。大显存说明可以装下更多激活，不保证计算
更快；该严格 FP32 路径在 B32 之后已经没有明显批量收益。

正式接口有限首轮因软 KL 停止只更新 Actor 1 次，核心轮约 52.1 秒；恢复轮更新 3 次，
核心轮约 66.1 秒。之前另一有限恢复轮更新 4 次约 69.3 秒。不同更新次数和物理执行
时长不能直接与旧普通轮 55.95 秒做纯速度比；目前不能宣称 GPU 重构已经加速整轮。
环境数矩阵、最新故障验收和最终选型结果仍在补齐。

## 运行与恢复

服务器 1 对应目录：`/home/user/liwei/GENMO-bumi-stage10-gpu-vectorized`。
入口脚本调用已验证的 Python 环境并配置 NCCL，无需修改系统环境。

```bash
cd /home/user/liwei/GENMO-bumi-stage10-gpu-vectorized
bash scripts/train_stage10_gpu_vectorized_server1.sh \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/你的新目录 \
  --stop-after-iteration 1
```

完整恢复必须使用同一代码、配置和真实已发布恢复点：

```bash
bash scripts/train_stage10_gpu_vectorized_server1.sh \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/同一目录 \
  --resume latest --stop-after-iteration 2
```

`latest` 只指向实际完整 checkpoint。每 500 轮之外的未持久化尾部在崩溃后明确回退，
已耗执行和优化预算不退。不得在运行中 `git pull`；代码变化后应建新 run，不能修改
身份文件伪装成原 run。正式一万轮尚未启动；原额度保留，不自动扩大预算保证一万轮。

吞吐矩阵入口是 `tools/benchmark_stage10_gpu_vectorized.py`。`--mode collection`
固定每 rank 20 条，可配 `--updates` 做真实学习；`--mode capacity` 才允许 512/1024。
固定全局 160 条时，每 rank 同时最多有 20 条有效任务，因此 512/1024 环境的容量数据
只能说明更大任务批次的潜力，不能计作当前 160 条训练的加速。
