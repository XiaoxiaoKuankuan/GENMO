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
提前完成的环境继续执行原参考并记录真实反馈，最多到共同片段边界或仍支持前缀和
GMT前瞻的最后决策点；尾段奖励和实际时长合并到最后一条上层转移，不额外生成动作。
到参考边界的环境保留有效下一条件，明确行政截断并在下轮reset。活动环境遇到连续
迟到、结果不可能赶上已承诺deadline时，同样最多执行到合法bootstrap边界并丢弃
未提交票据。不得在参考耗尽后补零、重复末帧或让整个共享场景继续非法推进；零控制步
行政动作不能计入160条真实转移。此行为绑定`available_reference_fragment_boundary.v3`，
相较旧流程的真实控制量可能改变，必须分别报告，不能冒充等物理工作量提速。

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
- `vector_devices.py`：Isaac使用全局PCI编号，不对Kit子进程施加单卡CUDA掩码；
  每个世界的真实设备UUID必须与所属GENMO rank对应。这样修复Omniverse与CUDA枚举
  不一致导致的启动失败。Kit可能在其他可见卡建立少量上下文，但物理张量及策略
  计算设备明确绑定并核验，不把这些上下文当作额外并行训练进程。
- `vector_reward_math.py`、`vector_reward_adapter.py`：GPU FP64批量计算七项连续奖励，
  原音乐/活动窗、权重和事件逻辑复用；首控制步及每100步独立标量重算完整证据。
  GPU原始时钟在控制步入口clone，避免后续物理推进改变尚未落盘的奖励身份。

## 数值和物理边界

概率门槛维持 log-prob `1e-4`、ratio `1e-3`、独立 Gaussian `1e-8`。
固定真实旧 rollout 在 B=1/8/16/32/64/128 和 CFG 合批下通过零更新检查。
B=128 曾因单个 `gate_msa` 梯度不通过而被拒绝；修复门控反向归约后，全部未裁剪
梯度、参数和 Adam 状态通过原容限。源归档 SHA 在测试前后不变，没有重采样旧链。

冻结 GMT GPU 批量与原 CPU ONNX 在 B=1/2/4/8/16/32/64/128/256/512/1024、
不同输入尺度和置换下通过 `atol=rtol=1e-4`，最大绝对误差约 `3.05e-5`。
输入归一化、原始动作和裁剪后动作都参与检查，八卡模块指纹保持不变。
这证明输入输出对照，不能推出
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

参考行政边界、GPU连续奖励和原生设备绑定各有独立合同。旧GPU实验即使模型结构
相同，也不能通过修改身份JSON冒充当前实现的完整恢复；可显式读取权重建立新run。

## 已有有限证据

结果根目录：
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/gpu_vectorized_validation_20261009`。
这些完整成功/失败证据按本任务交付要求保留，不进入 Git 或正式训练目录。

| 测试 | 结果及限制 |
|---|---|
| `gmt_torch_exact_history.json` | 八卡冻结 GMT B1..32 对照通过 |
| `gmt_torch_batch1024_final.json` | 八卡冻结 GMT B1..1024，四种输入尺度、置换、原始及裁剪动作通过 |
| `worlds_n1`、`worlds_n8` | GPU 状态、局部 reset、物理诊断对照通过 |
| `saved_learning_large_v2.json` | B32/64/128 全部梯度、Adam、参数和原概率通过 |
| `vector_train8_v14_first` / `vector_train8_v14_resume` | 真实 160 条，停止进程后完整恢复第 2 轮通过 |
| `production8_v3_short` | 原正式管理接口的首轮/独立进程恢复/短评估/保存/归档均通过 |
| `condition_graph_v1.json` | 无梯度条件图逐元素一致；收益很小，默认关闭 |
| `cpu_gpu_replay150_v1` | 状态对照通过，Mine 奖励 RMS 门槛未通过，保留原结论 |
| `vector_reward_math_v2.json` | GPU连续奖励同输入逐字段对照通过，最大误差1.78e-15，含延迟持久化时钟别名回归 |
| `reward_integrated_matrix_v7` | 当前设备/奖励/参考边界合同下，8环境真实两轮、GAE/BC/完整KL通过 |
| `environment_matrix_final_v1` | 每卡1/2/4/16/32个分配环境均通过真实两轮，32只激活20个任务 |
| `production8_final_v1` | 正式入口、完整32任务评估，首轮保存退出后恢复第2轮通过；历史产物缺新独立审计字段，保留原文件 |

固定同一批旧数据、4 次 Actor 更新、不含 BC/物理的实测：B32 为 19.32/19.53 秒，
B64 为 19.54/19.52 秒，B128 为 19.82/19.82 秒；最终完整 KL 约 1.01/1.03/1.06 秒。
峰值显存约 17.5/30.5/54.9 GB（十进制）。大显存说明可以装下更多激活，不保证计算
更快；该严格 FP32 路径在 B32 之后已经没有明显批量收益。
这里的四次更新只用于固定计算量计时，显式关闭软停止且不发布模型；该固定数据
四次更新后的KL会超过0.03，不能称为已接受训练。真实训练始终保留软停止及最终
完整KL检查。数值一致性单独对照相同的一次更新，包含全部未裁剪梯度和Adam状态。

当前设备/奖励/边界合同下的固定160条真实转移矩阵如下。采样取八rank最大墙钟，
核心轮排除checkpoint、归档和独立评估；每格两个数字分别对应第1、2轮。
Actor列是实际接受更新次数，并非用减少更新来制造固定四步的速度提升。

| 每卡分配/活动环境 | 采样秒 | Actor秒 / 次数 | 最终完整KL秒 | 核心整轮秒 | 全局真实控制步 |
|---|---|---|---|---|---|
| 1 / 1 | 34.84 / 33.03 | 5.85（1次） / 11.01（2次） | 1.09 / 1.03 | 52.08 / 52.94 | 3981 / 4000 |
| 2 / 2 | 22.65 / 23.22 | 5.70（1次） / 5.54（1次） | 1.09 / 1.04 | 39.10 / 37.73 | 4000 / 3976 |
| 4 / 4 | 28.44 / 27.73 | 5.69（1次） / 14.97（2次） | 1.09 / 1.07 | 45.86 / 52.82 | 7928 / 7856 |
| 8 / 8 | 37.28 / 32.92 | 6.13（1次） / 15.48（2次） | 1.03 / 1.07 | 56.65 / 59.85 | 12400 / 12510 |
| 16 / 16 | 58.69 / 44.59 | 14.93（2次） / 23.34（4次） | 1.07 / 1.04 | 89.98 / 79.80 | 21000 / 13450 |
| 32 / 20 | 52.24 / 19.30 | 6.48（1次） / 16.66（3次） | 1.09 / 1.03 | 75.80 / 45.56 | 23500 / 4550 |

Actor列表示整个Actor阶段的总耗时，括号为实际接受更新次数。
每轮Critic均80次，耗时约2.53～2.73秒。N4首轮出现1次真实物理失败并正确保留终止，
其余本表轮次物理失败为0；训练逻辑验收通过不表示所有轨迹都成功。
N32第二轮的控制量明显较少，不能单独挑该轮宣称最快；不同N的任务数、校准时延、
前缀和参考尾部等待导致物理工作量不同。当前有限样本中N2最快，但只跑两轮，
尚不足以宣称训练质量或长期稳定性最优。正式独立配置依用户要求仍为N8。

前缀也随实际部署关键路径延迟变化：首轮rank0的20条真实样本，N1/N2/N4/N8的P
分别为18/21/29/51；N16为84（16条）和44（4条），分配N32/活动N20为89。
对应八卡校准延迟范围约0.36～0.38、0.44～0.48、0.74～0.78、1.36～1.46、
2.54～2.84、3.06～3.32秒。因此更大的N不只增加吞吐，也改变本任务保留的延迟
反馈和自由动作坐标数量。P51已经超过明确绑定的s350000模型原训练P6～18，
也超过后续P6～30重训计划；不能仅看显存就认定N8或更大N适合长期策略优化。
这些观察来自`prefix_sampling_diagnosis_final.json`，不是对所有环境/长期训练的推断。
本次没有自动更换Stage1模型、截短前缀或隐藏实测生成延迟。

默认N8的20步GENMO批量生成共2.91～3.34秒，八卡梯度通信总计约0.39～0.69秒，
等待最慢采样卡约5.39～5.75秒。吞吐瓶颈已主要在共享场景闭环推进、真实参考等待、
状态/奖励和证据处理，不能再把全部采样耗时归因于扩散生成或NCCL。
相对旧普通轮55.95秒、采样27.14秒，当前N8没有整轮加速证据。

`production8_final_v1`正式入口首轮核心53.15秒，恢复轮65.20秒，实际Actor为1/3次。
相同新GPU评估合同的32任务初始/首轮/第二轮四来源等权回报为
41.5090 / 41.4466 / 41.3952，平均执行9.975秒、物理失败0。
两轮更新和存在实测时延抖动的结果不能证明相对Stage1提高，也不能仅凭这点差值判定退化。
最新独立产物审计、故障和容量复验结果在交付记录中分别列出，失败证据不覆盖。

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

完整矩阵命令如下，其中`--config`可指定本文件配套配置，`--output`必须为新目录。

```bash
/home/user/liwei/GENMO/.venv/bin/python -B tools/benchmark_stage10_gpu_vectorized.py \
  --mode collection --environments 1 2 4 8 16 32 --updates --rounds 2 \
  --config configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml \
  --gmt-repo /home/user/liwei/legged_lab_gmt-gpu-vectorized \
  --output /data1/user/liwei/GENMO_outputs/closedloop_stage10/新的采样矩阵目录

/home/user/liwei/GENMO/.venv/bin/python -B tools/benchmark_stage10_gpu_vectorized.py \
  --mode capacity --environments 1 2 4 8 16 32 64 128 256 512 1024 \
  --config configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml \
  --gmt-repo /home/user/liwei/legged_lab_gmt-gpu-vectorized \
  --output /data1/user/liwei/GENMO_outputs/closedloop_stage10/新的物理容量目录
```

collection工具的N32仅分配32、激活20个真实任务，正式训练入口拒绝分配超过每卡20个
环境。capacity则让全部N个机器人实际推进，包含完整执行证据，没有GENMO/DPPO。
两张表不能互相替代。所有测试都检查八卡空闲，逐规模独占运行；不要同时启动两个矩阵。
