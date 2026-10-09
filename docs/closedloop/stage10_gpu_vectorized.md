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
| `saved_learning_final_b2_32.json` | 最新八卡B1标量参考、B2/8/16/32全部梯度/参数/Adam通过；相同旧数据4次更新各重复2次计时，旧归档SHA未变 |
| `vector_train8_v14_first` / `vector_train8_v14_resume` | 真实 160 条，停止进程后完整恢复第 2 轮通过 |
| `production8_v3_short` | 原正式管理接口的首轮/独立进程恢复/短评估/保存/归档均通过 |
| `condition_graph_v1.json` | 无梯度条件图逐元素一致；收益很小，默认关闭 |
| `cpu_gpu_replay150_v1` | 状态对照通过，Mine 奖励 RMS 门槛未通过，保留原结论 |
| `vector_reward_math_v2.json` | GPU连续奖励同输入逐字段对照通过，最大误差1.78e-15，含延迟持久化时钟别名回归 |
| `reward_integrated_matrix_v7` | 当前设备/奖励/参考边界合同下，8环境真实两轮、GAE/BC/完整KL通过 |
| `environment_matrix_final_v1` | 每卡1/2/4/16/32个分配环境均通过真实两轮，32只激活20个任务 |
| `production8_final_v1` | 正式入口、完整32任务评估，首轮保存退出后恢复第2轮通过；历史产物缺新独立审计字段，保留原文件 |
| `production8_final_v2` | 真实6轮、20次Actor/480次Critic；独立审计29通过、1失败，失败项为关闭回执丢失。该失败报告保留，不当作完全通过 |
| `production8_final_v3` | 修复关闭后新建会话，首轮保存、退出进程、恢复第2轮；完整32任务评估及独立审计18项全部通过，0失败、0未执行 |
| `vector_kl_fault_v1` | 真正更新后注入KL超限，八rank Actor/Critic/两个Adam/BC/RNG逐字节恢复，已消耗优化尝试不退，不发布拒绝轮断点 |
| `worker_exit_final_v1` | 第一轮完整保存后杀死本测试rank2物理worker，八rank按预期非零退出；第一轮断点SHA不变、第二轮未发布、64环境预算单调且全部GPU进程清理 |

最新同一批旧数据、固定4次Actor更新、不含BC/物理的两次计时均值为：

| 学习microbatch | Actor秒 | 最终完整KL秒 | PyTorch峰值分配GB（十进制） |
|---:|---:|---:|---:|
| 2 | 25.3347 | 1.3573 | 5.44 |
| 8 | 20.8487 | 1.1072 | 7.70 |
| 16 | 19.4523 | 0.9970 | 10.97 |
| 32 | 19.2206 | 1.0190 | 17.46 |

B2→B32的Actor减少6.1142秒（24.13%），最终完整KL减少0.3382秒（24.92%）。
这里B16的最终KL略快于B32，不能声称所有分项都随B单调改善。
本轮B32零更新log-prob/ratio/独立Gaussian的报告最大误差均为7.28e-12，原门槛未改。
全部梯度、参数和Adam状态逐张量核对，未裁剪梯度保持atol3e-5/rtol2e-4。
旧归档SHA256为`814684df7891ac027bfa9f5f93c812cddfa746fb6b4bf4e8af1a78f2ab0e4b95`，
测试前后相同，没有重新采样、改写old概率或优势。摘要为`learning_batch_summary_final.json`。

另一次相同固定数据的B32/64/128实测：B32 为 19.32/19.53 秒，
B64 为 19.54/19.52 秒，B128 为 19.82/19.82 秒；最终完整 KL 约 1.01/1.03/1.06 秒。
峰值显存约 17.5/30.5/54.9 GB（十进制）。大显存说明可以装下更多激活，不保证计算
更快；该严格 FP32 路径在 B32 之后已经没有明显批量收益。
这里的四次更新只用于固定计算量计时，显式关闭软停止且不发布模型；该固定数据
四次更新后的KL会超过0.03，不能称为已接受训练。真实训练始终保留软停止及最终
完整KL检查。数值一致性单独对照相同的一次更新，包含全部未裁剪梯度和Adam状态。
所有本轮计时重复的`hard_kl_would_accept`均为false，报告如实保留；这是受控离线
计时结果，不是降低正式KL限制后的训练。正式配置始终保留软0.015/硬0.03。

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

为排除减少Actor更新次数的影响，`production8_final_v2`另连续运行至第6轮。
其中第4、5轮是没有初始化、恢复、保存和独立评估的普通轮，均有4次真实Actor更新。

| 项目 | 第4轮 | 第5轮 |
|---|---:|---:|
| 采样（八rank最大墙钟） | 32.53秒 | 35.63秒 |
| 批量GENMO生成（包含20步、各批次累计） | 2.90秒 | 3.05秒 |
| Actor（4次参数更新、含BC和必要KL） | 21.89秒 | 21.85秒 |
| Critic（80次） | 2.51秒 | 2.64秒 |
| 最终完整KL | 1.03秒 | 1.03秒 |
| 核心轮 | 62.39秒 | 65.37秒 |
| 逐轮墙钟 | 63.43秒 | 66.41秒 |
| 真实控制步 | 11925 | 10857 |
| 实际物理失败 | 0 | 0 |

普通轮墙钟均值为**64.9186秒**，当前配置10000轮仅普通轮即约**7.51天**，另加初始化、
周期评估、保存和可能的恢复。不能宣称相对用户给定55.95秒基线已经加速。
第4轮分项最大值：GMT GPU推理0.290秒，纯PhysX step墙钟2.023秒，完整物理控制阶段
5.586秒，物理/状态诊断2.046秒，边界证据展开/传输2.695秒。
这些分项有包含关系、跨线程重叠，不能直接相加复原采样墙钟。
GPU奖励计时是host发射时间，明确不能冒充独占GPU计算时间。
rank0梯度通信约0.99～1.00秒；采样阶段各rank最大等待约8.41～10.84秒。

此六轮run记录了真实时序和学习，但结束审计发现关闭回执缺失；独立报告为29通过、
1失败。随后的修复仅调整关闭回执、冻结核验和进程收尾，没有改变上述热路径。
`production8_final_v3`使用修复代码重新完成独立进程保存/恢复、完整评估，审计18项
全部通过。上述普通轮时间仍明确引用v2原产物，不冒称v3又跑过六轮。
v3两轮最终KL为0.01389/0.01697，Actor实际更新1/3次；第二轮出现1次真实物理失败，
终止被正常记录，不能把训练流程通过解释为所有机器人任务都无失败。
其保存结束轮墙钟为63.89/82.79秒，包含完整同步保存，不能与上表普通轮混用。

六轮run在相同新GPU评估合同下，32任务的初始/第1/第2/第6轮四来源等权回报分别为
41.8791 / 41.6588 / 41.6593 / 41.5077，平均执行9.975秒、物理失败均为0。
这是总回报，不是每秒奖励。存在实测生成时延抖动，且只有有限轮次，当前证据不能证明
相对Stage1改善，也不能仅凭这点差值判定退化。原CPU/GPU奖励门槛未通过的问题保留。

长期资源还有两个明确约束：原控制预算6000万步按约1.2万步/轮估算，不足支持10000轮
再加评估；完整执行归档按当前约0.67GB/轮估算，10000轮约6.7TB，超过配置两个归档
位置合计约4.34TB额度。因此本次没有自动扩大预算、丢弃证据或启动一万轮训练。
实现已具备有限真实闭环与恢复能力；长期启动仍需解决延迟导致的P覆盖、工作量和存储
预算匹配，并用新物理/时序基线评价策略收益。

## 1～1024环境的最新真实GPU容量

`capacity_matrix_final_v2`在GENMO `ee81f69`、GMT `2523830`上全部11档通过。
每档八张卡均运行，全部N个机器人实际推进；局部reset隔离、设备UUID、控制计数和
物理诊断通过。大规模标量诊断抽查32个均匀分布的环境，所有环境均参与状态、reset和
控制量检查，不能把抽查写成1024个环境逐字段全量标量对照。

下表每档重复3次，每次推进20个控制步，取各rank均值的最大值。三项时间的最大值
可能来自不同rank，总时间直接使用原始墙钟，不能按列重新相加。保留完整证据。
这张表**没有GENMO采样、Actor更新或KL**，不是固定160条上层转移的训练时间。

| 每卡环境数 | 控制/仿真段秒 | 证据展开秒 | 文件保存秒 | 全段秒 | 每卡有效机器人控制步/秒 | 整卡NVML峰值MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.695 | 0.060 | 0.050 | 0.800 | 25.00 | 5950 |
| 2 | 0.642 | 0.064 | 0.091 | 0.794 | 50.35 | 5950 |
| 4 | 0.647 | 0.088 | 0.182 | 0.911 | 87.86 | 5944 |
| 8 | 0.677 | 0.285 | 0.374 | 1.309 | 122.19 | 5980 |
| 16 | 0.678 | 0.396 | 0.794 | 1.854 | 172.60 | 5982 |
| 32 | 0.711 | 0.723 | 1.726 | 3.028 | 211.38 | 6034 |
| 64 | 0.823 | 1.327 | 3.415 | 5.388 | 237.57 | 6124 |
| 128 | 0.974 | 2.348 | 6.797 | 10.036 | 255.08 | 6316 |
| 256 | 1.312 | 4.756 | 14.895 | 20.829 | 245.82 | 6756 |
| 512 | 2.407 | 9.977 | 31.181 | 43.501 | 235.40 | 7662 |
| 1024 | 4.328 | 20.658 | 64.272 | 89.257 | 229.45 | 9422 |

纯PhysX step在N1/N8/N128/N1024分别约0.194/0.191/0.208/0.254秒，说明GPU物理
随环境数量扩展良好；对应GMT GPU推理约0.031/0.032/0.067/0.093秒。
完整控制/仿真段还包含观测、历史、参考和全部诊断，不等于纯物理算子时间。
整卡NVML包含Kit跨可见GPU创建的上下文，不能与GENMO PyTorch分配峰值直接相加。

在这组固定参考、完整证据保存的容量口径中，N128吞吐最高；N1024能运行但更慢，
主要增长在host证据展开和序列化。它不意味着应将当前DPPO配置改为每卡128环境：
当前每rank仅20条真实上层转移，超过20个活动环境无法在这一轮都产生训练样本。
固定160条的实际闭环矩阵中，有限结果更倾向N2，而不是最大显存占用。
没有以增大训练数据量、减少PPO更新或缩短物理执行来代替同工作量比较。

各档GPU利用率、整卡显存、测试进程CPU核数及RSS在每秒遥测中完整保留，摘要为
`capacity_summary_final_v2.json`。利用率均值涵盖启动和退出，不是稳定计算窗口的
GPU忙碌率；CPU仅统计本测试后代进程，RSS之和可能重复计入共享页。
容量使用完整短轨迹torch保存，正式训练另有两级journal和异步归档，不能把本表文件
保存时间直接当成正式训练单轮归档阻塞时间。

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

固定旧数据学习验收和故障验收可以分别复现，八卡必须空闲，输出必须是新路径：

```bash
bash scripts/validate_stage10_runtime_v4_server1.sh saved-learning \
  --iteration /data1/user/liwei/GENMO_outputs/closedloop_stage10/runtime_v4_validation_20261009/closedloop6_v2/sessions/8a6e3d6e-2d61-4b99-89b8-fb76007838a0/iterations/000001 \
  --weights /data1/user/liwei/GENMO_outputs/closedloop_stage10/runtime_v4_validation_20261009/closedloop6_v2/checkpoints/initial.pt \
  --stage1-config /data1/user/liwei/GENMO_outputs/closedloop_stage10/runtime_v4_validation_20261009/stage1_config.json \
  --assets /data1/user/liwei/GENMO_outputs/closedloop_stage10/runtime_v4_validation_20261009/assets \
  --output /data1/user/liwei/GENMO_outputs/closedloop_stage10/新的学习验收.json \
  --microbatches 2 8 16 32 --timing-repeats 2

bash scripts/validate_stage10_runtime_v4_server1.sh vector-collection \
  --config configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml \
  --gmt-repo /home/user/liwei/legged_lab_gmt-gpu-vectorized \
  --output /data1/user/liwei/GENMO_outputs/closedloop_stage10/新的故障验收目录 \
  --num-envs 8 --rounds 2 --updates --kill-worker-iteration 2
```

第二个命令故意杀死自己创建的rank2物理工作进程，**预期非零退出**，必须检查只保留
第1轮断点、未发布第2轮、已消费预算不回退和所有子进程被清理。它不代表正常训练成功，
也不会杀死其他训练。`--reject-iteration 2`可另行验证真正更新后的整轮KL拒绝与回滚，
不能与kill注入同时使用。
`worker_exit_final_v1_check.json`的8项实际核验均通过。第一轮checkpoint为
2,562,965,096字节，SHA256为`e0eac68f6ac9336318c5041f48639163faac6d066d7b2d428d9bb06c73a6ed11`。
预算使用原`replay_budget`只读重放SQLite持久事件并核对SHA链，没有把budget.json描述符
误当余额。此故障测试验证停止和持久化；完整恢复另由`production8_final_v3`验证。

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

## 交付索引与来源

本地可直接查看 `/home/weili/bumi-stage10-gpu-vectorized/验收结果_20261009`。
`delivery_manifest.json`列出精简报告的准确字节数、SHA256、服务器原始路径、最终代码
版本和两个仓库相对原功能分支的完整修改文件清单。训练模型、完整执行链、journal和
原始成功/失败日志留在服务器1上述独立validation目录，没有复制为正式训练结果。

关键报告分别为：`production8_final_v3_audit.json`（完整恢复18项审计）、
`production_summary_final_v2.json`（四次Actor更新的普通轮时序，注明原关闭失败）、
`production_summary_final_v3.json`（修复后的正式保存/恢复）、
`environment_matrix_summary_final.json`（固定160条真实转移）、
`capacity_summary_final_v2.json`（1～1024全部活动机器人容量）、
`learning_batch_summary_final.json`（固定旧数据B2/8/16/32同工作量对照）、
`worker_exit_final_v1_check.json`（真实进程退出故障）、
`prefix_sampling_diagnosis_final.json`（实际P范围）。失败项及严格数值原报告一并列入清单。

运行测试的代码来源分别记录：关闭修复后的正式恢复GENMO `52d6419` / GMT `2523830`；
最新容量、故障、固定旧数据学习GENMO `ee81f69` / GMT `2523830`。
后续交付提交仅补帮助文字、验收文档和记录，不更改物理、奖励或学习计算路径。
服务器1已完成启动脚本的shell语法检查；本地没有执行测试。

临时pytest依赖及本任务明确列出的`/tmp/genmo-vector-*`目录已按精确路径清理，服务器
代码目录无本任务测试缓存；用户要求的完整验收证据继续留存。以后若需清理证据，应先
按交付清单归档及核对SHA，再明确选择目录，不能按名称广泛删除正式模型或训练日志。
