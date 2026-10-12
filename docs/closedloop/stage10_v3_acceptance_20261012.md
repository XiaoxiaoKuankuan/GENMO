# GENMO–GMT Stage10 v3 重构与验收记录

本文记录 2026-10-12 恢复开发的真实证据。当前为开发验收记录，尚未宣布可正式训练、吞吐目标达成或强化学习收益改善。报告将在有限八卡验收完成后更新。

## 恢复时的仓库与中断点

原 GENMO 功能工作树 `/home/weili/bumi-closedloop-worktrees/GENMO` 位于 `feature/bumi-music-closedloop@832e19c`；Stage1 评估脚本及记录存在用户未提交修改。原 GMT 功能工作树位于 `feature/bumi-frozen-gmt-backend@f382356`，Isaac 播放脚本及文档存在未提交修改。恢复开发没有覆盖这些修改。

此前 v3 源码位于 `/home/weili/bumi-stage10-v3-20261012/{GENMO,legged_lab_gmt}`，两个工作树均为 detached HEAD。GENMO 有 26 个变更路径，GMT 有 9 个；恢复时和服务器1对应副本逐文件 SHA 完全一致。源码清单及此前小型验收 JSON 下载到 `/home/weili/bumi-stage10-v3-20261012/reports/recovery_20261012`。

为使后续开发具备可恢复提交身份，两仓库现使用独立分支 `refactor/stage10-v3-20261012`。首次提交为 GENMO `92caf69`、GMT `078251a`。后续 GENMO `ee73b08` 修复内存条件设备、轮末证据与异常保存等边界。GitHub 已接收提交；服务器新工作树为 `/home/user/liwei/GENMO-stage10-v3-verified-20261012` 和 `/home/user/liwei/legged_lab_gmt-stage10-v3-verified-20261012`。原服务器未提交源码保留。

恢复时最后一次 `collect4096_inprocess_v3` 于 15:32 退出 1：初始物理评估完成后，`mark_saved_evaluation()` 收到不支持的 `eligible` 参数。该运行尚未采集 4096 条训练转移。服务器1当时八卡空闲，旧正式 Stage10 会话中只看到日志查看进程。

## 实现与数据流

| 环节 | v3 实现 | 不变的训练语义及限制 |
|---|---|---|
| 初始化 | Stage1 `s595000` 及配套统计量；原生加载冻结 GMT `140500`，核验 ONNX 来源 SHA | 新 Actor/Critic 优化器；独立 v3 schema，不能透明恢复旧 v2 |
| 行为采样 | 每卡一个 Isaac GPU 世界，每卡 64 环境，每环境 8 决策；GENMO 与 GMT 同进程直接调用 | 50 Hz 控制、200 Hz 物理、每控制步四个真实物理子步 |
| 随机核 | 完整 20 步去噪，显式末端 std 日程；旧 mean/std/FP64 joint log-prob 留存 | 真实行为采样与重算采用同一实际核；不改写 old statistics |
| PPO | 自由有效坐标平均 log-ratio 代理；每 epoch 无放回 K8，按 20/K 校正完整目标 | 不称为真实联合概率；保留 Full20 与 joint_sum 对照入口 |
| 掩码与因果性 | 只排除已承诺前缀及 padding，保留完整 120 帧自由动作坐标 | 不按短执行窗口裁掉 GMT 前瞻及未来执行相关坐标 |
| KL | 直接高斯 log-ratio；完整旧路径上全链、全 20 步 KL，逐步均值/P95/均值变化 | old-path 条件高斯 KL 不等于最终动作边缘分布 KL；末端均值变化不等于机器人实测变化 |
| Critic | 从 Stage1 独立复制条件编码器并冻结；完整时序、时间与 mask、低维投影、两层 Temporal Transformer、Value Head | 参数及存储与 Actor 隔离；只读当前因果条件和音乐剩余时间 |
| DPPO＋BC | 原 Stage1 监督损失；继承 s595000 损失成熟度；局部累积、配对桶 SUM、全局梯度诊断、统一裁剪和 AdamW | BC 采样 RNG 从新运行开始；梯度范数比例不等于 Adam 更新贡献 |
| 数据生命周期 | 普通完整随机链驻 GPU 至本轮优化及最终 KL 完成；执行列保留只读视图 | 诊断/异常保存完整证据；未完成仿真 rollout 可以丢弃，恢复重建 PhysX |
| 保存 | 不可变 CPU 快照、单个在途异步写入、SHA、原子发布、实际评估模型权重 | 不将 mutable GPU 状态交给保存线程，不伪造中间物理状态恢复 |

主要实现文件为 `gem/closedloop/dppo/{probability_v3,policy_v3,pretrained_critic,gradient_merge_v3,updater_v3,runtime_v3,training_v3,checkpoint_v3}.py`。GMT 配套代码为 `bumi4340_policy_loader.py`、`bumi4340_native_frozen_policy.py` 及 `closedloop/inprocess_service.py`。

## 已核验的证据

服务器测试根目录为 `/data1/user/liwei/GENMO_outputs/closedloop_stage10/v3_validation_20261012`。用户要求交付全部测试结果，因此这些具名报告、失败日志和必要固定数据保留；pytest 临时夹具按精确路径清理。

| 检查 | 实际结果 | 证据范围 |
|---|---|---|
| 直接高斯概率公式、clip、无放回抽样归一化 | 八卡通过；此前直接 log-ratio 与联合概率相减最大差约 6.84e-10 | 数学正确性，不是学习效果 |
| 原生 FP32 flatten GEMM | 自身不同 batch 的 joint log-prob 差 0.011–0.018，失败 | 不采用此数值路径，不放宽门槛 |
| 固定归约 TF32x3＋SDPA math | 八卡 batch 1/4/8/16/32/64/128、置换、尾批、有/无梯度重算通过；最大概率差约 3.64e-12 | 尚需完整真实梯度、Adam 和实际更新对照 |
| BF16＋Flash 候选 | 无可用 kernel，失败 | 与后续 cuDNN 路径分别记录 |
| BF16＋cuDNN 候选 | 八卡自身采样/重算一致性通过 | 尚未选为主路径，不能代替完整梯度及学习效果 |
| Critic 基础 | 独立参数/存储、冻结无梯度、缓存一致、音乐顺序敏感通过 | 价值拟合及独立轨迹 EV 待后续验证 |
| 全局梯度与系统 | 不均匀/空 PPO rank、SGD、非空 Adam、内存 ACK、预算和不可变保存检查通过 | 小模型八卡检查，不替代真实 GENMO 更新 |
| GMT140500 | 八卡 GPU 原生 batch/单样本/置换与 CPU ONNX 对照通过 | 随机观测数值对照，不等于训练效果 |
| 单机器人延迟标定 | 64 有效样本，各卡 P50 0.214–0.237 秒，最大 0.238536 秒 | 新模型新合同采用 0.30 秒前缀预算；不将64环境训练墙钟充当部署时钟 |
| 旧路径回归 | `recovery_regression8_v2` 八 rank 各 269 项通过，总监督 63.09 秒 | 奖励、GAE、采样、缓存、预算、ACK、回滚、归档等回归 |
| 本次集成修复 | 25 项通过，3.26 秒，含实际 CUDA 条件设备交付 | 评估绑定、异常回读、Critic 缓存、轮末证据；不替代完整闭环 |

`collect4096_recovery_v4` 完成新模型初始固定 32 任务评估：四来源等权回报 **42.80773029688096**，平均真实执行 **9.975 秒**，物理失败 **0**。该值只能作为此新资产、随机核及部署时钟下的初始化基线，不与旧 v2 的 43.2109 直接作学习收益比较。初始未训练 Critic 的固定验证 MSE 为 319.0199、EV 为 -0.001390，不能据结构检查认定价值预测已可靠。

## 本次发现并修复的集成问题

1. 评估保存接口：只把 `best_observed`、`best_saved` 历史指针传给保存绑定函数，保留报告中的选择诊断；继续核验模型指纹、实际文件和最终报告 SHA。
2. GPU 内存条件：v4 已在每卡 64 环境完成各 8 次决策，但轮末 `UpperTransition.validate` 因 GPU free_mask 与 CPU context 比较失败。内存路径现交付采样 trace 内的同卡条件；旧磁盘路径继续用 CPU。
3. 轮末完整证据：额外真实等待控制步原先只追加奖励和计数，现在同时追加到内存 `execution_trace`，异常回放不会漏掉尾段。
4. 普通异常保存：原实现只在 KL 拒绝时持久化完整 rollout；现普通优化异常也先保存，再释放数据，并分别报告证据保存失败和学习状态回滚失败。
5. Critic 解冻边界：缓存只对确实冻结的编码器开放。实验性解冻时每个 optimizer step 重算自己的编码器，不复用旧可训练特征。
6. 数值与性能报告：BC 明确标记继承 Stage1 损失成熟度；计算对照增加独立真实 CPU/CUDA Profiler，测量扰动轮不纳入 P50/P95。

`collect4096_recovery_v4` 退出 1，总墙钟 410.71 秒，rank0 采集至轮末失败 117.91 秒。这批数据没有发布为成功固定数据，也不计作接受训练。修复后的 `collect4096_recovery_v5` 已在独立目录完成：4096条真实转移，控制131180次、物理524720子步；完整零更新概率最大误差9.094947e-13，独立高斯公式误差1.818989e-12，源文件未变，GMT冻结、世界正常关闭。采集及结算最大125.918秒，32.529条上层转移/秒；包含启动、完整概率检查、固定数据保存和关闭的总墙钟262.453秒。仅采集模式跳过已完成的初始评估，正常训练评估仍保留。

## 尚待完成及参数边界

- 完整真实梯度、非空 Adam、不同 microbatch/精度的参数与轨迹变化对照，以及实际 GPU Profiler、整轮吞吐和显存峰值。
- Critic 真实预热、本批/下一批/独立轨迹 MSE 与 EV，以及原 MLP 对照。
- 末端 std、PPO clip、Actor LR 和 KL 门槛的联合标定；确认 BC 梯度比例不压制 PPO。
- 坐标平均 K8、Full20 和严格 joint_sum 在相同真实交互预算下的有限学习对照；检查实际机器人执行分项、失败率与 Stage1 能力保持。
- 完整停止/恢复、整轮回滚和严格证据路径验收；按最终证据更新正式建议参数。

当前 YAML 中的 `lr=3e-6`、`clip=0.003`、末端 `std=0.03`、坐标平均软/硬 KL `0.001/0.002`、Critic 256维/2层/4头及8轮预热均为**待标定候选**。目前没有充分证据把它们宣布为正式推荐参数，也未启动长期正式训练。

## 固定真实数据计算对照（进行中）

`compute4096_tf32x3_v1` 从同一真实rollout和非空Adam出发，每组完成两epoch、8次全局Adam、65536内部去噪访问、512个BC样本，计时取预热后的三次。当前microbatch64的P50为103.033秒，峰值已分配显存22.944GiB；128的P50为107.724秒，峰值41.238GiB。128的全部梯度及Adam比较通过，两个LayerNorm权重与64参考相差1个FP32 ULP（1.192093e-7），末端均值最大差4.917383e-6，未通过原参数/输出更严门槛，不能写成全项通过。此处microbatch是内部去噪样本数，GEMM仍组合全部真实batch/time行，没有旧固定64网络行的Python切块。

原候选3e-6和BC初值.1尚未通过完整更新：第一步共同参数PPO范数0.274891、未加权BC范数3.154285，加权后BC0.286635，近1∶1而非目标0.1∶1；完成固定工作量后平均坐标KL0.030435，超过.002门槛。这些离线权重不被发布为训练策略。后续从原始学习状态有限扫描LR，并以.01作为有实际梯度依据的BC初值候选，验收门槛保持原值。

恢复边界又补齐同一边界重复保存的唯一发布身份、Actor/Critic/非空Adam/BC/控制器/RNG逐字节回滚核验、周期全概率检查，以及最终全链真实advantage有效clipping。新增11项CPU集成回归通过，具体输出在recovery_regression_v3/v4.log；这不代替真实恢复和学习效果验证。
