# 2026-10-21 Stage2 奖励尺度、关闭 BC 与扩散时间对照

本批按用户的两小时边界实施：开始于北京时间 17:45:43，19:45:43 前结束工作并报告实际结果。只运行独立 64 次尝试、一个宏迭代，不恢复旧五轮任务。新运行的物理结果不能与旧奖励的分数直接相减。

## 奖励合同

新配置为 `configs/closedloop/stage2_awr_rewardstd_nobc_minsnr.yaml`，奖励身份为 `stage2.gmt8_reward_std.v3`。GMT 仍输出原八项平方误差及原函数、原权重、原 `source_std`；CPU 与 GPU 奖励入口均强制核验来源。Stage2 仅用独立 `reward_std` 计算 `exp(-squared_error / reward_std**2)`，保持原权重、50Hz 积分和事件处理，不重复平方或乘 dt。

| 项目 | GMT source_std，不变 | Stage2 reward_std |
|---|---:|---:|
| anchor_pos | 0.30 | 0.20 |
| anchor_ori | 0.40 | 0.30 |
| body_pos | 0.30 | 0.15 |
| body_ori | 0.40 | 0.20 |
| body_lin_vel | 1.00 | 1.00 |
| body_ang_vel | 3.14 | 3.14 |
| foot_height | 0.07 | 0.07 |
| leg_joint_pos | 0.25 | 0.25 |

新版本禁止模糊的 `tracking.std`，必须显式给出完整两套八项字典；来源错误照常拒绝。旧配置、Replay、标签和模型保留原身份，不重标或混用。新证据同时保存两套尺度。活动度仍只诊断，奖励乘数为 1。没有修改 GMT 网络、控制、环境参数或八项误差函数。

## Actor 目标

`bc_enabled=false`、`bc_global_batch=0`、`bc_fixed_weight=0`：不创建 BC 训练采样器，不执行校准和 BC 前反向，不推进 BC 游标，checkpoint 的 BC 状态为 null。原 Stage1 验证损失可作为独立无梯度诊断读取真实标签，不参与优化。教师系数仍为 0，没有保留门槛；仅按物理奖励选择最佳模型。

既有梯度诊断发现高噪声 x0 回归梯度远大于低噪声梯度，本批增加可对照的 Min-SNR 时间权重：`h(t)=min(alpha_bar/(1-alpha_bar),5)`，再除以完整 1000 步时间表的均值。时间 t 仍均匀抽取；全表归一化是本项目工程选项，不是论文原始要求。它调整监督目标，不能声称已经消除 CFG 生成分布与重新加噪分布的所有差异。[Min-SNR 原论文](https://arxiv.org/abs/2303.09556)

各来源按 `N/(S*N_source)` 校正，避免较长音乐产生更多决策而控制整体训练目标。不进行优势优先采样。最终每样本系数是固定优势权重 × 固定来源因子 × 本次 t 对应的固定时间因子，仍除以真实全局 batch 数，不按 microbatch 重算均值。所有因子表与配置绑定 checkpoint 身份。

质量诊断按本计划真正进入 GMT 当前点/前瞻窗口的暴露权重统计 Root/Body/Joint 误差，修正原先误用生成所在区间的问题。原始奖励、MC、credit 和冻结优势标签的计算不因此改变。

## 有限实验

四组共享新奖励、原始 s595000、冻结 GMT、DDIM20/CFG2.5、相同 64 次采集 Replay 和 32 个原训练音乐评价任务：

1. A：冻结 Stage1。
2. U：无优势权重、均匀扩散时间目标、无 BC。
3. B：无优势权重、Min-SNR-5、无 BC。
4. C：AWR 优势权重、Min-SNR-5、无 BC。

U/B/C 均执行 5 Epoch，global batch 256，八卡每卡 microbatch 8，AdamW LR 1e-5、warmup 100 步、clip 1，来源因子相同。B−U 检验时间目标影响，C−B 检验优势权重影响。64 次尝试不静默补采；保留失败和行政中断；新旧实验不拼接。候选组和 group_hybrid 保持原方案，本批不声称已解决严格同状态候选覆盖稀疏的问题。

启动及查看（以最终运行副本为准）：

```bash
cd /home/user/liwei/awr-worktrees/GENMO-rewardstd-20261021
PYTHONDONTWRITEBYTECODE=1 /data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B tools/launch_stage2_awr_signal_server2.py \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/rewardstd_nobc_minsnr64_20261021_01 \
  --gmt-repo /home/user/liwei/awr-worktrees/legged_lab_gmt-signals-20261021
```

本文件初次提交时：本地 227 项回归通过；追加无 BC 组件恢复测试 4 项通过。GMT 本地 7 项通过、22 项因缺少 IsaacLab 数学源码跳过，须在 Server-2 补验。八卡与真实物理结果尚待执行，后续追加实际结果；不以测试通过宣称学习改善。
