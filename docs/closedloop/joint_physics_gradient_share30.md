# 后期物理参数梯度约30%的正式续训

用户选择的范围是**有物理更新的步**。保留H32每4个optimizer step执行一次，另外3步仍计算原数据监督；不将占比解释成所有步平均30%。

定义`g_s = grad(L_data + 0.1 L_future)`，`g_p = grad(L_exec)`；两者都必须先完成八卡全局平均。物理样本按全局有效样本数归一化，零有效样本rank仍参加归约。

目标为`lambda ||g_p|| / (||g_s|| + lambda ||g_p||) = 0.30`。因此`lambda = (0.3/0.7) ||g_s||/||g_p||`，不能直接设置lambda=0.3。该指标是裁剪前的分支参数梯度范数占比，不是Loss数值占比、合梯度方向的正交分解或AdamW最终参数位移占比。

## 课程和边界

- 从完整旧断点续训，旧lambda=0.02作为起点；首次实测占比作为课程起点，在20,000个全局optimizer step内线性提高目标到30%。总预算仍为200,000步。
- 每100步且有物理分支时测量一次全参数梯度；EMA系数0.9。单次权重变动通常不超过1.2倍，范围0.0001至5.0。实测批次占比超过40%时立即降低权重，此安全降低可突破1.2倍限制。
- 40%上限只对校准批次成立。其它物理批次复用最近权重，原grad_clip=1.0和非有限检查继续生效。目标因权重上限或数据波动不能达到时如实记录`weight_limited`、实测占比及校准步数。
- 零有效物理样本或零梯度不触发权重放大；数据监督继续。监督包含全部原Stage1 v5项与Future辅助监督。网络、AdamW、数据、噪声分布、H32执行目标、硬安全检查和冻结SM/WM均保持原合同。
- 校准步从前向起使用DDP `no_sync`，分别计算并平均两条参数梯度，最后联合backward后显式平均实际梯度一次。其它步保持原DDP。不会每一步额外执行两条完整梯度分解。

## 断点与运行

新配置为`configs/closedloop/joint_physics_formal_200k_gradshare30_8gpu_server2.yaml`。

首次切换使用`--continue-from <完整旧joint断点>`。只允许合同中增加`physics.gradient_balance`；其它合同差异仍拒绝。保留Actor、AdamW、学习率日程、global_step、八卡RNG、数据/Future/物理采样游标。新的输出目录写`continuation.json`标记这是改变权重课程的续训，不称作原目标不变的等价恢复。

新课程后续中断必须使用`--resume <新课程完整joint断点>`，严格恢复控制器EMA、起始步、当前lambda、初始占比和校准游标。每次打开新输出目录先写当前完整断点，后续仍每5000步保存。

```bash
export JOINT_PYTHON=/data0/user/liwei/envs/genmo_sm_runtime/bin/python
export JOINT_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash scripts/train_joint_physics.sh \
  configs/closedloop/joint_physics_formal_200k_gradshare30_8gpu_server2.yaml \
  --max-steps 200000 --continue-from <完整旧断点> --output <不存在的新目录>
```

`metrics.jsonl`和TensorBoard新增`gradient_balance/measured_share`、`target_share`、`measurement_step`、`supervised_norm`、`execution_norm`、`cosine`及各参数模块范数。日志步数与最近校准步数可能相差不足100步，不能把缓存的最近测量宣称成每一步即时测量。

## 必要验证

双进程Gloo回归覆盖校准/普通DDP交替、某rank零物理样本、全局零样本、全局未用参数以及实际AdamW更新与单进程全局均值一致。恢复测试比较下一批数据、噪声和更新后的全部参数逐位一致；额外检查旧完整状态显式切换保留数据及优化器演进。真实八卡启动与检查点结果追加至`记录文本.md`。

本改动提高物理梯度参与程度。是否降低真实执行误差、改善纯噪声生成与整曲跟踪，仍需后续独立评价，不能依据目标比例或短时Loss宣布成立。
