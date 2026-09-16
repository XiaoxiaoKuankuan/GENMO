# MotionMillion 同目标续训

本阶段从完整 `s210000.ckpt` 恢复模型、AdamW 动量、epoch 和绝对 global_step，
总目标 230000，即追加 20000 次优化器更新。数据、训练损失、8×256、BF16、
文本 dropout 0.1、梯度裁剪 0.5、4 worker 和 shard 分片规则沿用原正式训练。

## 学习率与恢复语义

使用 `exp=gem_smpl_motionmillion_continue`，必须提供 `resume_mode` 完整 checkpoint
路径。`pretrain_ckpt` 和 `ckpt_path` 保持 null。阶段调度器在 Lightning 恢复 optimizer
之后读取其真实 LR（s210000 为 2.2768240790631672e-6），前 1000 次更新平滑升至
1e-5，再用 19000 次更新余弦下降至 2e-6。旧 total_steps 不覆盖本阶段配置。

阶段内断点恢复使用相同配置和该阶段 checkpoint；阶段身份不一致会报错，不能把新的
学习率计划悄悄应用于旧阶段。`max_epochs=-1`，正式训练由 `max_steps=230000` 控制。
`quality_monitor/resume_audit.json` 记录恢复步数、LR、AdamW 状态数量及首个状态的
动量片段校验值，训练日志有 `[Continuation restored]`。

## 固定轻量质量监控

原 16 条无指标 Lightning validation 被关闭。新 callback 从官方 val 对应的转换
数据里确定性选择 32 条动作，每条中心裁剪、第一条 caption、4 个固定 seed；跨
8 个 rank 分担计算。样本增强、扩散噪声及生成 seed 固定，检查后恢复所有 RNG，
复用训练模型和预计算 T5，不加载第二份模型或运行 T5。

- 210000 建立起点基线，之后每 5000 步及训练结束检查。
- `quality/loss`：固定扩散噪声下、eval 模式的原加权验证损失，不等同训练 loss。
- `quality/diversity_m`：同文本不同 seed 的根相对 FK 关节轨迹两两距离均值。
- `quality/foot_slide_proxy_mps`：Y-up 足端近地且低垂直速度区间的水平速度。
  地面取该生成序列足端高度 5% 分位，故它是运动学代理，不是接触力或真实地面测试。
- `quality/contact_fraction` 与 `quality/acceleration_mps2`：辅助观察接触覆盖与抖动。
- 每个动作先对 4 个 seed 平均；与历史最佳做配对 bootstrap（2000 次）。损失相对
  改善至少 0.2% 且改善的 95% bootstrap 下界为正，才满足可靠改善条件。
- 同时要求多样性不少于起点的 90%、脚滑不高于起点的 110% 加 0.005 m/s、近地
  接触覆盖不少于起点的 50%。连续 3 次不满足则所有 rank 正常早停并补存 checkpoint。

这些阈值是本轮轻量监控的明确操作口径，不是经过任务质量标定的通用阈值。由于当前
官方 evaluator 资产未配置，本阶段不输出正式 FID/R-Precision/Matching Score；
不得称为已验证文本语义或正式最优模型。官方评测仍复用 tools/eval 中现有入口。

## 产物与复现

每 5000 步保存完整编号 checkpoint；正常结束或早停时额外刷新最终编号文件和 last。
旧训练目录与 checkpoint 不修改。新实验 `version_*/quality_monitor/` 保存：

- `cohort.json`：固定样本索引、文本/动作 manifest SHA、生成配置和判定规则。
- `resume_audit.json`：实际状态恢复证据。
- `history.json`：本阶段所有检查的数值，属于正式训练记录。
- `baseline.npz`、`latest.npz`：整套固定样本/seed 的 SMPL 动作，只有基线及最新结果。
  NPZ 的 metadata_json 包含文本、seed、帧数和身份，其余键按动作编号/seed/SMPL 字段
  保存数组。这组基线是本阶段固定对照，阶段结果被下一套验收结果替代后再清理。

通过源码入口 `tools/train/preflight_distributed.py` 执行每卡 CUDA 与 32/256 MiB
all-reduce/broadcast 检查。所有 smoke/回归测试输出只放系统临时目录，结束即清理。
正式运行命令遵循以下形式，输出路径及 checkpoint 必须替换成已核验的绝对路径：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=lo TORCH_NCCL_BLOCKING_WAIT=1 \
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train.py exp=gem_smpl_motionmillion_continue \
  resume_mode=/absolute/path/to/s210000.ckpt \
  output_dir=/absolute/path/to/new_stage_output
```

启动前先检查同分支工作树、GitHub 同步、GPU 空闲和 checkpoint 身份；依次完成本地
回归、推送、服务器 ff-only 同步、完整拓扑检查和隔离短程恢复测试，再正式启动。
服务器1以上四个 NCCL 环境变量沿用已完成的 215k 正式训练；完整拓扑检查也必须使用
相同环境。不能用 `NCCL_CUMEM_ENABLE` 替代 `NCCL_CUMEM_HOST_ENABLE`。
