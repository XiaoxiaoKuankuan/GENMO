# Server-2二十万步八卡联合训练

本记录对应2026-10-28新建的正式run；它从随机GENMO和空AdamW开始，不是旧10000步模型的续训。
分支为`feature/joint-diffusion-physics-from-scratch`，实际启动代码为
`12d3f0d40d54df83f08184205097d2088e49142b`；之后的交付文档提交不改变运行中进程。

## 正式配置与执行身份

配置：`configs/closedloop/joint_physics_formal_200k_bf16_8gpu_server2.yaml`。

| 项目 | 实际配置 |
|---|---|
| 网络 | 原GENMO 1024维、16层Transformer，随机初始化 |
| 表示/监督 | 60帧@30Hz、qpos30/contact2、完整Stage1 v5、1000步均匀扩散监督 |
| 神经计算 | BF16 autocast，仅音乐/历史/前缀/FutureEncoder/Transformer |
| 物理与优化器 | codec/FK/SM/WM/Loss、主参数及单个AdamW状态FP32 |
| GPU | Server-2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7，8个同步rank |
| 普通监督 | 每卡256，全局2048，每次global_step为一次全局optimizer更新 |
| 物理/Future | 各每卡4；各每4步，Future辅助权重0.1 |
| 物理课程 | H32，start=20000，warmup=30000，lambda_exec=0.02 |
| 无效物理窗口 | 保留数据监督；允许至少8步合法前缀，实际长度分开计数 |
| 优化器 | AdamW，峰值LR=1e-5，1000步LR warmup，grad_clip=1.0 |
| 总预算 | 200000次全局optimizer更新 |
| 保存/监控 | 每5000步完整checkpoint，每100步简要监控，每10000步数值生成诊断 |
| 采样核 | 完整DDIM20、CFG2.5、eta0，保持原FP32采样入口 |

北京时间2026-10-28 11:26启动一次，tmux为`joint-200k-bf16-20261028`。
pane PID为505373，torchrun PID为505377，八个rank PID为505404–505411。
这些PID是启动快照，后续查询需核验实时进程，不能当作长期固定身份。

输出根目录：

```text
/data0/user/liwei/GENMO_outputs/joint_physics_formal_200k_bf16_20261028/
  launch.json                 # 启动提交、预算、PID与配置
  launch_formal.sh             # 独立tmux启动脚本与flock防重复启动
  train.log                   # 简要训练输出
  run/config.json             # 实际解析配置
  run/metrics.jsonl           # 每100步聚合指标；样本身份仅详细诊断步
  run/joint_00000000.pt        # 已验证空AdamW、随机GENMO、8rank RNG
  run/joint_00005000.pt        # 第5000步才产生，依次保存到200000
  sidecar_increment/          # 新增真实离线状态与合法Future请求
  sidecar.log
  preflight_summary.json      # 八卡大batch、H32和恢复一致性证据
  precision_check.json        # BF16真实资产数值检查
  startup_checks.json
  temporary_cleanup.json
```

原10000步输出`joint_physics_formal_8gpu_20261027`保持不变。

## 本轮必要检查结果

- 17项针对性回归通过：批量Future输出及梯度、BF16边界、原完整FK与精简编码逐字段一致、
  CPU预取与恢复、增量原子发布、双rank零有效物理样本和完整checkpoint恢复。
  首次两项因本地stats默认路径在服务器不存在而失败，显式使用服务器真实stats后通过。
- 真实1024×16随机模型、4个合法真实请求、固定噪声：BF16 x0 RMSE=0.0001273204，
  contact RMSE=0.0001208583；完整v5 Loss相对差=5.7524e-7，非零代表梯度cos最小0.9999894。
  FP32/BF16都精确保留已提交前缀，目标/噪声/加噪结果一致，参数与物理输出FP32。
- 八卡每卡256运行两步；临时将物理课程设为step0，只用于启动检查。32个实际学生预测
  样本全部完成H32，一次执行梯度更新，x0执行梯度最大范数0.4442869，执行Loss0.7893301。
  此检查不表示物理收益或舞蹈质量，历史支持域超出仍按正式需求记录为诊断。
- rank0峰值分配18,099,695,616字节，约16.86GiB；第二步纯监督约2.13秒。
  实际正式运行另有GPU7离线采集，吞吐应以正式日志为准。
- 保存step1后恢复到step2：actor、AdamW、scheduler、8rank数据与噪声/RNG、课程、计数、
  下一批样本及loss/梯度范数均与不中断运行逐项精确一致；FP32优化器状态与有限性检查通过。
- 正式step0 checkpoint为869342691字节，确认global_step=0、AdamW为空、8rank状态完整，
  全库4287个训练source；正式step1→3持续增长，step3 data_loss=4.8362312。
  正式物理更新此时为0，因为未到20000步，启动检查更新不计入正式run。

BF16审计最初错误地对照GT目标检查已提交前缀，已改为严格对照实际`known_x`；首轮失败
JSON另存保留，未删除前缀门禁。临时测试checkpoint及数据已精确清理，摘要和失败说明保留。

## 数据加载与断点边界

每rank使用4个CPU装配线程、1批预取、4GB有界整曲缓存。采样权重和row索引预先缓存；
完整序列接触/地面复用原算法，窗口的qpos30编码使用已有不含额外FK的同数学入口。
后台线程仅处理显式source/frame计划；只有已消费批提交PCG游标，保存时不提前越过queued批。
恢复重建下一批，不保存大批排队张量。普通监督/Future保留原Stage1的rank等权batch归约；
物理项按全局有效样本数归一化，零有效rank仍参加同一backward。

增量索引由rank0读取并广播统一快照；断点保存该快照和每rank source/time/noise游标。
外部发布事件如果改变，后续刷新后的样本会相应变化，不承诺外部输入改变后的轨迹仍相同。

## 持续扩充真实sidecar

旧两个目录只读复用，第三目录由`joint-sidecar-200k-20261028`独立生产，启动PID371085。
四来源固定seed交错，目标新增128首，每首最多1000个50Hz控制步，失败与终止原样保留。
单写者锁、分片fsync后原子发布和既有source身份核验避免重复、半写入和覆盖。
训练初始化已读取19个有合法请求的source，后续每500步统一刷新，无需重启训练。

该进程使用GPU7，实测额外约0.85GB显存，与本run rank7共享计算资源；训练循环本身不启动
Newton。它属于离线数据生产，不是训练中的Newton评价或认证实验。当前128首任务完成后，
可继续扩到256首（已有source会跳过，运行中的同目录生产者会被锁拒绝）：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/joint-physics/GENMO
CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /data0/user/liwei/envs/genmo_sm_runtime/bin/python \
  tools/prepare_joint_physics_data.py \
  --config configs/closedloop/joint_physics_formal_200k_bf16_8gpu_server2.yaml \
  --split train --set prepare.max_sources=256 \
  --output /data0/user/liwei/GENMO_outputs/joint_physics_formal_200k_bf16_20261028/sidecar_increment
```

## 监控与恢复

以下命令在Server-2执行；当前正式任务已启动，不应再次执行新训练命令。

```bash
tmux attach -t joint-200k-bf16-20261028
tail -f /data0/user/liwei/GENMO_outputs/joint_physics_formal_200k_bf16_20261028/train.log
tail -f /data0/user/liwei/GENMO_outputs/joint_physics_formal_200k_bf16_20261028/sidecar.log
nvidia-smi
```

需要恢复时先确认原任务确已退出，然后使用最新**已存在**的完整编号checkpoint，并选择
尚不存在的输出目录。下例第5000步文件需实际生成后才能使用；200000是绝对global_step终点，
不是在恢复点额外再运行200000步：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/joint-physics/GENMO
JOINT_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
JOINT_PYTHON=/data0/user/liwei/envs/genmo_sm_runtime/bin/python \
bash scripts/train_joint_physics.sh \
  configs/closedloop/joint_physics_formal_200k_bf16_8gpu_server2.yaml \
  --max-steps 200000 \
  --resume /data0/user/liwei/GENMO_outputs/joint_physics_formal_200k_bf16_20261028/run/joint_00005000.pt \
  --output /data0/user/liwei/GENMO_outputs/joint_physics_formal_200k_bf16_20261028/resume_00005000
```

恢复命令同样应放在tmux中执行。勿将已有10000步FP32训练的checkpoint作为本run恢复点。
