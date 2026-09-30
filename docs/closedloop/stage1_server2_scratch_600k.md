# 服务器2 Stage1：当前Mine数据、随机权重、600000优化步

本次用户明确授权完成配置和数据准备后直接启动训练。600000表示全局优化步数，
不是epoch，也不是八张卡各自独立训练600000步。八个rank共同训练一个Actor。

## 固定配置

正式配置为`configs/closedloop/stage1_server2_scratch_600k.yaml`，启动脚本为
`scripts/train_stage1_8gpu_server2_scratch.sh`。仅改变用户指定的P分布、数据/统计量、
初始化方式和训练总步数；checkpoint间隔为磁盘容量调整。网络、历史代理、损失、
原学习率、四库采样权重和现有固定生成验证均保持原实现，没有新增对照实验。

| 项目 | 本次设置 |
|---|---|
| 初始化 | 随机初始化；不加载音乐权重、旧Stage1权重或旧optimizer/global_step |
| 前缀 | 15%请求P=0，85%均匀请求整数P=6～30；序列末尾按有效长度裁短 |
| 训练总步数 / cosine周期 | 600000 / 600000 |
| warmup / 基础学习率 | 500 / 1e-5 |
| GPU / batch / 精度 | 8卡DDP，每卡256，全局2048，BF16 |
| 模型 | 1024维、16层、8头；H50；120帧@30Hz；qpos30/contact2 |
| 完整断点 | 每5000步和最终步保存；120份约310GB，不删除旧训练结果 |
| 验证 | 每500步；四库各2个固定样本，DDIM20、CFG2.5 |
| Python | `/data0/user/liwei/envs/GENMO-cu128/bin/python`，Torch2.7.1+cu128 |

从随机权重开始允许架构中已有的零初始化残差/输出层，不等于从旧模型加载这些权重。
新run的`weight_loading_report.json`必须为`random_initialization`，optimizer和global_step
均未恢复；正式训练开始后用该报告、首批训练日志及八卡进程共同确认实际执行。

## 当前数据和新统计量

两台服务器继续使用同名目录：

`/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1`

2026-09-30从服务器1再次同步并通过rsync内容复核。服务器2四来源、train/val/test共
12套清单严格验证通过，包含payload、来源SHA、对齐、质量契约及关节限位。
Mine为`source_csv_root_z_preserved_v1`，高度恢复版本为
`genmo.mine_restore_csv_root_z.v1`；训练入口不读取`Mine.legacy_height_backup_20260928`。
99条Mine动作payload另逐条核验：均使用恢复后的高度身份、完整序列FK固定世界Z=0
派生的接触标签，contact形状为[T,2]且全部为二值有效标签。

新统计量使用既有`tools/data/bumi/compute_bumi_30d_stats.py`，以当前四库train清单、
120帧窗口和120帧stride重新计算；关节限位容差与训练配置一致为0.0001rad。
实际覆盖AIST++351、AIOZ-GDANCE3745、FineDance102、Mine89，共4287条train，
42404个窗口、5088202个有效特征帧；没有读取val/test计算统计量。

- 统计量：`/data0/user/liwei/GENMO_assets/stage1_scratch600k_20260930/qpos30_train_stats.json`
- SHA256：`c7c0acb0e7a38797adb38ce0be41ee72820e34daed4e780275c64fb55c3e5326`
- 数据验证报告及统计生成日志保存在上述资产目录，属于本次训练的数据来源记录。
- 旧stats和旧checkpoint保持原样。启动脚本同时绑定Dataset和Endecoder的新stats路径，
  并核对统计SHA、四库train清单/元数据SHA及Mine恢复版本；不绕过旧权重兼容性检查。

## 启动与监控

在`/home/user/liwei/GENMO-bumi-closedloop`执行：

```bash
bash scripts/train_stage1_8gpu_server2_scratch.sh --check-only
bash scripts/train_stage1_8gpu_server2_scratch.sh
```

`--check-only`只核验启动条件，不创建模型或训练输出。新训练输出目录固定为：

`/data0/user/liwei/GENMO_outputs/bumi_closedloop_stage1_90505_height20260928_p6to30_z15_scratch_b256_s600k_8gpu_20260930`

启动前确认GPU无其他计算任务和输出目录为空；正式运行放入独立tmux会话。运行产生的
`run_contract.json`、`weight_loading_report.json`、`config_from_s000000_to_s600000.yaml`、
`train_*.jsonl`、`tensorboard/`及`checkpoints/`均属于正式结果，不按测试缓存清理。

断点恢复必须继续使用本配置、8个rank、同一数据/统计量和本run的完整checkpoint，
通过既有`tools/train_closedloop_stage1.py --resume-checkpoint 路径`入口恢复全部状态；
新训练脚本会拒绝非空目录，避免把恢复和重新初始化混为一谈。

## 准备检查与执行边界

已完成两台服务器代码/数据同步、12套严格数据检查、新train统计量生成、99条Mine
高度/接触来源核验，以及启动脚本语法、配置合并和随机初始化入口检查。

2026-09-30 21:09:30（UTC+8）在服务器2 ZP-NC579正式启动，启动代码为`f2ca92f`。
tmux会话为`genmo_stage1_scratch600k_s2`，GPU0～7对应rank0～7，训练PID为
96637～96644。21:11:15实际核验已完成107个优化步，采样epoch=3、每rank offset=2816；
前107步的loss、梯度、学习率和耗时均有限，所有日志均包含8个rank且采样游标一致。
观测loss范围5.09466～5.52455，每卡峰值allocated约32329～32335MiB。

实际`weight_loading_report.json`为`random_initialization`，
`global_step_restored=false`、`optimizer_restored=false`。已解析训练器落盘配置确认
max_steps和scheduler.total_steps均为600000，三类已有权重/恢复入口均为null。
截至上述核验时刻，尚未到5000步保存点，checkpoint数量为0；不将目标步数冒充完成步数。

- console：`/data0/user/liwei/GENMO_outputs/launch_logs/stage1_scratch600k_server2_20260930.console.log`
- 启动和运行核验：资产目录中的`launch_manifest.json`、`launch_precheck.log`、`launch_audit.json`。
- 本地数据与启动证据：`/home/weili/bumi-closedloop-worktrees/stage1_server2_scratch600k_20260930/`。

查看日志或接入会话：

```bash
ssh 6000D-Server-2 'tail -n 20 /data0/user/liwei/GENMO_outputs/launch_logs/stage1_scratch600k_server2_20260930.console.log'
ssh -t 6000D-Server-2 'tmux attach -t genmo_stage1_scratch600k_s2'
```

早期loss有限和八卡运行只证明训练已正常开始，不代表600000步已完成或模型质量已达标。
