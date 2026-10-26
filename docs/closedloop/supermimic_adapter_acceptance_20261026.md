# 2026-10-26 Server-1 Adapter 修复、真实验收与未启动说明

本报告依据本轮实际文件、进程与真实 Newton/MuJoCo Warp 仿真。**正式 music Adapter 训练未启动，accepted_adapter_updates=0，第一次有效 Adapter 更新不存在。** 原因是 Critic 独立真实 MC 与拟合后的 WM 真实方向两道启动门槛均未通过；没有用数值测试、正则梯度或小预测误差替代合格结论。

用户最后明确指定 Server-1。本轮没有连接或修改 Server-2，没有切换共享 GENMO/GMT 工作树。开始时 Server-1 原任务占满 GPU；另一个已授权停止流程在18:48完成后，本轮才使用空闲 GPU。原任务停止证据位于 `/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_five_changes_validation_20261026_0330_launch/formal_resume/user_stop_final_audit_20261026.json`；本轮未发出停止原任务的命令。

**代码、资产与数据**

- 本地独立目录：`/home/weili/bumi-closedloop-worktrees/adapter/GENMO`；Server-1：`/home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO`。目标分支始终为 `feature/stage2-supermimic-adapter`。最后代码提交 `ce0401732cf795577005ac97a2a3e24e29c508dc`；物理 WM25 验收运行提交 `0fef182db80f32880c9806b1d665c0073035f036`，其后的代码改动只补完整评价分项。最终文档提交以交付时 Git HEAD 为准。
- SuperMimic 分支 `feature/genmo-worldmodel-bridge`，SHA `51b2172288d645ad930edaac6eac50ec6b16b772`，本轮未改其源码；本机、GitHub与Server-1已核对。
- 独立 Python：`/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python`，Python3.12.13 / Torch2.11.0+cu130 / Newton1.8dev0 / MuJoCo3.15 / Warp1.17。IsaacLab在 `/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab`，源码46d6cc3d7b115906fad12efd372fbf69f4e3d3ed；Newton使用固定71eeba16a717d4dcf0b9dd7056a7b6f863054dc3的已安装包副本，未改原训练环境。
- Stage1 s595000直接只读使用Server-1既有资产，SHA `51b522fbdeb0ec289e5ffc24e2fccca9888df90495f57f96714c1bb43c05f8e2`。SM024000及BUMI物理资产从已核验副本复制到独立资产根，SM SHA `172fbc720583f415fdf27560c91bb8c6f6efe8165738d961cf2455694789cb54`。全部实际路径、音乐/音频/初态SHA见 `assets/assets.json`。
- AIST++ / AIOZ-GDANCE / FineDance / Mine共20任务，8 train / 4 validation / 4 audit / 4 evaluation，song group、音频SHA、EDGE35内容SHA跨split互斥。所有任务都保存实际冻结GENMO接管计划与真实尾支持，音乐长度未缩短，不外推缺失名义轨迹。
- 固定音乐s0只用8首训练歌30993个真实控制步；4首验证歌7499步未进入标尺。s0范围0.4877417~1.9454427。12首训练/验证歌曲全部music_end，合计38492步；完整回放供MC/WM复用。
- 缓存准备的长停顿、连续晚到失败均保留。先处理有界no_grad生成过程的循环GC停顿，剩余3首声明200ms生成预约余量；这不修改Adapter十帧、仿真dt、音乐时钟或WM/Critic阈值。

**实现与原因**

| 文件/模块 | 改动及理由 |
|---|---|
| music_reward.py、test_music.py | 替换错误逐时刻BCE为因果软事件precision/coverage与多事件成本；保持真实执行qd、固定拍点与s0、同中心只结算一次。尖锐同步信号不再被高斯肩部反复惩罚；静止、恒速、微小活动、额外高频与缺拍有明确成本。 |
| quality.py、critic.py、trainer.py | Critic按完整真实MC与独立歌曲EV/MSE/失败附近质量解锁；保留跨轮最优模型/Adam。WM固定歌曲均衡时间窗口检查H1/H8/H16，至少25次真实监督后验收；执行梯度、参数/参考/动作变化与物理证据共同约束accepted计数。 |
| nominal.py、collector.py、contracts.py、replay.py | 保存实际存在的曲尾名义支持，音乐/物理终止优先；记录真实reset、前缀截断、真终止元数据，拒绝残缺MC及跨reset/版本窗口。失败生成也保存真实轨迹与原因。 |
| audit.py、imagination.py、dynamics.py | 真实同初态/前缀/RNG重放，直接检验270维残差和21维动作输入方向，报告重复性、接触切换和噪声。Adapter图仅使用独立WM，更新期间冻结WM/Critic参数，保留输入梯度。 |
| baseline.py、reference.py、checkpoint.py | 零策略真实回放一次采集，按最终固定s0因果重算音乐奖励；名义FK缓存只读数据，checkpoint共享存储去重，避免整曲资产逐步复制。 |
| entry.py、application.py、evaluation.py、独立config | 独立prepare/train/evaluate/audit；预检完整取证但不更新Adapter；绝对1000目标；每25小配对/保存、每100完整审计，真episode边界更新；实际物理指标和全部原始/加权奖励报告。 |
| tools/prepare_stage2_supermimic_adapter_assets.py | 使用真实四库manifest与Stage1/RobotSpec构建可追溯任务与初态，不使用/ABS占位数据。 |
| docs/closedloop/supermimic_adapter.md、记录文本.md | 方法、门槛、真实结果、失败边界与每批改动/验证均已更新。 |

保留2654→512→512→256→270的三隐藏层SiLU Adapter、10×27独立残差、SO(3)、真实第11名义点导数halo；原SM六项tracking、动作rate/-0.1和实际joint-limit/-2.0继续直接调用且dt不重复。没有PPO、GENMO BC、扩散微调或在线WM预瞄分支。

**已执行验证**

- 本地最终31项通过（14.22秒）；Server-1最终31项通过（16.31秒）。覆盖270/10×27/Goal570、SO(3)、零参考/速度/Actor/原SM奖励、时钟/halo/重叠/延迟、执行及music-only FD、故意断图负测、MC截断、坏门槛拒绝、microbatch与完整恢复。
- 未改原信号的排序：sharp_sync=-0.288448，高于sync=-0.760229、weak_sync=-0.930451、static/constant=-1.055085；quarter=-1.589025、half=-1.572300、highfreq=-6.903543、extra_peaks=-2.498622、low_activity=-1.055072，无beat按mask处理。
- 真实479点执行速度、最终s0、无未来补分：纯Torch音乐函数梯度0.972675580108，中心FD0.972675579680，相对误差4.40e-10（float64）。这是音乐函数导数证据，不等同真实动作方向。
- 真实起点GPU H16 microbatch1/2：目标差0，1822734个参数梯度最大差4.77e-7。零Goal最大差4.63e-8、Actor最大差4.77e-7；原冻结张量保持不变。
- 数值checkpoint下一更新逐参数一致；本次也实际恢复1.1GB真实完整checkpoint，随后完成25次独立WM监督及新物理reset审计。求解器隐藏状态未保存，恢复始终从新episode开始。

**真实门槛与阻塞**

| Critic 指标 | 实测 | 事前门槛 | 结论 |
|---|---:|---:|---|
| train / validation 样本 | 30993 / 7499 | ≥128 / ≥64 | 通过 |
| Explained variance | -0.474264 | ≥0.1 | 失败 |
| MSE / 常数基线MSE | 3.938287 / 1.999986 | 相对改善≥20% | 实际恶化96.92%，失败 |
| 真实失败附近样本 | 0 | ≥16，且MSE比≤0.8 | 数据不足，失败 |

Critic进行了5×20 epochs、12200次_fit_value调用，并恢复独立验证最优模型/optimizer；这些拟合调用不是accepted Adapter更新。仅继续增加epochs不能弥补失败样本缺失。

| 独立WM真实验证 | 根位置m | 根姿态rad | 关节位置rad | 关节速度rad/s |
|---|---:|---:|---:|---:|
| H1 | 0.00090492 | 0.00351076 | 0.00155627 | 0.09794997 |
| H8 | 0.00235585 | 0.01061444 | 0.00254273 | 0.06743108 |
| H16 | 0.00296732 | 0.02435899 | 0.00360112 | 0.09667812 |
| 门槛 | 0.05 | 0.1 | 0.1 | 2.0 |

以上32个固定验证窗口全部通过。独立WM已做25次真实8步监督，参数L2变化0.407725；原WM、Actor及normalizer逐张量未变。**预测误差合格不代表奖励方向正确。**

物理审计为4首独立歌×起点0/50×H8/H16，共16个case、32个残差/动作方向。原复制WM只有7/16通过；真实拟合25次后9/16通过，方向26正、4接触切换不确定、1明确负向、1低于噪声。保持原门槛，不把不确定记作通过。

| 未通过case | 证据 |
|---|---|
| AIOZ l5xqQDKyKCU_27，起点0，H8/H16 | 残差正负扰动跨接触分支；不能确认可导方向。 |
| AIOZ 同歌，起点50，H8 | 重复状态最大差0.000121713，超过1e-4。 |
| FineDance189，起点0，H16 | 动作方向模型导数+0.0233531，真实FD=-0.0119925；差值-0.000239849、SNR239.85、无接触切换，明确反向。 |
| FineDance189，起点50，H8/H16 | 残差扰动跨接触分支，不确定。 |
| Mine hani-baby，起点0，H16 | 重复性最大差0.0243931；动作方向低于噪声门槛。 |

**独立评价集零Adapter真实结果**

| 任务 | 控制步 | 音乐完成 | 原SM tracking（含dt） | beat F1 | 根位置RMSE m | P50/P95/P99 ms |
|---|---:|---|---:|---:|---:|---|
| AIST++_gHO_sBM_cAll_d19_mHO3_ch03 | 381 | 是 | 0.096350 | 0.5455 | 0.0504 | 22.77/23.26/23.63 |
| AIOZ-GDANCE_l5xqQDKyKCU_21_0_1620_dancer_03 | 2698 | 是 | 0.095371 | 0.4192 | 0.1500 | 22.94/23.29/23.83 |
| FineDance_172 | 3583 | 是 | 0.096721 | 0.4417 | 0.0980 | 22.79/23.13/23.84 |
| Mine_dance_2__看看世界有多大 | 1366 | 是 | 0.097313 | 0.4280 | 0.0820 | 22.82/23.19/24.18 |

4/4首完成，8028个真实控制步，全部超过20ms墙钟deadline。FK缓存将前期29.9ms降至此处约22.8ms，但仍不具备20ms完整实时执行证据。原SM六项、节拍密度/双向距离、幅度、jerk、真实接触脚滑、根姿态误差等完整值在 `preflight_all/zero_physical_baseline.json`；未把该零基线当训练提升。

最终s0下训练真实回放每步加权项：SM +0.08725249，music -0.01120734，reference-limit -0.000068895，activity -4.54e-8，magnitude/smooth/overlap均0。验证对应SM +0.09397368、music -0.00955963。music权重0.2的均值量级约为SM的12.84%；一条H16模型轨迹的参数梯度范数music33.36494、SM1.64787、terminal-value0.97156，零残差正则梯度为0。**0.2仍是未获真实质量验收的候选权重，不能仅凭均值小而批准训练。**

**进程、完整checkpoint和运行命令**

- 第一轮：tmux `genmo_adapter_preflight_20261026`，PID1562600，GPU5，退出码2。
- 拟合WM重验：tmux `genmo_adapter_wm25_preflight_20261026`，PID1908417，GPU5，退出码2。两会话和PID均已正常退出。
- 2026-10-26T20:27:34.857097+08:00现场：GPU计算进程为空；GPU5 UUID `GPU-c9bda0e6-905a-936c-dde1-93074571bacf`，0%利用率、6MiB显示基础占用。仅保留既有 `genmo-rolling-20261022_182720` tmux，本轮未动它。正式训练PID、tmux、run目录均未创建。
- 所有实际证据根：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026`；本机文本镜像：`/home/weili/bumi-closedloop-worktrees/adapter/acceptance_20261026`。验收目录不等于正式训练目录。
- 最终完整checkpoint：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/preflight_wm25/preflight_failed_000000_1793017228974384084.pt`，SHA `8c6d9635bb5cf278c9ae16491d669c1e7759e950e4d9b76c088ed11f2fc70a75`。包含Adapter/Critic/WM、三optimizer、RNG、12完整真实episode、音乐/残差/游标和失败报告。Adapter Adam state为空是因为0次Adapter更新；Critic与WM Adam分别6/14组状态。`critic_ready=false`，不能作为合格模型使用。

以下两条是已经实际执行的完整命令；对应launcher JSON和shell脚本保留于证据根。

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u tools/train_stage2_supermimic_adapter.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/assets/runtime_margin60.yaml --tasks /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/tasks_all.json --music-baseline /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/music_baseline_all.pt --baseline-replay /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard0/real_replay.pt /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard1/real_replay.pt /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard2/real_replay.pt /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard3/real_replay.pt --preflight-only --updates 1 --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/preflight_all --torch-device cuda:0 --visualizer none --deterministic
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u tools/train_stage2_supermimic_adapter.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/assets/runtime_preflight_wm25.yaml --tasks /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/tasks_all.json --music-baseline /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/music_baseline_all.pt --preflight-only --target-updates 1 --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/preflight_wm25 --torch-device cuda:0 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/preflight_all/preflight_failed_000000_1793016807909321215.pt
```

正式1000目标命令已具备完整实际资产路径，但**本轮未执行**；现有失败数据仍会被入口门槛拒绝。后续必须先补充失败附近真实数据、改进Critic独立校准及WM方向/重复性，再按相同阈值重验。不能绕过这两项直接将以下命令称为已启动。

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u tools/train_stage2_supermimic_adapter.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/assets/runtime_margin60.yaml --tasks /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/tasks_all.json --music-baseline /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/music_baseline_all.pt --baseline-replay /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard0/real_replay.pt /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard1/real_replay.pt /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard2/real_replay.pt /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/zero_baseline_shard3/real_replay.pt --target-updates 1000 --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/formal_music_1000_not_launched --torch-device cuda:0 --visualizer none --deterministic --formal
```

独立零策略评价命令（下一次显式评价使用新输出目录，当前已通过预检内部实际完成上述4首）：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
CUDA_VISIBLE_DEVICES=5 ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B tools/evaluate_stage2_supermimic_adapter.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/assets/runtime_preflight_wm25.yaml --tasks /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/tasks_evaluation.json --music-baseline /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/music_baseline_all.pt --variant zero --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/evaluation_zero_next --torch-device cuda:0 --visualizer none --deterministic
```

正式恢复沿用绝对`--target-updates 1000`并指定完整`--resume`，不能加预算。当前WM25失败checkpoint可用于继续预检取证；正式入口依旧检查Critic/WM/真实方向。no_music/music评价接口要求各自真实训练完整checkpoint，并用`--compare`检查相同名义/初态/噪声及更新/真实采集预算；本轮没有生成这两个已训练Adapter，未执行其配对收益评价。

本轮未运行：正式Adapter更新、第一次更新后物理评价、1000有效更新任务、已训练no_music/music消融、滚动模式新分布验收及实体机器人。代码/配置/日志均保留，不用旧H16结果自动解锁当前分布。

本机pytest临时模型目录已逐个清理；服务器pytest临时目录由入口清理。用户要求保留的真实轨迹、缓存、两份失败完整checkpoint、编译运行资产及所有门槛报告保留。

本轮实际改动文件（相对审计起点01d3c06）：

- `configs/closedloop/stage2_supermimic_adapter.yaml`
- `configs/closedloop/stage2_supermimic_adapter_criteria.example.json`
- `docs/closedloop/supermimic_adapter.md`
- `docs/closedloop/supermimic_adapter_acceptance_20261026.md`
- `gem/closedloop/reference_adapter/application.py`
- `gem/closedloop/reference_adapter/audit.py`
- `gem/closedloop/reference_adapter/baseline.py`
- `gem/closedloop/reference_adapter/checkpoint.py`
- `gem/closedloop/reference_adapter/collector.py`
- `gem/closedloop/reference_adapter/contracts.py`
- `gem/closedloop/reference_adapter/critic.py`
- `gem/closedloop/reference_adapter/dynamics.py`
- `gem/closedloop/reference_adapter/entry.py`
- `gem/closedloop/reference_adapter/evaluation.py`
- `gem/closedloop/reference_adapter/imagination.py`
- `gem/closedloop/reference_adapter/music_reward.py`
- `gem/closedloop/reference_adapter/nominal.py`
- `gem/closedloop/reference_adapter/quality.py`
- `gem/closedloop/reference_adapter/reference.py`
- `gem/closedloop/reference_adapter/replay.py`
- `gem/closedloop/reference_adapter/trainer.py`
- `tests/closedloop/reference_adapter/test_learning.py`
- `tests/closedloop/reference_adapter/test_music.py`
- `tests/closedloop/reference_adapter/test_quality.py`
- `tools/prepare_stage2_supermimic_adapter_assets.py`
- `记录文本.md`
