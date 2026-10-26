# GENMO Adapter八组单GPU真实训练交付（2026-10-26）

现场快照：**2026-10-26T23:48:25+0800，Server-1**。八个独立训练进程均存活，每进程2048物理环境；没有DDP、torchrun、NCCL AllReduce或共享可训练模型/优化器。目标每组3000次有效更新尚未完成。八组均已实际完成至少100次Adapter有效更新及对应真实整歌评价；当前GPU1、2的后续Adapter更新受真实有限扰动方向反例阻塞，进程继续各自真实采集、WM/Critic监督与定期复验。其他组合格时域继续更新。

同预算100次评价中，8组均完成4/4首歌，但跟踪分数全部低于各自零Adapter基线；不能宣称训练成功或优于基线。当前最佳候选仅指同预算候选之间的相对排名，不表示已达到改善目标。没有因评价分数下降回滚或终止训练。

## 代码、运行边界和实际文件

- 固定运行GENMO源码：`6b63450891f776ecb7ed18250f3a05705148ccdf`，分支`feature/stage2-supermimic-adapter`。
- 配套SM源码：`d9164b0e15934806ff8d750ff70911949c7f5171`，分支`feature/genmo-worldmodel-bridge`。
- 本地实现工作树：`/home/weili/bumi-closedloop-worktrees/adapter/GENMO`；Server-1常规工作树：`/home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO`。
- 每次启动建立detached运行快照，当前代码位于`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo`及`source_supermimic`；分支后续报告修改不会热修改运行中的训练。
- 原共享GENMO/GMT工作树仍为`feature/bumi-music-closedloop`和`feature/bumi-frozen-gmt-backend`，没有切换。原`genmo-rolling-20261022_182720`会话保留，未访问Server-2。

| 文件/模块 | 实际修改与理由 |
|---|---|
| `policy.py`, `rewards.py`, SM `mdp.py`/`bridge/runtime.py` | 保持2654→512→512→256→270；增加内部打包前向；原六项奖励与详细诊断共用一次真实FK，默认音乐权重1e-4。 |
| `vector_bank.py`, `vector_control.py` | 只读歌曲库一次校验和预计算；批量ID/时间索引；Goal570直接由根与关节构造，无修正身体FK及重复四元数转换；音乐窗口/关闭时刻缓存。 |
| `vector_collector.py`, `vector_replay.py` | 真正N环境整批控制和推进、局部复位、独立episode/时钟/历史；紧凑百万容量张量Replay，无歌曲副本；完整reset片段MC不冒充整歌。 |
| `vector_learning.py`, `vector_datasets.py`, `dynamics.py` | 训练集分组归一化Critic；独立真实WM多步监督和局部动作响应损失；按歌曲检查MC/EV/MSE；H1/H4/H8/H16单独验证。 |
| `vector_audit.py`, `vector_evaluation.py` | 同真实起点、因果历史与明确fresh-solver边界的物理配对；固定多尺度复测，不覆盖可靠反向；4首固定整歌真实评价。 |
| `vector_checkpoint.py`, `vector_entry.py` | 三网络/三优化器/Replay/RNG/音乐和残差全保存恢复；真实方向决定可用BPTT时域，Critic不合格时无终端价值；有效更新必须改变参数/Goal/Actor动作。 |
| `tools/*adapter*_vector.py`, `tools/*adapter_sweep.py` | 独立train/evaluate/audit入口、逐卡tmux启动、只读自动报告；报告继承各组自己的评价恢复链，按相同预算比较。 |
| `tests/closedloop/reference_adapter/test_vector.py` | 零/非零Goal、Actor、原SM奖励/梯度；音乐独立时钟；MC/reset/恢复；批量BPTT；局部方向边界和报告恢复计数。 |

代码提交依次：f497634（向量实现）、0c57cd7（真实压力与启动）、8d8be01（固定运行源码）、227d4d9（局部接触复测）、b7fde44（真实同起点配对/评价入口）、6b63450（固定扰动序列）。后续只读报告与速率修复另见Git历史；运行算法仍固定上述完整SHA。

## 冻结资产、任务与种子

SM checkpoint：`/data0/user/liwei/GENMO_assets/adapter_20261026/ckpt_024000.pt`，SHA256 `172fbc720583f415fdf27560c91bb8c6f6efe8165738d961cf2455694789cb54`。GENMO Stage1：`/data0/user/liwei/closedloop_stage10_v3_assets/20261012/stage1_s595000/s595000.pt`，SHA256 `51b522fbdeb0ec289e5ffc24e2fccca9888df90495f57f96714c1bb43c05f8e2`。SM Actor、normalizer、原bundle及名义缓存冻结，只有Adapter、新Critic和独立WM可训练。

20首冻结GENMO缓存及划分：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/tasks_all.json`；固定训练音乐标尺`music_baseline_all.pt`。8train/4validation/4audit/4evaluation按歌曲和原音频SHA隔离。模型seed=42，环境采样seed=20261026，首轮随机流配对；之后在线Replay各自独立。仅共享只读冻结资产和历史真实零Adapter基线（30993训练/7499验证转移），不共享其他实验在线数据。

| 划分 | 真实歌曲ID |
|---|---|
| train | AIST++:mMH2；AIST++:mLO0；AIOZ-GDANCE:O8T4hPMs2-U_03_0_1890；AIOZ-GDANCE:agELgQzrZqw_02_0_1620；FineDance:swoop；FineDance:qinghuaci；Mine:dance_2__shake-it-complete；Mine:dance_3__眉飞色舞 |
| validation | AIST++:mLO2；AIOZ-GDANCE:jyDJEaUbfZM_02_0_1440；FineDance:redeye；Mine:dance_3__BangBang |
| audit | AIST++:mKR5；AIOZ-GDANCE:l5xqQDKyKCU_27_0_1590；FineDance:breakupwithyourgf；Mine:dance_2__hani-baby |
| evaluation | AIST++:mHO3；AIOZ-GDANCE:l5xqQDKyKCU_21_0_1620；FineDance:bitchbetterhavemymoney；Mine:dance_2__看看世界有多大 |

训练初态40%均匀合法50Hz时刻、40%尾部50–499步前、20%原起点；25%困难状态扰动，真实执行器偏置±0.005rad，通过原RobotSpec/xi接口。严格区分完整随机reset片段与完整歌曲，未到真实终止或回放头部丢失不生成完整MC。没有制造失败标签，当前独立验证失败附近样本仍不足。

## 八组配置及当前进程

当前正式根目录：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3`。每组目录为其下`gpu0`至`gpu7`，完整运行配置为`config_gpu0.yaml`至`config_gpu7.yaml`，实际命令为`run_gpu0.sh`至`run_gpu7.sh`。每组2048环境，Replay实际999424条（100万向下取完整环境行）。

| GPU | 仓库配置 | LR | WM tangent | magnitude | 实际PID | tmux | accepted | WM更新 | 当前BPTT |
|---|---|---|---|---|---|---|---|---|---|
| 0 | [gpu0.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu0.yaml) | 1e-4 | .1 | .01 | 2884185 | genmo_adapter8p_20261026_g0 | 284 | 4428 | 1 |
| 1 | [gpu1.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu1.yaml) | 1e-4 | .1 | .03 | 2884190 | genmo_adapter8p_20261026_g1 | 100 | 4808 | 待复验 |
| 2 | [gpu2.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu2.yaml) | 1e-4 | .5 | .01 | 2884211 | genmo_adapter8p_20261026_g2 | 100 | 4796 | 待复验 |
| 3 | [gpu3.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu3.yaml) | 1e-4 | .5 | .03 | 2884234 | genmo_adapter8p_20261026_g3 | 299 | 4092 | 1 |
| 4 | [gpu4.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu4.yaml) | 2e-4 | .1 | .01 | 2884287 | genmo_adapter8p_20261026_g4 | 336 | 4540 | 1 |
| 5 | [gpu5.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu5.yaml) | 2e-4 | .1 | .03 | 2884343 | genmo_adapter8p_20261026_g5 | 118 | 4512 | 8 |
| 6 | [gpu6.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu6.yaml) | 2e-4 | .5 | .01 | 2884398 | genmo_adapter8p_20261026_g6 | 314 | 4052 | 1 |
| 7 | [gpu7.yaml](../../configs/closedloop/adapter_sweep_20261026/gpu7.yaml) | 2e-4 | .5 | .03 | 2884477 | genmo_adapter8p_20261026_g7 | 134 | 4172 | 4 |

共同参数：music=1e-4，50Hz控制/200Hz物理，reference_steps=10，initial_bptt_steps=8，gamma=.99，lambda=.95；WM batch2048/window8/每轮4更新，Adapter batch2048/microbatch32，Critic batch2048/3epochs；每轮采8向量步；目标3000 accepted，每100次真实评价并保存完整checkpoint。仓库配置默认1024，现场压力证据支持统一覆写2048，正式配置副本明确记载2048。

| GPU | 累计已记录真实转移 | 其中train | 本轮纯采集转移/s | 恢复后有效更新/小时 | NVIDIA观测峰值MiB | Torch分配峰值MiB |
|---|---|---|---|---|---|---|
| 0 | 97091584 | 18923520 | 111596 | 726.1 | 12010 | 9499.1 |
| 1 | 121645056 | 20480000 | 128293 | 0.0 | 12002 | 9483.6 |
| 2 | 118953984 | 20430848 | 114371 | 0.0 | 12004 | 9478.8 |
| 3 | 98537472 | 17547264 | 111104 | 851.7 | 11990 | 9502.5 |
| 4 | 101539840 | 19382272 | 108193 | 741.7 | 12030 | 9512.8 |
| 5 | 99266560 | 19267584 | 110928 | 155.7 | 11994 | 9480.9 |
| 6 | 97845248 | 17383424 | 105209 | 676.3 | 12024 | 9510.5 |
| 7 | 101064704 | 17874944 | 108669 | 295.6 | 12024 | 9512.7 |

累计真实转移包含train/validation/audit/evaluation/tangent，不能全部当作优化器使用的数据；历史导入38492条另计。纯采集吞吐不含反传/审计/保存，更新速率包含本进程预热和评价耗时。恢复后速率按新增accepted计算；旧运行status中的原速率字段会高估，只读报告器已纠正其展示。NVIDIA是离散观测峰值，Torch为CUDA allocator实际记录峰值。

现场GPU状态：

```text
index, uuid, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
0, GPU-52ce833e-60bf-79e3-a32c-2faa0bb2d232, 32 %, 12021 MiB, 85651 MiB
1, GPU-a73507ed-4d06-8a35-ee6e-431c42686802, 69 %, 12013 MiB, 85651 MiB
2, GPU-7e587103-ac60-9c30-ba63-31c06c6f4656, 64 %, 12016 MiB, 85651 MiB
3, GPU-59387dab-8f57-0851-737e-8c72e6632442, 63 %, 12005 MiB, 85651 MiB
4, GPU-363902b6-2fdc-7784-b64a-0b580298975b, 27 %, 12041 MiB, 85651 MiB
5, GPU-c9bda0e6-905a-936c-dde1-93074571bacf, 30 %, 12005 MiB, 85651 MiB
6, GPU-0c19dacc-4400-7607-6e21-8ef0bf6797ee, 28 %, 12035 MiB, 85651 MiB
7, GPU-35c92d4f-93a0-a531-8e45-ad6f17e59ef7, 29 %, 12035 MiB, 85651 MiB
```

## 测试、物理压力与数值合同

最终实际CPU全套40项通过（具体耗时见记录文本）；新增报告恢复链及速率前后的vector回归均通过。实际GPU使用SM真实024000权重，零/非零残差均完成Goal/Actor/梯度对照；原SM奖励与状态梯度最大差均0。零Goal误差4.63e-8、Actor4.77e-7；非零Goal1.997e-6、Actor3.28e-7；非零残差梯度最大绝对差1.16e-4，在原相对/绝对联合容差内通过。未更改奖励公式、dt或六项身体集合。

| 环境数 | 96步真实转移 | 真实转移/s | NVIDIA进程MiB | Torch峰值MiB | 物理/WM/BPTT |
|---|---|---|---|---|---|
| 64 | 6144 | 6617.20 | 8626 | 7862.67 | 通过 |
| 256 | 24576 | 24328.51 | 8658 | 7862.62 | 通过 |
| 512 | 49152 | 41907.27 | 8722 | 7864.44 | 通过 |
| 1024 | 98304 | 78381.25 | 8816 | 7866.80 | 通过 |
| 2048 | 196608 | 137856.91 | 9046 | 7873.83 | 通过 |

2048相对1024吞吐提升75.9%，并非凭85GB显存推断。额外长测1024向量步=2097152真实转移，131903.87转移/s，NVIDIA9040MiB；1m Replay真实分配5421875200字节、两次环绕。局部reset后其他环境state55最大差0、episode ID保持；8步真实窗口next_state与后继state逐元素一致。2198完整reset片段、87完整歌曲、280头部丢失终点不计完整MC。

长测分段均值：控制2.879ms、物理4.081ms、奖励/共享FK4.363ms、接触诊断0.005ms、Replay0.683ms、复位3.468ms，总15.479ms。这是2048环境训练批量路径，不等同于真实机器人传感器/通信/50Hz部署验收。每档还实际完成WM2048×8真实监督和H8/microbatch32反传；这些测试均未计为Adapter更新。

## 第一次有效Adapter更新与完整恢复

8组最初都在28次独立WM真实拟合、H1八case/16方向通过后才接受第一次更新，阶段dense_tracking_no_terminal_value。参数、Goal和Actor动作均有实际变化，不靠正则梯度或loss下降计数。

| GPU | 执行项梯度范数 | 参数L2变化 | Goal RMS变化 | Actor动作RMS变化 | 音乐梯度范数 |
|---|---|---|---|---|---|
| 0 | 0.003678797 | 0.02601211 | 0.005517056 | 0.01788117 | 5.874836e-05 |
| 1 | 0.003675458 | 0.02601251 | 0.00551649 | 0.01788845 | 5.867895e-05 |
| 2 | 0.003672436 | 0.02601253 | 0.00551219 | 0.01793716 | 5.614141e-05 |
| 3 | 0.003669486 | 0.02601104 | 0.005513818 | 0.01796748 | 5.614662e-05 |
| 4 | 0.003673455 | 0.05202547 | 0.01103485 | 0.03609569 | 5.868262e-05 |
| 5 | 0.003676421 | 0.05202613 | 0.01102761 | 0.03611944 | 5.873933e-05 |
| 6 | 0.003667002 | 0.05202424 | 0.01101916 | 0.03600637 | 5.614902e-05 |
| 7 | 0.003669037 | 0.0520248 | 0.01102497 | 0.03608351 | 5.619099e-05 |

完整checkpoint实际读回核验三优化器状态数Adapter8/Critic6/WM14，含训练归一化、Replay、验证Replay、局部响应数据、RNG、残差/音乐/控制历史；冻结bundle逐张量与原基线完全相同。已经实际进行了两轮完整恢复，物理求解器从新episode启动，不冒充恢复隐藏接触状态。

第一轮保存后迁移解决接触扰动；第二轮使用显式stop_after_checkpoint.request在当前迭代完整保存后正常退出，保存accepted为199/100/100/200/250/100/236/100。每组当前仅恢复自己的文件，旧目录与全部模型日志保留。第二轮停止是代码维护，不由评价变差触发。

## 同预算100次真实物理比较

以下8组均来自formal_8gpu_2048_local_v2各自evaluation_000100.json，使用同4首歌、原始起点、seed42、相同物理、无观测噪声，严格记到真实失败或音乐结束。Tracking为原SM六项乘dt后的歌曲均值；差值与该组同合同零Adapter比较。不同更新预算不混排。

| GPU | 完整完成 | 零Tracking | 100次Tracking | 相对变化 | Beat F1 | 根位置RMSE m | 根姿态RMSE rad |
|---|---|---|---|---|---|---|---|
| 0 | 4/4 | 0.096456 | 0.088297 | -8.46% | 0.4252 | 0.4017 | 0.0573 |
| 1 | 4/4 | 0.096434 | 0.087163 | -9.61% | 0.4112 | 0.5966 | 0.0519 |
| 2 | 4/4 | 0.096504 | 0.087401 | -9.43% | 0.4135 | 0.5126 | 0.0547 |
| 3 | 4/4 | 0.096501 | 0.087381 | -9.45% | 0.4023 | 0.5876 | 0.0541 |
| 4 | 4/4 | 0.096504 | 0.086308 | -10.57% | 0.4035 | 0.4631 | 0.0570 |
| 5 | 4/4 | 0.096524 | 0.087641 | -9.20% | 0.4110 | 0.5191 | 0.0515 |
| 6 | 4/4 | 0.096489 | 0.085614 | -11.27% | 0.4011 | 0.4167 | 0.0578 |
| 7 | 4/4 | 0.096496 | 0.084159 | -12.78% | 0.3976 | 0.7406 | 0.0533 |

8组均低于零Adapter，GPU0在候选之间的同预算Tracking最高，但没有证据优于零Adapter，暂不宣布最佳方法。音乐作为次要指标。此表不使用WM预测回报。

| GPU | body_pos | body_ori | body_linvel | body_angvel | root_pos | root_ori |
|---|---|---|---|---|---|---|
| 0 | 0.98986 | 0.77124 | 0.97822 | 0.92332 | 0.52488 | 0.97955 |
| 1 | 0.98987 | 0.78449 | 0.97643 | 0.92425 | 0.38315 | 0.98309 |
| 2 | 0.98932 | 0.76247 | 0.97714 | 0.92243 | 0.45611 | 0.98126 |
| 3 | 0.99005 | 0.78025 | 0.97642 | 0.92383 | 0.41530 | 0.98169 |
| 4 | 0.98777 | 0.73425 | 0.97216 | 0.90760 | 0.44754 | 0.97975 |
| 5 | 0.98993 | 0.80226 | 0.97238 | 0.91377 | 0.42405 | 0.98334 |
| 6 | 0.98650 | 0.68968 | 0.97228 | 0.90400 | 0.47728 | 0.97916 |
| 7 | 0.98656 | 0.71842 | 0.97121 | 0.90705 | 0.26717 | 0.98227 |

| GPU | 平均绝对关节速度 | 平均关节幅度rad | jerk RMS | 真实接触脚滑m/s |
|---|---|---|---|---|
| 0 | 0.59916 | 1.10034 | 779.61247 | 0.04051 |
| 1 | 0.60749 | 1.10764 | 796.72255 | 0.04235 |
| 2 | 0.60076 | 1.10025 | 800.94630 | 0.04240 |
| 3 | 0.60553 | 1.10908 | 780.42393 | 0.04272 |
| 4 | 0.59142 | 1.10454 | 808.41743 | 0.04163 |
| 5 | 0.60561 | 1.11843 | 835.06862 | 0.04249 |
| 6 | 0.58583 | 1.09820 | 810.10089 | 0.04344 |
| 7 | 0.59912 | 1.10341 | 813.82979 | 0.04537 |

逐歌曲失败时间、离散beat密度/双向匹配、动作幅度、jerk、脚滑、奖励各项及控制计时均在每个evaluation JSON内。上表无失败时间，因为这轮8×4首全部完成；不能据此推断训练困难初态或新歌曲都稳定。

## WM、Critic与明确未通过项

方向审计保持原噪声底1e-6、5倍基线回报差、重复状态1e-4及原状态误差门槛。较大残差扰动会被Actor放大到约0.15–0.25rad RMS，引起接触或非线性响应；固定使用1/5、1/25、1/125、1/625序列，需要连续两档同接触且超过噪声才确认局部方向，任意有效配对上的可靠负方向保留并否决。

并排世界各自重放50步存在漂移。现先真实执行前缀，再从同一主机器人当前真实state55/xi和因果控制/残差/音乐历史建立配对，重新初始化求解器；不使用想象状态或未来真实状态，也不声称保留求解器隐藏状态。实际GPU7诊断六变体初态最大差0、动作扰动误差最大6.03e-7。诊断仍保留H16 AIOZ/50步可靠负方向（真实回报差−0.00345981，噪声1e-6），H16未用于训练。

- GPU0：最近通过时域`[1]`，当前BPTT `1`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu0/directions_wm004092_adapter000200_posteval.json`。

- GPU1：最近通过时域`[]`，当前BPTT `None`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu1/directions_wm004800_adapter000100.json`。

  具体阻塞：Mine_dance_2__hani-baby、前缀50、H1、residual_offset，真实差-1.17942691e-05、模型导数0.014619818、SNR 11.79；初态差0。不通过选择更小的一次好结果掩盖此反例。

- GPU2：最近通过时域`[]`，当前BPTT `None`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu2/directions_wm004700_adapter000100.json`。

  具体阻塞：AIOZ-GDANCE_l5xqQDKyKCU_27_0_1590_dancer_00、前缀50、H1、residual_offset，真实差-0.000382505357、模型导数0.148128539、SNR 382.51；初态差0。不通过选择更小的一次好结果掩盖此反例。

- GPU3：最近通过时域`[1]`，当前BPTT `1`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu3/directions_wm003700_adapter000200.json`。

- GPU4：最近通过时域`[1]`，当前BPTT `1`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu4/directions_wm004396_adapter000300_posteval.json`。

- GPU5：最近通过时域`[1, 8]`，当前BPTT `8`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu5/directions_wm004444_adapter000100.json`。

- GPU6：最近通过时域`[1]`，当前BPTT `1`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu6/directions_wm003996_adapter000300_posteval.json`。

- GPU7：最近通过时域`[1, 4]`，当前BPTT `4`；方向证据`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu7/directions_wm004040_adapter000100.json`。

WM独立真实预测误差（最新验证，列为根位置m/根旋转rad/关节位置rad/关节速度rad/s；预测误差低不代替物理方向）：

| GPU0时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.00066904 | 0.0030868 | 0.0011979 | 0.10768 |
| 4 | 0.0011112 | 0.010178 | 0.0019884 | 0.068341 |
| 8 | 0.0024392 | 0.015469 | 0.0026707 | 0.070931 |
| 16 | 0.0044107 | 0.039394 | 0.0038822 | 0.090003 |

| GPU1时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.0002448 | 0.0026252 | 0.00088061 | 0.067442 |
| 4 | 0.0012202 | 0.0059047 | 0.0022494 | 0.073957 |
| 8 | 0.0022713 | 0.013613 | 0.0034916 | 0.048825 |
| 16 | 0.012454 | 0.074779 | 0.0059232 | 0.069299 |

| GPU2时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.00043688 | 0.0018012 | 0.0011334 | 0.072024 |
| 4 | 0.0010069 | 0.0057152 | 0.0020945 | 0.057567 |
| 8 | 0.0015078 | 0.010576 | 0.0024751 | 0.04911 |
| 16 | 0.0035616 | 0.023052 | 0.0035469 | 0.089212 |

| GPU3时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.00027978 | 0.0018678 | 0.00090218 | 0.072824 |
| 4 | 0.00125 | 0.008694 | 0.002512 | 0.060586 |
| 8 | 0.0043425 | 0.025258 | 0.003536 | 0.076414 |
| 16 | 0.0064565 | 0.043302 | 0.0052987 | 0.14009 |

| GPU4时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.0008993 | 0.0052773 | 0.0013406 | 0.15053 |
| 4 | 0.0022062 | 0.014742 | 0.0027279 | 0.084432 |
| 8 | 0.0039797 | 0.018874 | 0.0036485 | 0.10341 |
| 16 | 0.01155 | 0.036541 | 0.0042114 | 0.13459 |

| GPU5时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.00062648 | 0.0025481 | 0.0010516 | 0.06177 |
| 4 | 0.00089161 | 0.0062896 | 0.0018988 | 0.071219 |
| 8 | 0.0017081 | 0.016543 | 0.0024891 | 0.064212 |
| 16 | 0.0037201 | 0.031303 | 0.0029862 | 0.089906 |

| GPU6时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.00086768 | 0.006051 | 0.0017257 | 0.15357 |
| 4 | 0.0037862 | 0.0080278 | 0.0038227 | 0.16377 |
| 8 | 0.0044284 | 0.020836 | 0.0044351 | 0.13496 |
| 16 | 0.0080656 | 0.072629 | 0.0052864 | 0.15099 |

| GPU7时域 | root_p | root_R | q | qd |
|---|---|---|---|---|
| 1 | 0.00032238 | 0.0030064 | 0.00092483 | 0.081284 |
| 4 | 0.0010063 | 0.018209 | 0.0022543 | 0.078341 |
| 8 | 0.0017448 | 0.031011 | 0.0032361 | 0.067628 |
| 16 | 0.0080028 | 0.053213 | 0.0048201 | 0.081653 |

Critic只用训练集字段组和回报归一化；正式使用真实独立MC校准。当前所有组critic_ready=false，均不使用不可信终端值。失败附近样本不足及MSE改善不足仍是未通过项，不能将EV为正称为校准成功。

| GPU | 验证EV | 验证MSE | 常数MSE | 失败附近样本 | 门槛 |
|---|---|---|---|---|---|
| 0 | 0.34894 | 6.24617 | 8.53688 | 0 | False |
| 1 | 0.31201 | 6.50178 | 8.38836 | 0 | False |
| 2 | 0.32768 | 5.75955 | 8.01493 | 14 | False |
| 3 | 0.30169 | 6.36929 | 8.06405 | 0 | False |
| 4 | 0.44519 | 5.30313 | 8.65509 | 0 | False |
| 5 | 0.16779 | 8.36237 | 8.27652 | 0 | False |
| 6 | 0.39650 | 6.44998 | 9.76833 | 0 | False |
| 7 | 0.22082 | 7.31943 | 7.88620 | 0 | False |

| GPU0验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.19318 | 9.06717 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.47137 | 4.53116 |
| FineDance_204 | 0.50786 | 4.01045 |
| Mine_dance_3__BangBang | 0.15483 | 7.66248 |

| GPU1验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.05916 | 10.08630 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.47957 | 4.01701 |
| FineDance_204 | 0.51348 | 4.22304 |
| Mine_dance_3__BangBang | 0.14973 | 7.60996 |

| GPU2验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.09735 | 9.42031 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.53954 | 3.92677 |
| FineDance_204 | 0.50707 | 3.96250 |
| Mine_dance_3__BangBang | 0.06541 | 6.58501 |

| GPU3验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.07755 | 8.34142 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.46088 | 4.58630 |
| FineDance_204 | 0.44821 | 4.39674 |
| Mine_dance_3__BangBang | 0.09842 | 8.09937 |

| GPU4验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.09119 | 6.69679 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.58262 | 4.14507 |
| FineDance_204 | 0.58657 | 3.43054 |
| Mine_dance_3__BangBang | 0.25135 | 6.88208 |

| GPU5验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | -0.08841 | 12.04218 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.39580 | 4.59425 |
| FineDance_204 | 0.33368 | 5.78349 |
| Mine_dance_3__BangBang | 0.02113 | 10.06300 |

| GPU6验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.15786 | 9.26428 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.58544 | 4.26319 |
| FineDance_204 | 0.73846 | 2.57204 |
| Mine_dance_3__BangBang | 0.01616 | 9.83986 |

| GPU7验证歌曲 | EV | MSE |
|---|---|---|
| AIST++_gLO_sBM_cAll_d13_mLO2_ch02 | 0.02611 | 10.45826 |
| AIOZ-GDANCE_jyDJEaUbfZM_02_0_1440_dancer_01 | 0.42578 | 5.82757 |
| FineDance_204 | 0.32528 | 5.29935 |
| Mine_dance_3__BangBang | 0.00146 | 8.16060 |

没有实际完成3000次目标、独立多种子统计、无音乐已训练对照、滚动GENMO新分布或机器人硬件验收。当前只支持固定GENMO缓存的严格配对训练；本次没有外推名义尾部。

## 实际启动、评价、恢复和查看命令

8个当前进程已按以下完整参数实际启动（不是待执行建议）。命令也在launch_manifest.json和各run_gpuN.sh中：

GPU0，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu0`：

```bash
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu0.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu0 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu0/requested_exit_000199_1793028836386467706.pt
```

GPU1，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu1`：

```bash
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu1.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu1 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu1/requested_exit_000100_1793028842524754538.pt
```

GPU2，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu2`：

```bash
env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu2.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu2 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu2/requested_exit_000100_1793028850247731763.pt
```

GPU3，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu3`：

```bash
env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu3.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu3 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu3/requested_exit_000200_1793028839526486979.pt
```

GPU4，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu4`：

```bash
env CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu4.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu4 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu4/requested_exit_000250_1793028838169245444.pt
```

GPU5，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu5`：

```bash
env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu5.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu5 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu5/requested_exit_000100_1793028836540380241.pt
```

GPU6，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu6`：

```bash
env CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu6.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu6 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu6/requested_exit_000236_1793028836754174802.pt
```

GPU7，目录`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu7`：

```bash
env CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab /data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B -u /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/source_genmo/tools/train_stage2_supermimic_adapter_vector.py --config /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu7.yaml --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu7 --mode train --num-envs 2048 --visualizer none --deterministic --resume /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_local_v2/gpu7/requested_exit_000100_1793028836334426188.pt
```

以下只读查看命令可直接执行：

```bash
ssh 6000D-Server-1
tmux ls
nvidia-smi
cat /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/comparison_latest.json
tail -n 1 /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/gpu0/training.jsonl
```

后续独立评价/恢复需要选一张空闲GPU及全新输出目录；当前8卡已有训练，不直接重复启动。使用本分支最新工具，FULL选择相应组latest_checkpoint.json指向的完整文件：

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
export CUDA_VISIBLE_DEVICES=空闲卡编号
export ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab
export TORCHINDUCTOR_CACHE_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python
CFG=/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215/formal_8gpu_2048_true_pair_v3/config_gpu0.yaml
FULL=该组完整checkpoint绝对路径
"$PY" -B -u tools/train_stage2_supermimic_adapter_vector.py --mode evaluate --config "$CFG" --resume "$FULL" --output 全新评价目录 --num-envs 2048 --visualizer none --deterministic
"$PY" -B -u tools/train_stage2_supermimic_adapter_vector.py --mode audit --config "$CFG" --resume "$FULL" --output 全新审计目录 --num-envs 2048 --visualizer none --deterministic
"$PY" -B -u tools/train_stage2_supermimic_adapter_vector.py --mode train --config "$CFG" --resume "$FULL" --output 全新恢复目录 --num-envs 2048 --visualizer none --deterministic
```

独立evaluate入口复用已实际运行的整歌函数；本次整歌评价由训练入口自动完成，未为了验证命令再占用第九个GPU进程。每100次accepted真实评价+完整保存；纯监督等待完整保存最多每30分钟一次。需要维护时才显式创建stop_after_checkpoint.request，等待stopped_after_requested_checkpoint及实际进程退出；不会因Tracking/F1下降自动创建该文件。

## 证据位置与自动报告

服务器完整证据根：`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/vector8_20261026_2215`。保留pressure64/256/512/1024/2048、2048长压力、初始方向、局部接触与真实起点诊断、三个正式运行目录及全部checkpoint/log。CPU测试临时模型按精确路径清理，真实结果按用户要求保留。

本地文本/JSON镜像：`/home/weili/bumi-closedloop-worktrees/adapter/vector8_20261026_2215`，包含`evidence_first100.json`和`delivery_live_2349.json`（内部明确记录实际时间23:48:25）。当前只读报告tmux为`genmo_adapter8p_20261026_report`，每60秒输出`comparison_latest.json`，按各组恢复链中的共同真实评价预算排序；优先完整完成与原始名义Tracking，音乐其次。

## 2026-10-27 00:01:57现场追加：持续训练与最新真实效果

八个相同PID仍存活，GPU计算进程恰好每卡一个；无Traceback、CUDA OOM、device-side assert。报告器60秒刷新已验证，现场文件年龄8.37秒。当前共同评价预算仍为100，下面各组最新评价预算不同，不能混排为同预算结论。GPU2已通过新的H8验收恢复更新；GPU1仍因其真实有限扰动反例保留100次并监督复验。

|GPU|accepted最后落盘|当前BPTT|WM累计更新|最新评价预算|整歌完成率|真实Tracking|Beat F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|gpu0|450|4|5092|400|100%|0.0600897|0.3723|
|gpu1|100|待复验|6596|100|100%|0.0871794|0.4120|
|gpu2|138|8|5348|100|100%|0.0873832|0.4113|
|gpu3|399|4|4492|400|25%|0.0577092|0.3557|
|gpu4|483|4|5128|400|100%|0.0570819|0.3831|
|gpu5|174|8|4736|100|100%|0.0876406|0.4110|
|gpu6|513|8|4848|500|100%|0.0536627|0.3616|
|gpu7|297|1|4824|200|100%|0.0811401|0.4180|

GPU3的400次评价已完成而该轮状态/完整保存仍在收尾，因此这次瞬时快照status为399、评价预算为400；未把未评价的更新补入物理统计。Tracking下降和整曲失败均如实记录，不触发质量停止。目标3000仍未完成，不能宣称已找到优于零Adapter的配置。

- gpu3，AIOZ-GDANCE:l5xqQDKyKCU_21_0_1620，第40.40秒真实失败；对应评价预算400。
- gpu3，FineDance:bitchbetterhavemymoney，第12.10秒真实失败；对应评价预算400。
- gpu3，Mine:dance_2__看看世界有多大，第7.98秒真实失败；对应评价预算400。

最新已记录累计真实转移与速率（实际物理包含审计和评价，train列明确单列；恢复后速率已扣除导入accepted）：

|GPU|全部真实转移|train转移|本进程新增accepted|新增accepted/小时|NVIDIA观测峰值MiB|Torch峰值MiB|
|---|---:|---:|---:|---:|---:|---:|
|gpu0|120158208|21643264|251|732.5|12010|9509.5|
|gpu1|173910016|27803648|0|0.0|12002|9497.6|
|gpu2|134432768|22691840|38|111.1|12004|9484.0|
|gpu3|110352384|19185664|199|582.1|11994|9503.0|
|gpu4|114126848|21790720|233|682.2|12030|9512.8|
|gpu5|100186112|20185088|74|216.8|11994|9481.8|
|gpu6|121454592|20643840|277|812.4|12026|9513.3|
|gpu7|112898048|20545536|197|578.3|12026|9515.9|

当前训练固定6b63450，报告器已单独切换到63201af，不中断八个训练进程。新运行目录若尚未达到下一个100次完整保存，使用manifest.json中--resume指向的本组父checkpoint（例如当时GPU5为父目录100次）；不会伪造未写出的新checkpoint。完整现场机器可读证据为本地final_live_20261027.json，含8条/proc命令、nvidia-smi、tmux、checkpoint、错误扫描和最新评价来源。
