# Stage2 AWR 操作与数据契约

本实现运行于独立 `feature/stage2-awr` 工作树。训练顺序为 Collect → Critic/FVI →
Advantage Labeling → Weighted Diffusion Training → Evaluation。宏迭代与 PPO iteration 无关。
初始配置为 512 个完整 Episode、FIFO 512、Actor 5 Epoch、全局 256 AWR＋128 BC、
Actor LR 1e-5；默认最多一个宏迭代。`smoke` 为独立 64 Episode 数据集。

## Server-2 命令

```bash
cd /home/user/liwei/awr-worktrees/GENMO
bash tools/run_stage2_awr_server2.sh preflight
bash tools/run_stage2_awr_server2.sh run --profile smoke \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/<独立64集运行名>
bash tools/run_stage2_awr_server2.sh run --profile pilot \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/<独立512集运行名>
bash tools/run_stage2_awr_server2.sh stop --run-dir <现有运行目录>
bash tools/run_stage2_awr_server2.sh resume --run-dir <现有运行目录>
bash tools/run_stage2_awr_server2.sh evaluate --run-dir <现有运行目录> --all-unique
bash tools/run_stage2_awr_server2.sh evaluate --run-dir <现有运行目录> \
  --checkpoint <该运行checkpoints下的JSON引用>
```

`ablate` 与 `run` 一样执行配置中的同 Replay B/C 对照流程，必须使用新目录。
`stop` 写该运行的停止标志，不发进程信号；采集停止发新任务并排空，Actor 保存完整优化步。
`resume` 读取已保存配置、源码提交与资产身份，不隐式增加预算。重建 PhysX 世界，
完整已发布 Episode 幂等登记，未完成任务从初态重新采集，不恢复假想物理中间态。
检查点采用原子文件发布与摘要绑定，只加载本运行可信的 PyTorch 断点。

启动器只设置当前进程环境，显式使用两份 AWR 仓库，不修改 editable 安装。GPU 有计算
进程时预检失败，不抢占。校准使用 AWR 同款单请求 DDIM20/eta0/CFG2.5；延迟曲线必须
实际生成，配置中的空 `deployment_profile` 不能用原 DPPO 曲线填充。

## Replay 和动作语义

- `AWRTransition` 绑定 session/env/episode/decision/plan、Actor 版本和指纹、原始
  Stage1 条件、生成 canonical qpos30、有效/未知 Mask、Contact 诊断、Root Anchor、
  Critic 生成前 Tracking90、50Hz 真实奖励/事件、提交和消费证据。
- `AWRCompleteEpisode` 只接收同一物理 Episode 内连续且不重复的真实控制区间，行为
  版本在 Episode 内固定。自然结束、真实失败尾值为零，行政截断没有完整 MC。
- `episodes/<id>/manifest.json` 为结构索引；同 dtype 数值数组合并成少量 `.npy`，
  支持 mmap。SQLite 只保存身份/FIFO/引用；大张量不转 JSON。发布目录原子 rename，
  主 rank 登记；快照引用防止仍用于训练/对照/恢复的数据被回收。
- 实际参考由 GMT `awr_reference_audit` 返回。保护前缀可按原逻辑精确恢复，自由源
  姿态必须逐元素不变。拒绝/未实际进入 GMT 输入的参考不进入 Actor 回归，相关时间、
  失败和奖励仍记录。随后才消费的计划只补充消费证据，不搬移其他决策区间奖励。
- 保存的 Contact 是概率输出，绝不作为真值。原 Stage1 BC 使用真实预训练标签及
  完整 Loss。AWR 回归整个 60 帧窗口；末端无 halo 的 Root XY 沿用无效标签规则。

## 数学与优化

控制折扣为 `gamma_c = 0.99 ** (1/25)`。每区间按真实控制步数积分折扣，事件只计一次，
奖励已经包含 dt，不再乘 dt。`G[k] = R[k] + gamma_c**n[k] * G[k+1]`。

双 Critic 独立复制原 Stage1 条件编码，冻结音乐/前缀，训练 Tracking GRU、Temporal
Transformer 和 Value Head。group/audio 连通组固定分两折，每折内部留出音乐用于
Early Stopping。先拟合 MC，再用冻结上一轮值构造 16-step FVI；不跨 Episode。只有
两折内部验证的真实 MC MSE 均下降且绝对偏差不升时选择 FVI，否则整轮采用 MC-only。
Critic 是 Replay 行为混合分布的价值基线。

`A = G - V_crossfit`，整轮 FP64 总体标准差归一化，不减均值。权重为
`exp(min(A/std/beta, log(w_max)))`；默认整轮均值归一化后，上界是
`w_max / mean(raw_weight)`。非有限、下溢、近零方差或全部权重退化拒绝 Actor 阶段。
权重在本轮固定，Replay 每 Epoch 均匀遍历一次，不再按优势抽样。

单条样本重新抽一个 t 和高斯噪声，只执行单时间步监督前反向。先对各自有效未知坐标
求 MSE，再乘对应权重；AWR/BC 分别除实际全局样本数。各 rank 累积后 SUM，再裁剪、
更新；全局无监督参数保持 grad=None。尾批和空 rank 不重复样本补齐。
BC 系数由 8 批真实梯度诊断校准，之后宏迭代内固定。B/C 使用相同数据、噪声、Batch
顺序、BC 随机流和系数。参数变化不代表物理改善。

## 验收证据

`artifacts/` 保存计划、折、优势/权重、价值验证、模块变化、训练成本和物理比较；
`evaluation/` 保存完整任务及失败；`replay/`、`checkpoints/` 与测试临时目录分开。
固定 32 任务与最终全量 54 任务都使用固定种子、初态、冻结 GMT 和新建物理世界。
成对收益按来源平衡音乐组聚类计算 95% CI。

Best 要求完成率/时长不降、失败不增、没有新增成对失败、活动度至少保留 80%、
完整任务回报改善。未满足时保留 Stage1。AWR 对 Stage1 与普通自蒸馏的物理收益
都必须用独立评价判断；代码测试、Loss 或 Critic EV 不能替代该结论。

2026-10-20 首批验证：24 项 AWR CPU 测试通过，含八进程 Gloo；公共回归首次 69 项
通过。更大范围得到 1032 通过、10 跳过；2 项 PPO 旧失败在基线复现，8 项旧测试依赖
独立工作树外 s350000 资产。GMT 28 项参考 CPU 测试通过。NCCL、真实物理和首轮
学习结果以随后 `记录文本.md` 与运行报告为准，不把上述 CPU 结果称为已完成物理验收。
