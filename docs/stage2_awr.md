# Stage2 AWR 操作与数据契约

本实现运行于独立 `feature/stage2-awr` 工作树。训练顺序为 Collect → Critic/FVI →
Advantage Labeling → Weighted Diffusion Training → Evaluation。宏迭代与 PPO iteration 无关。
初始配置为 512 个完整 Episode、FIFO 512、Actor 5 Epoch、全局 256 AWR＋128 BC、
Actor LR 1e-5；默认最多一个宏迭代。`smoke` 为独立 64 Episode 数据集。

2026-10-20：64集及独立512首轮、三组完整音乐对照、真实八卡梯度与续算验收已完成。
全54任务Stage1/自蒸馏/AWR成功数为50/49/52，但AWR成对回报收益CI跨零，且Root、
Joint和滑动指标变差；固定32任务时长门槛未通过，`best_saved`仍为Stage1。
详见[最终实现与有限验收报告](stage2_awr_acceptance_20261020.md)。未启动第二宏迭代。

## Server-2 命令

最新交付源码放在`/home/user/liwei/awr-worktrees/GENMO-delivery`，使用同一AWR分支，
与冻结运行目录独立。以下新建运行命令从交付目录执行；不改共享环境的editable指向。

```bash
cd /home/user/liwei/awr-worktrees/GENMO-delivery
bash tools/run_stage2_awr_server2.sh preflight
bash tools/run_stage2_awr_server2.sh verify --report <独立证据目录>/distributed.json
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

512首轮固定使用`/home/user/liwei/awr-worktrees/GENMO`的`f417545`，该目录保留原源码。
对此运行的停止/恢复/重新评价必须从原目录调用；原版本恢复前设置当前shell软限制
`ulimit -Sn 65536`，不改硬限制。新版入口自动预留句柄，但不能用新版冒充旧运行恢复。
原64集源码为`ea81029`，重新物理评价也须使用该原提交的独立工作目录及匹配路径配置。

`ablate` 与 `run` 一样执行配置中的同 Replay B/C 对照流程，必须使用新目录。
`stop` 写该运行的停止标志，不发进程信号；采集与评价停止发新任务并排空，Actor 保存完整优化步。
`resume` 读取已保存配置、源码提交与资产身份，不隐式增加预算。重建 PhysX 世界，
完整已发布 Episode 幂等登记，未完成任务从初态重新采集，不恢复假想物理中间态。
检查点采用原子文件发布与摘要绑定，只加载本运行可信的 PyTorch 断点。
独立`evaluate`保留训练原状态与完成时间；同一Actor和任务计划使用稳定评价身份，
再次执行该命令会接续未完成任务或读取已完整提交的报告。它不扩展宏迭代预算。
已完成运行的`resume`直接返回，不重写原完成时间或重新执行评价。

最终断点可在原运行源码下用归档工具进行只读CPU组件恢复核验：

```bash
python -B <归档工具路径>/verify_stage2_awr_checkpoint.py \
  --run-dir <已完成运行目录> \
  --runtime-repo /home/user/liwei/awr-worktrees/GENMO \
  --output <新验收报告路径>.json
```

工具逐元素比较权重、AdamW矩和Scheduler，绑定固定/全量物理评价指纹，检查八rank
CPU随机流与保存的CUDA状态；它不执行新的优化步或CUDA随机流，不能代替八卡续训测试。

有限八卡续算工具`verify_stage2_awr_resume.py`使用三条真实Replay及一条真实BC，包含
空rank。连续两步，在第一步原子保存全部rank状态，再恢复重算第二步，逐元素比较
Actor、AdamW、Scheduler、BC及CPU/CUDA RNG。共三个独立诊断优化步、零物理交互；
验收后恢复输入Actor指纹，正式训练计数与checkpoint不变。归档时须同时复制
`verify_stage2_awr_checkpoint.py`作为比较助手。

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

本实现参考[GTI §II-E、Figure 3、Equation (1)–(3)](https://arxiv.org/html/2609.31577v1#S2.SS5)
的离策略回放、交叉优势和加权生成监督。GENMO保留DDPM的单时间步`x0`监督；
这是扩散监督适配，不是精确扩散策略对数似然优化，也不提供严格策略改进保证。
论文的`(1-gamma)`价值缩放和非失败截断尾部不移植：本任务使用原积分奖励单位，
音乐自然结束是真正有限任务终止，行政截断排除于完整MC之外。论文Table II的地形
奖励、条件扰动也不移植。N=16、beta=1、上限20和学习率均为项目候选参数。
普通自蒸馏对照用于检验优势加权的增量作用，对应[GTI §III-B](https://arxiv.org/html/2609.31577v1#S3.SS2)
提出的比较问题；结论以本项目独立物理评价为准。

## 验收证据

`artifacts/` 保存计划、折、优势/权重、价值验证、模块变化、训练成本和物理比较；
`evaluation/` 保存完整任务及失败；`replay/`、`checkpoints/` 与测试临时目录分开。
固定 32 任务与最终全量 54 任务都使用固定种子、初态、冻结 GMT 和新建物理世界。
成对收益按来源平衡音乐组聚类计算 95% CI。

Best 要求完成率/时长不降、失败不增、没有新增成对失败、活动度至少保留 80%、
完整任务回报改善。未满足时保留 Stage1。AWR 对 Stage1 与普通自蒸馏的物理收益
都必须用独立评价判断；代码测试、Loss 或 Critic EV 不能替代该结论。

2026-10-20最终代码验收源码`2f65089`：本地torch2.6环境62项AWR CPU测试通过，Server-2
原torch2.7.1环境同62项通过，含八进程Gloo、真实小型Stage1与BC合同、FVI/Actor
恢复、评价逐任务停止恢复。服务器验证禁用CUDA可见设备，临时目录自动清理。
实际导入目录及全部冻结资产SHA亦通过；证据为
`docs/validation/stage2_awr_delivery_final_cpu_tests_20261020.json`和
`docs/validation/stage2_awr_delivery_assets_20261020.json`，此预检不宣称GPU空闲。
公共回归69项通过。更大范围1032通过、10跳过，2项PPO旧失败在指定基线复现；
8项缺s350000资产的旧测试使用临时只读链接后，相关9项通过，链接和临时目录已移除。
GMT 28项参考接口和18项USD并发隔离测试通过。历史详细过程保留于`记录文本.md`。

Server-2真实s595000＋原Stage1 BC八卡NCCL测试通过：9 AWR／3 BC 的相对梯度L2误差
4.72e-7，3 AWR／1 BC（含空rank）为5.14e-8；更新后各rank参数摘要完全一致。
耗时16.73秒，单卡PyTorch峰值显存约1.65–1.73 GiB。原始报告见
`docs/validation/stage2_awr_distributed_20261020.json`。这是计算验收，物理结果另报。
小型64集只执行每组32个固定任务；全量54任务放在独立512集首轮最终对照中执行。

## 只读结果汇总与资源统计

```bash
python3 tools/report_stage2_awr.py --run-dir <运行目录> --output <新的验收JSON路径>
python3 tools/audit_stage2_awr_replay.py --run-dir <运行目录> \
  --max-episodes 24 --output <新的回报审计JSON路径>
bash tools/run_stage2_awr_server2.sh retention \
  --run-dir <已完成运行目录> --runtime-repo <原运行提交的GENMO工作树> \
  --output-dir <新的独立验收目录>
```

汇总只读取已提交的状态和JSON报告，不需要GPU，不重算评价或选择模型。它保留实际
运行的源码提交、报告摘要及失败门槛；进行中的阶段不会补写结果。已有输出文件拒绝覆盖。
独立Replay审计还核验完整manifest及数组SHA、物理区间连续性、原始生成目标和实际提交，
并逐50Hz控制步逆推MC，不调用训练的区间回报函数。省略`--max-episodes`时核验调用时
所有已原子发布Episode；正在运行时未完成的Episode不会被当成完整数据。报告明确抽查
数量和当时发布总量，不能将抽样通过说成全部Episode通过。
新运行在`resources/gpu_<attempt>.jsonl`每秒记录整卡显存与利用率，结束发布同名summary；
恢复生成新attempt。整卡峰值是离散采样的下界，包含GMT和驱动开销；Actor训练报告
另给PyTorch allocator峰值及统计范围，二者不混用。
报告还读取运行旁的`*.launch.json`/`*.exit.json`，保留失败启动和恢复的完整墙钟，
单列缺少退出回执的暂定耗时。已知八卡入口的预约GPU小时包括CPU读取、I/O和空闲，
不冒称GPU实际忙碌时间；有尝试重叠或未知设备数时不计算这个估算。原始GPU JSONL
即使因异常退出没有summary也参与峰值汇总，并报告最大采样间隔、查询失败和不完整
末行；不能仅选择成功尝试的显存记录。

`retention`在完整物理运行结束、GPU空闲后进行单GPU只读推理，默认从原Stage1 val
每来源选择4个窗口、使用2个固定生成种子。三份Actor共享保存的原条件与同一噪声，
比较完整Stage1生成损失、未知qpos30误差、真实Contact BCE和关节活动度/输出变化。
使用原Dataset的真实标签及ground metadata，不构造Replay伪标签。该小批固定探针
不冒称整个Stage1验证集或物理收益；条件分片、模型指纹和工具SHA均在独立输出留存。

Replay `array_tree.v2`保留物理回执的共享列，兼容旧v1；训练视图只展开必要条件和事件，
读取前仍核验完整manifest和全部数值数组SHA。完整原始证据未删除，训练视图明确禁止
回写为完整Episode。单样本的真实无损编码与读取验证见
`docs/validation/stage2_awr_storage_20261020.json`，其本地读速不代表整轮Server-2吞吐。

512先导的一条266决策、6643控制步的真实分片为447,160,691字节，其中JSON结构
索引164,336,908字节。并发登记期间、已有页缓存的只读抽查中，完整读取/摘要/投影
耗时3.661秒，记录校验0.135秒，MC及其校验0.113秒，MC逐元素一致。
证据见`docs/validation/stage2_awr_pilot512_read_profile_20261020.json`。这说明结构索引
解析仍是工程瓶颈，不能用短Actor反向时间宣布整轮训练提速；该单例也不是无负载的
整体吞吐基准。后续应在独立数据副本验证更紧凑的结构索引编码，不改写本轮正式Replay。

## 代码与恢复边界

| 文件 | 实际职责 |
|---|---|
| `awr/sampler.py`、`actor.py` | 完整Stage1 DDIM采集；独立单时间步监督内核；原Stage1完整标签校验保留 |
| `awr/collection.py`、`records.py` | 原始参考与提交/消费身份绑定、真实终止和完整Episode连续性 |
| `awr/io.py`、`replay.py` | dtype二进制分片、列共享、SHA、SQLite事务、FIFO、快照引用与只读训练视图 |
| `awr/returns.py` | FP64真实MC及可变时长N-step目标，调用公共600Hz时间审计 |
| `awr/folds.py`、`tasks.py` | group/audio连通音乐折、内部验证、来源采样及持久化任务计划 |
| `awr/critic.py`、`fvi.py` | 两份独立原非对称网络、MC/FVI比较、冻结目标和交叉标签身份 |
| `awr/advantages.py`、`dataset.py` | 整轮固定标签、样本身份严格对齐、均匀回放及Contact诊断隔离 |
| `awr/losses.py`、`trainer.py`、`distributed.py` | 每样本加权、真实BC、全局分母、梯度累积/SUM、裁剪及优化 |
| `awr/checkpoint.py`、`runtime.py` | 原子checkpoint及五阶段调度，同Replay自蒸馏/AWR对照 |
| `awr/evaluation.py`、`physical_metrics.py`、`metrics.py` | 完整任务/失败、成对CI、安全门槛、保留能力和探索诊断 |
| `awr/world.py`、`config.py`、`resources.py` | 独立PhysX世界、AWR时钟校准、资产/资源检查及GPU采样 |
| `closedloop/common/` | 原物理事务、奖励、Episode、Critic、BC与评价的算法无关实现；旧DPPO路径保留兼容导出 |

`state.json`只引用已经提交的阶段产物。Actor断点保存参数、AdamW和Scheduler、
Epoch/Batch游标、优化步数、数据身份、固定BC系数及各rank独立BC/RNG状态。
Critic保存两份参数、Optimizer/Scheduler、拟合随机状态、MC/FVI候选与冻结目标身份；
恢复边界是一次已完成的MC或FVI拟合，未完成拟合从上一已提交边界重做。

正常停止在采集和评价阶段停止发任务并排空当前Episode，在Actor阶段保存完整优化步；
Critic在已提交的一次MC/FVI拟合边界响应。评价逐任务原子登记，恢复先核验模型、
任务/物理身份及数组SHA，保留原rank/slot/任务索引，只执行尚未登记的任务。
评价失败也登记，未完成报告不能选优；该账本提交前强制中断的任务需从初态重做，
不把孤立分片冒充已经提交的评价。进程异常时保留原始日志、已发布Episode和
已提交checkpoint；恢复不拼接旧世界的中间物理状态。若固定任务发生预热失败或行政
截断，采集验收会失败并保存排除原因，不静默补成成功Episode或换一首音乐。
本条是最新代码的停止语义，512物理运行固定的f417545仍在完整评价阶段结束后响应；
该运行没有热更新，也未拿CPU调度测试冒充真实物理中途停止验收。

配置与两个源码提交均参与恢复身份。更新源码后重新评价旧模型时，应使用原运行提交的
独立工作树及其配置；只读汇总脚本可跨版本读取报告，但不能改变运行身份。
后续多宏迭代由新运行配置的`max_macro_iterations`控制；已有运行的resume不会扩大预算。

## 显式追加采集尝试

若原固定计划全部执行完，但存在首次GENMO决策前的真实预热失败，则这些任务没有
Actor动作或完整MC样本。默认采集仍拒绝缺额，不自动换歌。用户明确允许追加预算后，
可使用独立入口；2026-10-20的512首轮授权为最多26次额外尝试，补足512条有效Episode。

```bash
bash tools/run_stage2_awr_server2.sh complete-collection \
  --run-dir <原运行目录> --runtime-repo <原运行提交的GENMO工作树> \
  --max-extra-attempts 26 \
  --authorization '用户允许保留全部失败，最多26次额外尝试，补足512条有效完整Episode' \
  --tool-commit <补采工具所在的已提交40位Git提交> --continue-training
```

先等待原进程退出，入口仍检查八卡空闲和原源码/资产身份。工具可以从已推送提交按原样
归档到运行的`acceptance/tools/<commit>/`，使用同一八卡torchrun环境执行，并显式
`--runtime-repo`导入原提交；此方式不更新运行工作树或关闭身份校验。报告另外绑定工具
文件SHA与Git提交，物理、奖励、采样及训练代码继续绑定原提交。

`macroNNN/supplement/plan.json`保存原采样机制生成的追加候选，原计划和request不改。
每批先预留不超过剩余缺口的任务，成功和真实物理失败均接纳，只有无动作预热失败
不能充当Actor样本。已预留但中断的任务也占额外尝试上限，防止恢复时无限重试；
预算用尽仍不足会保存`exhausted.json`并停止，不能自动扩额。

原采集已完整验证并写入SQLite的数据按不可变摘要接续，新增分片仍执行原生产Replay的
SHA/时间/MC/隔离校验；封存复核全部manifest，Actor训练读取再次核验全部数组。
达到目标后保存失败快照摘要、完整
尝试账本、授权与精确快照，发布`*_collection_amendment`和`*_replay`阶段产物。
此后原`resume`可继续Critic/Actor；已经开始价值拟合或Actor更新的数据不得替换。
报告同时列原任务数、额外尝试数、失败数和有效Episode数，不将补采后的512冒称只有
512次物理尝试。行政截断或未执行完的原任务应走正常恢复，不能用此入口静默替换。
封存还验证本宏迭代全部记录的Actor指纹一致，并分别汇总Actor回报区间、成功进入
采集的Episode初始化区间、失败预热区间的真实控制秒数；初始化不伪造为Actor MC。

## Replay字段和校验

| 数据 | 字段与含义 |
|---|---|
| Actor条件 | `context`保存原有十个Stage1条件张量，含音乐/有效Mask、proprio48历史/时间、决策时间、已知前缀/Mask、未来时间/Mask；单记录Batch维为1 |
| 原始目标 | `target_qpos30 [60,30]`为生成的物理表示；`target_valid`和`free_mask`同形状且bool，后者严格等于有效未知坐标；`contact [60,2]`仅诊断 |
| 提交与消费 | `execution.source_audit`含原始/实际提交世界参考摘要与数值；另存Root Anchor、坐标身份、窗口首次观察tick、当前点消费tick和奖励参考tick |
| Critic状态 | `metadata.critic_context`及`remaining_music_seconds`来自生成前状态；包含Tracking90历史、时间/Mask和原条件，不加入该动作、种子或未来奖励 |
| 真实反馈 | `rewards [n]`为实际50Hz已积分奖励；`control_tick_begin/end`、实际控制步数、终止/截断及原因；完整物理trace、奖励分项和事件保存在原始metadata |
| 行为身份 | `identity`含session/env/episode/decision/plan、policy/actor版本和Actor指纹；metadata另存采样器、生成种子及源码；Episode含source/group/audio、任务和冻结契约 |
| 完整标签 | `returns [K]`为这条Episode的K个上层状态MC；标签分片以唯一sample_id绑定跨折Value、原始/标准化Advantage、raw/final权重及Critic身份 |

连续性检查采用整数tick：每个控制周期12 tick、4个物理子步；区间首尾必须相接。
已知前缀逐坐标相等，目标不得与未来有效Mask冲突，行政截断没有完整MC。
生成目标与实际提交自由坐标不一致时拒绝接纳；自然结束和物理失败均保存，
未执行候选不参与Actor回归。完整60帧监督与实际消费范围分开记录。

SQLite记录Episode顺序、摘要、来源、行为版本、Transition/Actor样本数、字节量及
快照引用；数组目录先原子发布再登记。重复同一摘要幂等，身份相同内容不同立即报错。
淘汰按整条Episode进行；被训练、对照或恢复快照引用的旧数据暂不删除，
达到字节预算即停止接纳并要求处理引用，不能无限增长。

## 结果的判定范围

小型64集的音乐折每来源、每折只有两个音乐组，其中一个用于内部验证，所以该验收
实际内部留出比例为50%；正式512使用全训练音乐清单预先构建的约20%映射。报告给出
实际Episode、状态和音乐组数，不能用配置比例代替实际统计。

MC/FVI内部验证改善只决定价值基线选择。Actor权重永远是实际MC减交叉Value，
不用FVI目标替代MC。BC数据不进入Critic或标签，Contact Head直接监督来自真实BC。
原始权重截顶与均值归一化后的上界不同；报告同时给两种权重分布及上界。

梯度诊断和参数变化将`denoiser.final_layer`（qpos30 Head）、
`denoiser.static_conf_head`（Contact Head）与其余denoiser参数分开统计，避免输出头
变化被大网络总范数掩盖。分组互不重叠；Contact没有Replay真值，不能据此新增其AWR监督。

八卡计算验证见`docs/validation/stage2_awr_distributed_20261020.json`。
公共奖励在6545个真实控制步上的独立基线重算见
`docs/validation/stage2_awr_real_reward_parity_20261020.json`。
物理改善必须同时查看完成数、所有失败、时长、累计回报、活动度及成对置信区间；
自蒸馏/AWR损失、参数变化和EV不承担此结论。

GMT `agent.yaml`和Isaac契约文件的内容摘要由预检自动计算，保存到运行和物理身份；
每次新建世界与恢复均复核。只修改同一路径文件的内容也会被拒绝，无需用户填写SHA。

2026-10-20小型64集五阶段真实闭环已完成，进程退出码0。详细结果见
`docs/validation/stage2_awr_smoke64_20261020.json`；模型恢复与评价指纹绑定见
`docs/validation/stage2_awr_real_checkpoint_restore_20261020.json`。32任务AWR相对Stage1
及自蒸馏的来源平衡音乐组聚类收益CI均为正，并通过全部物理门槛。这仅是小型协议
的结果；独立512首轮及54任务评价已完成，以[最终报告](stage2_awr_acceptance_20261020.md)
及其绑定的独立运行证据为准，不能以64集的正收益替代512结论。

## 已完成的小型64集结果

实际运行GENMO `ea81029`、GMT `4ca256f`，固定32任务，数据与后续512独立。

| 指标 | 冻结Stage1 | 自蒸馏＋BC | AWR＋BC |
|---|---:|---:|---:|
| 完成任务 | 29 | 30 | 31 |
| 物理失败 | 3 | 2 | 1 |
| 实际执行秒数 | 1779.100000 | 1725.940000 | 1798.460000 |
| 累计回报 | 4049.709943 | 4004.027662 | 4247.027692 |
| 来源平衡每秒回报 | 2.395235 | 2.423448 | 2.445147 |
| Root Position误差 m | 0.325756 | 0.451497 | 0.450207 |
| Yaw误差 rad | 0.317202 | 0.379652 | 0.273572 |
| 关节位置RMSE rad | 0.099055 | 0.104760 | 0.104576 |
| 原支持球滑动代理 m/s | 0.023152 | 0.028290 | 0.028432 |

AWR相对Stage1的来源平衡音乐组聚类成对累计收益为5.715245，95%CI
[3.210320, 8.178376]；相对自蒸馏为6.144275，CI [0.528749, 15.128622]。
AWR通过本轮既定物理门槛，没有新增成对失败；其实际关节活动度为Stage1的1.193369倍。
同时Root Position、关节误差和滑动代理增加，不能将回报提升解释为所有跟踪质量改善。
这些平均误差还受各任务执行长度变化影响，完整任务级数据保留在原报告中。

采集64集耗时929.57秒，真实物理执行2406.2秒，形成4842次上层决策及4801个Actor样本。
两组均训练5 Epoch、95次优化，B/C约168.64/165.12秒。旧密集Replay读取687.17秒，
回放体积12.85 GB；运行从创建到阶段完成4490.06秒，不能用Actor训练时间代替全闭环成本。
旧运行未采集整卡全过程显存，PyTorch累计峰值约4.01 GiB；512首轮有独立整卡采样。

512首轮在`pilot512_20261020_01`运行，固定GENMO `f417545`、GMT `4ca256f`。
原512任务产生504条有效完整Episode和8次首次生成前的预热失败。用户明确追加最多
26次尝试后，实际仅补采8次且全部自然结束，随后停止发任务。最终520次物理尝试中，
426次音乐结束、86次生成后物理失败进入512条Replay；另外8次预热失败保留快照，
不虚构Actor动作或MC标签。

已封存快照为`87e89f9282d2684aaa79b905c11d053cc59fb1ab37b83f86f3a7189120c9177e`，
包含44722条决策、44447条Actor样本、74984836087字节数据。Actor回报区间22260.02秒，
完整Episode初始化512秒，失败预热0.72秒，合计1138637个真实控制步；与后端实际
活动控制计数和预算扣除逐项一致。物理采集两次合计2352.756秒，索引登记/封存和
后续训练读取另计。证据见`docs/validation/stage2_awr_pilot512_collection_20261020.json`。

512快照首次全量训练读取暴露了进程句柄上限：Server-2默认`RLIMIT_NOFILE=1024`，
rank 1–7的常驻npy mmap触发`EMFILE`，在Critic/Actor更新前同步退出。新入口在每个
rank加载前按`max(4096, 1024+32×Episode容量)`预留句柄，只提高进程软限制；硬限制
不足时提前报错，并保存各rank实际限制。同步错误立即输出原始traceback，避免等待
较慢rank时看不到原因。当前固定f417545的512运行通过启动父进程设置软限制65536恢复，
保留原源码/Replay/任务/配置身份；不需要重新物理采集，也没有修改系统硬限制。
恢复后训练与最终54任务对照已完成，B/C各实际更新870步，均为5 Epoch，数据预算、
学习率、BC系数及八rank最终BC随机流配对一致。C全54成功52/54，累计回报6987.649650，
Stage1为50/54、6960.838710；C相对Stage1来源平衡任务回报差1.475730，
95%CI[-4.296548,7.188531]。固定32的执行时长比Stage1少3.10秒，未满足既定晋级门槛；
最佳发布模型继续保留Stage1，未进入第二宏迭代。


512条完整Episode独立审计已通过：44722条决策、44447条Actor样本、1113001个奖励
控制步；原始生成目标最大差0，独立逐控制步MC最大绝对差8.526512829121202e-13，
全部manifest和数组SHA通过。包含全部86次生成后失败，另8次预热失败保留于原尝试
账本。证据见`docs/validation/stage2_awr_pilot512_full_replay_audit_20261020.json`及其
绑定的服务器完整报告。1367.088秒为并发加载期间CPU审计耗时，不是隔离读取基准。


`backward-profile`在已完成运行、八卡空闲时执行1次预热和默认3次有限前反向，读取
本轮真实Replay与固定权重、最终AWR Actor和原BC真标签，分别测量AWR、BC、SUM和
裁剪，不调用优化器step，前后校验Actor指纹不变。该离线测量不能替代整轮成本：

```bash
bash tools/run_stage2_awr_server2.sh backward-profile \
  --run-dir <已完成运行> --runtime-repo <该运行原源码目录> \
  --output-dir <新的独立验收目录> --steps 3
```

新Trainer逐步拆分BC前反向、梯度归约、裁剪及优化器耗时，CUDA阶段末同步以免仅测
到排队耗时；保留旧BC＋更新合计字段。已完成的512运行保留f417545，其历史BC时间
只有合计字段，不能事后伪造分项。实际有限八卡测量已通过：AWR前反向0.157401秒、
BC前反向1.367170秒，不含优化器的总时间1.714306秒；仅3个测量batch、0优化器更新，
不能替代整轮成本。结果见`docs/validation/stage2_awr_pilot512_backward_profile_20261020.json`。

最终B/C断点组件恢复及真实八卡续算均已通过，参数、AdamW、Scheduler、BC和CPU/CUDA
RNG逐元素一致；3个独立诊断优化步不计入正式训练，0新物理交互。临时验收checkpoint
已精确清理，正式checkpoint未变。证据为
`docs/validation/stage2_awr_pilot512_checkpoint_restore_20261020.json`与
`docs/validation/stage2_awr_pilot512_distributed_resume_20261020.json`。
# 2026-10-21 修复验收边界

正式五轮运行已由用户停止，完成四轮；原目录和源码保持冻结。新AWR配置使用
`stage2.gmt8_no_activity_gate.v2`，Tracking分数不再乘活动度，仍报告活动度。
旧奖励Replay不能直接作为新版Critic标签；旧Actor条件/参考可以由隔离诊断工具
以全1权重只读用于自蒸馏，不携带旧优势进入新训练。

Replay现在监督与部署相同的CFG2.5预测。Stage1真实BC保持原完整损失和真实Contact。
`awr.actor.reference_weight`可为冻结起始Actor的两个去噪分支增加保留约束；0表示关闭。
它是新增工程正则，不冒充原AWR公式。`max_optimizer_steps`为独立短实验显式上限。
普通自蒸馏先通过`awr.retention_gate`的完整音乐检查，才进入AWR；物理退化则停止后续宏迭代。

Server-2隔离诊断（在最新、干净AWR交付工作树；原正式源码不更新）：

```bash
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 tools/diagnose_stage2_awr_distillation.py train \
  --source-run /data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01 \
  --output /data1/user/liwei/GENMO_outputs/stage2_awr/distillation_repair_20261021_01
```

`train`最多三组×120优化步，保存全部实际步数和固定val探针；`evaluate`使用同一命令
替换操作并追加`--candidate bounded`，重建Stage1与候选的32任务物理基线。两步均不会
启动后续AWR或长期训练。数学修复、离线生成保留、物理保留、强化收益必须分开报告。
