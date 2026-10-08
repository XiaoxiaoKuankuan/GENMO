# 第二阶段八卡训练 v2 实施与运行说明

本版按最终授权配置实施。Actor 固定学习率 **5e-9**，正式训练不扫描候选、不自动降低学习率。正式入口 `scripts/train_stage10_8gpu_server1.sh` 默认使用 `configs/closedloop/stage10_8gpu_server1_v2.yaml`，默认仅运行一轮；长期训练必须显式给定停止轮次。

2026-10-08性能重构已完成真实八卡重复验收。固定执行形状2、microbatch=2、CFG合批的两组四轮均完成首轮保存、退出恢复和后续更新，独立审计分别通过22项。相同1次Actor更新的核心时间从70.04秒降到平均41.60秒；相同2次从88.08秒降到46.50秒。四次更新的新版核心时间为54.64秒，旧版没有同工作量实测基线。此前[KL 0.03验收报告](stage10_v2_kl003_acceptance_20261008.md)保留为旧标量路径基线，不再代表新版吞吐。

本轮改变了数值执行和持久化契约。正式训练应使用**新运行目录**、配置明确指定的原Stage1模型初始化；不能把旧scalar/NPZ训练目录当成本版原样续训。相同新版身份的保存、退出、恢复已经实际验证。没有自动启动七天训练，也没有扩大正式预算。

## 更新流程和配置

八个 rank 各自持有一个冻结 GMT/CPU PhysX 后端，使用同一模型版本分别采集 20 条上层转移（全局 160）。真实执行时长决定 GAE，环境/episode/连续区间之间不串接；old log-prob、old/next value、returns 和全局标准化 advantage 整轮固定。每条链 20 步，完整链组成 1600 内部转移的优化器 minibatch，2 个 epoch 最多 4 次 Actor 参数更新。microbatch上限为4，候选还必须不超过固定执行形状；当前正式形状2，启动校验2、1并记录八卡共同通过的配置，独立于优化器minibatch。固定形状4已完成四轮真实验收，因采集延迟更大而未选为正式默认值。

Actor `AdamW(lr=5e-9, weight_decay=0)`，PPO clip 0.01；BC 每次参数更新由 rank0 计算全局 2 条样本、权重 0.1。PPO 按全局实际有效内部样本数归一化，再做梯度 SUM；BC 只加入一次，不乘或除以 8。Critic 默认 80 次、全局 batch32、学习率 1e-4；可显式配置20/40/80进行对照。`critic_update_mode: rank0_broadcast` 启用单卡更新后广播的性能对照，默认 `distributed` 不变。`x0_diagnostic_every: N` 显式开启每N轮更新前后真实x0输出诊断，默认0关闭，报告额外全链前向成本。

随机核默认 eta=0.1、std_floor=0.001、CFG=2.5，不删除末端去噪 loss。精确 joint_sum 为默认概率目标；free_coordinate_mean 必须显式确认学习率/clip/BC配置，属于算法对照，真实 joint KL 仍保留。std_schedule 改变会改变核身份，不允许混用旧轨迹。

## 拒绝、冻结与恢复

每个 minibatch 对本轮采集策略检查 KL，达到0.015停止后续步；按2026-10-08用户明确修改，最终全 rollout、全20步平均内部 joint KL不得超过0.03。20步固定时平均链KL阈值换算为0.6，不是额外独立门槛。链KL是旧策略采样路径估计，不是最终动作分布或机器人轨迹的精确KL。最大内部/逐步均值/链最大KL为独立可空阈值。Actor固定学习率仍为5e-9，未加入自动降学习率或回退搜索；旧0.02验收报告保留原值，新配置须在新run中验收。

任何更新阶段数值异常或最终KL拒绝均恢复整轮 Actor、Critic、优化器、BC与更新RNG并停止；已经发生的优化尝试和物理消耗保留。故障诊断会写 `rejected_*.json`，不会暗中换小学习率。

load_actor保留构造时的冻结规则，优化器排除固定编码表。旧Stage2若编码污染，使用 `tools/repair_stage2_position_encoding.py --help` 的审计和新产物weights-only分支，原件不变，旧Adam/RNG不冒充已修复完整恢复。新v2断点与旧v1拓扑不允许交叉完整续训。

## 保存、预算和证据

完整恢复点按外层300、600、900轮同步原子发布；初始化、正常结束、状态一致的受控退出可额外保存。保存共享模型/优化器和8份本地RNG、采样游标及执行计数。latest只指向已发布完整断点。崩溃后未保存尾部写入 superseded_tails；预算账本不回退。

执行证据在所有journal关闭、immutable seal建立后进入单个独立CPU归档进程，线程只负责有界调度，在途与排队合计最多4轮。压缩逐成员校验并原子发布后才回收原件；失败背压/停止，恢复可幂等补齐。每轮轻量JSONL和TensorBoard不会替代完整执行证据。初始化模型、每300轮模型和最近两份完整断点保留。

根进程预占整轮各卡最大执行额度，各卡先持久化实际执行再ACK，正常完成后只退还可证明没有使用的额度。原总预算未扩大；八路160条/轮不保证仍能完成旧额度对应的外层轮数。

## 延迟、评估和曲线

`deployment_critical.v2` 将必要的前缀请求、条件构建、传输、GENMO推理、坐标转换和GMT参考准备计入 `critical_ready_seconds`。训练链保存、journal持久化和归档单独计时；commit单列。模拟参考到达和校准使用同一新口径，保留执行结果先持久化再ACK的恢复机制。新旧计时曲线必须分开，计时改动不能算作学习收益。

初始化、每100轮和正常结束使用独立评估状态，四来源各4条固定验证样本、42/1729两种子、每条10秒，共32任务，评估不额外保存模型。使用内存权重，恢复训练RNG和游标。最佳观测与最佳已保存模型分别记录；物理失败不增加、平均时长不下降后才比较四来源等权回报。

固定初始链漂移用于诊断相对Stage1概率变化，不能替代闭环配对评估。训练曲线同时看reward/执行秒、分项奖励、真实执行时长、失败/拒绝、KL、clip、真实Actor步数、PPO/BC梯度、裁剪系数及Critic新批误差，不能仅凭loss平稳认定学到或退化。每次Actor step都记录PPO/加权BC/合并梯度范数及夹角，代价是rank0临时保留一份PPO梯度快照，不随完整概率检查频率关闭。周期子集比较必须显式 `--mode periodic_subset`，不冒称全验证集验收。

运行耗时区分三种口径：不可变轮次汇总的 `seconds` 是采集、更新及更新后校验的核心时间；`iteration_walltime/seconds` 进一步包含封存、checkpoint、归档入队背压及当轮周期评估；正常退出后的最终评估 `wall_seconds` 和 `shutdown/archive_drain_seconds` 单列，不能直接把并行归档工作时间加到每轮前台时间上。控制台分别显示 `core_seconds` 与 `wall_seconds`。初始、周期和终态评估成功后登记相对路径与报告SHA，正式验收同时检查原始分片及任务计划，不以训练会话退出码替代效果评估。

并行GMT的 `asset_conversion_dir` 由入口分配到各rank私有目录，避免固定环境种子导致IsaacLab默认秒级USD目录碰撞。路径不参与实际物理指纹；不更换URDF或转换选项。通信耗时明确为梯度SUM的CUDA stream event时间，不冒称全部Gloo/RPC通信；异步归档仅由训练主线程写入TensorBoard。

## 命令（服务器1本机）

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
source /home/user/liwei/GENMO/.venv/bin/activate
bash scripts/train_stage10_8gpu_server1.sh --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/新运行目录 --stop-after-iteration 1
bash scripts/train_stage10_8gpu_server1.sh --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/同一运行目录 --stop-after-iteration 2 --resume latest
python -B tools/eval/audit_closedloop_stage10.py --help
TASK_RUN=/data1/user/liwei/GENMO_outputs/closedloop_stage10/同一运行目录
TASK_SESSION=替换为实际会话ID
python -B tools/eval/compare_closedloop_stage10.py --mode periodic_subset \
  --initial-report "$TASK_RUN/evaluation_baseline.json" \
  --final-report "$TASK_RUN/sessions/$TASK_SESSION/evaluations/000100.json" \
  --output "$TASK_RUN/periodic_000100_comparison.json"
```

初始化汇总固定在运行根目录的 `evaluation_baseline.json`；周期汇总位于本次会话的
`sessions/<session_id>/evaluations/000100.json`、`000200.json` 等文件。正常结束的
额外评估可能使用 `final_000301.json` 这样的名称，以实际文件为准。比较时选择当前
有效训练分支的报告，不能使用已在 `superseded_tails` 中作废的历史尾部报告。

在已激活环境的服务器1终端启动 TensorBoard，日志目录是同一运行根目录下的
`tensorboard`，横轴使用外层训练轮次：

```bash
tensorboard --logdir "$TASK_RUN/tensorboard" --host 127.0.0.1 --port 6006
```

需要从本地浏览器查看时，在本地终端建立端口转发，再打开 `http://127.0.0.1:6006`：

```bash
ssh -N -p 50030 -L 6006:127.0.0.1:6006 user@112.65.216.193
```

完整val评估入口继续保留，并支持v2权重的显式只读加载，不恢复训练优化器/RNG。
下列 `--eval-count all` 覆盖完整验证样本池及配置中的两种子，但当前v2配置仍按每条
音乐的前10秒窗口评估；它不等于评估每条音乐的全部时长：

```bash
CUDA_VISIBLE_DEVICES=0 python -B tools/train_closedloop_stage10.py --config configs/closedloop/stage10_8gpu_server1_v2.yaml --mode eval --checkpoint 完整模型.pt --eval-count all --output-dir 独立完整验证目录
```

本轮实际服务器有限验收结果、同步提交及尚未通过的条件以 `记录文本.md` 最终记录为准。上述命令说明不代表已经开启长期训练，也不代表相对Stage1效果已提升。

## 性能重构的文件、原因与行为

| 问题 | 修改位置与实现 | 守住的边界 |
|---|---|---|
| 校准混淆自身错误与跨实现舍入 | `execution_profile.py::probe_profiles` 分开报告自身、跨执行和训练重算；覆盖全部20步、CFG分支、全模块梯度及临时Adam | log-prob/ratio/独立高斯门槛仍为1e-4/1e-3/1e-8；临时更新恢复原权重 |
| 同形状但行号改变仍会放大末端概率误差 | `policy.py::transition_parameters` 使用固定batch形状和 `step % batch_size` 行位置，采样、学习、KL一致 | 新核执行身份 `fixed_shape_step_lane_single_condition_fp32.v1`；不等同旧标量恢复 |
| 学习数据反复cat和H2D | `tensor_cache.py::RolloutTensorCache` 每轮每rank一次组织完整链/条件/旧概率/旧核/掩码/目标；超预算显式分块 | 实测每卡20链12,806,700字节，一次上传；更新结束关闭，不广播完整链 |
| 每步重复编码可学习条件 | `ConditionGraphCache` 同optimizer step每链编码一次，累计条件叶子梯度后回传原编码图 | 历史、前缀、音乐均有梯度；不跨step缓存、不用retain_graph |
| Actor各卡工作量不均 | `updater_v2.py::balanced_epoch_order` rank内打乱、跨rank轮转分组 | 80链分成各卡10链；无丢失/重复，尾批和空rank仍使用实际全局分母 |
| BC两次小前后向和昂贵梯度取值 | `trainer.py::SupervisedAnchor._backward_batch` 两条合批，逐样本损失等权；梯度范数/夹角在GPU汇总 | BC仍全局2条、0.1，只加一次；独立且可恢复的批处理RNG身份 |
| 通信桶反复分配 | `distributed_runtime.py::sum_gradients` 复用布局和单桶缓冲 | 全局归一化后的SUM，不额外除以8；保留PPO/BC两次SUM以正确分解全局梯度 |
| 最后minibatch KL重复计算 | `analytic_kl_local` 与 `KLResultCache` 绑定rollout/参数版本/核/索引/聚合定义 | 最后80链复用、其余80链补算，最终全部160链全20步验收 |
| reward_substeps重复物理诊断 | GMT `closedloop/backend.py::advance` 只执行当前需要的 `collect_reward` | 保留50Hz控制、200Hz物理、四子步奖励和原终止逻辑 |
| RPC NPZ大量Python成员开销 | `runtime/closedloop_protocol.py` 原始连续数组v2：dtype/shape/offset/size描述符，兼容旧NPZ读取 | 回复字段/值逐项相同，解码结果独立可写；journal规范SHA及ACK语义不变 |
| 同一回复JSON编码两次 | `run_management.py` 复用不可变规范JSON字节供容量/SHA/SQLite | 执行回复持久化成功后才ACK |
| 扩散链和rollout重复保存 | `rollout_storage.py::BlockRolloutWriter` 真正16条块落盘，完整链仅在raw_samples保存一份 | 引用含SHA/size/策略身份；v1、v2读入及归档独立审计均覆盖 |
| old/next value逐条推理 | `parallel_training.py::_collect` 采集后固定Critic批量计算 | 终止bootstrap、GAE及实际执行时长不变 |
| 节拍/资产重复加载 | `rewards.py` 缓存音乐节拍；`asset_cache.py` 按文件身份缓存不可变字节与SHA，周期清空 | 文件stat身份变化立即重新核验，不共享可变张量 |
| 预算历史每轮整体复制写盘 | `budget_ledger.py` SQLite FULL追加事务、SHA链、唯一lease/接受身份；8rank一起预占/结算 | 零退款仅校验不写盘；崩溃未知消耗不退还；完整checkpoint保留可重建全状态 |
| 扫目录、监控和计时口径 | 增量容量账本；启动/恢复/每100轮全扫描；常规曲线缓冲；`performance.py` 分离本地计算与等待 | 恢复关键事件同步持久化；嵌套CPU/CUDA区间不直接相加；总墙钟包含preflight |

## 为什么正式选择2而不是4

原故障不是高斯公式错误。FP32矩阵计算会随batch形状和行位置产生微小差异，末端标准差0.001将均值误差放大成明显的联合log-prob差异。新路径固定形状及行位置后，真实20步自身重算误差约1e-12；跨旧标量路径仍存在约1e-2量级差异，因此必须使用新执行身份和新基线。

固定形状4、CFG合批确实能正确训练，Actor单次约5.27秒，比形状2的约6秒略快；但单环境采样每步只能产生一个后继状态，其余位置是保持数值路径的填充。实测生成延迟约0.59秒，跨过0.5秒决策间隔，真实物理执行量增加，整轮约77～90秒。形状2将生成延迟降到约0.32～0.33秒，校准预算0.42秒，恢复正常约25控制步/转移，整轮明显更快。这是端到端实测选择，不是概率门槛放宽，也不是microbatch偷偷退回1。

两组独立形状2验收均实际产生1、2、3、4次Actor更新，未强行关掉KL软停止来制造满四次更新。每类只有两次观测，下面P95仅描述这两次，不代表七天长训尾延迟。

| 实际Actor步数与场景 | 核心均值 / P50 / P95（秒） | Actor均值（秒） | 最终KL均值（秒） | 完整前台均值（秒） |
|---|---:|---:|---:|---:|
| 1，首轮完整概率检查、结束保存 | 41.60 / 41.60 / 41.86 | 6.01 | 1.74 | 49.72 |
| 2，恢复首轮完整概率检查、不保存 | 46.50 / 46.50 / 46.63 | 12.06 | 1.63 | 47.09 |
| 3，连续普通轮、哨兵概率检查 | 48.73 / 48.73 / 48.83 | 17.46 | 1.62 | 49.28 |
| 4，连续普通轮、结束保存 | 54.64 / 54.64 / 54.82 | 23.10 | 1.62 | 63.82 |

八卡采集最慢rank平均26.27～27.63秒，Critic约2.6秒；Actor梯度SUM加Critic SUM约0.39/0.60/0.80/1.00秒（1/2/3/4步），另有本地打包和等待，不能把此数字称作全部通信时间。峰值allocated显存约4.06～4.46GiB/卡。上层转移核心吞吐约3.85/3.44/3.28/2.93条/秒；Actor内部训练转移约265～277条/Actor秒。后三者均按实际更新次数统计，没有用减少更新带来虚假加速。

旧版1、2步核心时间分别70.04、88.08秒，Actor分别15.91、35.76秒，最终KL均约6.47秒。对应核心加速约1.68、1.89倍，Actor加速约2.65、2.97倍。旧版两轮均保存，新版第二轮不保存，因此完整墙钟不能直接当同保存工作量比较。旧版没有四步、普通哨兵轮或正式32任务周期轮的足量对照，报告不补造旧数值。

真实单条advance回复的10次只读codec对照：NPZ编码均值49.62ms、解码315.69ms、2,266,704字节；原始数组v2编码22.10ms、解码12.23ms、1,508,703字节，字段/类型/shape/值完全一致。约25.8倍是该回复的解码加速，不是整轮加速。

正式32任务计划另做两轮独立周期夹具：每卡4个10秒任务、全部8卡参与，初始化107.28秒，随后两次周期评估107.79、107.95秒（均值/P50=107.87秒，样本P95=107.94秒）。该夹具14项适用审计全部通过、0失败0未执行，明确使用 `--no-require-resume`，恢复能力另由前面两组四轮各22项审计证明。第一轮1次Actor更新、核心41.13秒；第二轮2次更新、核心42.93秒，后者使用普通哨兵检查，不能与恢复完整检查的46.50秒混组。第二轮含正常结束checkpoint和32任务评估的完整前台为160.75秒。正式每100轮一次评估时，当前观测折算平均约1.08秒/轮，不能将整个评估成本加到每个普通轮上；第100/300轮叠加完整概率/指纹/保存的全部组合尚未真实跑到，该触发逻辑由CPU测试覆盖。

32任务四来源等权回报从初始42.8151到第一轮42.9635、第二轮43.0126，三次各32任务的物理失败均为0。这是有限任务执行与更新的健康性证据，只有两次更新后的评估，不足以证明泛化或Stage2质量提升。原形状2两组短验收8任务的结果与这组32任务分别报告，不混算奖励均值。

## 剩余成本与每卡多环境方案

首组形状2第4轮rank0：20链GENMO生成约6.21秒；500控制步的纯PhysX约1.57秒、GMT ONNX约0.57秒、控制步结束诊断约3.36秒；全部后端控制步约9.47秒，其中包含物理和诊断，不能再次相加。journal与RPC仍占可见时间。纯物理并非采样主导，因此本轮保留CPU PhysX和CPU ONNX。

已定位但没有实施的进一步修改：50步GRUCell融合、BC异步预取、健康状态对象通信合并、同控制步last_trusted/terminal状态快照复用。BC准备约0.03秒/更新，异步预取须隔离全局Python/NumPy/Torch随机状态和可恢复游标；不能由后台线程直接进入现有全局RNG作用域。源码SHA约0.02秒/轮、seal约0.4秒/轮，完整核验继续保留，没有用弱指纹替代。整轮回滚快照约0.3秒，保留为KL拒绝时恢复Actor/Critic/Adam所需。小Critic的默认80次更新和两次Actor梯度SUM也未因提速更改。

多环境后续方案是先引入显式 `(rank, env_id, episode_id, request_id, policy_version)` 状态容器，每环境独立GMT状态、时间线、历史、音乐游标、终止和预算；ready队列只合并独立环境已经就绪的推理请求，保持各环境内部因果顺序。Actor批处理还需要新的环境槽位数值契约：当前按去噪step确定行位置，不允许直接把多个相同step塞进同一槽位。GMT动态batch须先核验ONNX的动态轴、按环境拆分obs/history/action，并与batch1逐项比较，再测试独立终止、故障传播和多环境GAE。没有这些验收，单改 `num_envs` 或切换GPU provider是不完整的实现。

部署延迟改变了前缀分布：本次形状2短评估通常P=21，而配置仍明确使用原P6～18的Stage1 s350000模型。这是需要长期配对验证的分布边界，不能把新旧运行的奖励差异全算成策略学习收益；本轮未擅自替换为服务器2新模型。短评估中的小幅奖励变化不能证明Stage2优于Stage1。

正式预算仍为10000接受轮、30000优化尝试、100万次生成、2500万控制步、1亿物理步。仅160次生成/轮已将理论生成上限压到6250轮以下，初始化和周期评估还会消耗额度；若每轮四次Actor更新，优化尝试也先于10000轮耗尽。预算会按原规则预检查并停止，本轮没有自动扩容。按每轮约55秒估算10000轮核心时间约6.4天只是算术外推，不表示现有预算能执行10000轮，也未包含周期评估、保存和故障停机。

## 证据、验证与兼容性

本地完整交付目录为 `/home/weili/bumi-closedloop-worktrees/analysis_reports/stage10_performance_refactor_20261008`，包含 `performance_comparison.json`（逐轮、逐rank及分组均值/P50/P95）、`efficiency_resolution.json`（原35项逐项闭环）、`shape2/numeric_probe.json`、`shape2/rpc_benchmark.json`、逐阶段JSON、带来源SHA的小型捕获及测试日志。`build_report.py` 只读重建对照，不运行训练。小型捕获不包含完整模型或原始执行归档。

新增执行路径的单卡真实模型20步梯度对照：各可训练模块梯度相对L2约1e-7～5e-7，最大参数差7.45e-9，Adam状态差约4e-9；自身联合概率误差约1e-12。完整CPU回归715项通过，另有GMT39项通过。曾出现6个同一Gloo测试fixture超时，堆栈在通信组销毁处，双rank结果均已保存；添加成功路径屏障后定向6项和全套715项通过，未放宽数值/超时或删除测试。旧接口的autocast弃用提示保留，不影响当前验收。

新块rollout与旧逐条格式均可读取、审计；旧NPZ帧可读取，v2传输明确标记；新增预算账本能验证旧v1初始快照及重建新事务。旧完整checkpoint仍可用原执行路径评估，本版新性能身份拒绝旧身份的无变化恢复。新版增量账本不支持旧的自动预算扩展写入路径，本任务未授权或执行预算扩展。

八卡性能数据对应GENMO `c0c28bd`、GMT `e02ff51`。交付检查补齐 `tools/train_closedloop_stage10.py::_sources` 对五个新增运行模块的逐文件SHA清单（performance、tensor_cache、asset_cache、rollout_storage、budget_ledger），另45项来源变更检测/性能/生命周期/断点测试通过。该补项不改变学习计算，但会改变新run的来源身份；不将旧测试run迁移成新身份。新增测试第一次因fixture缺少 `source_manifest_sha256` 等报告字段出现5失败40通过，补齐夹具后45通过，生产核验门槛没有放宽。

依用户要求保留完整原始执行证据，以下服务器1独立验收目录作为交付结果保留，不能按普通临时缓存清理：

- `/data1/user/liwei/GENMO_outputs/closedloop_stage10/tmp_performance_refactor_20261008`：形状4四轮及恢复证据。
- `/data1/user/liwei/GENMO_outputs/closedloop_stage10/tmp_performance_refactor_b2_20261008`：形状2首组四轮及恢复证据。
- `/data1/user/liwei/GENMO_outputs/closedloop_stage10/tmp_performance_refactor_b2_repeat_20261008`：形状2重复四轮及恢复证据。
- `/data1/user/liwei/GENMO_outputs/closedloop_stage10/tmp_performance_periodic32_20261008`：首次32任务初始化评估及预算预检拦截记录。初始化106.64秒，原50万控制步测试额度不足60.48万步保守预占，实际训练0轮，退出码0不代表训练验收通过。
- `/data1/user/liwei/GENMO_outputs/closedloop_stage10/tmp_performance_periodic32_bound_20261008`：按完整预占计算的新独立两轮32任务周期评估夹具；测试控制额度100万，仍低于正式2500万，没有扩展任何已有run。仅测试配置周期为1，正式周期100不变。

这些目录的后续清理条件是用户确认不再需要独立审计或完整备份已核验。本次pytest临时目录和传输tar已精确清理；没有删除旧训练或覆盖旧checkpoint。代码同步以 `记录文本.md` 最终提交与服务器HEAD核对为准。
