# 第二阶段八卡训练 v2 实施与运行说明

本版按最终授权配置实施。Actor 固定学习率 **5e-9**，正式训练不扫描候选、不自动降低学习率。正式入口 `scripts/train_stage10_8gpu_server1.sh` 默认使用 `configs/closedloop/stage10_8gpu_server1_v2.yaml`，默认仅运行一轮；长期训练必须显式给定停止轮次。

## 更新流程和配置

八个 rank 各自持有一个冻结 GMT/CPU PhysX 后端，使用同一模型版本分别采集 20 条上层转移（全局 160）。真实执行时长决定 GAE，环境/episode/连续区间之间不串接；old log-prob、old/next value、returns 和全局标准化 advantage 整轮固定。每条链 20 步，完整链组成 1600 内部转移的优化器 minibatch，2 个 epoch 最多 4 次 Actor 参数更新。计算 microbatch 启动依次校验 4、2、1 并记录共同通过的配置，独立于优化器 minibatch。

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
