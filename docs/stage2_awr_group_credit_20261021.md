# 训练音乐上的同状态候选与局部参考信用实验

本实验按2026-10-21新要求实施，取代新训练入口的教师约束和保留门槛。旧正式配置、
五轮任务、Replay、checkpoint和历史修复证据保持不变。当前代码只按来源平衡的完整
任务回报选择best；失败率、执行时长、活动度和Root/Yaw/Body/Joint误差仍完整报告，
不阻止AWR更新。默认只运行64次预定尝试、一个宏迭代，不能据此声称学习有效。

## 自蒸馏与漂移的边界

无权重自蒸馏是将Replay样本权重全部设为1，使用同一去噪Loss和原Stage1 BC。
教师一致性是额外比较当前网络与冻结原网络的两个分支预测；它会阻止变化，但不能证明
原训练机制正确。本实验reference_weight=0，实际不创建教师网络，不计算一致性Loss。

即使目标由原模型生成，有限20步DDIM的输出重新经过Stage1前向加噪后，也不要求原
去噪器在每个随机时间点都将其精确重建。因此自蒸馏的期望梯度没有实现保证等于零。
实际数据还包含闭环状态分布、已知前缀、CFG与原预训练BC之间的目标差异。此前证据
只证明单分支拟合CFG目标存在接口不一致，且修复后仍会漂移；不能说CFG是唯一根因，
也不能用强教师或极小学习率下的保留结果证明根因消失。当前继续保留同CFG前向和
Replay关闭Dropout的修复，通过无教师B/C对照直接测量更新后的真实表现。

## 随机流审计

- 旧tasks._task把seed、rank、样本等放入task_id，再由task_id派生noise_index_start；
  公共生成入口使用stable_noise_seed(1729, index)，每次生成递增。因此改task.seed
  同时改变reset请求和全部生成噪声，不是只改变一次动作。
- 当前GMT vector_service.reset_episode接收seed但不使用它。config固定cfg.seed=42；
  vector_backend.reset使用站立或配对动作的确定性初态，关闭reset/domain randomization、
  观测噪声、外力和延迟。本系统不声称恢复了PhysX内部求解器随机状态。
- 新候选任务显式保存reset_seed、generation_seed、group/index/count、公共未来key、
  comparison_start_tick和初态指纹。首次噪声由generation_seed提供；后续仅由组key和
  decision_index派生，排除rank、slot、task_id、重试次数。
- 每组4个Episode，只有第一次生成噪声不同。group_fraction=0.25指Episode配额占比；
  64次尝试中16个候选Episode形成4组，其余48个普通Episode。不会变成64×4次尝试。
- 实际首次reserve后保存根位姿、根线/角速度、关节位置/速度、全身净接触力、Actor
  全部条件、音乐起点、前缀长度和Anchor。默认状态atol=1e-5、接触力atol=1e-3 N；
  bool、整数和条件时间严格相等。指纹是精确内容哈希，容差比较另外记录各字段最大差。
- 组成员缺失、行为/物理/奖励身份不同、初态超容差或未实际执行首动作均不能产生组内
  标签。保留为普通完整Replay并回退；缺首动作的预热失败保留排除证据，不伪造Episode。

## 标签数学和账本

原50Hz奖励与完整MC不变：gamma_c=0.99^(1/25)，R_k为真实控制样本折扣和，Gamma_k
取实际执行步数。正常终止与失败终止尾值为0；非零步事件已在原reward数组中，零步
事件仅从event_reward补入一次。

Actor N-step是独立于Critic FVI的新标签：

    A_n(k) = Σ_j Gamma(k,j) R_(k+j) + Gamma(k,n) V(s_(k+n)) - V(s_k)

支持4、8、16、full_mc；超出真实终止取消bootstrap。一个Episode当前与未来的全部
Value来自同一未训练该音乐组的Critic快照，随标签冻结；不在Actor更新后重算。

新reference_ledger按二进制保存每个控制步的600Hz tick、21点参考plan索引和时间、
中心点、奖励参考、原始跟踪误差及commit来源。验证连续、唯一、同Episode及先commit
后观察。current_only每步把1分给中心plan；current_plus_lookahead默认中心0.5，
未来10点各0.05。相同plan的份额相加，每步总量恒为1，bootstrap份额不转送给新动作。
在决策区间j内按真实控制步平均得到c_ij，零步没有伪造暴露。

    delta_j = R_j + Gamma_j V(s_(j+1)) - V(s_j)
    A_credit(i) = Σ_j Gamma(i,j) c_ij delta_j

c全1时退化为普通N-step，单元测试使用可手算可变时长及零步失败验证。c描述计划
在输入中的暴露，包括其保护前缀，不是已证明的物理因果贡献；不声称无偏策略梯度。
旧Replay缺逐步证据时不能伪造credit，按credit_fallback回退nstep或mc。

同状态候选使用同起点的2秒/4秒真实折扣回报，默认2秒打分。失败后的缺失控制步为
零吸收尾部，记录缺失数量与真实事件；不对存活时间平均。A_group=G_window-group_mean，
只标首次动作，后续动作不被当成同状态比较。

MC、N-step、有效credit、有效group分别以本轮FP64总体标准差归一化，不减均值。
缺失/近零方差分量明确记无信息并按规则回退，不能用除极小数制造巨大权重。
普通hybrid=(1-alpha)norm(MC)+alpha norm(credit或fallback)；group_hybrid在有效首动作
再以group_alpha混合norm(group)。grouped只替换有效首动作，其余用标准MC。
最后仅指数加权一次：beta=1、raw cap=20、整轮均值归一化。混合结果不按batch再次归一化。

AdvantageLabelSet保存mc/nstep/credit/group/final_advantage、final_weight、credit_valid、
credit_exposure、group_valid、source和label_mode。身份绑定Replay snapshot、两Critic
参数指纹与fold、gamma、N-step、credit模式/比例、alpha、候选配置、奖励、采样器及
实际分量指纹。Dataset拒绝用旧标签静默运行新模式。

## 配置与操作

`configs/closedloop/stage2_awr_group_credit.yaml`是完整独立实验配置：DDIM20/CFG2.5，
64次任务尝试/64集FIFO、1宏迭代、Actor LR1e-5、5Epoch、AWR/BC全局batch256/128、
microbatch8/4、warmup100、clip1、原Stage1 BC保留。BC系数通过本轮8批真实梯度诊断
校准后固定；B/C共享同系数、数据顺序与噪声。没有45步上限、极低LR或教师约束。

`stage2_awr_signal_*.yaml`是明确的独立对照配置，继承同一基础物理与优化参数，分别
提供MC、N-step4/8/16、current/前瞻credit、hybrid alpha0/0.5/0.7、grouped。
`extends`仅用于配置装配；运行request.json保存展开后的配置。

以下命令在新且干净的Server-2两个AWR目录执行；不修改任何已有运行源码目录。

```bash
cd /home/user/liwei/awr-worktrees/GENMO-signals-20261021
ulimit -n 131072
bash tools/run_stage2_awr_server2.sh preflight \
  --config configs/closedloop/stage2_awr_group_credit.yaml \
  --gmt-repo /home/user/liwei/awr-worktrees/legged_lab_gmt-signals-20261021

# 前台有限验收，建议在独立tmux会话内运行；自动保存launch/log/exit回执。
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/launch_stage2_awr_signal_server2.py \
  --gmt-repo /home/user/liwei/awr-worktrees/legged_lab_gmt-signals-20261021 \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/group_credit64_20261021_01

# 同一冻结Replay/交叉Value的11种标签比较，不训练、不重跑物理。
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/audit_stage2_awr_signals.py \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/group_credit64_20261021_01 \
  --output /data1/user/liwei/GENMO_outputs/stage2_awr/group_credit64_20261021_01/signal_audit

bash tools/run_stage2_awr_server2.sh stop --run-dir <本次运行目录>
bash tools/run_stage2_awr_server2.sh resume --run-dir <本次运行目录>
bash tools/run_stage2_awr_server2.sh evaluate --run-dir <本次运行目录>
```

resume要求相同源码、物理资产与保存的配置，不拼接中断的PhysX状态。候选缺成员时不
生成伪group；已确认的预热失败不会被重新抽歌，且不再计为已完成动作Episode。

## 验收记录

本地代码批次：167项GENMO/AWR/Stage1/评价回归通过，17项GMT快照、调度和列式回执
测试通过。临时目录已自动清理。真实Server-2结果随后追加；以上CPU结果不证明候选
物理初态可比或跟踪误差改善。
