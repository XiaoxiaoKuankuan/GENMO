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

新实验使用干净的Server-2交付目录；不修改任何已有运行源码目录。下面的启动命令会
运行新一次64次尝试验收，不是继续旧五轮训练；每次须指定全新的输出目录。

```bash
cd /home/user/liwei/awr-worktrees/GENMO-signals-delivery-20261021
ulimit -n 131072
bash tools/run_stage2_awr_server2.sh preflight \
  --config configs/closedloop/stage2_awr_group_credit.yaml \
  --gmt-repo /home/user/liwei/awr-worktrees/legged_lab_gmt-signals-20261021

# 前台有限验收，建议在独立tmux会话内运行；自动保存launch/log/exit回执。
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/launch_stage2_awr_signal_server2.py \
  --gmt-repo /home/user/liwei/awr-worktrees/legged_lab_gmt-signals-20261021 \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/<新的运行目录>

# 同一冻结Replay/交叉Value的11种标签比较，不训练、不重跑物理。
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/audit_stage2_awr_signals.py \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/group_credit64_20261021_01 \
  --output /data1/user/liwei/GENMO_outputs/stage2_awr/group_credit64_20261021_01/<新的审计目录>

bash tools/run_stage2_awr_server2.sh stop --run-dir <本次运行目录>
bash tools/run_stage2_awr_server2.sh resume --run-dir <本次运行目录>
bash tools/run_stage2_awr_server2.sh evaluate --run-dir <本次运行目录>
```

resume要求相同源码、物理资产与保存的配置，不拼接中断的PhysX状态。候选缺成员时不
生成伪group；已确认的预热失败不会被重新抽歌，且不再计为已完成动作Episode。
本次已完成运行的原始源码是`GENMO-signals-20261021@56014c3`，其恢复/重评必须使用
该目录；新交付目录包含后续离线审计精度修正，不能冒充原运行源码。

## 验收记录

本地首批167项GENMO/AWR/Stage1/评价回归通过，17项GMT快照、调度和列式回执通过。
Server-2最终交付代码1381c4d执行175项回归全部通过（12.46秒），只有既有autocast
弃用警告；测试进程不可见GPU，临时目录自动清理。后续83661a0仅改文档与注释，
执行AST未改变。上述结果不代替下面的真实物理结论。

## 2026-10-21真实64集结果：实现贯通，学习收益未成立

运行目录：`/data1/user/liwei/GENMO_outputs/stage2_awr/group_credit64_20261021_01`。
原始运行GENMO为`56014c3cb964b314b8d96f9c580a04aa669a31d9`，GMT为
`a1b624490fff0468ac25dcd59daeeb55e259a20b`，全程未热更新。运行正常退出0，
完成1宏迭代；旧formal5保持停止。数据、失败、B/C模型、原标签及原始回执完整保留。
可移植指标与证据SHA见[验收归档](validation/stage2_awr_group_credit_20261021.json)。

64次尝试得到64个完整Episode：56自然结束、8偏航失败，没有预热失败或补采。
4796次决策中4757条是有效Actor样本；39条未实际暴露的生成不用于Actor监督，
其物理账本和完整Episode仍保留。完整Replay为8,075,937,929字节，快照身份
`15342906688024d8daf0509beff17c51eeab706a3f20cee47eded2ec4e260121`。

物理评价是同一组原Stage1 **train** 音乐16首×2种子=32任务。四来源train清单SHA
与s595000的qpos统计文件一致；早期发行元数据中的original_split不代表s595000
实际训练划分。没有使用当前val/test清单；本次不报告泛化能力，也不是全训练集穷举。

| 指标 | A：冻结Stage1 | B：无权重自蒸馏＋BC | C：Group-hybrid AWR＋BC |
|---|---:|---:|---:|
| 来源平衡平均完整任务回报↑ | 117.11446 | 113.26210 | 112.52847 |
| 32任务原始奖励总和↑ | 4317.17280 | 4175.77545 | 4148.67175 |
| 来源平衡每秒回报↑ | 2.58727 | 2.48940 | 2.47504 |
| 完成/物理失败 | 30/2 | 30/2 | 30/2 |
| 总执行时长，秒 | 1730.54 | 1744.92 | 1745.54 |
| Root位置误差，米↓ | 0.24353 | 0.30200 | 0.28356 |
| Body位置RMSE，米↓ | 0.14559 | 0.17755 | 0.16742 |
| Yaw误差，弧度↓ | 0.30994 | 0.26836 | 0.29619 |
| Joint位置RMSE，弧度↓ | 0.12585 | 0.14534 | 0.14627 |
| Joint速度RMSE，弧度/秒↓ | 0.82401 | 0.95205 | 0.96176 |
| 固定train监督探针Stage1完整Loss↓ | 0.16197 | 0.19214 | 0.18961 |

完整任务回报先在同音乐组内平均种子，再在各来源内平均音乐组，最后等权平均来源；
跟踪误差列取原物理报告的全控制步统计。95%置信区间按来源平衡、音乐组聚类重采样：

- B−A：−3.85237，95%CI [−5.64513, −1.91636]。
- C−A：−4.58599，95%CI [−6.44054, −2.56179]。
- C−B：−0.73363，95%CI [−1.24325, −0.32536]。

因此这次C相对A、B的奖励都下降。C相对B的Root/Body误差有所减少，但Joint/Yaw
更差，且仍未达到原Stage1的Root/Body/Joint表现。三个模型都保留全部失败任务。
best保留Stage1的唯一选择理由是回报更高；完成率、活动度和执行时长没有作为门禁。
B的负结果没有中断C，C实际完成了所有训练和评价。

## 候选组和信用信号的实际覆盖

4个候选组/16个Episode全部成员完整，只有FineDance和AIOZ两组通过严格初态检查，
全部被检查字段最大差精确为0。AIST和Mine两组初态不符，正确回退普通样本。
不符字段包含实际qpos、速度、接触和输入历史；AIST组最大关节速度差0.02988rad/s、
净接触力差0.68294N，Mine组分别0.03596rad/s、0.32772N，明显超过预设严格容差。
没有放宽容差或把同seed当成同状态。预热/环境求解状态的具体差异来源尚未隔离，
本系统仍不具备任意时刻克隆完整PhysX状态的证据。

有效组内优势仅8条首次动作，占4757条有效Actor样本的0.168%。后续动作不冒充
同状态比较；两组无效候选不产生group训练标签。FineDance有效组2秒回报为
4.8937854/4.8815046/4.8886129/4.8957286，窗口各100控制步；四集均在之后失败，
这些失败全部保留。纯grouped标签权重随共同窗口回报单调递增。group_hybrid同时
混合MC/credit，可能改变这个排序；例如FineDance候选0的混合权重大于候选3，
不能宣称混合信号一定遵从短窗口排序。

首次计划账本实例如下：commit=744tick，首次进入GMT前瞻=792tick，首次成为当前
参考=912tick。时间全部来自实际600Hz账本，未把生成时刻或前瞻首次观察当作中心消费。
最终4757条样本具备有效前瞻信用，ESS=3646.10（76.65%），权重均值1、P50=0.79334、
P95=2.22250、P99=2.60282、最大3.04724，原始cap触发比例0。
各来源权重质量占比：AIOZ15.91%、AIST4.36%、FineDance42.87%、Mine36.86%。
按来源观察，高权重组通常具有较低Root/Body/Yaw误差，但Joint排序仍不稳定，
FineDance高权重四分位Joint误差0.15911rad，高于低权重组0.15331rad。
这些是暴露/质量诊断，不是严格物理因果证明。

同一冻结Replay/Critic完成11模式离线标签审计，耗时238.76秒，没有新增训练或物理：
MC、nstep4/8/16、current/current+lookahead credit、hybrid0/0.5/0.7、grouped、
group_hybrid。c全1与普通N-step最大差1.63425e−13；MC与alpha0权重逐值相同。
后续FP64直接构造修正仅改变group分量中间舍入：原运行与新代码最终权重最大差
3.58705e−8，原运行标签没有重写。各模式身份独立；11种标签审计不等于11组Actor
物理消融，尚未据此确定最佳混合系数。

## 更新、分布式和成本证据

B/C都从s595000开始，使用同一4757条Replay，5Epoch各95次更新；全局AWR/BC
batch=256/128，八卡microbatch=8/4。名义LR1e−5、warmup100，实际95步依次使用
1e−7至9.5e−6；保存的下一步调度值为9.6e−6。无教师一致性，不设置45步上限。
BC系数0.2720249901在本轮固定，八批校准AWR梯度范数22.59599、BC原始8.30659，
平均夹角余弦0.15147，BC加权后梯度范数比例0.1。Clip=1；原Contact真实监督由BC提供。

Critic两折均选择MC+FVI：内部验证MSE分别1420.07→469.74、1105.26→354.19，
绝对偏差也减少。每折16训练Episode/4音乐组、16内部验证Episode/4音乐组，
为另一折32Episode/8组标注；这是本64集小样本配置，不能解释为正式20%留出结果。
优势当前/未来Value都来自未拟合该音乐组的同一冻结Critic。Critic改善不等于Actor改善。

八卡真实Stage1/真实BC验收通过：9个AWR+3个BC的非整除尾批和3个AWR+1个BC
的空rank场景，最大梯度相对L2误差分别1.84036e−6和5.23153e−8，更新后的八rank
参数指纹相同，耗时13.58秒。B/C最终断点独立只读恢复检查耗时6.00秒，320个Adam
状态均为95步，Actor/Optimizer/Scheduler逐元素精确恢复，8rank BC随机流及训练
样本身份相同，恢复Actor指纹与真实物理评价一致。此检查没有额外执行续训或PhysX。

| 阶段 | 实测墙钟 |
|---|---:|
| 64集采集 | 638.20秒 |
| 双Critic | 120.37秒 |
| 标签生成 | 7.88秒 |
| B训练，5Epoch/95步 | 约189.9秒，约1.95秒/步 |
| C训练，5Epoch/95步 | 约187.0秒，约1.92秒/步 |
| Stage1/B/C各32任务评价 | 584.40/561.32/561.86秒 |
| 整次启动至退出，含初始化/I/O | 3381.20秒，56.35分钟 |

采集约7.45条有效重规划/墙钟秒；生成批次平均0.235秒。Replay首次读取8.08GB耗时
158.07秒（约51.1MB/s），训练缓存读取另计。Actor进程Torch峰值每卡4,982,733,824
字节（约4.64GiB）；整个运行nvidia-smi离散采样设备峰值11,838–12,322MiB，属于
总设备显存的采样下界，不能与Torch分配峰值混用。两次5秒采样超时及最大7.23秒
空隙保留。整次八卡分配成本约7.51GPU小时，不等于GPU持续繁忙时间。

## 尚未解决的问题

已经排除了“B/C使用不同Replay/BC预算”“八卡尾批错误归一化”“把FVI目标冒充MC”
等本轮可检验的问题；仍没有证据把普通自蒸馏漂移归结为唯一原因。关闭教师并按正常
更新预算训练后，B仍退化，说明必须继续检查生成样本再加噪的监督目标、扩散时间
分布、闭环历史条件以及BC之间的梯度作用。不能靠教师、极小LR或门禁把退化藏起来。

下一步应在相同Replay和更新预算上拆分Actor目标/分扩散时间诊断，独立比较纯MC、
N-step、credit与grouped的真实Actor结果，同时提高严格初态一致候选的有效覆盖。
本轮不自动扩大为512集或恢复旧五轮；所有负结果与现有实现均已保留供后续定位。
