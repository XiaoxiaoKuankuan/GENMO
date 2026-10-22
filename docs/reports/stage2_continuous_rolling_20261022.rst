Stage2 Asymmetric DPPO 连续滚动生成
==================================

用途与选择的算法
----------------

本实现位于 GENMO ``feature/stage2-asymmetric-denoising`` 和 GMT
``feature/stage2-tracking-history``。直接扩展这两个分支的现有调度代码，没有从
``feature/bumi-music-closedloop`` 复制训练实现。

当前 ``tools/train_closedloop_stage10.py`` 默认仍为
``configs/closedloop/stage2_random20s_gamma095_server2.yaml``，固定2Hz；新模式显式使用
``configs/closedloop/stage2_continuous_rolling_server2.yaml``。两份原始YAML除新增的
``runtime.generation_schedule`` 外逐字段一致，测试比较了完整PPO优化绑定。
``stage2_credit_server2.yaml`` 仍为旧continuous_tracking.v2配置，当前配置解析器不接受；
不得将该旧实验的坐标平均目标或学习率混入本次完整链实验。

保留8192完整链mini-batch、epoch1、全16层、目标感知回溯、BC关闭、gamma=.95和
lambda=.98。Actor仍为48x50实际历史，Critic保留Tracking90；Stage1 s595000、20步
15+4+1、真实噪声核、CFG2.5、60帧30Hz和冻结GMT完全不改。所有奖励、随机化及物理
终止条件沿用当前配置。

时间与参考事务
--------------

新调度合同为 ``genmo.continuous_rolling.v1``；参考合同为
``gmt.executed_prefix_replaceable_lookahead.v1``，均写入已有runtime/训练checkpoint身份。
旧断点不能作为新配置的普通resume。默认Stage1在新目录初始化，并重新建立对应物理基线
和价值校准；不把旧完整音乐评价或旧调度MC标签当作等价成果。

每个环境每次只进行一个生成事务，600Hz整数tick下每60tick允许新决策。部署到达时刻
来自原单机器人实测profile，和实际训练GPU排队、日志、CPU参考准备墙钟无关。例如0秒
发起、0.26秒到达并提交，0.30秒读取最新状态再次生成；延迟0.32秒则0.40秒再读状态。
这不是固定0.3秒周期，也没有添加平滑或0.1秒动作过渡。

GPU批量生成在离线世界调度器中计算候选；模拟到达之前仍逐20ms执行旧参考和四次
PhysX子步。没有异步实时墙钟部署的性能承诺。不同环境的到达时刻由共享世界按最早
事件推进；不能因为一个环境未准备好而漏掉另一个活动环境的物理控制。

前缀使用本次可复现的预计到达时间，最少6帧。保护包含提交边界、中心差分右点和
30Hz插值右点；不将读过的未来10个参考永久锁定。以0.26秒到达为例，需要10个源帧，
不是18帧。生成时刻和整条动作时间起点不平移。到达截止仍受旧参考可支持的完整GMT
前瞻范围限制；迟到或接缝/版本不一致则拒绝，保留旧参考并在后续合法网格重试。

reserve绑定父计划和内容版本；prepare从精确已知前缀独立重算FK、SLERP、线速度和
角速度；commit只在50Hz控制边界检查后原子发布。此前执行的参考逐元素保留，当前位置、
姿态和速度须在既有严格容差内一致。CPU/GPU内容版本在物理推进前核对，逐控制步
GMT输入与奖励参考的版本必须相同。read_watermark只作为未来曾被观察过的证据。

窗口、奖励与存储
----------------

8秒世界窗口保持400个控制步。正常episode跨窗口续跑；20秒上限从1秒真实预热后开始，
超时使用重置前状态Value，失败和音乐结束仍零尾值。新reset按100ms世界网格开始；
如果所有环境都提前结束，真实推进空闲PhysX到合法网格，不改变世界时钟相位或虚构奖励。

窗口处未到达的候选明确取消，已完成的部分控制区间全部保存并自举。学习batch只接纳
同一Actor版本的新动作；此前已提交参考可以继续作为实际状态的一部分，其原版本和
来源保留。原GAE按真实控制步数换算gamma/lambda，无需改变算法公式。

ReferenceCreditLedger和compact_execution记录生成/到达/提交、21点窗口、当前中心、
奖励参考及覆盖区间。计划身份中可能含继承前缀，因此“该计划进入窗口”不能直接解释为
“新自由坐标已执行”；自由源帧归属仍需递归前缀血缘分析。奖励保留在唯一真实区间，不
按照plan_id重新分配。

资源推导上界改为8秒/0.1秒=80条/环境，而非固定16条；1024个环境的理论上界81920条
不是预设实际样本数。每轮优化数仍为ceil(实际有效完整链数/8192)，epoch1。新模式将完整
链和条件存放CPU，使用原有有界设备块缓存，避免频率增加后整轮GPU驻留超限。没有减少
真实8秒交互或裁去去噪链。报告包含实际链数、时长直方图、前缀长度、提交/拒绝/取消、
原训练性能与显存峰值。

操作命令
--------

先做有限真实采集（三个窗口覆盖20秒边界），不更新模型::

  cd /home/user/liwei/GENMO-bumi-closedloop
  bash scripts/train_stage10_8gpu_server1.sh \
    --config configs/closedloop/stage2_continuous_rolling_server2.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/rolling_new_capture \
    --gmt-repo /home/user/liwei/legged_lab_gmt \
    --collect-only --collect-rollouts 3 --stop-after-iteration 2

新训练从Stage1初始化，明确独立目录与累计轮数::

  bash scripts/train_stage10_8gpu_server1.sh \
    --config configs/closedloop/stage2_continuous_rolling_server2.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/rolling_new_training \
    --gmt-repo /home/user/liwei/legged_lab_gmt --stop-after-iteration 2

仅同一连续模式训练目录可恢复::

  bash scripts/train_stage10_8gpu_server1.sh \
    --config configs/closedloop/stage2_continuous_rolling_server2.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/rolling_new_training \
    --gmt-repo /home/user/liwei/legged_lab_gmt --resume latest --stop-after-iteration 2

本次只执行有限验收，以上正式训练命令为操作说明。

本次实际验收
------------

GENMO必要回归70项通过（5.07秒）；GMT参考/调度/列式回归49项通过（2.36秒），
包括真实CUDA缓存读回与CPU权威参考逐字段零容差一致，以及奖励关节FP64缓存。
固定模式默认配置与完整链优化绑定不变，旧checkpoint普通resume及显式旧片段迁移
均不能绕过新调度身份。

服务器2使用8张RTX6000D，每卡128个环境，共1024环境。三窗采集使用GENMO
``61ea68f``、GMT ``ff7c781``；随后在GENMO ``3e6dda2``、GMT ``4ae7370`` 上对最后的
到达事件推进优化再做独立一窗复测。两次都从原Stage1开始，不更新Actor或Critic；
本次Critic未经新模式MC拟合，相关Value只用于检验真实末态及自举数值路径，不能作为
价值估计质量或训练效果证据。

三窗逐卡均准确推进400个控制步，共得到24537、26727、25282条完整链，总76546条。
GENMO接管区间共有1121703个真实20ms奖励和4486812个PhysX子步；预热、失活等待
单列，不把这些计数冒充全部已分配世界步数。所有保留区间按物理episode连续，无重复，
GMT输入和奖励参考版本一致，行为版本均为0。

成功提交72684次，无参考事务拒绝；3862个未到达候选在窗口、终止或超时边界取消。
21次接管后姿态失败、25次接管前预热失败、479次音乐自然结束均保留统计。
603次片段超时全部为接管后20.000秒，并保留重置前tick12600、Critic时间21秒的状态。
真实终止与超时共1103个边界目标独立重算，最大误差8.882e-16；真实终止零尾值，
超时为实际区间折扣奖励加折扣后的next_value。

三窗全部前缀为10帧，与所用实测0.26/0.28秒延迟一致。72684条正常区间为15控制步
即0.3秒，其他均为真实音乐/失败/窗口/超时的部分区间；单元测试另外覆盖0.2、0.4、
0.5秒的不同到达情况，不能由本批延迟池推断调度被写死为0.3秒。

递归源帧审计覆盖2109个物理episode，来源链全部完整、控制区间全部连续。新自由源帧
在0.26～0.28秒进入未来窗口，在0.32秒成为当前中心；不是计划身份的0.28秒。
有2158个计划的自由源帧消费跨越rollout；完整可观察2秒范围内的计划通常只有9个
源帧成为当前参考，后续未来部分被新计划替换。全部决策自身区间结束前尚未成为当前
中心，约89.87%的控制步甚至发生在该动作进入前瞻之前。这是实际生成延迟带来的
跨transition收益，不是奖励错位；原半马尔可夫GAE继续向前传递后续收益，未按plan_id
搬移奖励、裁掉未来坐标或改造PPO目标。生成频率提高并不自动解决全部信用分配问题。

三窗主采集最长rank耗时分别395.33、431.31、421.58秒；包含targets和预算核对为
404.37、451.72、437.12秒；整个collect-only任务含启动、概率检查与固定数据落盘
共1614.54秒。NVIDIA进程级采样峰值每卡约17.2～17.9GiB。这些是服务器批量训练
墙钟，不是单机器人生成到达延迟。

首次有限采集曾因预付余量仅5步，而末端预热RPC合法申请25步，在465.73秒触发预算
错误退出；修复为400+25预付、未执行部分退款后重试成功，真实窗口始终400步。
没有通过减少8秒物理时间或忽略失败让验收通过。

最后版本与八卡优化结果
----------------------

最后一版等待到达时直接推进至事件，单次RPC最多25控制步；下一次生成仍按60tick
网格发起。独立一窗得到24536链、357831控制步、23512次提交，无事务拒绝，真实
物理失败9次、音乐结束1次。主采集最长rank为368.60秒，连targets375.52秒；整个
独立collect-only进程624.87秒。曾同时运行CPU只读审计，不能将这一次耗时差异当成
严格性能基准。

与三窗任务首窗逐条比较，共同的24536条记录任务、到达时刻、初始噪声、完整链和
奖励张量完全相同，最大差异0。唯一不同是rank4/slot13失败重置后的失活等待多5个
控制周期，因此8秒窗口尾部少一条5步未提交候选，前一条被标记为窗口末端；这不是
活动环境漏步。两个世界均400步。真实重复运行不宣称整体文件逐字节完全一致。

以最后版本真实一窗数据运行原完整链数学入口：8192全局mini-batch、epoch1、clip=.1、
名义LR3e-7、原目标感知回溯、BC关闭。24536条链产生3个mini-batch，实际执行7次
Adam候选尝试，4次按原规则回滚、3次保留；最终完整rollout联合KL均值.00983716，
检查通过。三个接受LR分别1.875e-8、2.34375e-8、2.9296875e-8，没有为验收放宽KL。

完整rollout目标为0 → .00466144 → .02082096 → .02992639；末步Active Clip41.29%，
16层Transformer每步均有实际参数变化，BC采样0。旧概率及独立Normal误差9.095e-13；
真实网络16条链的完整Autograd/global SUM与流式microbatch1/3/8梯度相对误差均小于
4.424e-7。数学进程511.63秒，其中Actor更新阶段341.63秒；PyTorch峰值27.63GiB，
NVIDIA进程级采样峰值30.98GiB。诊断后Actor、Adam、控制器、BC与各rank RNG逐字节
恢复，无正式候选被晋级，没有启动长期训练。

固定数据数学入口复现方式（CASE为重新有限采集产生的实际目录）::

  bash scripts/validate_stage2_dppo_suspicions_8gpu.sh full-chain \
    --case "$CASE" \
    --config configs/closedloop/stage2_continuous_rolling_server2.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/rolling_new_math \
    --learning-rates 3e-7 --epochs 1 --adaptive-kl

只读物理审计和源帧归属入口分别为 ``tools/analyze_stage2_rolling.py``、
``tools/analyze_stage2_execution_credit.py``。后者明确区分继承前缀、真正新自由帧、
前瞻和当前中心，不使用前瞻首次观察冒充中心执行。

结论与证据边界
--------------

独立滚动调度、可替换未来参考、动态前缀、8秒共享世界、真实变长回报/自举和八卡
完整链更新均已通过本次有限验收。优化目标改善只证明这批数据上的PPO数值更新正常，
不能证明物理回报或跟踪质量已经改善；本次没有训练后固定任务物理对照，也未完成
新滚动模式的MC价值校准。沿用实测单机器人延迟不等于服务器批量推理能实时部署。

可恢复的有限非法参考、迟到、保护范围或参考版本冲突继续使用旧参考；NaN、损坏的
概率链和不可恢复的设备/协议错误仍中止完整性检查，不伪造有效PPO样本。

机器可读数值摘要、失败启动记录、逐rank文件SHA和最终状态校验保存在同目录
``stage2_continuous_rolling_20261022.json``。按仓库临时测试清理约定，本次独立验收
根目录的临时固定数据和模型在审计完成后清理，历史正式训练/模型/日志不动；原路径
仅作为来源追溯，不作为仍存在的checkpoint下载链接。源代码、配置、数值摘要及
复现命令随两个实验分支正常提交推送。
