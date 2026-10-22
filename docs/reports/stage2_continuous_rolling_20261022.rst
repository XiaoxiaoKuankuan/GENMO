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
