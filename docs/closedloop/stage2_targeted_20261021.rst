2026-10-21 Stage2两小时定向优化与真实验证
================================================

本轮时间范围为北京时间16:34:45—18:34:45。固定GENMO分支
feature/bumi-music-closedloop和GMT分支feature/bumi-frozen-gmt-backend。
这是限时算法与物理验证，不自动恢复旧正式训练；旧4096任务在第19轮保存后正常退出0。
新的物理目录stage2_nobc_rewardstd_20261021_physical从原Stage1 s595000开始，
只做新奖励MC预热及两个连续8秒真实rollout，Actor训练迭代仍为0。

配置与行为边界
----------------

Stage2的BC反向、样本消费和BC梯度控制器更新关闭；实际BC样本、权重和梯度为0。
原Stage1配对监督、已知前缀训练及只读保留能力验证代码保留，没有新增教师或蒸馏。
全部16层Transformer仍参与允许的PPO更新，Actor输入、历史GRU、音乐和前缀结构不变。
GMT及Stage1前15步/末步副本冻结。20步、60帧/30Hz、0.5秒重规划、50Hz GMT与200Hz
物理时钟、真实噪声标准差调度保持；没有删除远期概率、裁剪ratio或恢复ESS硬门限。

奖励版本明确分离：

* v2：stage2.gmt8_reference_limits.v2；原tracking.std及原评分保持可复现，完整旧配置
  保存为stage2_transformer_groups_reward_v2.yaml。
* v3：stage2.gmt8_reward_std.v3；source_std严格核验冻结GMT原误差来源，reward_std
  单独决定Stage2评分，不能声称数值仍与原GMT奖励相同，也不能直接恢复混用v2模型。

四个改动的source_std/reward_std依次为anchor_pos 0.30/0.20、anchor_ori 0.40/0.30、
body_pos 0.30/0.15、body_ori 0.40/0.20。其余仍为body_lin_vel 1.00、body_ang_vel 3.14、
foot_height 0.07、leg_joint_pos 0.25。原八项权重0.5/0.5/1/1/1/1/0.8/0.5除以6.3，
track_weight不变。CPU标量和GPU向量共享exp(-squared_error/reward_std**2)，
传入的已经是平方误差，未再次平方。

只读奖励消融
--------------

analyze_stage2_reward_std.py输出线性及对数尾部曲线、CSV、JSON。A为旧std指数；
B为用户指定的新std指数，是默认正式候选；C为0.95乘B加0.05/(1+squared_error/sigma**2)，
仅在离线工具出现，未接入训练。误差横轴是原GMT平方误差的平方根；身体项为原聚合
误差的平方根，不能解释成某一个关节或身体的单独世界位移。

历史第10轮90216个真实控制步按同一原始误差反事实重评分，anchor_pos低于0.001
的比例由13.8235%增至27.1681%，anchor_ori由8.0485%增至13.7958%；body_pos与
body_ori在这批数据均没有低于0.001的样本。新std提高小误差分辨率，也使大误差更快
饱和；B的body_pos在约0.394米、anchor_pos在约0.526米后低于0.001。
这些是同一旧物理轨迹的重评分，不是新奖励或关闭BC已经改善机器人执行的证据。

真实参考消费与延迟
------------------

analyze_stage2_execution_credit.py读取两份不可变真实case，合并同一物理episode的
实际控制窗口，递归继承前缀原始来源，输出逐计划及环境/来源/前缀/8秒区间统计。
逐控制JSONL压缩证据保留中心参考对应的原始计划、源帧及线性插值权重。
位置/关节、四元数SLERP支持和速度差分/FK依赖分开；GMT新增只读
source_velocity_support并与真实重采样和FP32差分验证，实际调度与参考值不改。

首批结果：30120个实际生成计划、749915个真实控制步，compact来源记录缺失0。
25997个完整保留观察时域的计划，其原始新生成中心源帧数全部为15；实际前缀18帧、
自由42帧。此处15来自实际控制与递归来源，非固定预设。2042个计划跨8秒rollout边界
继续被消费。commit相对决策约0.26—0.28秒，首次原始自由帧进入当前中心为0.58秒。

当前transition的控制区间早于本计划自由帧首次中心消费；按整个保留范围统计，
约76.09%的控制还早于其首次前瞻消费。缺失首次时间的未提交/终止/右截断计划保持
可识别，不能据此宣称永不执行。中心消费、GMT前瞻作用及Transformer跨帧耦合是
不同概念；未直接中心消费的概率因子依旧保留，不能把其KL统称无效。

GAE与价值证据
--------------

原GAE根据真实50Hz控制数折算gamma/lambda，真正终止关闭自举，正常rollout截断
保留next_value并切断当批递推；按环境/episode/策略版本分段后才做全局优势标准化。
本轮没有按plan_id搬动奖励，也未把不同Actor版本混合为on-policy batch。
只读工具把同一行为版本的两个连续窗口用于诊断，比较原固定GAE目标、连续GAE与
lambda=1回报；后者只有观察到真正终止的尾部才无自举，其余明确是右截断自举，
不能冒充完整物理episode MC。真实物理失败附近与音乐自然结束、普通rollout边界分组。

新奖励完整MC预热正常退出0，96首训练音乐乘2种子，独立32任务2447状态验证：
纯MC候选获选，MSE125.979744、EV0.919005，相对初始MSE下降96.6235%。
paired motion warmup和standing分组EV分别0.916017与0.976475。
这是当前固定Stage1行为的价值预测证据，不是Actor优化效果。细分未执行参考造成的
价值偏差仍需要结合后续同协议数据，不能只凭整体EV排除局部偏差。

完整链PPO与候选
----------------

保留四步联合FP64 log-ratio及原clip/KL。主配置audit_full_rollout_objective=true，
更新前、每次接受Adam后记录完整不可变rollout的真实clipped目标。来源、优势正负、
普通/KL尾部成员分别报告相对行为与相对前一步的目标变化，尾部变化按同成员比较。
StepDiagnostics记录真实FP32 Adam增量与当前minibatch的负PPO损失梯度、加权BC
梯度和合成梯度的内积/余弦，不能把minibatch梯度冒称完整rollout梯度。

compare_stage2_optimizer_batches.py公平比较4096、8192、完整rollout累积及4096
objective-aware回溯，均从相同Actor、原分组Adam、BC、RNG和真实行为数据开始。
每次回溯仍检查原真实联合KL；目标感知候选额外要求整批目标不下降，并在有限合格
试探中选真实目标最好者。无合适候选正常结束该次更新，恢复对应Actor/Adam/BC/RNG；
原KL-only逻辑保留为对照，不放宽阈值，不删尾部，不截断重要性权重。

首轮真实GPU对照在第一次Adam前发现新增审计路径的CPU records/CUDA索引错误，
修复仅把小型[N,4,8] FP64记录移到Actor设备；原GPU路径不拷贝。真实CUDA回归
证明CPU记录与GPU记录所得全部梯度逐位相同。原失败目录/退出码保留，重跑另用
batch_comparison_retry，同一原始case未重采或改写。

时间残差是独立候选：TemporalResidualPolicy在15—18步使用
mean_stage1+gate*(mean_trainable-mean_stage1)，从known_mask及真实消费证据选择
支持范围，边界余弦平滑。采样和重算共用均值、全部自由概率因子保留，零gate均值
严格等冻结先验。原完整60帧PPO仍为默认。冻结最后一步仍有跨帧耦合，因此未声称
零gate保证最终远期动作固定；同增量反事实探针与独立门控训练/PhysX收益严格区分。

采样协议
----------

1. 正式随机训练及PhysX评估：实际eta=0.1及现有std/floor，前15冻结、15—18可训练
   随机高斯转移、19冻结确定性；完整20步。
2. 固定噪声随机链评估：与1相同核及innovation分布，固定任务/种子使噪声协议可重现；
   不是把innovation置零。原行为链探针固定旧state15及四步真实innovation。
3. UpperEnvironment确定性展示：保留混合网络及原均值系数，只把innovation置零，
   明确display_only、不入PPO。已修复原直接actor.sample绕过混合网络的路径。
4. 交互sim2sim：原实现明确为eta=0、仅初始噪声随机的混合网络展示，先核验训练核
   再显式切换展示核；不与1的正式随机物理结果混比。本轮未进行本机实时MuJoCo验收。

验证与结果边界
----------------

服务器1大范围隔离CPU177 passed/3 skipped；其后概率证据复用/原候选导出定向
12 passed，新增真实GPU边界回归所在定向11 passed，分别留存日志，不把重复用例
加总成独立测试数。GMT使用实际训练合同28 passed。隔离源码和pytest临时目录自动
清理；正式checkpoint、原日志、原失败任务全部保留。

八卡候选对照、同协议物理配对与门控反事实的最终数值，在本轮截止前据实补充。
当前不宣称BC一定有害、不宣称新std或目标感知回溯已经改善机器人执行，也不自动
启动正式训练。

证据目录
----------

服务器：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_optimization_20261021_analysis
本地：outputs/analysis/stage2_optimization_20261021
新物理采集：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_nobc_rewardstd_20261021_physical

查看有限对照（不是正式训练）：

::

   ssh 6000D-Server-1
   tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_optimization_20261021_analysis/batch_comparison_retry.log

