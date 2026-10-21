4096完整链PPO、取消活动度系数和偏航终止的新实验
================================================

用户2026-10-21明确要求开启正式训练：全局批量4096，使用新运行目录，关闭偏航
超限终止，奖励不再乘活动度系数。全部16层Transformer参与15～18步学习；
从Stage1 s595000开始，原54轮模型、日志和旧评估保留。此文记录实际行为和证据，
启动完成前不把准备工作写成已完成Actor训练，也不宣称物理收益已提升。

代码与奖励边界
--------------

rewards.py和columnar_reward.py删除跟踪/音乐分项的活动度乘子，分项gate及
activity_gate恒为1；活动度RMS、配对目标和旧门控比值保留诊断。音乐条件、
活动度数据完整性检查、Stage1 BC保留；音乐奖励和独立稳定性奖励仍关闭。
原八项跟踪、Alive、各物理代价、事件罚分、权重、物理时钟均不修改。

奖励语义明确记为stage2.gmt8_reference_limits.v2，并同步当前七个Stage2配置、
评估采集检查和夹具。新奖励不再伪装成旧活动度门控奖励；历史结果原样保留。
当前正式stage2_transformer_groups.yaml的actor_minibatch_chains为4096，
16层主干倍率0.1、输出头1、共同名义LR1e-6、Clip0.10、Epoch1、梯度裁剪1，
BC每rollout2048，名义4次Adam各512样本，实际尾批按完整链数处理。
逐rollout重试名义LR、保留Adam和模块倍率、正常结束有限拒绝逻辑不变。

偏航终止yaw_error_rad=null，保留根高、非yaw倾斜和末端高度终止；
偏航误差继续记录，原姿态跟踪奖励继续评价偏航。未更新冻结GMT权重，
未改Actor/Critic结构、随机核、已知前缀和50Hz/200Hz/0.5秒时钟。

服务器1执行安排
---------------

运行目录：/data1/user/liwei/GENMO_outputs/closedloop_stage10/
stage2_transformer_4096_ungated_20261021_physical

分析/启动日志：/data1/user/liwei/GENMO_outputs/closedloop_stage10/
stage2_transformer_4096_ungated_20261021_analysis

新实验累计上限从创建起为200。使用现有八卡入口，自动完成新协议的完整MC
价值校准：96首独立训练音乐×2种子，训练重置分布下采到真实终点；独立32任务
验证。固定质量评估以16首音乐×2种子从片段开头执行，建立新Stage1基线。
新奖励和终止会改变回报标签，因此不直接导入旧协议MC；源码版本不作为门禁。

第一会话累计停在2轮，检查实际Actor更新、完整checkpoint、正常退出后，同一
run与同一Adam恢复至第10轮，完成提前固定评估，再自动恢复至200。
没有“任意一首新增失败就禁训”或ESS门禁，不要求两轮立即奖励提升。
50/100/150/200轮固定评估，100/200轮里程碑保存，2/10轮受控结束也完整保存。
同一tensorboard_compact目录维持一个run，旧基线/旧reward不能作为新协议收益对照。

登录后查看启动与训练：

::

  tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_4096_ungated_20261021_analysis/launcher.log
  tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_4096_ungated_20261021_analysis/initial.log

回归验证
--------

服务器1隔离CPU264 passed/1 skipped（23.74秒），覆盖零/低活动和不同配对
活动度时奖励不打折、GPU奖励数学的CPU执行、标量/列式一致、真实4子步、
完整链概率梯度、分组Adam/回滚、MC监督、预热导入、评估保存和源码仅记录。
首轮8项失败为旧门控预期、夹具一次超过25控制步及旧策略构造替身接口；
按新奖励语义和实际接口修正后全量通过，没有放宽数值容差或检查。

临时源码/tmp/stage2_4096_ungated_i57_8zjp与/tmp/stage2_4096_ungated_26q37upn
和服务器传输patch已精确清理，本地patch在本批结束前清理。原始失败与通过日志
保留上述analysis目录，作为本轮必要证据。正式模型与训练日志不做清理。


真实启动记录
------------

13:22:24通过tmux genmo_transformer4096_ungated_train启动本次流程，实际代码
03f469c；首会话69a9841d-31f1-4758-b757-dc7039208290，torchrun父PID3139333，
LOCAL_RANK0～7对应worker3139339～3139346，均已存在且创建GPU上下文。
13:23保存Stage1初始完整checkpoint，尚无Actor更新。

startup_verified.json核对实际training_identity：4096、all、595000、reward v2、
yaw=null、MC匹配训练reset，18组LR倍率正确。真实Isaac TerminationManager
日志仅含anchor_pos、ee_body_pos、non_yaw_anchor_ori三项，偏航项未注册。
当前已进入preheat_val物理采集；此处不将MC准备状态冒充PPO正式更新。

本次运行启动器保存在analysis/launcher.py（本地同名outputs/analysis内有副本），
中文头注释说明用途、数据流和边界，已通过compile语法检查；launcher.log及每阶段
launch/exit/verified JSON保留实际进度。启动器只执行既有统一入口和同目录恢复，
对源码版本、ESS、新增音乐失败、奖励立即改善均不设置门禁。

后续阶段日志为through10.log、formal.log；本次全部阶段都写同一运行目录，
前2轮和第10轮完整保存后自动恢复，累计上限保持200，不重置Adam或预算。


新奖励真实MC与系数移除验证
-------------------------

13:45完成192/192训练、32/32独立验证任务采集，13:46新MC预热通过并保存完整
checkpoint。总墙钟1397.18秒（含物理采集与价值拟合），选中mc_ratio=1纯MC监督；
独立验证2447条上层状态MSE4243.5067→140.5231，EV-0.0002967→0.9240160，
MSE相对下降96.6885%。配对运动reset子集EV0.919890、站立子集0.979962。
这些只评价Stage1下Critic价值初始化，不是Actor物理收益，不与旧奖励MC数值直接比较。

从preheat_val/rank07一个完整episode的无损控制侧车重算800控制步，跟踪分项
gate/activity_gate均为1，weighted_rate与score×weight误差低于1e-12。
其中旧活动度诊断比值最低0.22576398，该步跟踪奖励率仍为2.29494006，确认
实际PhysX采集后的消费路径也不再打折。actual_reward_gate_verified.json保存样本
路径和结果。未改变配对活动度完整性、参考/GMT时钟或其他权重。

已开始phases/eval_initial的32完整任务Stage1基线；此处仍未记录PPO收益。


新基线与正式PPO已执行
--------------------

13:56新Stage1基线32任务全部完成，status=passed、物理失败0，来源均衡episode
回报134.76824495，平均实际执行56.385秒；完整报告为本run/evaluation_baseline.json，
墙钟511.10秒。基线仍是Stage1，取消活动折扣及偏航终止后的数值不能和旧run直接
相减声称收益。两种协议的原始结果都保留。

14:02出现真实Actor Adam接受记录，14:06正式接受第1轮：14315有效链，
计划4次/实际接受3次、实际Adam8次；应用12288条链，BC实际1536样本。
第4小批次有限KL拒绝且下一步低于候选下限，正常结束本批剩余更新，保留前3步，
不是整轮异常或回滚。最终全部14315链KL均值0.05775855、最大1.54782996通过。
最后接受LR1.953125e-7、最后试探1.220703125e-7、名义LR恢复1e-6。

首小批次16层Transformer均有真实更新，各层changed_fraction最低0.99974385，
最小Adam增量RMS4.8631651e-8，更新传到最终参考。首轮整批PPO目标增益
-0.00980845，明确记录尚无收益证据，不将旧奖励下4096的正增益移植为本轮结论。
原预定流程继续第2轮，并不新增立即目标改善/个别新增失败门禁。

首轮墙钟555.51秒，其中采集236.76秒，Actor更新事务200.49秒，Critic10.31秒；
额外概率审计等计入整轮墙钟，分项存在包含关系不能盲目相加。峰值allocated
显存28282096640字节；前两轮不冒充所有200轮的内存稳定保证。

当前第2轮正在执行，首个训练后完整checkpoint计划在第2轮结束评估后发布。
latest暂仍指向iteration0的evaluated_initial，不能用它否认已接受的第1轮，
也不能把它冒充训练后模型。真实进度见metrics/<session>.jsonl和initial.log。


第2轮完成、配对物理结果与真实续训
--------------------------------

14:12第2轮正式接受：15804条链全部参与，计划/接受4/4，实际Adam5次、BC2048；
全链KL均值0.05976578、最大1.07106691通过。新rollout首个候选从1e-6试探，
实际接受LR依次5e-7、6.25e-7、7.8125e-7、9.765625e-7，离开恢复名义1e-6，
全部16层每次真实更新，确认临时步长未拖累下一批。整批PPO目标增益-0.00592847，
本轮墙钟392.42秒，峰值allocated28490522112字节。累计2轮/7次接受更新。

14:23完成训练后模型32任务评估、保存、正常退出0。原会话summary为passed，
启动器initial.verified.json核对配置、基线、实际16层更新与完整保存后自动恢复。

checkpoint：checkpoints/stage10_r000002_a000007_01792563802018114743_f89246b78882_evaluated_final_000002+controlled_end.pt

SHA256：5e7bf84bad78f80b359a2fa61840ae49b42572025f8d933d54f47bbdc428c988。

独立只读mmap核验完整checkpoint：Actor Adam所有已建状态step=7，18个参数组
均保留lr_multiplier，实际lr等于1e-6×倍率；未因换rollout或会话清空Adam。
通过既有加载检查后，新会话edd1c92d-ae51-45d5-9f47-d250a83f2def写出
resolved_config.yaml，8个worker502730～502737对应rank0～7，父torchrun502689。
同run已实际恢复到累计目标10，完成后启动器继续200，不重做MC或Stage1基线。

同协议物理结果：基线0/32失败→第2轮0/32，参考拒绝0。来源均衡episode回报
134.76824495→135.27276820（+0.3744%）；按每秒配对增益0.00440438，
95%区间[-0.00011091,0.00995773]跨0。活动度保留99.8976%。两轮没有证明持续收益，
不得用小幅回报变化宣称稳定改善；PPO目标仍负的事实也保留，训练按授权继续观察。

以下为相同32任务、来源均衡的八项RMS及根位置尾部指标，负相对变化表示误差降低：

* Tracking/root_position_rms_m: 0.36320115 → 0.34760217，-4.2949%。
* Tracking/root_orientation_rms_rad: 0.53626452 → 0.52812515，-1.5178%。
* Tracking/body_position_rms_m: 0.02635105 → 0.026280792，-0.2666%。
* Tracking/body_orientation_rms_rad: 0.13752492 → 0.13757153，+0.0339%。
* Tracking/body_lin_vel_rms_m_s: 0.16254515 → 0.16309394，+0.3376%。
* Tracking/body_ang_vel_rms_rad_s: 1.0560918 → 1.0569907，+0.0851%。
* Tracking/foot_height_rms_m: 0.010437941 → 0.010429589，-0.0800%。
* Tracking/leg_joint_position_rms_rad: 0.089291733 → 0.089496136，+0.2289%。
* Tracking/root_position_p95_m: 0.80720511 → 0.80697396，-0.0286%。

完整配对指标、checkpoint/Adam核验和续训进程证明保存在本地及服务器analysis的
paired_iteration2_summary.json、checkpoint2_optimizer_verified.json、formal_resume_verified.json。
临时源码/patch清理完成，正式模型和全部32任务证据保留。

当前查看继续到第10轮的训练：

::

  ssh 6000D-Server-1 'tail -n 30 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_4096_ungated_20261021_analysis/through10.log'

查看自动续训阶段：

::

  ssh 6000D-Server-1 'tail -n 20 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_4096_ungated_20261021_analysis/launcher.log'

第10轮结束后查看formal.log；TensorBoard始终为同一physical/tensorboard_compact，
没有为2/10/200拆成三个run。旧54轮实验、旧模型和56轮异常证据完全保留。
