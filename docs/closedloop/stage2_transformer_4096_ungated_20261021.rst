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
