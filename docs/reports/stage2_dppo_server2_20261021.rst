2026-10-21 Server-2 原 DPPO 正式续训
========================================

用户明确要求离开AWR，回到feature/stage2-asymmetric-denoising与
feature/stage2-tracking-history，开启正式八卡训练。只在本地独立dppo工作树修改，
共享原功能分支和全部AWR数据/模型不动；服务器对应旧分支原目录均已核实空闲、干净。

继承与新实验边界
----------------

父任务stage2_stepkl_20261020/train停在22轮、87次Actor Adam更新，正常退出0。
父完整模型SHA为93b8f175af25cd52fb90d97d44a6efeaf996f619bfe6c0ca4ac01d4ccf40d346。
continuation.py显式核对SHA、Stage1/GMT资产、15+4+1采样核及参数结构，继承Actor、
非空Adam和原LR控制器（基础LR为2.304e-7），保留22轮/87更新计数。
由于奖励和终止协议变更，新目录重新采完整MC训练Critic，旧Critic/标签/BC状态与
旧物理中间态不进入新任务；新交互预算单独计数。原文件全部只读保留。
初始物理比较基线为“旧DPPO第22轮在新协议下的结果”，明确不是原始Stage1。

本次修改
--------

* 保留原Clipped DPPO：20步15+4+1，CFG2.5，eta0.1，4个可训练随机步，
  自由坐标平均代理和clip0.01，epoch1，统一AdamW及原逐步KL回溯/跨轮LR调度。
  不合并后续全链联合PPO、分组LR、AWR候选/回放/扩散监督代码。
* GMT权威source_std不变；Stage2 reward_std为anchor_pos0.20、anchor_ori0.30、
  body_pos0.15、body_ori0.20、body_lin_vel1.00、body_ang_vel3.14、
  foot_height0.07、leg_joint_pos0.25。逐项源函数和source_std继续核对。
* 新v3奖励不乘活动度；音乐奖励仍关闭。原v1配置和门控保留，仅供历史复算。
  标量、GPU数学、列式路径使用同一新尺度，控制积分与事件惩罚不变。
* BC完全关闭，不建立数据采样器、不执行监督前反向、不更新BC控制器。
  Actor Adam保留父任务历史动量；关闭BC指本次新优化步不再计算BC梯度。
* yaw_error_rad=null，仅取消单独偏航超限失败。根高误差0.2m、非yaw倾角0.6rad、
  末端相对高度误差0.15m仍终止；八项跟踪仍惩罚姿态误差，GMT权重/控制不动。
* 评价为当前train与重划分前original_train交集，固定16音乐×2seed，完整歌曲。
  只按来源均衡完整任务回报选模；失败、时长、活动度仍记录，不再限制晋级或停训。
* 移植已核验的GENMO/GMT批量CPU搬运循环引用释放，避免GPU张量跨轮滞留；
  并行评估的Critic诊断先按物理Episode排序再计算原GAE，防止交错环境假切断。
* MC每卡32环境收集原96音乐×2seed规模；质量阈值只报告，不作为Actor开训门禁。
  真实非有限值、错误奖励/资产和DDP不一致仍必须如实报错，不能伪造成功。

正式运行参数与操作
------------------

配置configs/closedloop/stage2_dppo_server2_formal.yaml。8卡×128环境=1024，
每轮共享世界8秒，0.5秒重规划，60帧30Hz；控制50Hz/PhysX200Hz。名义16384条链，
全局4096链一次更新，通常每轮4次Actor Adam；真实条数由实际有效执行决定。
LR调度沿用原5e-8至1e-6范围；Critic历史GRU1e-5、其他层1e-4，训练32epoch。
首次实际更新校验八rank Actor/Critic完整指纹并保存；之后每5轮保存、每20轮整曲
评估、每100轮完整MC诊断。累计上限200，即从22轮继续最多178轮。

正式根目录：
/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_dppo_return_20261021

服务器启动（仅一个监督器；本任务已明确授权执行）::

  cd /home/user/liwei/GENMO-bumi-closedloop
  /data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
    tools/run_stage2_training_supervisor.py \
    --config configs/closedloop/stage2_dppo_server2_formal.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_dppo_return_20261021/train \
    --status-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_dppo_return_20261021/monitor \
    --stop-after-iteration 200

日志monitor/formal.log；监督状态monitor/supervisor.json；完整快照train/latest.json；
逐轮指标train/metrics；固定评价在train/sessions/<session>/evaluations；
TensorBoard在train/tensorboard_compact。用户停止时向supervisor.json的pid发送SIGTERM，
监督器和当前进程组停止，不再自动拉起。正常使用相同命令会恢复latest完整状态。

监督器保留每次真实退出码，异常后30秒重启；若启动阶段没有完整checkpoint，
先取得原目录排他锁，将该次不完整启动目录原样归档，再重采MC。不会清锁或覆盖
旧证据，不会把只有监督器PID当成训练更新。

验证
----

本地169项GENMO回归通过、2项CUDA跳过（CPU限定环境）；涵盖真实小Actor四步PPO、
BC零调用/控制器不动、非空Adam继承、奖励三路径、评估/恢复/原配置兼容。
GMT36项通过、1项已安装Isaac管理器测试因本地无Isaac跳过，服务器补验。
早期发现并修复测试参数误含冻结参数、空评估身份、train_evaluation曲线scope，
以及继承Actor的新奖励基线需显式区别Stage1的问题。首次更新额外保存/指纹修改
另做针对性回归。正式运行和实测轮速在启动后追加，当前不宣称已改善物理收益。
