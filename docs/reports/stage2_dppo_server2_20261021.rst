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
逐轮指标train/metrics；固定评价在train/sessions/<session>/phases/eval_<label>；
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

正式启动与新奖励预热实测
------------------------

运行源码为GENMO 921b563afafd50dd6abfa057397684ef2a15e750、
GMT 9be857ce314fc6721eb71fe37fd7f415756e7824，均已推送并在服务器2快进同步。
服务器原Isaac环境补验GENMO 86 passed（6.62秒）、GMT 37 passed（0.90秒）；
本地首次更新额外保存与指纹回归55 passed（3.80秒）。这些批次存在交集，不相加。
正式入口八卡NCCL SUM=36检查通过，所有模块导入路径均为指定的旧DPPO分支工作树。

2026-10-21 22:33:38（北京时间）启动tmux stage2-dppo-server2-20261021；
监督器初始PID2168863，torchrun初始PID2171114。启动会话为
1befc54c-f832-485f-9599-78e2ce6251c9，目标累计200轮，不设置两小时截止。
PID仅为启动证据，实时运行应查询monitor/supervisor.json、GPU进程和新轮指标。

实际读取父模型和新initial完整checkpoint，逐张量核对Actor、非空Adam及LR控制器
完全一致，保留22轮/87次Actor更新；新Critic Adam为空、Critic更新数为0，
8rank的BC状态均为genmo.stage2.bc_disabled.v1。详细证据为运行根目录的
continuation_state_audit.json，不能把initial文件落盘视为新增Actor更新。

新奖励完整MC预热耗时1472.824秒，192个train任务全部有有效完整记录，包含8次
non_yaw_anchor_ori物理失败；32个独立Critic验证任务全部自然结束。上层决策真实控制步
分别333739和60841；失败样本没有被剔除。三组Critic混合候选实际比较后选择MC比例1.0，
验证MSE3843.748484降至159.921476、EV由-0.000315升至0.911718，原质量诊断通过。
该结果只证明价值拟合，不证明Actor或物理性能改善。预热checkpoint SHA为
ae439f2a7d3bdf4984c39c3bec5d8cfbc011152f613f9b59cb30a0ba2b10bfd7。

查看正式训练，无需重新启动::

  ssh -t 6000D-Server-2 tmux attach -t stage2-dppo-server2-20261021

只看主日志::

  ssh 6000D-Server-2 'tail -n 60 -F /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_dppo_return_20261021/monitor/formal.log'

日志中的accepted_iteration只在真实Actor更新接受后前进；预热、物理采集、被撤销的
Adam候选均不冒充新轮次。tmux使用Ctrl+b、再按d离开，不结束任务。

首次真实Actor更新与持续运行
--------------------------

新协议初始整曲基线32任务全部执行，28个音乐完成、4次物理失败，来源均衡任务回报
125.584386，按实际控制时间汇总的每秒回报2.250327；平均执行65.096秒，评估耗时
598.694秒。失败证据完整保存，任务继续。这一基线对应继承的第22轮模型，不与旧奖励、
旧评估音乐或AWR采样器的数值直接比较。

2026-10-21 23:19（北京时间）核实第23轮已经完成：14313条有效链，409407个真实
控制步、1638400个分配物理子步；轮内8次物理失败照常计入。4个Actor Adam步骤均首次
接受，无重试、无撤销，累计Actor更新87→91。全局batch为4096/4096/4096/2025，
实际BC样本、BC损失及BC梯度均为0，实际统一LR2.304e-7。四步PPO梯度范数分别约
0.320268/0.318609/0.334412/0.528122，梯度裁剪倍率均为1；不能误读BC关闭后的空
shared梯度分组为整个PPO梯度为零。

完整八rank Actor与Critic指纹全部一致；新Actor指纹与基线不同。最终KL均值
0.000138565、最大步均值0.000239241、P95为0.000307179，均通过原DPPO阈值。
完整首次更新checkpoint实际读取并核对91次更新、八rank禁用BC状态、Actor指纹及文件SHA；
SHA为432da7068bd997e970e5ac7e19fee0122551d83c11f1bd21ea8f2709fb4dd06a，文件为::

  train/checkpoints/stage10_r000023_a000091_01792595825879716712_89044b6664e9_continuation_first_update.pt

首轮墙钟368.033秒（6.13分钟）；物理采集连同目标/预算238.365秒、Critic10.186秒、
Actor116.964秒，Actor项已经包括候选KL，不能再次相加。最大PyTorch分配峰值28.685GiB，
不含PhysX外部分配。首次检查后的普通轮速仍需后续记录验证；按当前速度并计入每20轮约
10分钟整曲评估，正常8小时可新增约60～70轮，属于估算，不是已完成轮数。

随后已进入train_000024，监督器仍为attempt1。未执行用户停止，也没有修改正在运行
的源码/配置；后台持续到既定累计200轮。首次更新只证明实际学习步骤和存储正确，
收益是否改善等待后续同条件整曲评价，不能以KL或参数变化替代物理效果。

运行根formal_startup_verification.json保存完整首次更新证据；本地小型证据同步到
outputs/stage2_dppo_return_20261021/，与continuation_state_audit.json一起保留用于此次
正式任务追溯，不属于临时pytest产物。主模型、全量日志和物理证据只保留在服务器正式根。
