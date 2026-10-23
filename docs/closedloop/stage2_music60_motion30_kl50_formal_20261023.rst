两秒音乐、一秒动作：仅KL接受的正式八卡运行
===========================================

用途与边界
----------

按2026-10-23用户要求恢复两秒音乐，动作仍为一秒，启动Server-2正式八卡训练。
本文件记录配置差异、运行身份和查看方法；启动不代表已经完成Actor更新或获得物理收益。
旧实验目录、模型和日志保留，训练使用独立配置快照。

配置与算法
----------

配置：``configs/closedloop/stage2_ddim10_last5_motion30_music60_klrelaxed_server2.yaml``。
相对上一版音乐30帧配置，只将音乐恢复60帧，并换回同核60音乐/30动作的实测部署profile。
音乐仍30Hz，不把两秒压成一秒；运动扩散变量保持30帧，额外音乐上下文不产生额外运动参考。

* 扩散10步：前4步冻结，中间5个随机步骤微调，末1步冻结确定性收尾。
* 最高候选学习率3e-6；KL相对原值放宽50倍，hard mean/P95/P99/max=10/30/45/100，soft=5。
* 首个数值有效且KL合格的候选直接接受；完整批次PPO目标仅记录，下降不作为拒绝条件。
* 完整链PPO，8192链Mini-batch，Epoch1；BC关闭，gamma0.95/lambda0.98。
* 8卡各128环境；8秒物理采集窗口，连续滚动，随机配对运动起点，20秒超时保留Value自举。
* 冻结GMT及现有奖励、物理终止保持；根位置/姿态reward std仍为0.3/0.4。
* 原训练音乐32任务初始评价及每20轮评价，按奖励挑选checkpoint。

真实单请求64样本标定中位0.135999秒，最慢0.141559秒；不重复声称短动作会将耗时减半。
27项短窗/真实Adam末尾检查/时钟回归通过；新配置解析、接受身份、部署profile一致性通过。

运行身份与查看
--------------

北京时间2026-10-23 18:34:34启动；预算200个外层训练轮次，真实Actor更新次数另行记录。
GENMO分支 ``feature/stage2-asymmetric-denoising``，启动提交 ``be3d30b``；
GMT分支 ``feature/stage2-tracking-history``，提交 ``03d9503``。
从原始s595000与新优化器开始，不带旧checkpoint初始化或resume参数，自动重新采集MC并校准Critic。
监督器启动PID1740276，torchrun PID1740340；后续PID以实际supervisor.json为准。

正式目录::

    /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_music60_motion30_kl50_formal_20261023_183434

监督与配置快照目录为上述路径追加 ``_supervision``；其中 ``launch.json`` 绑定代码、配置SHA和命令。
远程查看日志::

    ssh 6000D-Server-2
    tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_music60_motion30_kl50_formal_20261023_183434_supervision/formal.log

真实已发布轮次以正式目录 ``latest.json`` 为准，并同时核对Actor更新计数；
预热、初始评价和待提交优化步均不能当作已完成正式轮次。
监督器从同目录最新完整checkpoint恢复意外退出；不依赖终端会话存活。
运行目录内的checkpoint、评价及日志为正式资产，长期保留。

启动验收
--------

18:36:43核验八个rank均已真实生成和推进物理，MC校准采集账本累计8980个控制步、747次生成；
八卡分别占用约13.6～14.3GiB显存，监督器仍为attempt1，主日志无Traceback/RuntimeError。
初始checkpoint已实际加载检查：Actor与Critic两个Adam状态均为空，Actor LR3e-6，
外层轮次/Actor更新/Critic更新/策略版本全为0，Actor指纹与原Stage1一致。
实际运行identity确认60帧音乐/30帧动作、放宽KL、BC关闭及部署采样核SHA一致。
此时尚在MC预热采集，未完成Actor正式轮次；不能据此声称学习效果改善。
证据保存在正式目录 ``startup_verification.json``。
