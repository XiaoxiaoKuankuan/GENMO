服务器2连续滚动DPPO正式训练启动约定
==================================

本记录说明2026-10-22用户明确要求启动正式训练后的配置、进程管理、日志路径和
恢复边界。实际进程PID、启动时间、代码SHA与状态以服务器运行目录中的launch.json、
supervisor.json、训练metrics和latest.json为准；后台监督器存活不等于Actor轮次推进。

配置与训练流程
--------------

GENMO分支feature/stage2-asymmetric-denoising，配套GMT feature/stage2-tracking-history。
使用已通过有限八卡验收的configs/closedloop/stage2_continuous_rolling_server2.yaml，
生产实现对应GENMO17d2c16、GMT88b43da。本批只增加启动记录，不改变配置或算法。

从Stage1 s595000新建独立实验，累计最多200个接受的PPO外层迭代；不使用有限数学
诊断的临时Actor，不把旧固定2Hz训练checkpoint冒充滚动模式恢复。训练入口先自动
完成当前调度的MC Critic预热与初始32任务评价，再执行真实采集、Critic和Actor更新。
现有MC质量指标仍记录，enforce_quality_gate=false，不新增训练停止条件。

保持8GPU、每卡128环境、每轮8秒世界窗口、接管后20秒片段和真实1秒预热；连续
滚动使用60tick启动网格、单机器人实测延迟、动态前缀。完整链PPO8192链mini-batch、
epoch1、clip=.1、名义LR3e-7及原目标感知回溯、BC关闭、gamma=.95/lambda=.98不改。
每20轮进行固定物理评价，每5轮或1800秒保存checkpoint；原数值完整性检查保留。

正式目录与启动
--------------

正式根目录::

  /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_rolling_formal_20261022

其中train为训练/模型目录，supervisor为后台状态及完整控制台日志目录。此目录是
用户授权的正式实验，不属于临时测试清理范围。服务器使用原环境及现有监督器::

  cd /home/user/liwei/GENMO-bumi-closedloop
  /data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
    tools/run_stage2_training_supervisor.py \
    --config configs/closedloop/stage2_continuous_rolling_server2.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_rolling_formal_20261022/train \
    --status-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_rolling_formal_20261022/supervisor \
    --stop-after-iteration 200

启动器应以独立会话后台运行并将PID、实际SHA和完整参数写入launch.json；终端断开
不影响训练。意外退出时现有监督器等待30秒，从本实验latest恢复；不改Adam、LR或
概率/奖励定义，不删除锁文件。每次退出码都写formal_attempt_NNN.exit.json。

查看与停止
----------

查看控制台::

  tail -n 80 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_rolling_formal_20261022/supervisor/formal.log

查看监督器、已保存轮次和GPU::

  cat /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_rolling_formal_20261022/supervisor/supervisor.json
  cat /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_rolling_formal_20261022/train/latest.json
  nvidia-smi

latest是最近完整保存点，不一定等于已完成最新轮次；最新训练进度同时查看
train/metrics下当前session的JSONL。MC预热和初始评价期间iteration为0是正常阶段，
不能据此报告Actor已经训练。TensorBoard目录为train/tensorboard_compact。

需要停止时给launch.json记录的监督器PID发送SIGTERM，监督器会转发给当前训练器，
并停止自动拉起；不要只杀一个worker。以后只恢复同一滚动实验目录，禁止覆盖旧目录。

启动前只读检查确认：两仓库分支/HEAD正确且无未提交修改，八卡无计算进程，/data1
剩余约1.2TiB；实际资产SHA由训练入口在创建CUDA/Isaac世界前再次强制核对。
