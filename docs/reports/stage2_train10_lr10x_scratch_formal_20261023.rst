服务器2十倍学习率、十步去噪的Stage2从头正式训练
================================================

用户在2026-10-23明确要求“用10倍的配置重头开始训练，开启正式训练”。本次从原始
Stage1 s595000权重开始新的Stage2实验，重新创建Critic、Actor/Critic Adam和训练RNG，
外层轮次及更新计数从零开始；不继承第64轮Stage2权重、Adam或有限倍率试验候选。
这里的“从头”是重新训练Stage2，不是丢弃Stage1预训练并随机初始化整个生成器。

配置与范围
----------

使用现有 ``configs/closedloop/stage2_continuous_train10_lr10x_server2.yaml``，
配置文件SHA为 ``0ebad0c4a141068295b6d6521d5c3ff0baa6585e6c0f1c09733a9347d308e7bc``。
不修改算法、阈值或已有配置。本次明确启动至200个采集更新外层轮次，沿用此前
正式任务预算；不使用入口默认的2轮有限验证模式。

* 八卡，每卡128个环境；8秒真实世界采集窗口，连续滚动生成。
* 原始s595000初始化；9冻结+10可训练随机步+1冻结确定性末步，总计20步。
* 名义Actor LR和回溯上限为3e-6，完整链联合Clipped PPO，8192链Mini-batch，Epoch 1。
* 保留该配置已有的目标感知回溯和数值核验；实际接受LR、尝试次数、零更新采集轮
  必须如实记录。名义3e-6不表示每一步必定以3e-6接受。现有iteration会在正常零更新
  采集轮后推进，但actor_updates和policy_version不会增加；必须分别报告，不能仅凭
  iteration增加就宣称Actor执行了有效训练。
* 根位置/根姿态reward_std为0.30/0.40，Body位置/姿态为0.15/0.20，其余保持。
* BC关闭，gamma=0.95、lambda=0.98；随机配对运动起点，站立起点比例0，真实预热1秒，
  GENMO接管后最多20秒，超时保留重置前Value自举。
* 固定32任务来自训练音乐，每20个Actor轮次物理评价；奖励作为模型选择依据。
* 每5轮或1800秒保存checkpoint。MC校准重新采集，quality gate不作为强制停机条件。

正式启动先完成新行为的MC Critic预热与初始物理评价，再自动进入采集、Critic更新和
Actor更新。预热期间Actor轮次为0，不能把预热GPU占用或后台PID当作Actor已学会。
既有监督器处理意外进程退出；不因单次奖励下降、普通物理失败或活动度下降新增停机规则。

目录和命令
----------

正式根目录::

  /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_train10_lr10x_scratch_formal_20261023

首次启动只使用新目录，不带resume或initialize-from-checkpoint::

  cd /home/user/liwei/GENMO-bumi-closedloop
  /data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
    tools/run_stage2_training_supervisor.py \
    --config configs/closedloop/stage2_continuous_train10_lr10x_server2.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_train10_lr10x_scratch_formal_20261023/train \
    --status-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_train10_lr10x_scratch_formal_20261023/supervisor \
    --stop-after-iteration 200

实际以独立后台会话启动，启动PID、命令、代码SHA和配置SHA保存于根目录launch.json。
运行后的身份核验保存于startup_verification.json。旧正式运行和有限对照目录全部保留。

查看训练日志::

  tail -n 80 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_continuous_train10_lr10x_scratch_formal_20261023/supervisor/formal.log

已保存进度和后台状态分别位于train/latest.json、supervisor/supervisor.json；最新采集和
优化指标查看train/metrics下的当前session JSONL。曲线目录为train/tensorboard_compact。
latest是最近完整保存点，可能落后于实际迭代，不据此单独判断是否卡住。

如需显式停止，将SIGTERM发送给本目录launch.json中的监督器PID；监督器会转发给
当前训练子进程并停止自动重启。恢复只能恢复本新实验，且先确认原监督器和worker
已退出，再使用同一命令；监督器只在本目录存在latest时添加resume。禁止同时启动
两份监督器，禁止用旧第64轮checkpoint替换本实验恢复点。

启动证据边界
------------

启动前已核验Server-2两仓库分支和干净状态，八张RTX6000D均空闲，/data1约1.1TiB
可用。配置解析、200轮预算及首次命令无旧checkpoint/resume检查通过。入口会在创建
CUDA/Isaac世界前逐一核对Stage1、qpos统计、运动学和冻结GMT资产SHA。
实际进程、初始checkpoint空Adam、source_stage1身份及物理阶段进度由启动后现场
核验，不以本文命令或“配置已准备好”代替已启动证明。

启动后现场核验
--------------

2026-10-23北京时间13:58:53已启动，监督器PID 3101045，torchrun PID 3101046，
首次attempt=1；启动代码GENMO e4871b86、GMT 88b43daa。八个主worker以及对应GMT
校准世界已就绪，MC预热独立数据采集阶段已进入，现场检查未见Traceback或子进程退出。

14:01读取新初始checkpoint并核验SHA：
``c6f8b2f5a5accbfad45a299369492ee9f393686a090c626039a3992c8a7f9a12``。
actor_initialization为source_stage1/595000，Actor指纹与冻结原始Stage1指纹完全相同；
Actor与Critic的Adam state均为空，iteration/actor_updates/critic_updates/policy_version
均为0，优化器初始LR确为3e-6，可训练步为9～18。完整证据在同名JSON及正式根目录
startup_verification.json。此证据证明从头启动正确，不表示Actor已经完成更新或产生收益。

本次为持续后台正式运行。文档中的状态是上述时刻的快照，后续实际轮次、有效Actor
更新、接受LR和物理结果应读取运行目录的新日志，不能用本段静态文字替代实时检查。
