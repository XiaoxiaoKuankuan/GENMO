第二阶段正式训练：按累计 PPO 迭代停止
====================================

本说明对应当前功能分支上的 iteration_horizon 配置和 checkpoint 身份。保留
Stage1 s595000 初始化 Actor、冻结 GMT140500、真实 Isaac 物理反馈、DPPO＋BC
以及独立 Critic。每次中断后从完整 checkpoint 恢复学习状态，重新创建 PhysX
世界和物理 episode，不恢复中途执行状态或未完成 rollout。

本轮参数与停止条件
------------------

正式配置是 ``configs/closedloop/stage10_server1.yaml``。Actor 学习率为 1e-7，
Critic 为 1e-4。实际 60 帧/2秒生成、每卡 64 环境、4 秒共享世界窗口、20 步
行为扩散、无放回 K8、eta=0.1、原 20 项噪声调度、PPO clip=0.003、平均坐标
KL 上限 0.002、完整 20 步最终 KL 和 BC 梯度目标比例 0.1 保持原设置。

``--stop-after-iteration`` 指累计已接受的 PPO 外层迭代上限，不是本次启动
额外获得的轮数。正式运行指定 500；预热和被拒绝的更新不计为接受轮数。
训练没有总墙钟期限。SIGINT/SIGTERM 仍在安全边界停止并保存完整 checkpoint；
资源用量上限、磁盘余量/容量检查、显存不足处理、单次采集超时、参考提交截止、
单机器人推理延迟合同和物理失败终止全部保留。耗尽资源或更新未通过验收时仍
受控停止，不以达到 500 轮为由绕过安全检查。

旧实现清理与终止语义
------------------

当前 WorldEnvironmentCollector 仅使用统一世界时间采样。删除无调用的
``_fragment_flow``、旧配额等待配置以及没有生产者的行政 drain/wait 汇总；实际
世界步数、有效转移数、非 PPO 控制步和物理预算仍由真实反馈记录。
旧 ``distributed_runtime`` 的学习协调器和 ``vector_runtime`` 已移除；当前
通信类集中在 ``collectives.py``，有效 Buffer 价值固定函数移到当前 ``runtime.py``。
GAE 默认返回原始优势，完整采集后才按所有 rank 有效样本做一次全局标准化。

GMT 向量后端的正常物理失败设置对应环境 ``done=True, terminated=True``，
不把它变成行政截断；实际四物理子步和失败步保留在回复中。站姿启动失败不产生
虚假上层动作，只让对应环境重新开始，其他环境继续。行政截断需要同一 episode
未经自动 reset 的可信终态和合法只读 preview 才能 bootstrap；真实终止价值为零。
执行缺步、计数矛盾、错误 episode 或非法参考仍拒绝，不能吞掉错误伪造学习数据。

恢复与 Critic 初始化
-------------------

新配置身份为 ``genmo.closedloop.stage10.iteration_horizon.v1``，checkpoint
schema 为 ``genmo.closedloop.stage10.iteration_horizon.checkpoint.v1``。
新身份拒绝旧总时限版本的等价恢复；不改写旧 checkpoint、源码 SHA 或资产身份。
新 checkpoint 保存 Actor、Critic、两套优化器、BC 控制器、各 rank BC 游标/RNG、
累计接受轮数、策略版本和资源计数。恢复使用同一运行目录与配置：

::

    bash scripts/train_stage10_8gpu_server1.sh \
      --config configs/closedloop/stage10_server1.yaml \
      --output-dir /绝对路径/本次独立正式运行 \
      --stop-after-iteration 500 --resume latest

首次启动去掉 ``--resume latest``，输出目录必须不存在。已有 MC Critic
初始化文件经过严格源码审查后，发现其物理源码两项与当前仓库不一致，不能
直接复用。本次新运行重新固定 Stage1 Actor 收集完整歌曲 MC，执行原有
MC/TD 配比独立验证和质量门槛，通过后才允许 Actor 更新；没有修改预热门槛。

验证边界
--------

CPU 回归覆盖配置拒绝、500轮上限、无局部默认标准化、真实双进程非空Adam恢复、
旧schema拒绝、64环境单个启动失败隔离及终止/截断自举。八卡通信与 checkpoint
检查及正式运行的实际启动、恢复结果写入根 ``记录文本.md``。旧验收报告可从
Git 历史追溯，不把参数调整或成功启动本身当作已经证明学习收益。

本次实际启动与恢复验收（2026-10-13）
----------------------------------

代码实现为 GENMO ``190790e``、GMT ``2bf9d7a``，都在原功能分支提交、推送并
同步服务器1。CPU DPPO整套878通过、13项GPU相关跳过，后续定向57项和20项通过；
GMT调度/参考44通过、2项运行环境依赖跳过。新身份八卡系统检查8/8通过，
11.18秒，覆盖空PPO分片、非空Adam、梯度归约、预算、不可变checkpoint及SHA。

旧MC成果因物理源码指纹不一致未复用。本次重新采集32条完整训练episode和
32条独立验证episode，无行政截断标签；训练真实失败3条、验证真实失败6条，
其余正常歌曲结束。验证MSE为7276.57，原有MC比例0/0.5/1候选分别得到
6521.25/2291.93/655.74；按独立验证选择1，解释方差0.687，原质量门槛通过。
采集与拟合共2243.44秒，选中Critic累计1024次更新，Actor保持Stage1权重。

首段完成第1轮后正常退出，随后同目录恢复并实际接受第2轮：

============================== =============== ===============
项目                           第1轮           恢复后第2轮
============================== =============== ===============
真实上层转移                   3072            3072
本轮Actor更新                  6               6
累计Actor更新                  6               12
累计Critic更新                 1120            1216
整轮墙钟秒                     163.54          137.99
最终全20步平均自由坐标KL        0.000791        0.001160
本轮BC样本                     512             512
============================== =============== ===============

两轮均通过最终全部20步检查，分别覆盖61440内部转移，优化期间PhysX暂停；
第2轮没有重新MC预热，累计交互继续增加。恢复建立新世界，世界窗口从0推进到
200控制步，原物理中间状态未伪装恢复。首段checkpoint实际包含非空Actor/ Critic
Adam、八rank BC游标、动态BC控制器及完整RNG，SHA独立核验通过，无wall_budget。

新的固定32任务初始回报34.587400，首轮后34.565889，均0/32物理失败。这是
10秒固定评估子集，不能代替长歌曲成功率；微小下降尚未证明Actor学习收益。
后续沿原交互量周期评估，不用权重变化或KL通过宣称任务效果已改善。

正式运行目录为：

::

    /data1/user/liwei/GENMO_outputs/closedloop_stage10/formal_h60_k8_lr1e7_500iter_20261013

续跑会话为 ``stage10_500iter_20261013``，实际使用 ``--resume latest`` 和
``--stop-after-iteration 500``，没有总墙钟期限或自动失败重试。运行仍受原资源、
KL和物理安全约束。控制台在同级 ``iteration_training_20261013/resume_segment.log``；
每轮指标位于正式目录 ``metrics/``、``curves.jsonl`` 和 ``tensorboard/``。
周期checkpoint仍按原30分钟间隔和安全停止边界保存，latest不等同于最新内存轮数。

便携核验证据保存在本地
``/home/weili/bumi-closedloop-worktrees/reports/iteration_training_20261013``，包含
八卡系统结果、初始与首轮完整checkpoint核验、MC数据/拟合和实际恢复记录。
正式模型、完整MC数据及日志留在服务器独立运行目录。

目录清理与当前依赖（2026-10-14）
------------------------------

按用户明确要求删除已过期的第二阶段开发工作树、旧训练及测试模型、日志、
rollout、重复代码快照、旧配置和报告。当前第二阶段仅保留本分支工作树、
``stage10_server1.yaml``、本说明和当前运行。旧方法由 Git 历史追溯，不新建
备份工作树或归档目录继续保留已撤下方案。

当前正式运行及同级 ``iteration_training_20261013`` 控制日志和验收记录保留。
旧 ``continuous_validation_20261013`` 只保留正式配置绑定的
``stageD_h60_deployment_calibration/deployment_calibration.json``；其路径和 SHA
属于恢复身份，不能随旧实验模型一起删除。带 ``v3`` 名称的 Python 环境和
s595000/GMT140500 资产仍为当前运行依赖，目录名称不会导致它们被判定为过期。

清理不修改当前运行绑定的 ``gem/``、GMT 物理源码、训练入口、启动脚本和正式
配置；源码 SHA、真实控制时钟、奖励及 checkpoint 身份保持不变。当前通用
概率、奖励、参考、归档核验及恢复测试仍保留；只撤下旧训练入口和旧容量规划
工具专属测试，避免把当前回归测试误当作生成产物删除。

逐路径删除清单及实际占用在本地当前证据目录的 ``cleanup_20261014.json``，
服务器删除记录在 ``iteration_training_20261013/cleanup_20261014.json``。
