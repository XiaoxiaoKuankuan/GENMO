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
检查及正式运行的实际启动、恢复结果写入根 ``记录文本.md``。历史验收图保持
原测量含义，不把参数调整或成功启动本身当作已经证明学习收益。
