# 2026-10-28：基于已完成物理对照的八组正式训练

本轮按用户“设计8组或4组并实际开启正式训练”的新指令执行。选择八组独立单GPU，
不引入双卡通信或其它策略算法；每组2048真实环境，冻结GENMO、SuperMimic Actor及
其normalizer，保持270维十帧参考Adapter和WM+BPTT。旧50次实验不覆盖、不停止。

## 当前结论与参数选择依据

14:20只读核验：旧修复实验有效更新45/45/30/29/20/20/20/50；GPU0～6仍运行，
GPU7正常完成且空闲。H8第40次整歌已完成，H16已完成第20次；H32根姿态开启组
正在第20次真实整歌评价，不能用其前缀冒充完整结果。

只依据validation确定本轮范围：H8无噪声20→40次的根RMSE改善由16.48%回落至10.35%，
H8噪声.02由9.46%回落至4.15%；H16第20次根RMSE改善14.13%/17.04%，关节误差
下降7.06%/7.01%。关闭根姿态的H32到50次validation根收益也由20次14.09%回落到2.72%。
这支持保留H8/H16，并加入更小的Adapter学习率；不宣称降低学习率已经解决根轨迹。
独立evaluation历史仍完整报告，不能以其结果反复调整在线候选或best选择。

现有证据说明小幅参考补偿有真实关节/姿态收益，但长期根轨迹和WM梯度可靠性仍有
边界。本轮是正式长预算比较，不预先宣称全面成功；best只由真实validation选择，
保留真实零Adapter，latest退化时不能用更新数覆盖best。

## 八组固定矩阵

|GPU|BPTT|Adapter LR|采集动作噪声std|根姿态修正|种子|目标有效Adam|
|---|---:|---:|---:|---|---:|---:|
|0|8|1e-5|0|开|42|1000|
|1|8|1e-5|0.02|开|42|1000|
|2|8|3e-5|0|开|42|1000|
|3|8|3e-5|0.02|开|42|1000|
|4|16|1e-5|0|开|42|1000|
|5|16|1e-5|0.02|开|42|1000|
|6|16|3e-5|0|开|42|1000|
|7|16|3e-5|0.02|开|42|1000|

配置位于`configs/closedloop/stage2_formal_20261028/gpu0.yaml`至`gpu7.yaml`。
各组从零Adapter和同一原预训练WM初始化，不恢复已经退化的旧Adapter，不共享在线Replay。
根XY关闭，根姿态开启；0.1大噪声占比为0。没有将根姿态的计算成本作为永久禁用理由。

共同训练：WM真实动作H8递归监督、batch2048、LR3e-5、每轮10次；Adapter batch1024、
microbatch32；gamma.99、grad clip1；每轮2/3个向量控制步，UTD.4；先64向量步采集，
100轮WM-only预热。Replay100万、50Hz控制、seed42，64首train及已有validation/evaluation
划分不变。统一原SM六项/action-rate、music1e-4，原精度和参考正则权重不变；软关节限位、
额外body_w Huber关闭。Critic终端和优化全部关闭，配置保留的Critic LR不代表实际更新。

投影沿用已验证预算：普通关节附加速度1rad/s、加速度10rad/s²，踝0.5/5；高度0.12m/s、
1.2m/s²；根旋转幅度.05rad，附加角速度.3rad/s、附加角加速度3rad/s²。仍有实际输出
精度证书和有限数值保护，不扩大运动约束。新增异常证据持久化在新运行中生效。

## 评价、保存与启动

0/5/10为256步固定短评价，20/50/100/200/400/700/1000为整歌真实评价；所有评价关闭
探索噪声。每50次保存完整checkpoint，初始、预热结束、首Adam和评价点额外保存。
保留真实零模型、最近两份完整模型和validation best，编号轻量模型及报告保留。
所有有效计数只在真实有限Adam完成时增加，排队/预热/WM更新不计Adapter更新。

本轮根目录：
`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/formal_h8_h16_20261028_1425`。
八组tmux为`smformal8_20261028_g0`～`g7`，输出为根目录下`gpu0`～`gpu7`。

```bash
cd /home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B tools/launch_stage2_alternating.py \
  --root /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/formal_h8_h16_20261028_1425 \
  --config-dir configs/closedloop/stage2_formal_20261028 \
  --session-prefix smformal8_20261028 --wait-for-free-gpus --poll-seconds 30
```

启动器冻结本次GENMO和SM运行工作树及八配置；无CUDA等待器逐卡检查实际compute PID，
在旧任务自然退出后启动相应实验。排队只占CPU，不占GPU，不发送停止信号。互斥锁
避免本启动器对同一GPU重复分配；GPU UUID变化或查询失败不会误判为空闲。
`queue_gpuN.json`明确区分等待PID、真实训练PID、训练中和结束/失败，`gpuN.exit`
记录训练退出码。不自动重启失败任务，也不覆盖已有日志。

首轮应先启动空闲GPU7，其余排队；这不等于已实际启动八个训练子进程。后续真实启动
状态以queue、训练status、compute PID和日志共同核验，并在记录文本追加实际结果。

## 本次修改与验证边界

只新增八配置、扩展已有启动器的显式配置目录/资源等待、补充CPU回归及文档。
不改奖励、WM监督、Adapter梯度、Actor或投影数值实现。资源等待6项CPU回归通过，
覆盖忙卡释放、查询失败、UUID变化、失败不重启、日志保留和完整八配置矩阵；测试无
GPU或真实训练。完整113项算法回归和物理修复证据见上一轮报告，不冒充本轮重新运行。
临时测试输出`/tmp/genmo_formal_queue_20261028`已精确删除并核对无残留。
