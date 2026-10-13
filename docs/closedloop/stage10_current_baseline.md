# 第二阶段当前开发基线

2026-10-13 按用户要求，GENMO 统一在 `feature/bumi-music-closedloop` 开发，
GMT 统一在原 `feature/bumi-frozen-gmt-backend` 开发。原重构分支代码已完整合回，
后续在这两个原功能分支直接修改，不自动另建分支或开发工作树。本文中的“当前基线”
表示目前继续优化的起点，并不表示强化学习收益已经达到最终验收目标。

## 当前入口

服务器 1 使用原功能工作树 `/home/user/liwei/GENMO-bumi-closedloop` 和
`/home/user/liwei/legged_lab_gmt`。训练入口只有
`tools/train_closedloop_stage10.py`，八卡环境由
`scripts/train_stage10_8gpu_server1.sh` 设置。主配置为
`configs/closedloop/stage10_server1.yaml`。

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
bash scripts/train_stage10_8gpu_server1.sh \
  --config configs/closedloop/stage10_server1.yaml \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/自行指定的新运行目录 \
  --stop-after-iteration 2
```

该命令示范有限运行，不会因配置里存在 500 轮预算而自动启动长期训练。
每个实验必须给出未使用的输出目录。改变代码或算法身份后开始新实验；当前加载器只
接受当前完整 checkpoint schema 和一致训练身份，不自动转换旧运行。相同当前运行
发生异常时，可以 `--resume latest` 从最近完整边界恢复并重新建立物理世界。

`STAGE10_PYTHON` 可以明确指定已经验证的统一 Python。现有运行环境和模型资产目录
名称仍包含历史版本字样，保留这些实际路径是为了继续使用已核验的同一批文件。
物理时钟及列式反馈协议里的版本号也继续表示其数据合同，不表示有第二套训练方法。

## 当前实现

- `training.py`、`runtime.py`：固定行为采样、Critic 预热、DPPO＋BC、完整 KL 验收、
  回滚和 checkpoint。每张 GPU 一个向量化 Isaac 世界，优化时暂停物理推进。
- `dense_policy.py`、`gaussian_probability.py`、`updater.py`：行为完整 20 步，默认每条
  链每个 epoch 无放回抽 8 步计算 PPO 梯度；有效自由坐标平均 log-ratio 是代理目标，
  原始 joint probability 与全部 20 步 joint KL 独立保留。
- `pretrained_critic.py`：从 s595000 复制独立冻结条件编码器，256 维、两层、四头的
  时序价值头参与学习；Actor 和 Critic 不共享更新参数。
- `gradient_merge.py`、`retention_anchor.py`：PPO 与原 Stage1 BC 梯度本地累计、统一
  跨卡同步和裁剪；BC 继承 s595000 的损失成熟度，权重按有界平滑梯度比例标定。
- `memory_execution.py`、`training_checkpoint.py`：普通训练链保留至本轮更新结束，
  诊断或异常才保存完整证据；完整 checkpoint 使用一致 CPU 快照和单在途异步写盘。
- `execution_state.py`、`evaluation_runtime.py`、`optimization_support.py`：从旧训练器
  抽出仍在使用的执行计数、独立物理评估和概率检查，当前主循环不再导入旧训练入口。

旧 Stage10 单卡/并行训练入口、旧 PPO 更新器、旧配置及仅适用于它们的启动和性能
脚本已经撤下。必要的公共测试已迁移，旧方法专属测试随方法退休。Stage1 功能、
当前使用的 Stage9 数据/奖励/概率基础组件以及历史报告读取工具按其用途保留。
历史报告中的旧文件名、命令与配置需结合对应历史提交阅读，不能当作当前启动说明。

## 当前参数与证据边界

主配置为八卡、每卡 64 环境、每环境 8 次连续上层决策，共 4096 条真实链；PPO 两个
epoch、每个 optimizer minibatch 1024 条完整链，最多 8 次 Actor 更新。
Actor 学习率 `5e-8`、clip `0.003`，计算微批 64、完整 KL 微批 256；采用已通过概率
一致性验证的 TF32x3 路径，主参数保持 FP32。Critic 每批 32 epoch，默认先采 8 轮
独立真实轨迹预热。长 episode、完整回报、双种子及噪声配置是同一实现的诊断条件。

此前相同真实数据的 Actor 计算对照中，K8 约 104 秒，Full20 约 246 秒，计算速度约
提升 2.37 倍。K8 只减少梯度使用的去噪步，实际生成始终完成 20 步。严格 joint K8
改变的是 PPO 概率归约方式，用于算法对照，不是另一条开发分支。
现有物理结果尚未证明稳定的正向强化学习收益；长轨迹价值预测仍存在偏差。
这些结论没有因合并分支、修改入口或命名而改变。原始结果见
[2026-10-12 验收报告](stage10_v3_acceptance_20261012.md)。

本轮只收敛代码与开发流程，验证当前入口和依赖。具体测试、提交和同步结果按完成时间
追加到根目录 `记录文本.md`。正式训练 checkpoint、日志、数据和历史验收产物不覆盖。


## 本轮迁移验证（2026-10-13）

当前 CPU 回归覆盖 870 个通过用例，13 项在 CPU 环境跳过；首轮的一处旧测试引用已
迁移到当前高斯统计并通过相关 14 项定向复测。五组真实八卡检查全部通过，覆盖概率、
梯度/Adam 恢复、Critic、批量重算与冻结 GMT。最大零更新概率误差为 `1.82e-12`。

统一入口实际启动八个 Isaac 世界，完成 64 次单请求测量并正常关闭；最大请求耗时
`0.254s`，低于沿用的 `0.30s` 前缀预算。冻结 GMT、执行 ACK、400 个控制步与
1600 个物理子步均通过边界检查。本轮没有更新 Actor，不新增强化学习效果结论。
原用户修改保留，独立测试输出与缓存已清理；详细证据位于本轮仓库外验收目录，路径
和原始失败/修复过程见 `记录文本.md`。
