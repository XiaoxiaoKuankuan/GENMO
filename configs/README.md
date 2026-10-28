# 配置入口

当前实验及必要依赖见 [closedloop/README.md](closedloop/README.md)。2026-10-28已移除被替代的闭环实验配置；旧实验由Git历史保存。

`closedloop` 以外的目录提供Hydra模型、数据集、网络、机器人资产、损失和调度组合，属于共享模块。`tests/closedloop/fixtures` 中少量固定参数只服务CPU回归，不能作为训练入口。
