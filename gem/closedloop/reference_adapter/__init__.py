"""十帧参考 Adapter 的独立训练路径。

冻结 GENMO 提供只读名义参考；本包将可观测反馈映射为十帧姿态修正，再调用原 SM
Actor。真实转移监督独立 WM，想象闭环只训练 Adapter 和新 Critic。导入不启动仿真、
训练或资产下载，不复用旧 GENMO Trainer，不改变共享 SuperMimic 源码。
"""
