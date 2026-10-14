"""GENMO音乐条件生成与冻结GMT在MuJoCo中的实时闭环。

本包只用于本机推理、交互和物理验证；复用Stage1条件、Stage2行为随机核以及
GMT参考提交合同，不加载Critic/优化器，不进入Isaac训练恢复流程。实际执行与
已有轨迹的运动学回放分别标识，所有产物集中保存于outputs/sim2sim。
"""
