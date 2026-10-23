"""RoboFFT-D 第二阶段动作级代理优化的独立算法身份。

本包只替换 Actor 学习目标及其数据记录；物理世界、真实时长 GAE、独立 Critic、
冻结 GMT 和连续滚动参考仍由 closedloop/dppo 中的中立基础设施负责。行为采样核
不变，优化身份与行为核分开绑定；旧 Gaussian PPO checkpoint 不能完整恢复到本算法。
"""

ALGORITHM = 'robofft_d.v1'
VERSION = 'genmo.closedloop.stage2.robofft_d.v1'
OBJECTIVE = 'robofft_d_action_clipped_ppo'
CHECKPOINT_SCHEMA = 'genmo.closedloop.stage2.robofft_d.checkpoint.v1'
