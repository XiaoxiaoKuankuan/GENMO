"""仅用于用户明确选择的Newton预览电机目标限幅，不改变训练或冻结SM参数。

读取实际StreamingPhysicalEnv已装配的default_q、xi偏置、alpha缩放及SM关节顺序，
把SM原始归一化动作转换为同一执行器的目标角，在权威机械上下限处饱和后反算动作。
反算后的动作仍经原step_physics缩放一次并驱动原Newton；previous_action记录实际
发送的动作。原始策略动作、请求目标和限幅目标由调用方分别保存，不能混称原策略效果。

本模块不修改生成参考、不裁剪仿真测得的状态、不重置接触或求解器、不改变SM/WM权重。
Newton软约束可能产生有限角度穿越，预览可仅记录而不因角度终止；跌倒/非有限/求解器
故障仍由原入口处理。此模式只说明加入执行限幅后的仿真表现，不作为裸模型安全证明。
"""

import torch


def clamp_motor_action(action, env, lower, upper):
    """使用实际执行器实例的同一缩放/偏置；不修改传入策略动作或环境状态。"""
    if action.ndim != 2 or action.shape[-1] != len(env.alpha):
        raise ValueError('预览动作必须与实际执行器的SM关节维度一致')
    base = env.default_q+env.xi[:, 1:1+action.shape[-1]]
    requested = base+env.alpha*action
    target = torch.maximum(lower, torch.minimum(upper, requested))
    executed = (target-base)/env.alpha
    saturated = (requested < lower) | (requested > upper)
    return executed, requested, target, saturated
