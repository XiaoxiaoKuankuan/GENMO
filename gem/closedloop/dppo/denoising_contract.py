"""定义完整扩散步数与可训练随机区间，统一采样、PPO、KL和数据形状。

保留20步的15+4+1和9+10+1实验；新增10步的4+5+1：索引4..8为最后五个
随机步骤，索引9为冻结Stage1确定性映射。末步没有高斯密度，不能假称为PPO步骤。
另支持10步5+5：索引5..9全部为可训练随机步，末步具有真实正方差并直接输出。
每个配置的真实时间步映射、噪声和冻结权重进入kernel身份；不同链不能混用或resume。
本模块不平均链概率、不修改物理奖励，也不使用帧裁切伪装扩散计算提速。
"""
from __future__ import annotations


SUPPORTED_INTERVALS = {(20, 15): tuple(range(15, 19)),
                       (20, 9): tuple(range(9, 19)),
                       (10, 4): tuple(range(4, 9)),
                       (10, 5): tuple(range(5, 10))}


def trainable_steps(frozen_prefix_steps, total_steps=20):
    if type(frozen_prefix_steps) is not int or type(total_steps) is not int:
        raise ValueError('Denoising step counts must be integers')
    try:
        return SUPPORTED_INTERVALS[total_steps, frozen_prefix_steps]
    except KeyError as error:
        raise ValueError('Supported schedules are 15+4+1, 9+10+1, 4+5+1, or terminal 5+5') from error


def validate_trainable_steps(steps):
    steps = tuple(steps)
    if steps not in SUPPORTED_INTERVALS.values():
        raise ValueError('Trainable Gaussian interval does not match a supported schedule')
    return steps


def steps_for_count(count):
    if type(count) is not int:
        raise ValueError('Trainable count must be an integer')
    matches = [steps for steps in SUPPORTED_INTERVALS.values() if len(steps) == count]
    if not matches:
        raise ValueError('Complete PPO chains require 4, 5, or 10 trainable Gaussian steps')
    return matches[0]


def step_strategy(steps, total_steps):
    """显式区分包含最终输出的末五步与冻结确定性收尾之前的中间区间。"""
    steps=validate_trainable_steps(steps)
    return ('terminal' if steps[-1]==total_steps-1 else 'intermediate')+str(len(steps))


def calibration_scope(count):
    steps_for_count(count)
    return {4: 'four', 5: 'five', 10: 'ten'}[count] + '_trainable_steps_only'
