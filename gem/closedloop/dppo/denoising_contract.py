"""显式定义二十步扩散中的可训练随机区间，统一采样、PPO、KL和数据形状。

旧实验使用15个冻结随机步、4个可训练随机步和1个冻结确定性末步；新实验使用
9个冻结随机步、10个可训练随机步和相同末步。索引均为零起始，训练区间必须连续
并结束于18。该模块不更换噪声、不平均链概率、不改变梯度分母或物理调度；不同
区间必须通过真实策略kernel身份区分，旧四步rollout不能作为十步行为数据复用。
"""
from __future__ import annotations


def trainable_steps(frozen_prefix_steps):
    if type(frozen_prefix_steps) is not int or frozen_prefix_steps not in (9, 15):
        raise ValueError('Supported policies are frozen15/train4/final1 or frozen9/train10/final1')
    return tuple(range(frozen_prefix_steps, 19))


def validate_trainable_steps(steps):
    steps = tuple(steps)
    if steps not in (trainable_steps(15), trainable_steps(9)):
        raise ValueError('Trainable Gaussian steps must be exactly 15..18 or 9..18')
    return steps


def steps_for_count(count):
    if type(count) is not int or count not in (4, 10):
        raise ValueError('Complete PPO chains require exactly 4 or 10 trainable Gaussian steps')
    return trainable_steps(19-count)
