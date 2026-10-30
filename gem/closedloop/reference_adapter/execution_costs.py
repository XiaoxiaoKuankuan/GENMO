"""执行状态的轻量jerk成本与名义支撑窗口足端速度成本。

jerk由同一真实episode或同一想象轨迹的三个相邻关节速度计算，单位rad/s³；
真实复位后第一步没有足够历史，显式不计分。上一速度在BPTT内部保留计算图，
不向2654维Adapter输入增加字段，也不增加WM预测头。

足端项复用原SM奖励已经计算的实际状态FK速度。支撑权重仅来自冻结GENMO参考的
足端高度和速度：接近该歌曲名义低位、且名义水平速度小的窗口权重较大。损失
比较实际与名义足端水平速度，不强迫摆腿停住；权重不依赖可学习的执行状态，
避免通过抬脚逃避成本。它是名义支撑相位的可微代理，不能冒称求解器真实接触
脚滑；正式评价继续使用Newton真实接触。原SM六项、动作及参考均不改变。

本模块只返回未乘dt、未乘权重的成本。调用方统一乘一次0.02，固定归一化标尺
只允许来自训练集；所有向量化运算保留对执行速度及历史的真实梯度。
"""

import torch


def huber_mean(value):
    return torch.nn.functional.huber_loss(value, torch.zeros_like(value), reduction='none').flatten(1).mean(-1)


def execution_jerk(state, previous_state, previous_qd, valid, scale):
    value = (state[:, 34:]-2*previous_state[:, 34:]+previous_qd)/.02**2
    mask = valid.to(value)
    return huber_mean(value/scale)*mask, value.square().mean(-1).sqrt()*mask


def nominal_stance_weight(position, velocity, floor, *, height_band=.04, height_tau=.01, speed_scale=.2):
    height = position[..., 2]-floor
    near_floor = torch.sigmoid((height_band-height)/height_tau)
    slow = torch.exp(-velocity[..., :2].square().sum(-1)/speed_scale**2)
    # 只使用名义低位且近乎静止的高置信窗口；其它时段不当作支撑标签。
    return ((near_floor*slow-.8)/.2).clamp_min(0.)


def stance_foot_velocity(actual_velocity, nominal_position, nominal_velocity, floor, *, scale=.1,
                         height_band=.04, height_tau=.01, speed_scale=.2):
    gate = nominal_stance_weight(nominal_position, nominal_velocity, floor,
        height_band=height_band, height_tau=height_tau, speed_scale=speed_scale)
    error = (actual_velocity[..., :2]-nominal_velocity[..., :2])/scale
    component_cost = torch.nn.functional.huber_loss(error, torch.zeros_like(error), reduction='none')
    cost = (component_cost*gate[..., None]).mean((1, 2))
    speed = actual_velocity[..., :2].square().sum(-1).sqrt()
    return cost, dict(stance_weight=gate, stance_weight_mean=gate.mean(-1),
                     stance_foot_speed=(speed*gate).sum(-1)/gate.sum(-1).clamp_min(1e-8))
