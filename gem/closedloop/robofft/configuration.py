"""RoboFFT-D 显式配置与恢复身份验证，不借用 Gaussian KL 门限。

生产配置仍经过原物理/时序/数据/资源检查；此处仅验证新 Actor 目标的 MC、动作级
比值、固定分组学习率和关闭的监督支路。参数是待标定候选，不作为最优或安全结论。
返回的绑定进入训练 checkpoint；源码 SHA 仍只是追溯，算法、权重形状和执行合同
才是恢复依据。本轮不支持启用 BC/best-policy anchor，避免隐式接回旧学习路径。
"""
from __future__ import annotations

import copy
import math
from . import OBJECTIVE, VERSION


def optimization_binding(stage):
    t = stage['training']
    return dict(schema=VERSION, objective=OBJECTIVE, candidate_not_calibrated=True,
        robofft=copy.deepcopy(t['robofft']), actor_lr=t['actor_lr'],
        actor_lr_multipliers=copy.deepcopy(t['actor_lr_multipliers']),
        ppo_epochs=t['ppo_epochs'], global_minibatch_actions=stage['scale']['actor_minibatch_actions'],
        action_microbatch=t['action_microbatch'], ppo_clip=t['ppo_clip'], gradient_clip=t['grad_clip_norm'],
        bc_enabled=False, anchor_enabled=False, true_gaussian_kl_control=False)


def validate_configuration(config, maximum_steps):
    s, t = config['stage10'], config['stage10']['training']
    r = t['robofft']
    if (s['version'] != VERSION or s['actor_optimization'] != {'schema': VERSION} or
            t['actor_objective'] != OBJECTIVE or t['actor_lr_control'] != 'fixed' or
            t['objective_logprob_reduction'] != 'mc_mean_logratio_action_clip' or
            t.get('bc_enabled') is not False or r.get('anchor_enabled') is not False or
            r.get('use_per_sample_ratio') is not False or r.get('delta_log_snr') is not True or
            r.get('sample_t') != 'uniform_sampling_indices_9_18' or
            r.get('mc_samples') != 16 or r.get('rng_device') != 'cpu' or
            t['frozen_prefix_steps'] != 9 or t['ppo_steps_per_chain'] != 10 or
            t['actor_trainable_scope'] != 'all' or maximum_steps < 1):
        raise ValueError('Explicit RoboFFT-D fixed-MC action PPO and frozen9/train10/final1 contract required')
    retired = {'trust_region', 'kl_rejection', 'kl_soft_stop_mean_coordinate', 'kl_microbatch',
               'max_active_clip_fraction', 'log_probability_reduction'}
    if 'kl' in s or retired & t.keys():
        raise ValueError('RoboFFT must not depend on Gaussian KL or backtracking configuration')
    for name in ('beta_neg', 'logprob_normalizer', 'logratio_alpha'):
        if type(r[name]) not in (int, float) or not math.isfinite(r[name]) or r[name] <= 0:
            raise ValueError('Invalid RoboFFT '+name)
    if r['beta_neg'] > 1 or type(r['mc_microbatch']) is not int or not 1 <= r['mc_microbatch'] <= 16:
        raise ValueError('Invalid negative-advantage discount or MC microbatch')
    stop = r.get('surrogate_kl_stop')
    if stop is not None and (not math.isfinite(stop) or stop <= 0):
        raise ValueError('surrogate_kl_stop must be null or positive')
    from ..dppo.actor_optimizer import validate_lr_multipliers
    validate_lr_multipliers(t)
    return optimization_binding(s)


class DisabledAnchor:
    """保持中立恢复协议的显式关闭状态；不加载监督数据、不提供训练 loss。"""
    def state_dict(self):
        return dict(schema='genmo.robofft.disabled_anchor.v1', enabled=False)

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError('Cannot restore an enabled BC anchor into RoboFFT')
