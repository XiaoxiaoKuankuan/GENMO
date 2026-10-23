"""仅供固定数据对照的 Stage1 等分布步骤扩展证明，不改写任何旧采样记录。

旧行为完整保存了19个随机转移。仅当学习状态的可训练Actor逐张量等于冻结Stage1，
且新旧DDIM核、噪声、条件、精度与冻结资产完全一致时，允许只读工具把可训练范围
由15..18扩到9..18：初始时两种路由实际调用相同权重，行为分布不变。随后仍须对
全部真实链重新核验19步旧概率，不能仅用此结构证明跳过数值检查。
本模块不更改chain、old_mean/std、log_prob、mask、优势或数据身份。授权仅挂在当前
临时policy对象上，生产build_policy、checkpoint恢复与物理采集均不自动启用。
已训练Actor、其他核变化或其他步数都拒绝；结果只用于反事实优化比较，不能当作
新奖励物理训练、同一checkpoint恢复或已完成部署标定。
"""
from __future__ import annotations
import copy
import torch


def bind_stage1_step_expansion(policy,behavior_kernel):
    if tuple(policy.trainable_steps)!=tuple(range(9,19)):
        raise ValueError('Only explicit Stage1 four-to-ten diagnostic expansion is supported')
    old=copy.deepcopy(behavior_kernel);new=copy.deepcopy(policy.kernel_config)
    for kernel,count in ((old,4),(new,10)):
        prefix=kernel.get('frozen_prefix',{})
        if (prefix.get('steps')!=19-count or prefix.get('gaussian_kl_steps')!=list(range(19-count,19))
                or prefix.get('schema')!=f'genmo.stage2.frozen{19-count}_train{count}_deterministic1.v1'):
            raise ValueError('Unexpected source/target stochastic step routing')
        for field in ('steps','gaussian_kl_steps','schema'):prefix.pop(field)
    if old!=new:raise ValueError('Step expansion must not change the actual Gaussian kernel or frozen prior')
    actor=policy.actor.state_dict();frozen=policy.frozen_policy.actor.state_dict()
    if actor.keys()!=frozen.keys() or any(not torch.equal(v,frozen[k]) for k,v in actor.items()):
        raise ValueError('Diagnostic expansion requires exactly the original Stage1 Actor, not a trained checkpoint')
    policy._diagnostic_behavior_kernel=copy.deepcopy(behavior_kernel)
    return dict(schema='stage2.stage1_step_expansion_diagnostic.v1',source_steps=list(range(15,19)),
        candidate_steps=list(range(9,19)),identical_stage1_parameters=True,unchanged_actual_sampling_kernel=True,
        requires_all_chains_all19_zero_update_check=True,behavior_data_rewritten=False,training_resume=False)
