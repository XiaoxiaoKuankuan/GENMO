"""根位置/姿态新奖励及单卡完整旧行为批次重放的回归测试。

输入为当前连续滚动配置和模拟的多rank真实数据结构；测试只重新分配计算所有权，
确保行为链对象、有效标记与已经全局归一化的优势不被复制改写或重新标准化。
同时核对10倍、50倍学习率可通过实际配置加载，组倍率、噪声核、GMT权重和其他
六项奖励保持原值。文件仅写入pytest临时目录，不启动正式训练或物理仿真。
"""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.rewards import ROOT_RELAXED_REWARD_VERSION,ROOT_RELAXED_REWARD_STD
from tools.compare_stage2_optimizer_batches import combine_rollout_shards
from tests.closedloop.test_stage1_actor import actor_factory


def shards():
    manifest=[dict(owner_rank=r,local_index=i,valid=i!=1,has_free=True)
              for r,n in enumerate((3,2)) for i in range(n)]
    return [dict(kernel={'same':True},global_manifest=manifest,transitions=[object() for _ in range(n)],
        targets=dict(advantages=torch.arange(n,dtype=torch.float64)+10*r,
                     valid=torch.tensor([i!=1 for i in range(n)]),advantages_normalized=True,
                     advantage_global_count=3,advantage_global_mean=1.2))
        for r,n in enumerate((3,2))]


def test_complete_single_device_reassignment_preserves_behavior_and_advantages():
    parts=shards();result=combine_rollout_shards(parts)
    assert result['transitions']==sum([p['transitions'] for p in parts],[])
    assert torch.equal(result['targets']['advantages'],torch.tensor([0.,1.,2.,10.,11.],dtype=torch.float64))
    assert result['targets']['advantage_global_count']==3
    assert [m['local_index'] for m in result['global_manifest']]==list(range(5))
    assert all(m['owner_rank']==0 for m in result['global_manifest'])
    assert result['targets']['valid'].tolist()==[True,False,True,True,False]
    assert parts[1]['global_manifest'][3]['owner_rank']==1


def test_incomplete_or_inconsistent_shards_are_rejected():
    parts=shards()
    with pytest.raises(ValueError,match='All original ranks'):combine_rollout_shards(parts[:1])
    parts[1]['targets']['advantage_global_mean']=1.3
    with pytest.raises(ValueError,match='normalization'):combine_rollout_shards(parts)


@pytest.mark.parametrize('factor',[10,50])
def test_requested_lr_and_reward_configuration_preserve_other_contracts(tmp_path,factor):
    path=Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_continuous_rolling.yaml'
    original=yaml.safe_load(path.read_text());changed=copy.deepcopy(original)
    t=changed['stage10']['training'];t['actor_lr']=1e-6*factor;t['trust_region']['maximum_lr']=t['actor_lr']
    target=tmp_path/'candidate.yaml';target.write_text(yaml.safe_dump(changed))
    c=configuration(target);stage=c['stage10']
    assert stage['reward']['version']==ROOT_RELAXED_REWARD_VERSION
    assert c['stage9']['reward']['tracking']['reward_std']==ROOT_RELAXED_REWARD_STD
    assert c['stage9']['actor_lr']==pytest.approx(1e-6*factor)
    for field in ('actor_lr_multipliers','std_schedule','frozen_prefix_steps','ppo_steps_per_chain','bc_enabled'):
        assert c['stage9'][field]==original['stage10']['training'][field]
    assert stage['reward']['tracking']['weights']==original['stage10']['reward']['tracking']['weights']


def test_bounded_condition_encoding_preserves_all_probabilities_and_global_gradient(actor_factory):
    from tests.closedloop.dppo.test_clipped_dppo import case
    from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.execution_checks import policy_phase
    from gem.closedloop.dppo.optimization_support import probability_check_local
    policy,rows,targets=case(actor_factory)
    for row in rows:row.transition_valid=True
    cache=RolloutTensorCache(rows,targets,'cpu')
    try:
        with policy_phase(policy):
            before=evaluate_chain_records(policy,rows,cache,microbatch=3)
            a=backward_chain_surrogate(policy,rows,cache,before,clip=.01,global_chain_count=11,microbatch=3)
            grads={n:p.grad.clone() for n,p in policy.actor.named_parameters() if p.grad is not None}
            policy.actor.zero_grad(set_to_none=True);policy.condition_cache_max_chains=2
            after=evaluate_chain_records(policy,rows,cache,microbatch=3)
            b=backward_chain_surrogate(policy,rows,cache,after,clip=.01,global_chain_count=11,microbatch=3)
        # FP32条件编码批形变化会在高维joint log-ratio中累积；沿用生产1e-4重算容差。
        torch.testing.assert_close(before,after,rtol=1e-6,atol=1e-4)
        torch.testing.assert_close(a['loss'],b['loss'],rtol=1e-6,atol=1e-8)
        for name,p in policy.actor.named_parameters():
            if name in grads:torch.testing.assert_close(p.grad,grads[name],rtol=1e-5,atol=1e-7)
        audit=probability_check_local(policy,rows,denoising_microbatch=3,tensor_cache=cache)
        assert audit['passed'] and audit['condition_cache_chunks']==3 and audit['checked_chains']==5
        assert audit['checked_steps']==list(range(19))
    finally:cache.close()
