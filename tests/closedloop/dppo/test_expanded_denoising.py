"""十步完整链新实验的配置和显式权重迁移验收。

检查三份独立配置只改变授权的根奖励、可训练扩散区间及学习率；用小型真实Adam
快照验证动量继承、学习率覆盖和旧物理/噪声身份拒绝。该测试不接触训练模型或服务器，
临时checkpoint只写pytest目录，不把通过数学及身份测试当作物理学习收益。
"""
import copy
from pathlib import Path

import pytest
import torch

from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.training_checkpoint import SCHEMA, initialize_expanded_policy
from gem.closedloop.dppo.rewards import ExecutionReward,REWARD_STD_VERSION,ROOT_REWARD_STD_VERSION
from tests.closedloop.test_stage1_actor import actor_factory

ROOT=Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('suffix,lr',[('base',3e-7),('lr10x',3e-6),('lr50x',1.5e-5)])
def test_ten_step_configuration_keeps_full_chain_physics(suffix,lr):
    config=configuration(ROOT/f'configs/closedloop/stage2_continuous_train10_{suffix}_server2.yaml',genmo_repo=ROOT)
    t=config['stage10']['training']
    assert t['frozen_prefix_steps']==9 and t['ppo_steps_per_chain']==10
    assert t['denoising_steps']==20 and t['ppo_step_strategy']=='intermediate10'
    assert t['actor_lr']==lr and t['trust_region']['maximum_lr']==lr
    assert t['actor_objective']=='full_chain_clipped_ppo' and t['objective_logprob_reduction']=='joint_sum'
    assert not t['bc_enabled'] and t['gamma_upper']==.95 and t['lambda_upper']==.98
    assert config['stage10']['scale']['actor_minibatch_chains']==8192
    reward=ExecutionReward(config['stage10']['reward']).config['tracking']
    assert reward['reward_std']['anchor_pos']==.3 and reward['reward_std']['anchor_ori']==.4
    assert reward['reward_std']['body_pos']==.15 and reward['reward_std']['body_ori']==.2
    assert reward['source_std']['body_pos']==.3 and reward['source_std']['body_ori']==.4


def fixture_payload():
    actor=torch.nn.Linear(3,2)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=3e-7,weight_decay=0.)
    actor(torch.ones(1,3)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    identity=dict(kernel=dict(noise=[.1],frozen_prefix=dict(schema='old',steps=15,
        gaussian_kl_steps=[15,16,17,18],content_sha256='same-frozen-stage1')),
        training=dict(actor_lr=3e-7,frozen_prefix_steps=15,ppo_step_strategy='intermediate4',
            ppo_steps_per_chain=4,trust_region=dict(maximum_lr=3e-7),gamma_upper=.95),
        runtime={},timing=dict(deployment_profile={'measured':'old'}),kl=dict(calibration={}),
        reward=dict(version=REWARD_STD_VERSION))
    payload=dict(schema=SCHEMA,identity=identity,actor=copy.deepcopy(actor.state_dict()),
        actor_optimizer=copy.deepcopy(optimizer.state_dict()),actor_fingerprint='old-actor',
        state=dict(iteration=64,actor_updates=253))
    target=copy.deepcopy(identity)
    target['kernel']['frozen_prefix'].update(schema='new',steps=9,gaussian_kl_steps=list(range(9,19)))
    target['training'].update(actor_lr=3e-6,frozen_prefix_steps=9,ppo_step_strategy='intermediate10',ppo_steps_per_chain=10)
    target['training']['trust_region']['maximum_lr']=3e-6
    target['reward']['version']=ROOT_REWARD_STD_VERSION
    return actor,optimizer,payload,target


def test_expansion_only_inherits_actor_adam_and_sets_new_lr(tmp_path):
    actor,optimizer,payload,target=fixture_payload()
    path=tmp_path/'source.pt';torch.save(payload,path)
    with torch.no_grad():actor.weight.add_(1.)
    optimizer.param_groups[0]['lr']=3e-6
    report=initialize_expanded_policy(path,identity=target,actor=actor,actor_optimizer=optimizer)
    torch.testing.assert_close(actor.weight,payload['actor']['weight'],atol=0,rtol=0)
    assert optimizer.param_groups[0]['lr']==3e-6
    assert all(float(v['step'])==1 for v in optimizer.state.values())
    assert report['source_iteration']==64 and not report['critic_and_adam_preserved']
    assert not report['old_rollout_reused'] and report['critic_recalibration_required']


@pytest.mark.parametrize('changed',['noise','gamma','source_std','repeat_transfer'])
def test_expansion_rejects_unrelated_changes_before_mutation(tmp_path,changed):
    actor,optimizer,payload,target=fixture_payload()
    if changed=='noise':target['kernel']['noise']=[.2]
    elif changed=='gamma':target['training']['gamma_upper']=.99
    elif changed=='source_std':target['reward']['tracking']={'source_std':{'anchor_pos':.2}}
    else:target['kernel']['frozen_prefix']['steps']=15
    path=tmp_path/'source.pt';torch.save(payload,path)
    before=copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError):initialize_expanded_policy(path,identity=target,actor=actor,actor_optimizer=optimizer)
    for name,value in actor.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)


@pytest.mark.parametrize('prefix',[9,15])
@pytest.mark.parametrize('use_trust',[False,True])
def test_complete_configured_chain_adam_update_with_tail_batch(actor_factory,tmp_path,prefix,use_trust):
    import torch.distributed as dist
    from tests.closedloop.dppo.test_clipped_dppo import case
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.continuation import DisabledBC
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.updater import actor_update
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    policy,rows,targets=case(actor_factory,frozen_prefix_steps=prefix)
    cache=RolloutTensorCache(rows,targets,'cpu')
    optimizer=torch.optim.AdamW([p for p in policy.actor.parameters() if p.requires_grad],lr=1e-9,weight_decay=0.)
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(len(rows))]
    trust=(dict(maximum_lr=1e-9,growth=1.25,margin=.8,max_attempts=3,
        minimum_trial_lr_ratio=.0625)
        if use_trust else None)
    try:
        result=actor_update(policy,optimizer,rows,targets,cache,manifest,BCGradientController(),
            distributed=DistributedCollectives(0,1,device='cpu'),anchor=DisabledBC(),
            generator=torch.Generator().manual_seed(4),minibatch_chains=3,microbatch=3,
            sampled_steps=19-prefix,step_strategy=f'intermediate{19-prefix}',
            bc_enabled=False,bc_samples_per_rollout=0,audit_full_rollout_objective=True,
            trust_region=trust,kl_limits=dict(chain_joint_mean=.2,chain_joint_p95=.6,
                chain_joint_p99=.9,chain_joint_max=2.))
        assert result['optimizer_steps']==2 and result['actual_global_bc_samples']==0
        assert result['internal_sample_visits']==5*(19-prefix)
        assert result['steps'][-1]['global_chain_count']==2
        assert result['steps'][0]['denoising_steps']==list(range(prefix,19))
        assert result['steps'][0]['candidate_checked_chains']==5
        assert result['steps'][0]['post_step_chain_scope']=='full_rollout'
        if use_trust:
            assert all(step['trust_region']['accepted'] for step in result['steps'])
        policy.assert_frozen_identity()
    finally:
        cache.close();dist.destroy_process_group()


def test_fixed_lr_accepts_lower_objective_when_kl_passes():
    from tools.validate_stage2_full_chain import candidate_update_checks
    config=configuration(ROOT/'configs/closedloop/stage2_continuous_train10_base_server2.yaml',genmo_repo=ROOT)
    kl=dict(chain_joint_kl=dict(mean=.05,p95=.08,p99=.1,max=.2),active_clip_fraction=.2,
        mean_coordinate_kl=0.,maximum_step_mean_coordinate_kl=0.,coordinate_kl={'p95':0.})
    report=dict(optimizer_steps=1,steps=[dict(optimizer_step=1,
        full_rollout_before={'objective_mean':0.},full_rollout_after=dict(kl,objective_mean=-.0795))])
    hard,eligible=candidate_update_checks(report,kl,config)
    assert hard['passed'] and eligible['passed']
    assert not eligible['failures']
    report['steps'][0]['full_rollout_after']['objective_mean']=.01
    assert candidate_update_checks(report,kl,config)[1]['passed']
