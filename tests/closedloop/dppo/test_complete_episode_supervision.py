"""完整物理 MC 回报与片段/行为边界的独立 CPU 验收。

逐50Hz控制步手算gamma时间折扣，验证积分奖励没有二次乘dt、失败事件没有重复
积分，以及普通PPO截断可在固定行为下形成MC但行政截断和跨Actor版本会被拒绝。
测试读取当前配置，确保物理安全上限、采集秒数与生成长度各自独立。
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
from gem.closedloop.dppo.complete_episode import complete_mc_returns, CompleteEpisodeBank
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.returns import compute_gae


def test_mc_training_selection_uses_distinct_music_and_audited_longest_sample(tmp_path):
    import json,hashlib
    from gem.closedloop.dppo.mc_calibration import complete_mc_plan
    rows=[]
    for name,group,frames in [('short','song1',30),('long','song1',90),('second','song2',60)]:
        rows.append(dict(sample_id=name,num_frames=frames,split='train',fps=30,
            resplit_provenance=dict(group_id=group),source_audio_sha256=hashlib.sha256(group.encode()).hexdigest()))
    manifest=tmp_path/'Mine/manifests/train.jsonl';manifest.parent.mkdir(parents=True)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    samples=[dict(dataset='Mine',split='train',group_id=r['resplit_provenance']['group_id'],row=r) for r in rows]
    catalog=SimpleNamespace(data_root=tmp_path,identity={'test':'catalog'},samples={'train':{'Mine':samples}})
    plan=complete_mc_plan(catalog,'train',samples_per_source=2,seeds=[42,1729],selection_seed=461)
    assert plan['selected_unique_group_count']==plan['selected_sample_count']==2
    assert plan['selected_unique_audio_count']==2 and plan['task_count']==4
    assert {t['sample_id'] for t in plan['tasks']}=={'long','second'}
    assert plan['critic_mc_supervision'] and plan['episode_seconds'] is None
    assert len(catalog.samples['train']['Mine'])==3
    samples[1]['row']=dict(samples[1]['row'],num_frames=91)
    with pytest.raises(ValueError,match='audited training catalog'):
        complete_mc_plan(catalog,'train',samples_per_source=2,seeds=[42],selection_seed=461)


def row(begin, rewards, *, terminal=False, truncated=False, version=0, event=0.):
    return SimpleNamespace(identity=dict(backend_session_id='b',episode_id='e',policy_version=version),
        context={'causal':torch.tensor([[begin]])},next_context=None,transition_valid=True,
        rewards=torch.tensor(rewards,dtype=torch.float64),executed_control_steps=len(rewards),
        control_tick_begin=begin,control_tick_end=begin+12*len(rewards),terminated=terminal,truncated=truncated,
        reason='music_end' if terminal else 'collector_fragment_boundary' if truncated else None,
        metadata=dict(event_reward=event,remaining_music_seconds=10.,next_remaining_music_seconds=9.,
            critic_context={'causal':torch.tensor([[begin]])},next_critic_context=None,
            physical_episode_administrative_truncation=False,sampler_trace=dict(kernel_config={'kernel':'fixed'})))


def test_complete_mc_matches_low_time_discount_and_gae_lambda_one():
    a=row(600,[.04]*25);b=row(900,[.03]*10,terminal=True)
    b.rewards[-1]-=5.
    gamma=.99**(1/25)
    expected=(torch.cat((a.rewards,b.rewards))*gamma**torch.arange(35,dtype=torch.float64)).sum()
    result=complete_mc_returns([a,b])
    assert result[0]==pytest.approx(float(expected))
    targets=compute_gae([a.rewards,b.rewards],[7.,19.],[19.,0.],[25,10],[True,False],[True,False],
        gamma_upper=.99,lambda_upper=1.,normalize=False)
    torch.testing.assert_close(result,targets['returns'],rtol=1e-12,atol=1e-12)


def test_zero_control_failure_is_a_real_event_without_fake_time():
    r=row(600,[],terminal=True,event=-5.)
    assert complete_mc_returns([r]).tolist()==[-5.]


@pytest.mark.parametrize('damage',['administrative','gap','actor_version','incomplete'])
def test_mc_rejects_invalid_complete_episode_contract(damage):
    a=row(600,[.02]*25);b=row(900,[.02]*25,terminal=True)
    if damage=='administrative':b.metadata['physical_episode_administrative_truncation']=True
    if damage=='gap':b.control_tick_begin+=12;b.control_tick_end+=12
    if damage=='actor_version':b.identity['policy_version']=1
    if damage=='incomplete':b.terminated=False
    with pytest.raises(ValueError):complete_mc_returns([a,b])


def test_bank_spans_ppo_fragments_but_old_actor_labels_are_invalidated():
    bank=CompleteEpisodeBank(policy_version=0,kernel_identity={'kernel':'fixed'})
    a=row(600,[.02]*25,truncated=True);b=row(900,[.02]*25,terminal=True)
    task=dict(task_id='t');bank.append(task,a);bank.append(task,b)
    samples,targets=bank.training_data(policy_version=0,kernel={'kernel':'fixed'})
    assert len(samples)==len(targets)==2 and bank.report()['real_control_steps']==50
    assert set(samples[0].context)=={'causal'}
    with pytest.raises(ValueError,match='different fixed Actor'):
        bank.training_data(policy_version=1,kernel={'kernel':'fixed'})


def test_administrative_physical_episode_is_excluded_not_bootstrapped_as_mc():
    bank=CompleteEpisodeBank(policy_version=0,kernel_identity={'kernel':'fixed'})
    r=row(600,[.02]*25,truncated=True);r.reason='physical_safety_limit'
    r.metadata['physical_episode_administrative_truncation']=True
    bank.append(dict(task_id='t'),r)
    assert bank.report()['complete_states']==0 and len(bank.report()['excluded'])==1


def test_current_configuration_separates_three_time_concepts():
    root=Path(__file__).resolve().parents[3]
    c=configuration(root/'configs/closedloop/stage2_full_chain.yaml')
    assert 'episode_seconds' not in c['stage10']['training']
    assert c['stage10']['physical_episode']['safety_limit_seconds'] is None
    assert c['stage10']['ppo_rollout']['world_seconds']==8.
    assert c['stage10']['generation_horizon']==dict(frames=60,fps=30)
