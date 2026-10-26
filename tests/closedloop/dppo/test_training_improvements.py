"""五项训练改进的数值、生命周期、恢复与抽样分布回归测试。

使用独立pytest临时清单和小型真实Actor/Adam，不读取正式模型、不创建物理世界。
检查质量报告不改变权重或触发停止、代理饱和保留实际更新、按音乐聚合统计，
以及八rank共享困难度统计、概率上限、覆盖序列、终止分类和checkpoint随机序列。
通过这些测试仅证明实现与恢复行为，不将短评估或代理损失当成真实物理收益。
"""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.adaptive_sampling import AdaptiveTimeWeights, adaptive_settings
from gem.closedloop.dppo.time_sampling import StratifiedTimePool
from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler, SOURCES
from gem.closedloop.dppo.learning_guards import record_quality_result, retain_quality_stop_history, quality_stop_blocks
from tests.closedloop.dppo.test_full_dataset import make_catalog_data
from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_robofft import cpu_threads, make_policy, make_rows, local_distributed, settings


def feedback(*, failed=False, timeout=False, reason=None, severity=0.):
    detail=dict(dt_s=.02, components=dict(track=dict(raw=dict(leg_joint_pos_squared_error=.04*severity))),
        diagnostics=dict(physical_state=dict(root_height_error_m=.1*severity,non_yaw_orientation_error_rad=.3*severity)))
    return NS(executed_control_steps=1, reason=reason, metadata=dict(reward_details=[detail],
        terminal_snapshot=dict(terminated=failed), physical_time_limit_reached=timeout))


def task(source='Mine', index=0):
    return dict(sampling=dict(time_bin=dict(source=source,bin_id=index)))


def pool(catalog, rank=0):
    return StratifiedTimePool(catalog,seed=42,rank=rank,world_size=8,minimum_remaining_seconds=4.,
        bin_seconds=1.,standing_start_probability=0.,adaptive=dict(enabled=True))


def test_quality_warning_continues_and_preserves_historical_stop(tmp_path):
    state=dict(iteration=10, quality_stop=dict(iteration=5,reasons=['old']))
    c=NS(output=tmp_path,stage=dict(evaluation=dict(quality_action='report_only')),state=state,
         distributed=local_distributed())
    (tmp_path/'quality_stop.json').write_text('{"historical":true}')
    retain_quality_stop_history(state,c.stage['evaluation'])
    result=dict(physical_guard=dict(passed=False,reasons=['one_song'],relative_reward_change=-.01))
    record_quality_result(c,result,'test')
    assert 'quality_stop' not in state and state['quality_stop_history'][0]['iteration']==5
    assert not quality_stop_blocks(tmp_path,c.stage['evaluation'])
    assert json.loads((tmp_path/'quality_stop.json').read_text())==dict(historical=True)
    assert json.loads((tmp_path/'quality_warning.json').read_text())['action']=='report_only'
    assert quality_stop_blocks(tmp_path,dict(quality_action='stop'))


def test_supervisor_report_mode_does_not_stop_on_old_marker(tmp_path,monkeypatch):
    from tools import run_stage2_training_supervisor as supervisor
    (tmp_path/'quality_stop.json').write_text('{}')
    (tmp_path/'latest.json').write_text('{"iteration":200}')
    config=tmp_path/'settings.yaml';config.write_text('stage10:\n  evaluation:\n    quality_action: report_only\n')
    monkeypatch.setattr('sys.argv',['supervisor','--config',str(config),'--output-dir',str(tmp_path),
        '--status-dir',str(tmp_path/'status')])
    assert supervisor.main()==0
    assert json.loads((tmp_path/'status/supervisor.json').read_text())['status']=='complete'


def test_saturation_retains_actual_actor_and_adam_update(actor_factory,monkeypatch):
    from gem.closedloop.robofft import updater
    from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer
    policy,context=make_policy(actor_factory)
    rows,targets,cache,manifest=make_rows(policy,context)
    opt=make_actor_optimizer(policy.actor,settings())
    updater.prepare_mc(policy,rows,cache,torch.Generator().manual_seed(19))
    before={n:p.detach().clone() for n,p in policy.actor.named_parameters() if p.requires_grad}
    policy.robofft['update_guard']=dict(max_ppo_zero_gradient_fraction=.45,
        max_joint_change_rms_rad=100.,max_near_joint_change_rms_rad=100.,max_normalized_change_rms=100.)
    original=updater.diagnostics
    def saturated(*args,**kwargs):
        result=original(*args,**kwargs)
        result['ppo_zero_gradient_fraction']=.8
        return result
    monkeypatch.setattr(updater,'diagnostics',saturated)
    result=updater.actor_update(policy,opt,rows,targets,cache,manifest,distributed=local_distributed(),
        generator=torch.Generator().manual_seed(33),ppo_epochs=3,minibatch_actions=2,microbatch=2)
    assert result['stop_reason']=='surrogate_saturation_finish_epoch_without_rollback'
    assert result['optimizer_steps']==result['attempted_optimizer_steps']>0
    assert result['epoch_guard'][0]['action']=='retain_and_finish' and opt.state
    assert any(not torch.equal(before[n],p) for n,p in policy.actor.named_parameters() if p.requires_grad)
    cache.close()


def test_outcome_semantics_and_warmup_do_not_invent_actor_failures():
    a=AdaptiveTimeWeights({'Mine':list(range(5))},dict(enabled=True))
    a.observe(task(index=0),feedback(reason='ppo_world_window_boundary'))
    a.observe(task(index=1),feedback(timeout=True,reason='physical_safety_limit'))
    a.observe(task(index=2),feedback(reason='music_end'))
    a.observe(task(index=3),feedback(failed=True,severity=1.))
    a.observe(task(index=4),warmup_failure=True)
    a.merge([copy.deepcopy(a.pending)])
    totals=a.summary()['totals']
    assert totals['physical_failures']==totals['timeouts']==totals['music_ends']==totals['warmup_failures']==1
    assert totals['transitions']==4 and 'Mine/4' not in a.levels
    assert a.probabilities('Mine')[3]>a.probabilities('Mine')[0]


def test_distributed_feedback_counts_once_and_all_probabilities_match():
    ranks=[AdaptiveTimeWeights({'Mine':list(range(20))},dict(enabled=True)) for _ in range(8)]
    for rank,a in enumerate(ranks):a.observe(task(index=rank),feedback(failed=rank==7,severity=rank/7))
    shards=[copy.deepcopy(a.pending) for a in ranks]
    for a in ranks:a.merge(shards)
    for a in ranks:
        np.testing.assert_array_equal(a.probabilities('Mine'),ranks[0].probabilities('Mine'))
        assert a.summary()['totals']['transitions']==8
        a.merge([{} for _ in range(8)])
        assert a.summary()['totals']['transitions']==8


def test_probability_cap_survives_normalization_and_music_aggregation():
    a=AdaptiveTimeWeights({'Mine':list(range(1000))},dict(enabled=True))
    a.levels={f'Mine/{i}':1. for i in range(20)}
    p=a.probabilities('Mine')
    assert p.sum()==pytest.approx(1.) and (p>0).all()
    assert p.max()<=3/1000
    assert p[:20].sum()<=3*20/1000
    mixed=.8/1000+.2*p
    assert mixed.max()<=1.4/1000


def test_adaptive_checkpoint_restores_feedback_and_exact_next_draws(tmp_path):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=600))
    original=pool(catalog)
    sampler=FullMusicSampler(catalog,seed=4,minimum_remaining_seconds=4.,time_pool=original)
    for _ in range(25):
        t=sampler.next_task();sampler.record_execution(t,1);sampler.record_outcome(t,feedback(severity=.8))
    original.sync_feedback(local_distributed())
    # 待合并增量也必须恢复，不能丢掉跨边界最后一段。
    t=sampler.next_task();sampler.record_execution(t,1);sampler.record_outcome(t,feedback(failed=True))
    saved=original.state_dict();lane=sampler.state_dict()
    expected=[original.draw('Mine') for _ in range(200)]
    restored=pool(catalog);restored.load_state_dict(saved)
    assert [restored.draw('Mine') for _ in range(200)]==expected
    restored.load_state_dict(saved)
    sampler2=FullMusicSampler(catalog,seed=4,minimum_remaining_seconds=4.,time_pool=restored)
    sampler2.load_state_dict(lane)
    assert restored.adaptive.pending==saved['adaptive']['pending']
    assert restored.adaptive_draws>0 and restored.summary()['adaptive']['updates']==1


def test_coverage_component_keeps_disjoint_sweeps_when_adaptive_enabled(tmp_path):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=600))
    all_bins=[]
    for rank in range(8):
        p=pool(catalog,rank);seen=[]
        while len(seen)<len(p.shards['Mine']):
            _,_,meta=p.draw('Mine')
            if meta['sampling_component']=='coverage_without_replacement':seen.append(meta['time_bin']['bin_id'])
        assert len(set(seen))==len(seen)
        all_bins.extend(seen)
    assert len(all_bins)==len(set(all_bins))==len(p.bins['Mine'])


def test_music_statistics_average_seeds_before_counting_wins():
    from gem.closedloop.dppo.periodic_monitor import paired_summary
    rows=[];tasks=[]
    for music in range(4):
        for seed in (42,1729):
            key=f'{music}-{seed}'
            tasks.append(dict(task_id=key,sample=dict(group_id=str(music))))
            rows.append(dict(task_id=key,dataset='Mine',seed=seed,reward=10.,executed_seconds=20.,physical_failure=False))
    base=dict(plan_sha256='same',plan=dict(tasks=tasks),episode_metrics=rows)
    current=copy.deepcopy(base)
    for i,row in enumerate(current['episode_metrics']):row['reward']+=1 if i<6 else -4
    result=paired_summary(base,current)['music_summary']
    assert result['count']==4 and result['improved']==3 and result['regressed']==1
    assert result['improvement_fraction']==.75 and result['reward_delta']['p50']==1


def test_critic_difficult_groups_distinguish_absolute_music_time_from_continuous_execution():
    from gem.closedloop.dppo.critic_diagnostics import subgroup_labels
    rows=[]
    for i in range(3):
        row=feedback(severity=2.,failed=i==2)
        row.identity=dict(backend_session_id='session',env_id=0,episode_id=1)
        row.control_tick_begin=600+i*300;row.control_tick_end=row.control_tick_begin+300
        row.terminated=i==2;row.truncated=i==1
        row.metadata.update(music_start_frame=3300,consumed_plan_ids=['old','new'])
        rows.append(row)
    flags=subgroup_labels(rows)
    assert 'observed_failure_before_5s' in flags[0] and 'reference_switch' in flags[0]
    assert 'rollout_boundary_1s' in flags[0] and 'large_leg_error' in flags[0]
    assert 'music_position_after_100s' in flags[0] and 'continuous_execution_after_100s' not in flags[0]


def test_formal_adaptive_warmstart_preserves_models_and_adam_and_refreshes_sampler(tmp_path):
    from gem.closedloop.robofft import OBJECTIVE
    from gem.closedloop.robofft.checkpoint import initialize_optimization_variant
    from gem.closedloop.dppo.training_checkpoint import schema_for_identity
    actor,critic=torch.nn.Linear(2,2),torch.nn.Linear(2,1)
    ao,co=torch.optim.Adam(actor.parameters()),torch.optim.Adam(critic.parameters())
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    identity=dict(critic=dict(type='asymmetric_tracking_committed_temporal',complete_mc=dict(enabled=False)),
        training=dict(actor_objective=OBJECTIVE,robofft=dict(mc_samples=16)),runtime={},dataset=dict(split='train'))
    payload=dict(schema=schema_for_identity(identity),identity=identity,
        actor=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),
        state=dict(iteration=10,actor_updates=62,mc_preheat_complete=True))
    path=tmp_path/'source.pt';torch.save(payload,path)
    target=copy.deepcopy(identity);target['dataset']['adaptive']=adaptive_settings(dict(enabled=True))
    report=initialize_optimization_variant(path,identity=target,actor=actor,critic=critic,
        actor_optimizer=ao,critic_optimizer=co,future_critic_formal=True)
    assert report['sampling_distribution_changed'] and report['critic_and_adam_preserved']
    assert not report['rollout_reused'] and not report['inherited_mc_preheat_for_unchanged_value_contract']
    assert next(iter(ao.state.values()))['step']==1 and next(iter(co.state.values()))['step']==1
    with pytest.raises(ValueError,match='dataset'):
        initialize_optimization_variant(path,identity=target,actor=actor,critic=critic,
            actor_optimizer=ao,critic_optimizer=co)


@pytest.mark.parametrize('bad',[dict(fraction=1.),dict(max_priority=4.),dict(ema_alpha=0.),dict(enabled=1)])
def test_adaptive_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):adaptive_settings(bad)
