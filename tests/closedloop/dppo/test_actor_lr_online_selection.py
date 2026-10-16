"""验证高学习率有限在线实验的选择边界，不启动训练或伪造物理收益。

以合成的32任务总量检查：每秒分数提高不能掩盖早失败；活动度、参考/实际硬限位
都参与经验筛选；原KL候选优先，探索代表必要时补足另一目标且实际步长大于低LR
基线，总数仍最多四组；已有扫描参数与
物理任务身份必须保持一致。测试结果只证明编排规则，不能替代真实物理验收。
"""
import copy
import json
import pytest
from tools.eval.run_stage2_actor_lr_online import assess,select_candidates,verify_online,sha


def analysis():
    limits={name:dict(hard=dict(max_excess_rad=.01,violation_fraction=.0001))
            for name in ('actual','consumed_reference','generated')}
    return dict(plan_sha256='same32',aggregate=dict(episode_count=32,valid_episode_count=32,
        infrastructure_failure_count=0,administrative_truncation_count=0,
        music_completion_count=29,reward_sum=100.,executed_seconds=100.,
        activity=dict(actual_activity_rad_s=dict(mean=1.)),joint_limits=limits))


def fixtures():
    cases=[];values={'stage1':analysis()}
    for objective in ('coordinate_dppo','joint_score_pg'):
        for rate in (5e-8,3e-7,1e-6,3e-6,1e-5):
            for scale in (1,3,10):
                name=f'{objective}_lr{rate:g}_kl{scale}'
                cases.append(dict(name=name,screening={'passed':True},objective=objective,actor_lr=rate,
                    kl_multiplier=scale,path=name+'.yaml',sha256=name,
                    backtracking=dict(accepted_lr=rate,attempt_count=1)))
                values[name]=analysis()
    return dict(candidates=cases),values


def test_early_failure_activity_and_three_limits_are_not_hidden_by_rate():
    baseline=analysis();candidate=analysis()
    candidate['aggregate'].update(reward_sum=120.,executed_seconds=75.,music_completion_count=28)
    result=assess(baseline,candidate)
    assert not result['eligible']
    assert set(result['failures'])=={'fewer_complete_tasks','execution_seconds_regression'}
    for domain in ('actual','consumed_reference','generated'):
        candidate=analysis();candidate['aggregate']['joint_limits'][domain]['hard']['max_excess_rad']=.2
        assert not assess(baseline,candidate)['eligible']
    candidate=analysis();candidate['aggregate']['activity']['actual_activity_rad_s']['mean']=.8
    assert 'actual_activity_regression' in assess(baseline,candidate)['failures']
    candidate=analysis();candidate['plan_sha256']='other_song'
    with pytest.raises(ValueError,match='unpaired'):assess(baseline,candidate)


def test_original_gate_primaries_and_at_most_one_explicit_exploration():
    scan,values=fixtures();plan=select_candidates(scan,values)
    assert [r['name'] for r in plan['selected']]==[
        'coordinate_dppo_lr5e-08_kl1','coordinate_dppo_lr1e-06_kl1','joint_score_pg_lr1e-06_kl1']
    values['coordinate_dppo_lr1e-06_kl1']['aggregate']['music_completion_count']=27
    values['coordinate_dppo_lr3e-07_kl1']['aggregate']['reward_sum']=105.
    for name in ('coordinate_dppo_lr1e-06_kl3','joint_score_pg_lr3e-06_kl10'):
        values[name]['aggregate']['reward_sum']=103.
    plan=select_candidates(scan,values)
    assert len(plan['selected'])==4 and len(plan['assessments'])==30
    assert plan['selected'][1]['name']=='coordinate_dppo_lr3e-07_kl1'
    assert sum(r['kl_multiplier']>1 for r in plan['selected'])==1
    assert all(r['eligible'] for r in plan['selected'])
    broken=copy.deepcopy(values);del broken['joint_score_pg_lr3e-06_kl10']
    with pytest.raises(ValueError,match='Missing'):select_candidates(scan,broken)


def test_online_completion_requires_real_checkpoint_and_current_mc(tmp_path):
    session=tmp_path/'sessions'/'one';session.mkdir(parents=True)
    (session/'mc_calibration').mkdir()
    summary=dict(status='passed',exit_code=0,source_unchanged=True,
        initial_state=dict(critic_initialization={'strict':True}),
        final_state=dict(iteration=8,actor_updates=8,policy_version=8,optimizer_attempts=15),
        evaluation_artifacts=[])
    for iteration in (0,4,8):
        report=tmp_path/f'eval{iteration}.json';report.write_text('{}')
        summary['evaluation_artifacts'].append(dict(iteration=iteration,path=report.name,sha256=sha(report)))
    for version in (4,8):
        (session/'mc_calibration'/f'policy_{version}.json').write_text(json.dumps(dict(
            behavior_version=version,fit=False,after=dict(episodes=32,
            scope='independent_complete_MC_current_fixed_behavior_no_bootstrap'))))
    (session/'summary.json').write_text(json.dumps(summary))
    (tmp_path/'evaluation_baseline.json').write_text('{}')
    weights=tmp_path/'complete.pt';weights.write_bytes(b'unit-test-placeholder-not-model')
    (tmp_path/'latest.json').write_text(json.dumps(dict(path=weights.name,sha256=sha(weights))))
    result=verify_online(tmp_path)
    assert result['accepted_actor_steps']==8 and result['optimizer_attempts']==15
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):verify_online(tmp_path)


def test_objective_coverage_never_relaxes_physical_gates():
    scan,values=fixtures()
    for name,value in values.items():
        if name.startswith('coordinate_dppo'):
            value['aggregate']['music_completion_count']=28
    values['coordinate_dppo_lr3e-06_kl10']=analysis()
    values['coordinate_dppo_lr3e-06_kl10']['aggregate']['reward_sum']=102.
    values['joint_score_pg_lr3e-07_kl3']['aggregate']['reward_sum']=104.
    plan=select_candidates(scan,values)
    names={r['name'] for r in plan['selected']}
    assert names=={'joint_score_pg_lr1e-06_kl1','joint_score_pg_lr3e-07_kl3',
                   'coordinate_dppo_lr3e-06_kl10'}
    assert len(plan['selected'])<=4
    values['coordinate_dppo_lr3e-06_kl10']['aggregate']['joint_limits']['actual']['hard']['max_excess_rad']=.2
    plan=select_candidates(scan,values)
    assert all(r['objective']=='joint_score_pg' for r in plan['selected'])
    values['joint_score_pg_lr5e-08_kl3']['aggregate']['reward_sum']=150.
    plan=select_candidates(scan,values)
    assert 'joint_score_pg_lr5e-08_kl3' not in {r['name'] for r in plan['selected']}
