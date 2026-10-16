"""pilot审计的独立统计与真实奖励核对测试。

不加载模型、不启动物理、不访问服务器。主要检查同曲增加种子不会被错误当成更多
独立音乐而缩窄区间，任务身份不一致必须拒绝，以及50Hz已积分奖励不能再次乘dt。
"""
import copy
import pytest
from tools.eval.audit_stage10_pilots import (add_launcher_timing, episode_metrics,
    paired_statistics,joint_boundary_sample,evaluation_control_steps,summarize_run,
    summarize_complete_evaluation,compare_models,export_run)


def resumed_run(tmp_path):
    """构造目录名逆序的两次真实发布关系，验证计数衔接而非UUID字典序。"""
    import hashlib,json
    root=tmp_path/'run';(root/'sessions').mkdir(parents=True);(root/'metrics').mkdir()
    (root/'training_identity.json').write_text('{}')
    checkpoint=root/'final.pt';checkpoint.write_bytes(b'immutable checkpoint fixture')
    publication=dict(path='final.pt',sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        size_bytes=checkpoint.stat().st_size,iteration=2)
    (root/'latest.json').write_text(json.dumps(publication))
    def state(i):
        return dict(iteration=i,rollouts_completed=i,actor_updates=8*i,critic_updates=32*i,
            policy_version=i,optimizer_attempts=8*i,training_control_steps=100*i,
            training_physics_steps=400*i,training_upper_transitions=4*i)
    for sid,start in [('z_first',0),('a_resume',1)]:
        directory=root/'sessions'/sid;directory.mkdir();(directory/'resolved_config.yaml').write_text('{}')
        summary=dict(session_id=sid,run_id='run',status='passed',exit_code=0,source_unchanged=True,
            initial_state=state(start),final_state=state(start+1),evaluation_artifacts=[],
            world_shutdown=dict(policy_unchanged=True),checkpoint_shutdown=publication)
        (directory/'summary.json').write_text(json.dumps(summary))
        (root/'metrics'/(sid+'.jsonl')).write_text(json.dumps(dict(mode='dppo_bc',state=state(start+1)))+'\n')
    return root


def test_resumed_export_uses_counter_order_and_all_sessions(tmp_path):
    result=export_run(resumed_run(tmp_path))
    assert [s['session_id'] for s in result['sessions']]==['z_first','a_resume']
    assert [r['state']['iteration'] for r in result['training']]==[1,2]
    assert result['initial_state']['iteration']==0 and result['final_state']['iteration']==2


@pytest.mark.parametrize('damage',['missing_summary','counter_gap','failed','source_changed','duplicate_iteration','publication'])
def test_resumed_export_rejects_incomplete_or_inconsistent_history(tmp_path,damage):
    import json
    root=resumed_run(tmp_path);path=root/'sessions/a_resume/summary.json'
    summary=json.loads(path.read_text())
    if damage=='missing_summary':path.unlink()
    else:
        if damage=='counter_gap':summary['initial_state']['actor_updates']+=1
        elif damage=='failed':summary['exit_code']=1
        elif damage=='source_changed':summary['source_unchanged']=False
        elif damage=='publication':summary['checkpoint_shutdown']['iteration']=1
        elif damage=='duplicate_iteration':
            metrics=root/'metrics/a_resume.jsonl';metrics.write_text(metrics.read_text()*2)
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):export_run(root)


def test_earlier_final_evaluation_keeps_its_own_session_interaction_count():
    data=dict(final_state=dict(training_control_steps=500000))
    evaluation=dict(label='final_000100',iteration=100,training_control_steps=100000)
    assert evaluation_control_steps(data,evaluation)==100000


@pytest.mark.parametrize('damage',[None,'sha','missing','physics'])
def test_compact_physical_extractor_keeps_authoritative_journal_guards(tmp_path,damage):
    import hashlib,sqlite3
    from gem.closedloop.dppo.journal_codec import encode_binary
    from tools.compare_stage10_physical import load_physical
    path=tmp_path/'execution.sqlite'
    identity=dict(backend_session_id='backend',env_id=0,episode_id='episode')
    rows=[dict(identity,tick=t,state_valid=True,completed_physics_steps=4,actual=.9) for t in (612,624)]
    if damage=='missing':rows.pop()
    if damage=='physics':rows[0]['completed_physics_steps']=3
    payload=encode_binary(dict(trace=rows));digest=hashlib.sha256(payload).hexdigest()
    if damage=='sha':digest='0'*64
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE replies(sha256 TEXT,payload BLOB)')
        db.execute('INSERT INTO replies VALUES(?,?)',(digest,payload))
    report=dict(episodes=[dict(task_id='task',decisions=[dict(identity=identity,
        authoritative_journal_path=str(path),controls=[dict(tick=t) for t in (612,624)])])])
    if damage:
        with pytest.raises(ValueError):load_physical(report,extractor=lambda row:dict(boundary=row['actual']))
    else:
        result,evidence=load_physical(report,extractor=lambda row:dict(boundary=row['actual']))
        assert result['task']['boundary'].tolist()==[.9,.9]
        assert evidence[0]['matched_controls']==2 and evidence[0]['verified_reply_count']==1


def test_actual_and_reference_joint_boundaries_are_independent():
    raw=dict(hard_limits_rad=[[-1.,1.]]*21,safe_limits_rad=[[-.9,.9]]*21,numeric_tolerance_rad=1e-6,
        actual_joint_position_rad=[.95]+[0.]*20,consumed_reference_joint_position_rad=[1.01]+[0.]*20)
    values=joint_boundary_sample({'raw':raw})
    assert values['actual_soft_any'] and not values['actual_hard_any']
    assert values['actual_max_hard_excess_rad']==0.
    assert values['reference_hard_any'] and values['reference_max_hard_excess_rad']==pytest.approx(.01)


def paired_rows(repeats):
    before, after = [], []
    for group, difference in [('song_a', 0.), ('song_b', 2.)]:
        for seed in range(repeats):
            row = dict(task_id=f'{group}_{seed}', dataset='source', sample_id=group,
                seed=seed, music_group=group, audio_sha=group, music_start_frame=0,
                noise_index_start=seed, reward=10.)
            before.append(row)
            after.append(dict(row, reward=10.+difference))
    return before, after


def test_repeated_seeds_do_not_create_independent_music():
    one = paired_statistics(*paired_rows(1), 'reward', draws=1000)
    many = paired_statistics(*paired_rows(20), 'reward', draws=1000)
    assert one['mean'] == many['mean'] == 1.
    assert one['ci95'] == many['ci95']
    assert one['ci90'] == many['ci90']
    assert one['ci95'][0] <= one['ci90'][0] <= one['ci90'][1] <= one['ci95'][1]
    assert many['music_groups_by_source'] == {'source': 2}


def test_evaluation_axis_uses_measured_controls_including_post_rollout_mc():
    data=dict(initial_state=dict(training_control_steps=200000),
        final_state=dict(training_control_steps=1500123),
        training=[dict(mode='dppo_bc',state=dict(iteration=10,training_control_steps=1223687))])
    assert evaluation_control_steps(data,dict(label='initial',iteration=0))==200000
    assert evaluation_control_steps(data,dict(label='periodic_r000010_c000001223687_a000010',iteration=10))==1223687
    assert evaluation_control_steps(data,dict(label='final_000010',iteration=10))==1500123
    assert evaluation_control_steps(data,dict(label='extra',iteration=10))==1223687
    with pytest.raises(ValueError,match='交互计数'):
        evaluation_control_steps(data,dict(label='extra',iteration=9))


def test_disabled_gradient_diagnostic_is_missing_not_zero_in_completed_report():
    before,after=paired_rows(1)
    for row in before+after:
        row.update(reward_per_second=row['reward']/10,actual_activity_rad_s=1.,activity_ratio=.8,
            activity_gate=.9,physical_failure=False,end_reason='music_end')
    kl=dict(covered_internal_transitions=40,covered_chains=2,wall_seconds=1.,
        mean_coordinate_kl=.0001,maximum_step_mean_coordinate_kl=.001,objective_clip_fraction=.05,
        per_denoising_step=[dict(step=i,mean_coordinate_kl=dict(mean=.0001),joint_kl=dict(mean=.1)) for i in range(20)])
    state=dict(selected_actor_lr=1e-7,training_control_steps=200)
    record=dict(mode='dppo_bc',full20_kl=kl,wall_seconds=5.,actor_seconds=2.,critic_seconds=1.,
        actor=dict(optimizer_steps=1,steps=[dict(learning_rate=1e-7)],gradient_snr=None),
        collection_by_rank=[dict(through_targets_and_budget_seconds=1.)])
    data=dict(run_id='example',summary_status='passed',exit_code=0,source_unchanged=True,
        checkpoint_sha_verified=True,training=[record],evaluations=[dict(rows=before),dict(rows=after)],
        initial_state=state,final_state=state)
    result=summarize_run(data)
    assert result['full20_coverage_passed']
    assert result['training_metrics']['half_gradient_cosine'] is None
    assert result['gradient_snr_observed_iterations']==0
    record['actor']['gradient_snr']=dict(half_split_cosine_pooled=.02)
    result=summarize_run(data)
    assert result['training_metrics']['half_gradient_cosine']['mean']==.02
    assert result['gradient_snr_observed_iterations']==1
    kl['covered_internal_transitions']=39
    with pytest.raises(ValueError,match='20步'):summarize_run(data)
    kl['covered_internal_transitions']=40;kl['per_denoising_step'].pop()
    with pytest.raises(ValueError,match='20步'):summarize_run(data)
    kl['per_denoising_step'].append(dict(kl['per_denoising_step'][0]))
    with pytest.raises(ValueError,match='20步'):summarize_run(data)


def complete_rows():
    rows=paired_rows(1)[0]
    for i,row in enumerate(rows):
        controls=100 if i==0 else 200
        row.update(reward=10.,reward_per_second=10./(controls/50),seconds=controls/50,
            actual_activity_rad_s=1.,activity_ratio=.8,physical_failure=i==0,truncated=False,
            end_reason='yaw_anchor_ori' if i==0 else 'music_end',control_steps=controls,
            joint_boundaries=dict(available=True,control_samples=controls,
                actual_soft_control_fraction=.1,actual_hard_control_fraction=.03 if i==0 else 0.,
                actual_max_hard_excess_rad=.001,reference_soft_control_fraction=.2,
                reference_hard_control_fraction=.3,reference_max_hard_excess_rad=.2))
    return rows


def test_full_song_report_keeps_early_failure_and_separates_reference_boundaries():
    result=summarize_complete_evaluation(dict(rows=complete_rows()))
    assert result['overall']['completed']==result['overall']['physical_failures']==1
    assert result['overall']['completion_rate']==.5
    assert result['overall']['reward_per_second']==3.75
    assert result['overall']['actual_hard_control_fraction']==pytest.approx(.01)
    assert result['overall']['reference_hard_control_fraction']==pytest.approx(.3)
    assert result['overall']['actual_max_hard_excess_rad']==.001
    assert result['overall']['reference_max_hard_excess_rad']==.2


@pytest.mark.parametrize('damage',['truncated','evidence','failure_flag'])
def test_full_song_report_rejects_noncomplete_or_unverified_inputs(damage):
    rows=complete_rows()
    if damage=='truncated':rows[0]['truncated']=True
    elif damage=='evidence':rows[0]['joint_boundaries']['available']=False
    else:rows[1]['physical_failure']=True
    with pytest.raises(ValueError):summarize_complete_evaluation(dict(rows=rows))


def test_complete_model_comparison_requires_same_execution_and_includes_failures():
    evaluation=dict(rows=complete_rows(),plan_sha256='plan',actor_identity='stage1',
        evaluation_identity={k:'same' for k in ('kernel','timing_contract','execution_profile',
            'critical_latency_budgets','episode_seconds','physics')})
    baseline=dict(evaluations=[evaluation],summary_status='passed',exit_code=0,
        source_unchanged=True,checkpoint_sha_verified=True,
        identity={k:'same' for k in ('assets','source_files','physical_configuration','reward','generation_horizon')})
    result=compare_models(baseline,copy.deepcopy(baseline),copy.deepcopy(baseline))
    assert result['models']['P1']['overall']['physical_failures']==1
    assert result['paired_to_stage1']['P1']['reward_per_second']['ci90']==[0.,0.]
    changed=copy.deepcopy(baseline);changed['identity']['reward']='different'
    with pytest.raises(ValueError,match='执行身份'):
        compare_models(baseline,changed,baseline)


def test_pairing_rejects_same_task_id_with_changed_noise():
    before, after = paired_rows(1)
    after[0]['noise_index_start'] += 1
    with pytest.raises(ValueError, match='noise_index_start'):
        paired_statistics(before, after, 'reward')


def test_launcher_timing_includes_overheads_and_rejects_different_run():
    result = dict(runs={'P1': dict(accepted_iterations=40)})
    evidence = dict(pilot_progress='[2026-10-14_19:14:52] end P1 exit=0 wall=5781s\n')
    add_launcher_timing(result, [evidence])
    assert result['runs']['P1']['amortized_total_seconds_per_accepted_iteration'] == 144.525
    with pytest.raises(ValueError, match='不匹配'):
        add_launcher_timing(result, [dict(pilot_progress=evidence['pilot_progress'].replace('P1', 'P2'))])


@pytest.mark.parametrize('exit_code,wall_seconds',[(0,12000.5),(1,12000.5),(0,float('nan'))])
def test_native_launcher_exit_json_is_not_reconstructed_from_iteration_wall(exit_code,wall_seconds):
    result=dict(runs={'P1':dict(accepted_iterations=100)})
    evidence=dict(run_id='P1',exit_code=exit_code,wall_seconds=wall_seconds)
    if exit_code or wall_seconds!=wall_seconds:
        with pytest.raises(ValueError):add_launcher_timing(result,[evidence])
    else:
        add_launcher_timing(result,[evidence])
        assert result['runs']['P1']['amortized_total_seconds_per_accepted_iteration']==120.005


def test_integrated_rewards_and_terminal_event_are_not_multiplied_by_dt():
    task = dict(task_id='task', dataset='source', sample_id='sample', seed=42,
        music_start_frame=0, noise_index_start=1,
        sample=dict(group_id='song', row=dict(source_audio_sha256='audio')))
    control = dict(dt_s=.02, reward=-.03, continuous_reward=.07, event_reward=-.1,
        components=dict(alive=dict(integrated_reward=.01), tracking=dict(integrated_reward=.06)),
        activity=dict(actual_activity_rad_s=.2, target_activity_rad_s=.4, gate=1.))
    decision = dict(controls=[control], executed_control_steps=1, executed_physics_steps=4,
        zero_step_event_reward=-.2, reward_sum=-.23)
    episode = dict(task, decisions=[decision], reward_sum=-.23, physical_failure=True,
        truncated=False, end_reason='physical_failure')
    result = episode_metrics(episode, task)
    assert result['reward'] == pytest.approx(-.23)
    assert result['seconds'] == pytest.approx(.02)
    assert result['reward_per_second'] == pytest.approx(-11.5)
    assert result['activity_ratio'] == pytest.approx(.5)
    bad = copy.deepcopy(episode)
    bad['decisions'][0]['executed_physics_steps'] = 3
    with pytest.raises(ValueError, match='四子步'):
        episode_metrics(bad, task)
