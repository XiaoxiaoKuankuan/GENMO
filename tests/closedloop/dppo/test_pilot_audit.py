"""pilot审计的独立统计与真实奖励核对测试。

不加载模型、不启动物理、不访问服务器。主要检查同曲增加种子不会被错误当成更多
独立音乐而缩窄区间，任务身份不一致必须拒绝，以及50Hz已积分奖励不能再次乘dt。
"""
import copy
import pytest
from tools.eval.audit_stage10_pilots import add_launcher_timing, episode_metrics, paired_statistics,joint_boundary_sample


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
    assert many['music_groups_by_source'] == {'source': 2}


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
