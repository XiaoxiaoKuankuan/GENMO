"""紧凑真实消费来源与前瞻/中心区分的纯数据验收。

构造合法21点GMT窗口，验证首次进入前瞻不会误计中心消费、源帧插值权重守恒、
奖励时刻另算、未提交计划及来源缺失明确失败。缺少历史逐步记录时只能标未验证。
不创建物理环境，不读取正式模型或改动实际参考时间轴。
"""
import pytest
import torch
from types import SimpleNamespace
from gem.closedloop.dppo.execution_provenance import SCHEMA,source_frame_support,consumption_report,compact_execution


def test_source_interpolation_and_lookahead_is_not_current_consumption():
    assert source_frame_support(12,0)==[(0,.4),(1,.6)]
    assert source_frame_support(60,0)==[(3,1.)]
    with pytest.raises(ValueError):source_frame_support(1190,0)
    ticks=list(range(-108,133,12))
    plans=['bootstrap:x' if tick<24 else 'p' for tick in ticks]
    row=dict(control_tick_begin=12,control_tick_end=24,consumed_reference_ticks=ticks,consumed_plan_ids=plans,
        reference_tick=24,reference_plan_id='p')
    record=dict(schema=SCHEMA,plans={'p':dict(decision_tick=0,commit_tick=12,prefix_frames=0,
        parent_plan_id='bootstrap:x',generation_frames=60)},steps=[row])
    result=consumption_report([record]);p=result['plans']['p']
    assert p['first_window_tick']==12 and p['first_current_tick'] is None
    assert sum(p['current_frame_weight'])==0 and sum(p['reward_frame_weight'])==pytest.approx(1)
    assert sum(p['window_frame_weight'])==pytest.approx(10)
    backend_row={**row,'tick':row['control_tick_end']}
    packed=compact_execution([backend_row],ledger=SimpleNamespace(records=record['plans']),
        plan_id='p',decision_tick=0,commit={'effective_tick':12})
    assert 'consumed_plan_ids' not in packed['steps'][0]
    assert consumption_report([packed])==result
    assert consumption_report([None])['status']=='unverified'
    record['plans']['p']['commit_tick']=24
    with pytest.raises(ValueError,match='before successful commit'):consumption_report([record])


def test_consumed_frame_KL_association_does_not_invent_missing_history(tmp_path,monkeypatch):
    import tools.analyze_stage2_reference_consumption as tool
    (tmp_path/'rank00').mkdir()
    identity=dict(backend_session_id='backend',env_id=0,episode_id='episode',plan_id='p')
    metadata=dict(identity=identity,source='A',music_frame=0,prefix_frames=0,
        kl_by_frame=[1.]*60,log_ratio_by_frame=[-.1]*60)
    row=SimpleNamespace(identity=identity,metadata={},free_mask=torch.ones(60,30,dtype=torch.bool))
    monkeypatch.setattr(tool,'load_rollout',lambda path:{'transitions':[row]})
    result=tool.associate(tmp_path,dict(metadata=[metadata]))
    assert result['status']=='unverified'
    assert result['by_consumption']['unverified_missing_provenance']['joint_kl']==60
    ticks=list(range(-108,133,12));plans=['bootstrap:x' if tick<24 else 'p' for tick in ticks]
    row.metadata['compact_execution_provenance']=dict(schema=SCHEMA,plans={'p':dict(decision_tick=0,commit_tick=12,
        prefix_frames=0,parent_plan_id='bootstrap:x',generation_frames=60)},
        steps=[dict(control_tick_begin=12,control_tick_end=24,consumed_reference_ticks=ticks,consumed_plan_ids=plans,
            reference_tick=24,reference_plan_id='p')])
    result=tool.associate(tmp_path,dict(metadata=[metadata]))
    assert result['status']=='verified_in_retained_ranges'
    assert result['by_consumption']['current_consumed_source_support']['joint_kl']==0
    assert result['by_consumption']['window_observed_not_current']['joint_kl']==7
    assert sum(v['joint_kl'] for v in result['by_consumption'].values())==60


def test_interpolation_reaches_previous_plan_through_exact_committed_prefix():
    plans={'p0':dict(decision_tick=0,commit_tick=12,prefix_frames=0,parent_plan_id='bootstrap:x',generation_frames=60),
        'p1':dict(decision_tick=300,commit_tick=468,prefix_frames=17,parent_plan_id='p0',generation_frames=60)}
    ticks=list(range(504,745,12))
    row=dict(control_tick_begin=624,control_tick_end=636,consumed_reference_ticks=ticks,
        consumed_plan_ids=['p0' if tick<=612 else 'p1' for tick in ticks],reference_tick=636,reference_plan_id='p1')
    result=consumption_report([dict(schema=SCHEMA,plans=plans,steps=[row])])
    assert result['source_lineage_complete']
    assert sum(result['plans']['p0']['current_frame_weight'])==0
    assert result['plans']['p0']['current_generated_frame_weight'][31]==pytest.approx(.8)
    assert result['plans']['p1']['current_generated_frame_weight'][17]==pytest.approx(.2)
    assert sum(sum(v['current_generated_frame_weight']) for v in result['plans'].values())==pytest.approx(1.)
