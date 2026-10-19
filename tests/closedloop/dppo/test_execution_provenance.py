"""紧凑真实消费来源与前瞻/中心区分的纯数据验收。

构造合法21点GMT窗口，验证首次进入前瞻不会误计中心消费、源帧插值权重守恒、
奖励时刻另算、未提交计划及来源缺失明确失败。缺少历史逐步记录时只能标未验证。
不创建物理环境，不读取正式模型或改动实际参考时间轴。
"""
import pytest
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
    record=dict(schema=SCHEMA,plans={'p':dict(decision_tick=0,commit_tick=12)},steps=[row])
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
