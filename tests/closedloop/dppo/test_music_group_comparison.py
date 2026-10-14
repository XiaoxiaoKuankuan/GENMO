"""独立音乐组对照的关键口径测试。

验证同一音乐的多片段不增加独立样本数、最长样本的选择不依赖模型结果、种子
成组分片后无遗漏重复，以及完整MC按真实50Hz积分奖励折扣而非再次乘dt。
只使用内存夹具，无GPU、真实数据或训练目录读写。
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.eval.stage10_group_comparison import control_mc_returns, music_group_plan, partition_by_music, representatives


def sample(source, group, name, frames):
    return dict(dataset=source,split='val',group_id=group,
        row=dict(sample_id=name,num_frames=frames,fps=30,split='val',source_audio_sha256=group))


def test_longest_single_representative_and_tie_order():
    rows=[sample('A','g1','short',30),sample('A','g1','b',120),sample('A','g1','a',120),sample('A','g2','c',60)]
    assert [s['row']['sample_id'] for s in representatives(rows)]==['a','c']
    assert representatives(rows)==representatives(rows[::-1])


def test_group_plan_shards_keep_seed_pairs_and_exact_coverage():
    catalog=SimpleNamespace(data_root='/tmp/unused_group_fixture',identity={'fixture':True},samples={'val':{
        'A':[sample('A','g1','a',300),sample('A','g1','b',120),sample('A','g2','c',30)],
        'B':[sample('B','g3','d',60),sample('B','g4','e',90)]}})
    plan=music_group_plan(catalog,'val',[1729,2718])
    assert plan['selected_unique_group_count']==4 and plan['task_count']==8
    assert plan['complete_pool_count']==5
    assert {t['sample_id'] for t in plan['tasks']}=={'a','c','d','e'}
    shards=partition_by_music(plan,8)
    ids=[t['task_id'] for s in shards for t in s['tasks']]
    assert len(ids)==len(set(ids))==8
    for group in ['g1','g2','g3','g4']:
        assert sum(any(t['sample']['group_id']==group for t in shard['tasks']) for shard in shards)==1


def test_true_control_discount_and_no_double_dt():
    gamma=.99**(1/25)
    expected=np.dot(gamma**np.arange(75),np.ones(75))
    result=control_mc_returns([np.ones(25),np.ones(50)])
    assert result[0]==pytest.approx(expected)
    assert result[1]==pytest.approx(np.dot(gamma**np.arange(50),np.ones(50)))
    assert result[0]>70
    # 最后的实际事件罚分只计一次，不借未来值填充真实终止。
    assert control_mc_returns([np.array([.04,.04]),np.array([.04-5])])[0]==pytest.approx(.04+gamma*.04+gamma**2*(.04-5))


def test_no_nonfinite_mc_truth():
    with pytest.raises(ValueError):control_mc_returns([[np.nan]])
    with pytest.raises(ValueError):control_mc_returns([[1]],gamma_upper=0)
    assert control_mc_returns([np.array([1.]),np.array([])],event_rewards=[0.,-5.])[0]==pytest.approx(1-5*.99**(1/25))
