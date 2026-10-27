"""批量原Stage1评价的数据隔离与持久领取回归测试。

验证不同音乐尾部各自屏蔽、同请求噪声不随批大小变化、独立环境领取不重复。
只验证可解析的数据合同；真实Newton多环境等价性与速度另在服务器2实际测量。
"""
import json

import torch

from gem.closedloop.mb_control.batched_census import claim_batch,music_masks
from gem.closedloop.mb_control.stage1_census import connect,finish


def test_music_tail_is_masked_per_environment():
    ticks=torch.tensor([0,20,40,60,80]);lengths=torch.tensor([2,5])
    assert music_masks(ticks,lengths).tolist()==[[True,True,False,False,False],[True]*5]


def test_noise_identity_does_not_depend_on_batch_size():
    def noise(B,tick):
        g=torch.Generator().manual_seed(42+(tick//60)*1009)
        return torch.randn(1,60,30,generator=g).expand(B,-1,-1).clone()
    assert torch.equal(noise(1,300)[0],noise(32,300)[17])
    assert not torch.equal(noise(1,300),noise(1,600))


def test_batch_claims_independent_tasks_and_keeps_failures(tmp_path):
    tasks=[dict(id=str(i),census_key=str(i)) for i in range(5)]
    with connect(tmp_path) as db:
        db.execute('CREATE TABLE jobs(key TEXT PRIMARY KEY,ordinal INTEGER,task TEXT,status TEXT,attempts INTEGER DEFAULT 0,owner INTEGER,started TEXT,result TEXT)')
        for i,t in enumerate(tasks):db.execute('INSERT INTO jobs(key,ordinal,task,status) VALUES(?,?,?,?)',(str(i),i,json.dumps(t),'pending'))
    first=claim_batch(tmp_path,3);second=claim_batch(tmp_path,3)
    assert [t['id'] for t,_ in first]==['0','1','2']
    assert [t['id'] for t,_ in second]==['3','4']
    assert not claim_batch(tmp_path,32)
    finish(tmp_path,first[0][0],dict(status='physical_result',complete=False,reason='physical_failure'))
    assert not claim_batch(tmp_path,1)
