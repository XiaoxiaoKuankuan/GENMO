"""验证有限物理队列的筛选、精确权重去重及完整证据门禁。

使用合成清单检查调度逻辑，不启动GPU或Isaac，也不把合成结果作为物理质量证明。
不同KL门槛只有产生完全相同Actor指纹时才能共用执行；所有不同权重均须保留。
"""
import copy
import pytest
from tools.eval.run_stage2_actor_lr_physics import evaluation_jobs


def scan():
    rows=[]
    for objective in ('coordinate_dppo','joint_score_pg'):
        for lr in ('5e-08','3e-07','1e-06','3e-06','1e-05'):
            for scale in (1,3,10):
                name=f'{objective}_lr{lr}_kl{scale}'
                rows.append(dict(name=name,path=name+'.yaml',sha256=name,
                    screening={'passed':True},backtracking={'accepted':True},
                    final_acceptance={'passed':True},evaluation_actor=dict(
                        path=name+'.pt',sha256=name,actor_fingerprint=objective+lr)))
    return dict(status='completed_fixed_data_scan_not_physical_acceptance',
        same_stage1_actor=True,same_empty_adam=True,candidates=rows)


def test_only_identical_weights_deduplicated_all_survivors_retained():
    report=scan();jobs=evaluation_jobs(report)
    assert len(jobs)==11 and jobs[0]['actor'] is None
    assert all(len(j['aliases'])==3 for j in jobs[1:])
    report['candidates'][1]['evaluation_actor']['actor_fingerprint']='distinct'
    assert len(evaluation_jobs(report))==12
    report['candidates'][1]['screening']['passed']=False
    assert len(evaluation_jobs(report))==11


def test_incomplete_or_unaccepted_scan_cannot_launch_physics():
    for mutation in ('status','same_empty_adam','candidate'):
        report=copy.deepcopy(scan())
        if mutation=='status':report['status']='running'
        elif mutation=='same_empty_adam':report['same_empty_adam']=False
        else:report['candidates'][0]['final_acceptance']['passed']=False
        with pytest.raises(ValueError):evaluation_jobs(report)
