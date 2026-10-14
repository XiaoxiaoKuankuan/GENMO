"""单世界多环境评估、独立音乐任务与常驻生命周期的CPU合同验证。

检查长歌分配不丢样本、同样本多seed不跨rank、混合音乐起点产生独立正确噪声身份；
常驻评估使用同一个世界并轮换权威日志，退出评估时恢复训练后端而不关闭世界。
仅使用临时目录和替身，不启动Isaac；真实确定性和动力学差异由八卡物理验收报告。
"""
from types import SimpleNamespace
from pathlib import Path
import copy
from gem.closedloop.dppo.evaluation_world import assign_tasks,balanced_rank_tasks
from gem.closedloop.dppo.periodic_monitor import build_balanced_plan
from tests.closedloop.dppo.test_periodic_monitor import Catalog


def test_long_song_distribution_preserves_noise_and_groups(tmp_path):
    plan=build_balanced_plan(Catalog(tmp_path),samples_per_source=4)
    for i,t in enumerate(plan['tasks']):t['remaining_music_seconds']=1.+i//2
    before=copy.deepcopy(plan)
    ranks=balanced_rank_tasks(plan,3)
    assert plan==before
    assert sorted(t['task_id'] for r in ranks for t in r)==sorted(t['task_id'] for t in plan['tasks'])
    owners={}
    for rank,rows in enumerate(ranks):
        for task in rows:
            key=(task['dataset'],task['sample_id'])
            assert owners.setdefault(key,rank)==rank
    queues=assign_tasks(plan['tasks'][:2],4)
    assert sorted(i for q in queues for i,t in q)==[0,1]
    assert sum(not q for q in queues)==2


def test_distinct_music_mixed_start_preserves_seed_pairing(tmp_path):
    catalog=Catalog(tmp_path)
    for samples in catalog.samples['val'].values():
        for s in samples:s['row']['num_frames']=1800
    plan=build_balanced_plan(catalog,samples_per_source=4,start_mode='mixed',unique_music=True)
    assert plan['selected_unique_group_count']==16 and plan['selected_unique_audio_count']==16
    assert {t['music_start_frame'] for t in plan['tasks']}=={0,750}
    assert sum(t['music_start_frame']>0 for t in plan['tasks'])==16
    assert plan==build_balanced_plan(catalog,samples_per_source=4,start_mode='mixed',unique_music=True)


def test_persistent_world_restores_backend_and_rotates_journal(tmp_path,monkeypatch):
    from gem.closedloop.dppo import vector_evaluation as m
    events=[]
    def journal(path,*args,**kwargs):
        return SimpleNamespace(path=Path(path),close=lambda:events.append('journal_close'))
    handle=SimpleNamespace(backend=object(),journal=journal(tmp_path/'initial'),world=SimpleNamespace(),start_seconds=2.)
    context=SimpleNamespace(backend=object(),session=tmp_path,guard=None,stage={'evaluation':{'persistent_world':True}},
        distributed=SimpleNamespace(rank=0,all_gather_object=lambda x:[x]),overhead={})
    original=context.backend
    monkeypatch.setattr(m,'open_isolated_world',lambda *args:(events.append('open') or handle))
    monkeypatch.setattr(m,'close_isolated_world',lambda *args:events.append('close'))
    monkeypatch.setattr(m,'GuardedStepJournal',journal)
    monkeypatch.setattr(m,'local_call',lambda d,f:f())
    for label in ('initial','periodic'):
        with m.isolated_vector_evaluation(context,label):
            assert context.backend is handle.backend and context.active_evaluation_world is handle
        assert context.backend is original and context.active_evaluation_world is None
    assert events==['open','journal_close']
    assert handle.world.journal is handle.journal
    assert context.overhead['evaluation_world_start_seconds']==2.


def test_evaluation_publication_waits_for_matching_actual_checkpoint(tmp_path,monkeypatch):
    from gem.closedloop.dppo.training import _poll_checkpoint
    from gem.closedloop.dppo import periodic_monitor
    publications=[None,dict(iteration=3,path='checkpoints/three.pt',sha256='a'*64,
        metadata={'reason':'evaluated_000003'})]
    report={'evaluation_artifacts':[],'evaluations':[]}
    result={'iteration':3,'actor_identity':{'model_fingerprint':'frozen_snapshot_sha'},
        'selection':{'best_observed':None,'best_saved':None}}
    context=SimpleNamespace(output=tmp_path,session=tmp_path/'session',state={},overhead={},
        async_writer=SimpleNamespace(poll=lambda **kw:publications.pop(0)),
        pending_evaluation=(result,'000003',report,{}))
    def select(current,baseline,checkpoint,**kwargs):
        assert kwargs['actor_fingerprint']=='frozen_snapshot_sha'
        return {'best_observed':None,'best_saved':{'checkpoint':checkpoint}}
    monkeypatch.setattr(periodic_monitor,'mark_saved_evaluation',select)
    assert _poll_checkpoint(context) is None
    assert not report['evaluation_artifacts']
    _poll_checkpoint(context)
    assert context.pending_evaluation is None and len(report['evaluation_artifacts'])==1
    assert (tmp_path/report['evaluation_artifacts'][0]['path']).is_file()
