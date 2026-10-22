"""单世界多环境评估、独立音乐任务与常驻生命周期的CPU合同验证。

检查长歌分配不丢样本、同样本多seed不跨rank、混合音乐起点产生独立正确噪声身份；
常驻评估使用同一个世界并轮换权威日志，退出评估时恢复训练后端而不关闭世界。
仅使用临时目录和替身，不启动Isaac；真实确定性和动力学差异由八卡物理验收报告。
"""
from types import SimpleNamespace
from pathlib import Path
import copy
import pytest
from gem.closedloop.dppo.evaluation_world import assign_tasks,balanced_rank_tasks
from gem.closedloop.dppo.periodic_monitor import build_balanced_plan
from tests.closedloop.dppo.test_periodic_monitor import Catalog


def test_mc_parallel_capacity_does_not_change_actor_evaluation_protocol():
    from gem.closedloop.dppo.vector_evaluation import isolated_env_count
    c=SimpleNamespace(stage={'evaluation':{'envs_per_rank':4},
        'critic':{'complete_mc':{'envs_per_rank':32}}})
    assert isolated_env_count(c,'mc_preheat_train')==32
    assert isolated_env_count(c,'mc_preheat_val_retry1')==32
    assert isolated_env_count(c,'initial')==4
    assert isolated_env_count(c,'final')==4


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


def test_all_available_test_music_uses_longest_sample_without_repeating_groups(tmp_path):
    catalog=Catalog(tmp_path)
    catalog.samples['test']=copy.deepcopy(catalog.samples['val'])
    for source,samples in catalog.samples['test'].items():
        for sample in samples:sample['split']=sample['row']['split']='test'
        samples[1]['group_id']=samples[0]['group_id']
        samples[1]['row']['source_audio_sha256']=samples[0]['row']['source_audio_sha256']
        samples[1]['row']['num_frames']=300
    plan=build_balanced_plan(catalog,samples_per_source='all_unique',unique_music=True,split='test',seconds=None)
    assert plan['selected_unique_group_count']==16 and len(plan['tasks'])==32
    assert all(t['sample_id']!='0' for t in plan['tasks'])
    assert plan['training_buffer_allowed'] is False


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


@pytest.mark.parametrize('primary_failure',[False,True])
def test_cleanup_failure_does_not_hide_original_execution_failure(tmp_path,monkeypatch,primary_failure):
    from gem.closedloop.dppo import vector_evaluation as m
    handle=SimpleNamespace(backend=object(),start_seconds=0.,close_seconds=1.)
    context=SimpleNamespace(backend=object(),session=tmp_path,stage={'evaluation':{}},
        distributed=SimpleNamespace(rank=0,all_gather_object=lambda x:[x]),overhead={})
    original=context.backend
    monkeypatch.setattr(m,'open_isolated_world',lambda *args:handle)
    def close(*args):raise RuntimeError('incomplete execution or changed parameters')
    monkeypatch.setattr(m,'close_isolated_world',close)
    if primary_failure:
        with pytest.warns(UserWarning,match='primary failure preserved'):
            with pytest.raises(ValueError,match='original reference_underrun'):
                with m.isolated_vector_evaluation(context,'mc_replay'):
                    raise ValueError('original reference_underrun')
        import json
        record=json.loads((tmp_path/'phases/vector_eval_cleanup_mc_replay/rank00.json').read_text())
        assert record['primary_type']=='ValueError' and 'incomplete execution' in record['cleanup_message']
    else:
        with pytest.raises(RuntimeError,match='incomplete execution'):
            with m.isolated_vector_evaluation(context,'mc_replay'):pass
    assert context.backend is original and context.active_evaluation_world is None


def test_persistent_lane_preserves_episode_only_at_same_confirmed_watermark(tmp_path,monkeypatch):
    from gem.closedloop.dppo import evaluation_world as m
    class Lane:
        session='session';sequence=0;acked=0;episode=None
        def call(self,method,**p):
            if method=='hello':return dict(execution_protocols=['ack.v2'],backend_session_id=self.session,
                executed_seq=self.sequence,acked_seq=self.acked)
            if method=='ack':self.acked=p['through_seq'];return {}
            assert method=='execute' and p['expected_episode_id']==self.episode, 'stale_episode'
            assert p['backend_session_id']==self.session and p['mutation_seq']==self.sequence+1
            self.sequence+=1;self.episode=f'episode_{self.sequence}'
            return dict(backend_session_id=self.session,mutation_seq=self.sequence,operation=p['operation'],
                ok=True,result={'episode_id':self.episode})
    lane=Lane();handle=SimpleNamespace(journal=SimpleNamespace(path=tmp_path/'world.sqlite'))
    c=SimpleNamespace(active_evaluation_world=handle,builder=None,guard=None,env=SimpleNamespace(latency_budget_s=.3),
        state={'policy_version':0,'iteration':0})
    monkeypatch.setattr(m,'VectorUpperEnvironment',lambda config,backend,*args:SimpleNamespace(backend=backend))
    def create(label):
        d=m.EvaluationWorldDriver.__new__(m.EvaluationWorldDriver)
        d.c=c;d.config={'runtime':{}};d.budget=None;d.path=tmp_path/label
        d.collector=SimpleNamespace(lane_client=lambda slot:lane)
        resource=d._factory(0,None);d.collector.states=[SimpleNamespace(resource=resource)]
        return d,resource.env.backend
    first,backend=create('validation');backend.call('reset_episode');first.close()
    second,backend=create('training');assert backend.episode_id=='episode_1'
    backend.call('reset_episode');assert backend.episode_id=='episode_2';second.close()
    lane.sequence+=1;lane.acked=lane.sequence
    with pytest.raises(RuntimeError,match='changed session or acknowledged sequence'):create('unknown_mutation')
    del handle.lane_boundaries
    with pytest.raises(RuntimeError,match='without its confirmed episode identity'):create('missing_boundary')


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
