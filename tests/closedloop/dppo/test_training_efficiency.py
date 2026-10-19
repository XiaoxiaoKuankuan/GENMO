"""工程优化的完整性回归：KL凭据、单份控制证据和合并恢复点。

使用微型CPU网络和真实序列化/维护调度函数，验证权重、buffer、rollout变化会使KL
复用失效，任意rank失配会统一重算；无损控制记录能独立复算并检测篡改；同轮评估、
百轮保存和MC维护只能发布一份最终状态。测试不启动GPU、Isaac或真实训练。
"""
from types import SimpleNamespace as NS
import copy
import json
import pytest
import torch

from gem.closedloop.dppo.kl_reuse import seal_accepted_kl, reusable_accepted_kl
from gem.closedloop.dppo.compact_evidence import write_episode, read_episode
from gem.closedloop.dppo.evaluation import _publish_json
from gem.closedloop.dppo.training_checkpoint import AsyncCheckpointWriter, SCHEMA


class Collective:
    rank=0
    def all_gather_object(self,value):return [value]


def objects():
    actor=torch.nn.Linear(2,1)
    actor.register_buffer('counter',torch.zeros(1))
    policy=NS(actor=actor,kernel_config={'kernel':'fixed'},trainable_steps=(15,16,17,18),
        timestep_map=list(range(20)),eta=.1,guidance_scale=2.5,std_schedule=[.1]*20,
        assert_frozen_identity=lambda:True)
    cache=NS(host={'chain':torch.zeros(2,21,60,30)},blocks={},closed=False)
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(2)]
    report=dict(covered_chains=2,covered_internal_transitions=8,gaussian_kl_steps=[15,16,17,18])
    return policy,cache,manifest,report


@pytest.mark.parametrize('mutation',['none','parameter','buffer','cache','kernel','manifest','closed'])
def test_kl_reuse_binds_actual_model_data_and_complete_coverage(mutation):
    policy,cache,manifest,report=objects()
    token=seal_accepted_kl(policy,cache,manifest,report,clip=.003,reduction='free_coordinate_mean')
    with torch.no_grad():
        if mutation=='parameter':policy.actor.weight.add_(1.)
        if mutation=='buffer':policy.actor.counter.add_(1.)
        if mutation=='cache':cache.host['chain'].add_(1.)
    if mutation=='kernel':policy.guidance_scale=1.
    if mutation=='manifest':manifest[0]['valid']=False
    if mutation=='closed':cache.closed=True
    policy._accepted_kl_receipt=token
    result=reusable_accepted_kl({'complete_accepted_kl_available':True},policy,cache,manifest,Collective(),
        clip=.003,reduction='free_coordinate_mean')
    assert (result is not None)==(mutation=='none')


def test_any_rank_stale_forces_global_recomputation():
    policy,cache,manifest,report=objects()
    token=seal_accepted_kl(policy,cache,manifest,report,clip=.003,reduction='free_coordinate_mean')
    collective=NS(all_gather_object=lambda value:[value,False])
    policy._accepted_kl_receipt=token
    assert reusable_accepted_kl({'complete_accepted_kl_available':True},policy,cache,manifest,collective,
        clip=.003,reduction='free_coordinate_mean') is None


def test_control_archive_is_lossless_single_copy_and_rejects_corruption(tmp_path):
    controls=[dict(tick=12*(i+1),reward=.02*i,dt_s=.02,
        joint_limits=[float(i),0.],name='同一关节') for i in range(40)]
    episode=dict(task_id='task',reward_sum=sum(r['reward'] for r in controls),
        decisions=[dict(controls=controls,executed_control_steps=40)])
    path=tmp_path/'episode.json'
    write_episode(path,episode,publish_json=_publish_json)
    document=json.loads(path.read_text())
    assert 'controls' not in document['decisions'][0]
    restored=read_episode(path)
    assert restored['decisions']==episode['decisions']
    assert sum(r['reward'] for r in restored['decisions'][0]['controls'])==episode['reward_sum']
    archive=tmp_path/document['control_archive']['path']
    archive.write_bytes(archive.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='SHA'):read_episode(path)


def test_maintenance_coalesces_evaluation_milestone_and_mc_after_refit(monkeypatch):
    from gem.closedloop.dppo import training,mc_calibration
    events=[]
    c=NS(distributed=Collective(),overhead={},state=dict(iteration=100,rollouts_completed=100,
        policy_version=100,training_control_steps=1000,next_evaluation_control_steps=500,
        next_mc_calibration_control_steps=500,critic_updates=7),
        stage=dict(evaluation=dict(every_actor_iterations=100,every_control_steps=1000),
            storage=dict(checkpoint_interval_iterations=100,checkpoint_interval_seconds=1800),
            critic=dict(complete_mc=dict(enabled=True,every_actor_iterations=100,every_control_steps=1000))))
    monkeypatch.setattr(training,'_evaluate',lambda *args,**kwargs:events.append(('evaluate',kwargs['save'])))
    def mc(*args,**kwargs):c.state['critic_updates']=9;events.append(('mc',9))
    monkeypatch.setattr(mc_calibration,'calibrate_critic',mc)
    monkeypatch.setattr(training,'_snapshot',lambda c,reason:events.append(('save',reason,c.state['critic_updates'])))
    training._maintain_after_iteration(c,{},is_warmup=False,rejected=False,last_save=0)
    saves=[e for e in events if e[0]=='save']
    assert len(saves)==1 and saves[0][2]==9
    assert all(tag in saves[0][1] for tag in ['evaluated','iteration_000100','mc_calibrated'])
    assert events[0]==('evaluate',False) and events[1][0]=='mc'


def test_one_complete_publication_has_all_maintenance_roles(tmp_path):
    payload=dict(schema=SCHEMA,state=dict(iteration=100,rollouts_completed=100,actor_updates=100,critic_updates=9),
        actor={'weight':torch.ones(2)},actor_fingerprint='actor')
    writer=AsyncCheckpointWriter(tmp_path)
    try:
        writer.submit(payload,reason='evaluated_r100+iteration_000100+mc_calibrated')
        publication=writer.poll(wait=True)
        assert len(list((tmp_path/'checkpoints').glob('*.pt')))==1
        assert publication['metadata']['reasons']==['evaluated_r100','iteration_000100','mc_calibrated']
        assert torch.load(tmp_path/publication['path'],weights_only=False)['state']['critic_updates']==9
    finally:writer.close()


def test_engineering_migration_keeps_learning_content_and_rejects_algorithm_changes(monkeypatch):
    from tools.migrate_stage2_efficiency import digest,target_identity
    from gem.closedloop.dppo.training_config import configuration
    from pathlib import Path
    from tools import migrate_stage2_efficiency as migration
    config=configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_credit_server2.yaml')
    stage=config['stage10']
    keys=('training','critic','bc_gradient','kl','scale','dataset','evaluation','run_control','physical_episode','ppo_rollout','generation_horizon')
    parent={key:copy.deepcopy(stage[key]) for key in keys}
    parent.update(reward=config['stage9']['reward'],runtime=config['runtime'],timing=config['timing'],
        resource_limits=stage['limits'],assets=stage['assets'],physical_configuration={'physics':'same'},
        source_files={'genmo':{'gem/closedloop/dppo/training.py':'old'},'gmt':{}})
    monkeypatch.setattr(migration,'_physical_configuration_identity',lambda c:{'physics':'same'})
    monkeypatch.setattr(migration,'_sources',lambda c:{'genmo':{'gem/closedloop/dppo/training.py':'new'},'gmt':{}})
    identity,changes=target_identity(parent,config)
    assert identity['training']==parent['training'] and changes['genmo']
    assert digest({'x':torch.ones(4),'rng':(1,2)})==digest({'rng':(1,2),'x':torch.ones(4)})
    modified=copy.deepcopy(config);modified['stage10']['training']['actor_lr']=1e-6
    with pytest.raises(ValueError,match='algorithm'):target_identity(parent,modified)


def test_completed_run_compaction_keeps_controls_models_and_budget(tmp_path):
    import gzip
    import sqlite3
    from tools.compact_stage2_run import compact
    from gem.closedloop.dppo.training_checkpoint import file_sha256
    root=tmp_path/'run';root.mkdir()
    (root/'run_manifest.json').write_text(json.dumps({'schema':'genmo.closedloop.stage10.run.v1'}))
    phase=root/'sessions/s/phases/eval/rank00'
    episode_path=phase/'report/episodes/000000.json';episode_path.parent.mkdir(parents=True)
    episode=dict(task_id='task',reward_sum=1.,decisions=[dict(controls=[dict(tick=12,reward=1.,dt_s=.02)])])
    _publish_json(episode_path,episode)
    reference=dict(path=str(episode_path),sha256=file_sha256(episode_path),task_id='task')
    summary=root/'sessions/s/evaluations/initial.json';summary.parent.mkdir(parents=True)
    _publish_json(summary,dict(episodes=[episode],source_episode_manifests=[reference],aggregate={'reward_sum':1.}))
    session_summary=root/'sessions/s/summary.json'
    _publish_json(session_summary,dict(evaluation_artifacts=[dict(
        path=str(summary.relative_to(root)),sha256=file_sha256(summary))]))
    database=phase/'world.sqlite';connection=sqlite3.connect(database)
    connection.execute('create table evidence(value blob)');connection.execute('insert into evidence values (?)',(b'actual'*1024,))
    connection.commit();connection.close();original=database.read_bytes()
    budget=root/'budget.sqlite';budget.write_bytes(b'never delete budget')
    checkpoints=root/'checkpoints';checkpoints.mkdir()
    a=checkpoints/'a.pt';b=checkpoints/'b.pt';c=checkpoints/'c.pt'
    a.write_bytes(b'identical complete state');b.write_bytes(a.read_bytes());c.write_bytes(b'different complete state')
    result=compact(root,apply=True)
    assert read_episode(episode_path)['decisions']==episode['decisions']
    document=json.loads(summary.read_text())
    assert 'episodes' not in document and document['source_episode_manifests'][0]['sha256']==file_sha256(episode_path)
    assert json.loads(session_summary.read_text())['evaluation_artifacts'][0]['sha256']==file_sha256(summary)
    assert gzip.open(result['cold_sqlite'][0]['archive'],'rb').read()==original
    assert not database.exists() and budget.read_bytes()==b'never delete budget'
    assert a.stat().st_ino==b.stat().st_ino and c.read_bytes()==b'different complete state'
