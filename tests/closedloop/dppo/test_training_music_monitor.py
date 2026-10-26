"""训练集固定音乐监视、恢复和Stage1补测的独立CPU回归。

用四来源合成清单核验当前train与原train交集、16首两种子、完整分片和只读评估，
拒绝把MC监督或没有原始归属的样本混入质量报告。用真实小型Adam恢复验证仅修改
评估配置不会丢失动量和训练轮次，奖励或概率核变化仍被拒绝。最后在临时目录调用
真实补测协调器，以物理评估替身检查冻结Stage1视图、旧基线存档、指针发布和失败
时保留原证据。这里不调用Isaac、不启动训练，不把执行替身结果当作物理收益。
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from gem.closedloop.dppo.evaluation import build_evaluation_tasks, evaluate_policy
from gem.closedloop.dppo.full_dataset import SOURCES
from gem.closedloop.dppo.periodic_monitor import build_balanced_plan, shard_plan, merge_periodic_reports
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.training_checkpoint import SCHEMA, load_training_checkpoint
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.evaluation_rebaseline import ensure_training_progress_baseline, _stage1_context
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from tests.test_evaluation import Env, policy
from tests.closedloop.dppo.test_periodic_monitor import Catalog


def train_catalog(tmp_path):
    catalog=Catalog(tmp_path)
    catalog.samples['train']=copy.deepcopy(catalog.samples['val'])
    for samples in catalog.samples['train'].values():
        for index,sample in enumerate(samples):
            sample['split']=sample['row']['split']='train'
            sample['row']['resplit_provenance']={'original_split':'train' if index<4 else 'test'}
    return catalog


def train_plan(catalog):
    return build_balanced_plan(catalog,split='train',training_progress=True,original_split='train',
        samples_per_source=4,unique_music=True,seconds=None)


def test_original_train_selection_has_32_fixed_read_only_tasks(tmp_path):
    catalog=train_catalog(tmp_path)
    plan=train_plan(catalog)
    assert plan==train_plan(catalog)
    assert plan['task_count']==32 and plan['selected_unique_audio_count']==16
    assert plan['evaluation_purpose']=='training_progress' and plan['training_buffer_allowed'] is False
    assert not plan.get('critic_mc_supervision') and plan['episode_seconds'] is None
    assert all(t['split']=='train' and t['sample']['row']['resplit_provenance']['original_split']=='train'
        and t['music_start_frame']==0 for t in plan['tasks'])
    loaded=policy();paths=[]
    for rank in range(8):
        root=tmp_path/f'rank{rank}'
        evaluate_policy(Env(loaded),loaded,shard_plan(plan,rank,8),root,catalog=catalog)
        paths.append(root/'report.json')
    result=merge_periodic_reports(plan,paths)
    assert len(result['episode_metrics'])==32
    assert merge_periodic_reports(plan,paths,baseline=result)['paired_baseline']['new_failures']==0


def test_train_monitor_does_not_implicitly_allow_mc_or_missing_original_provenance(tmp_path):
    catalog=train_catalog(tmp_path)
    with pytest.raises(ValueError):build_evaluation_tasks(catalog,split='train')
    with pytest.raises(ValueError):
        build_evaluation_tasks(catalog,split='train',training_progress=True,critic_mc=True)
    catalog.samples['train'][SOURCES[0]][0]['row'].pop('resplit_provenance')
    with pytest.raises(ValueError,match='Insufficient'):train_plan(catalog)


def test_formal_configuration_explicitly_uses_original_and_current_train():
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    stage=configuration(root/'configs/closedloop/stage2_transformer_groups.yaml')['stage10']
    assert stage['evaluation']['split']==stage['evaluation']['original_split']=='train'
    assert stage['evaluation']['purpose']=='training_progress'


def test_stage1_view_matches_real_frozen_twenty_step_sampler_without_changing_actor(actor_factory,tmp_path):
    from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    loaded=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    with torch.no_grad():next(actor.parameters()).add_(.03)
    before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    c=NS(actor=actor,policy=loaded,state=dict(iteration=10,actor_updates=39,policy_version=10),writer=object())
    view=_stage1_context(c,tmp_path)
    with torch.no_grad():
        actual=view.policy.sample_rollout(_conditions(batch),generator=torch.Generator().manual_seed(42))
        expected=loaded.frozen_policy.sample_rollout(_conditions(batch),generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(actual['chain'],expected['chain'],rtol=0,atol=0)
    for key,value in actor.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    loaded.assert_frozen_identity()


@pytest.mark.parametrize('change',['evaluation','kernel','reward'])
def test_monitor_change_retains_adam_and_progress_but_not_old_best(tmp_path,change):
    actor,critic=torch.nn.Linear(2,2),torch.nn.Linear(2,1)
    ao,co=torch.optim.AdamW(actor.parameters()),torch.optim.AdamW(critic.parameters())
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(2,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    controller=BCGradientController()
    old=dict(kernel={'eta':.1},reward={'weight':2.5},evaluation={'split':'val'})
    payload=copy.deepcopy(dict(schema=SCHEMA,identity=old,actor=actor.state_dict(),critic=critic.state_dict(),
        actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),bc_controller=controller.state_dict(),
        state={'iteration':10,'actor_updates':39,'best_saved':{'old_val':True},
               'regression_monitor':{'iteration':10,'consecutive':2}}))
    path=tmp_path/'saved.pt';torch.save(payload,path)
    target=copy.deepcopy(old)
    target[change]=dict(split='train') if change=='evaluation' else {'changed':True}
    ao.state.clear()
    if change!='evaluation':
        with pytest.raises(ValueError):load_training_checkpoint(path,identity=target,actor=actor,critic=critic,
            actor_optimizer=ao,critic_optimizer=co,controller=controller)
        assert not ao.state
    else:
        loaded=load_training_checkpoint(path,identity=target,actor=actor,critic=critic,
            actor_optimizer=ao,critic_optimizer=co,controller=controller)
        assert loaded['state']=={'iteration':10,'actor_updates':39}
        assert ao.state and all(s['step']==1 for s in ao.state.values())
        torch.testing.assert_close(actor.weight,payload['actor']['weight'])
        assert torch.load(path,weights_only=False)['state']['best_saved']=={'old_val':True}


@pytest.mark.parametrize('fail',[False,True])
@pytest.mark.parametrize('physics_change',[False,True])
def test_rebaseline_archives_only_after_real_stage1_evaluation_and_preserves_training(tmp_path,monkeypatch,fail,physics_change):
    catalog=train_catalog(tmp_path);old=train_plan(catalog) if physics_change else build_balanced_plan(catalog)
    prior={'plan_sha256':old['plan_sha256'],'old_validation':True}
    if physics_change:prior['evaluation_identity']={'physics':{'evaluation_envs_per_rank':2}}
    (tmp_path/'evaluation_baseline.json').write_text(json.dumps(prior))
    (tmp_path/'fixed_diagnostic_reference.json').write_text(json.dumps({'old_reference':True}))
    (tmp_path/'best_saved.json').write_text(json.dumps({'old_best':True}))
    actor=torch.nn.Linear(2,2);frozen=copy.deepcopy(actor).requires_grad_(False)
    actor(torch.ones(1,2)).sum().backward();before=actor.weight.detach().clone();gradient=actor.weight.grad.clone()
    options=dict(split='train',purpose='training_progress',original_split='train',samples_per_source=4,
        seeds=[42,1729],episode_seconds=None,selection_seed=42,unique_music=True,envs_per_rank=4 if physics_change else 1)
    state=dict(iteration=10,actor_updates=39,policy_version=10,best_saved={'old_val':True},
        regression_monitor={'iteration':10,'consecutive':2})
    c=NS(output=tmp_path,session=tmp_path/'session',catalog=catalog,stage={'evaluation':options},state=state,
        actor=actor,policy=NS(actor=actor,frozen_policy=NS(actor=frozen),_phase_signature=None),
        writer=object(),generators={},identity={'source_stage1_step':595000},
        distributed=NS(rank=0,broadcast_object=lambda x:x),budget=NS(state_dict=lambda:{'used':123}))
    def evaluate(view,label,**kwargs):
        assert view.actor is frozen and view.policy.actor is frozen and c.policy.actor is actor
        assert view.state['iteration']==0 and view.writer is None
        assert c.state['iteration']==10 and c.state['actor_updates']==39
        torch.rand(5)  # 评估世界启动和执行即使消耗全局随机数，也必须恢复训练状态。
        if fail:raise RuntimeError('physics unavailable')
        plan=train_plan(catalog)
        (view.output/'fixed_diagnostic_reference.json').write_text(json.dumps({'plan_sha256':plan['plan_sha256']}))
        return {'plan_sha256':plan['plan_sha256'],'actor_identity':{'source':'Stage1'},
                'evaluation_identity':{'physics':{'evaluation_envs_per_rank':options['envs_per_rank']}}}
    monkeypatch.setattr('gem.closedloop.dppo.evaluation_runtime._evaluate',evaluate)
    training_rng=torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError,match='physics unavailable'):ensure_training_progress_baseline(c)
        assert json.loads((tmp_path/'evaluation_baseline.json').read_text())==prior
        assert (tmp_path/'best_saved.json').exists()
    else:
        change=ensure_training_progress_baseline(c)
        assert change['iteration']==10 and not change['optimizer_reset']
        assert json.loads((tmp_path/'evaluation_baseline.json').read_text())['baseline_provenance']['source_stage1_step']==595000
        suffix='_slots2' if physics_change else ''
        assert json.loads((tmp_path/'evaluation_protocols'/('previous_'+old['plan_sha256']+suffix)/'evaluation_baseline.json').read_text())==prior
        assert bool(state.get('requires_evaluation_protocol_start')) is physics_change
        if physics_change:assert change['reason']=='evaluation_physics_slots_change'
        assert not (tmp_path/'best_saved.json').exists() and 'best_saved' not in state
        assert 'regression_monitor' not in state
        assert ensure_training_progress_baseline(c) is None
    assert c.state['iteration']==10 and c.state['actor_updates']==39
    torch.testing.assert_close(actor.weight,before);torch.testing.assert_close(actor.weight.grad,gradient)
    assert torch.equal(torch.get_rng_state(),training_rng)


def _distributed_rebaseline(rank,path):
    from pathlib import Path
    import torch.distributed as dist
    import gem.closedloop.dppo.evaluation_runtime as runtime
    path=Path(path)
    dist.init_process_group('gloo',init_method='file://'+str(path/'collective'),rank=rank,world_size=2)
    try:
        def broadcast(value):
            objects=[value];dist.broadcast_object_list(objects,src=0);return objects[0]
        catalog=train_catalog(path);actor=torch.nn.Linear(2,2);frozen=copy.deepcopy(actor).requires_grad_(False)
        options=dict(split='train',purpose='training_progress',original_split='train',samples_per_source=4,
            seeds=[42,1729],episode_seconds=None,selection_seed=42,unique_music=True)
        c=NS(output=path,session=path/'parallel_session',catalog=catalog,stage={'evaluation':options},
            state=dict(iteration=10,actor_updates=39,policy_version=10),actor=actor,
            policy=NS(actor=actor,frozen_policy=NS(actor=frozen)),writer=None,generators={},identity={},
            distributed=NS(rank=rank,broadcast_object=broadcast),
            budget=NS(state_dict=lambda:{'actual_controls':4321}) if rank==0 else None)
        def evaluate(view,label,**kwargs):
            plan=train_plan(catalog)
            if rank==0:
                (view.output/'fixed_diagnostic_reference.json').write_text(json.dumps({'plan_sha256':plan['plan_sha256']}))
            return {'plan_sha256':plan['plan_sha256']}
        runtime._evaluate=evaluate
        change=ensure_training_progress_baseline(c)
        assert change['task_count']==32 and c.state['budget']=={'actual_controls':4321}
        assert c.state['iteration']==10 and c.policy.actor is actor
        (path/f'rank{rank}.passed').write_text('passed')
    finally:dist.destroy_process_group()


def test_rebaseline_broadcasts_root_only_budget_without_rank_local_manager(tmp_path):
    import torch.multiprocessing as mp
    catalog=train_catalog(tmp_path);old=build_balanced_plan(catalog)
    (tmp_path/'evaluation_baseline.json').write_text(json.dumps({'plan_sha256':old['plan_sha256']}))
    mp.start_processes(_distributed_rebaseline,args=(str(tmp_path),),nprocs=2,join=True,start_method='fork')
    assert all((tmp_path/f'rank{rank}.passed').exists() for rank in range(2))
