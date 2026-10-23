"""显式 RoboFFT 配置到 fake world 采集、学习、发布和恢复的完整 CPU 接线测试。

使用真实 WorldEnvironmentCollector、批量生成、UpperEnvironment 时钟/奖励、Buffer、
GAE、RoboFFT Actor、原 Critic 和原子 checkpoint；只用可计数后端替身代替 GMT/PhysX。
替身不证明真实跟踪、失败率或 GPU 性能。测试记录各阶段 CPU 耗时，输出仅在 tmp_path；
同时用 fail-on-call 禁止旧 Gaussian PPO/KL，验证独立 MC RNG、源码非门禁和 weights-only。
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from tests.closedloop.dppo.test_robofft import cpu_threads, settings, local_distributed
from tests.closedloop.dppo.test_env_adapter import adapter
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tracking_context import build_critic_context
from gem.closedloop.dppo.world_collector import WorldEnvironmentCollector
from gem.closedloop.dppo.world_flow import GenerationOperation
from gem.closedloop.dppo.memory_execution import DeviceRolloutBuffer
from gem.closedloop.dppo.buffer import RolloutBuffer
from gem.closedloop.dppo.runtime import freeze_buffer_targets
from gem.closedloop.dppo.returns import normalize_advantages_global
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.updater import critic_update
from gem.closedloop.dppo.training import _prepare_actor_rollout, _update_actor
from gem.closedloop.robofft import CHECKPOINT_SCHEMA
from gem.closedloop.dppo.training_checkpoint import (AsyncCheckpointWriter, immutable_cpu_snapshot,
    load_training_checkpoint, SCHEMA)
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.parallel_support import capture_local_rng, restore_local_rng
from gem.closedloop.dppo.training import _advance_training_iteration, _rollback_snapshot, _restore_rollback
from gem.closedloop.robofft.configuration import DisabledAnchor
from gem.closedloop.robofft.checkpoint import initialize_actor_weights_only
from gem.closedloop.dppo.reference_credit import ReferenceCreditLedger, validate_reward_intervals


def test_fake_world_config_to_resume(actor_factory,tmp_path,monkeypatch):
    config=configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_continuous_rolling_robofft.yaml')
    actor,batch=actor_factory(starts=(45,),motion_frames=60);_activate_branches(actor)
    # CPU 仅替换数值执行后端，真实生产配置仍严格要求 stable_tf32x3/GPU。
    policy=build_policy(actor,config['stage9'],dict(config['stage10']['precision'],linear_execution='native_gemm'))
    context=_conditions(batch)
    env,backend=adapter(tmp_path/'lane',elapsed=.26)
    env.config=config;env.policy=policy;env.rolling=True;env.decision_ticks=60
    env.policy_version=0;env.collector_env_slot=0
    env.sample={'dataset':'Mine','row':{'sample_id':'fake'}};env.music_start_frame=0
    from gem.closedloop.dppo.deployment_clock import clock_for_config
    env.deployment_clock=clock_for_config(config)
    env.reference_credit=ReferenceCreditLedger();env.reference_credit.reset('e')
    env.reference_credit.last_tick=600  # 后端替身从已完成的 1 秒 warmup 后开始。
    backend.MUTATIONS=set()
    class Builder:
        def build(self,snapshot,reservation,*args,**kwargs):
            out={k:v.clone() for k,v in context.items()}
            shift=snapshot['tick']/600-float(out['decision_time'][0])
            for key in ('decision_time','future_times','proprio_history_times'):out[key]+=shift
            out['proprio_history_times']=snapshot['tracking_history_ticks'].double()[None]/600
            out['proprio_history']=snapshot['tracking_history_values'][None,:,:48].clone()
            out['proprio_history_valid']=snapshot['tracking_history_valid'][None].clone()
            return out,dict(reservation,world_anchor=[0.,0.,0.])
        def build_many(self,snapshots,reservations,musics):
            results=[self.build(s,r,m) for s,r,m in zip(snapshots,reservations,musics)]
            return [r[0] for r in results],[r[1] for r in results]
    env.builder=Builder()
    class World:
        def __init__(self):
            self.commits=[(0,'bootstrap:fake')];self.pending=None
        def reply(self,request):
            result=backend.call(request['method'],**request['payload'])
            if request['method']=='commit_plan':
                self.commits.append((backend.tick,self.pending));result['plan_id']=self.pending
                result['effective_tick']=backend.tick
            if request['method']=='advance':
                def plan(t):return next((p for tick,p in reversed(self.commits) if t>=tick),'bootstrap:fake')
                for row in result['trace']:
                    ticks=list(range(row['control_tick_begin']-120,row['control_tick_begin']+121,12))
                    row.update(consumed_reference_ticks=ticks,consumed_plan_ids=[plan(t) for t in ticks],
                        reference_tick=row['tick'],reference_plan_id=plan(row['tick']))
            return result
        def call(self,method,**kwargs):
            if method=='exchange':
                return dict(replies=[dict(request_id=r['request_id'],ok=True,
                    result=self.reply(r)) for r in kwargs['requests']])
            return dict(world_window=dict(reached=method=='end_rollout',requested_world_controls=30))
    collector=WorldEnvironmentCollector(policy,None,World(),num_envs=1,generation_batch=1,training_storage='ephemeral_device')
    collector.states=[NS(resource=NS(env=env),transitions=0)]
    def flow(slot,version):
        result=[]
        for i in range(2):
            reservation=dict(request_id=f'fake{i}',parent_plan_id=collector.world.commits[-1][1],env_id=0,episode_id='e',
                decision_tick=backend.tick,deadline_tick=backend.tick+600,prefix_frames=6,protected_end_tick=backend.tick)
            generated,timing=yield GenerationOperation(dict(builder=env.builder,snapshot=env.snapshot,
                reservation=reservation,music=env.music,seed=100+i,decision=i))
            generated.update(prepared=dict(prepared_plan_id='ticket'),rejection=None,elapsed=.26,
                seed=100+i,raw_path=None,timing=timing)
            collector.world.pending=generated['meta']['plan_id']
            def ready(**kwargs):
                if False:yield
                return generated
            env.generate_flow=ready
            row=yield from env.step_flow()
            result.append(row);collector.states[0].transitions+=1
        result[-1].truncated=True
        return result
    collector._window_flow=flow
    def forbidden(*args,**kwargs):pytest.fail('Gaussian PPO/KL entered RoboFFT pipeline')
    for target in ('gem.closedloop.dppo.policy.masked_joint_log_prob',
        'gem.closedloop.dppo.training.actor_update','gem.closedloop.dppo.training.final_kl',
        'gem.closedloop.dppo.training.probability_check_local',
        'gem.closedloop.dppo.chain_ppo.evaluate_chain_records',
        'gem.closedloop.dppo.actor_backtracking.backtracked_step'):
        monkeypatch.setattr(target,forbidden)
    start=time.perf_counter();fragments,report=collector.collect(world_seconds=.6,policy_version=0)
    rollout_seconds=time.perf_counter()-start
    buffer=DeviceRolloutBuffer(fragments,policy,'cpu');rows=buffer.transitions
    assert len(rows)==2 and all(r.chain is None and r.action_x0.shape==(60,30) for r in rows)
    assert sum(r.executed_control_steps for r in rows)==sum(env.budget.actual)==30
    assert backend.tick==960 and all(r.identity['policy_version']==0 for r in rows)
    validate_reward_intervals(rows)
    for row in rows:row.validate()
    durable=RolloutBuffer(2)
    for row in rows:durable.append(row)
    critic=PretrainedConditionCritic(actor,dimension=32,layers=1,heads=4)
    targets=freeze_buffer_targets(buffer,buffer.fragments,critic,'cpu',critic_version=0,batch_size=8,
        gamma_upper=config['stage9']['gamma_upper'],lambda_upper=config['stage9']['lambda_upper'])
    targets=normalize_advantages_global(targets)
    assert targets['advantages_normalized'] and targets['advantage_global_count']==2
    cache=RolloutTensorCache(rows,targets,'cpu');d=local_distributed()
    manifest=[dict(valid=True,has_free=True,owner_rank=0,local_index=i) for i in range(2)]
    optimizer=make_actor_optimizer(actor,config['stage9'])
    value_optimizer=torch.optim.AdamW(critic.parameters(),lr=config['stage10']['critic']['learning_rate'])
    rngs={k:torch.Generator().manual_seed(seed) for k,seed in [('actor',11),('critic',22),('robofft_mc',33)]}
    controller=BCGradientController();anchor=DisabledAnchor()
    state=dict(iteration=0,rollouts_completed=0,policy_version=0,actor_updates=0,critic_updates=0)
    c=NS(actor=actor,critic=critic,actor_optimizer=optimizer,critic_optimizer=value_optimizer,
         controller=controller,bc=anchor,generators=rngs,state=state,robofft=True,policy=policy,
         settings=config['stage9'],stage=config['stage10'],distributed=d)
    original=_rollback_snapshot(c)
    actor_start=time.perf_counter()
    initial=_prepare_actor_rollout(c,rows,cache,manifest,args=NS(collect_only=False))
    assert initial['ratio']==pytest.approx(1.,abs=2e-6) and initial['surrogate_kl']<1e-12
    fixed=copy.deepcopy([r.mc_record for r in rows])
    actor_report=_update_actor(c,rows,targets,cache,manifest,label='fake',step_observer=lambda step:None)
    actor_seconds=time.perf_counter()-actor_start
    critic_start=time.perf_counter()
    value_report=critic_update(critic,value_optimizer,rows,targets,cache,manifest,distributed=d,
        generator=rngs['critic'],epochs=1,batch=3)
    critic_seconds=time.perf_counter()-critic_start
    from gem.closedloop.dppo.diagnostic_rollout import persist_rollout,load_rollout
    stored=persist_rollout(tmp_path/'compact_rollout',rows,targets,manifest,
        kernel=policy.kernel_config,identity={},full_execution=False)
    replay=load_rollout(tmp_path/'compact_rollout')
    assert stored['algorithms']==['robofft_d.v1'] and replay['transitions'][0].chain is None
    torch.testing.assert_close(replay['transitions'][0].mc_record.eps,rows[0].mc_record.eps,atol=0,rtol=0)
    assert backend.tick==960  # 所有优化期间物理暂停，没有重置连续世界。
    _advance_training_iteration(state,actor_report);state['rollouts_completed']=1;state['critic_updates']=value_report['optimizer_steps']
    assert state['policy_version']==1 and all(r.identity['policy_version']==0 for r in rows)
    identity=dict(training=config['stage9'],assets=config['stage10']['assets'],kernel=policy.kernel_config,
                  source_files={'fixture':'old-source'},actor_initialization={'kind':'explicit_weights_only'})
    payload=immutable_cpu_snapshot(dict(schema=CHECKPOINT_SCHEMA,identity=identity,config=config,
        actor=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=optimizer.state_dict(),
        critic_optimizer=value_optimizer.state_dict(),bc_controller=controller.state_dict(),state=state,
        rank_states=[dict(rank=0,rng=capture_local_rng(rngs),bc=anchor.state_dict())],budget={},actor_fingerprint='cpu-fixture'))
    writer=AsyncCheckpointWriter(tmp_path/'run');writer.submit(payload,reason='test_complete')
    publication=writer.poll(wait=True);writer.close();checkpoint=tmp_path/'run'/publication['path']
    assert json.loads((tmp_path/'run/latest.json').read_text())['sha256']==publication['sha256']
    expected_mc=policy.draw_mc(cache.get('action_x0',rows),rngs['robofft_mc'])
    assert _restore_rollback(c,original)['identical']
    changed_source=dict(identity,source_files={'fixture':'different-source'},actor_initialization={'kind':'source_stage1'})
    loaded=load_training_checkpoint(checkpoint,identity=changed_source,actor=actor,critic=critic,
        actor_optimizer=optimizer,critic_optimizer=value_optimizer,controller=controller)
    restore_local_rng(loaded['rank_states'][0]['rng'],rngs)
    actual_mc=policy.draw_mc(cache.get('action_x0',rows),rngs['robofft_mc'])
    assert torch.equal(expected_mc['eps'],actual_mc['eps'])
    for k,v in payload['actor'].items():assert torch.equal(actor.state_dict()[k],v)
    assert optimizer.state_dict()['state'] and value_optimizer.state_dict()['state']
    broken=copy.deepcopy(payload);broken['rank_states'][0]['rng']['generators'].pop('robofft_mc')
    broken_path=tmp_path/'missing_mc.pt';torch.save(broken,broken_path)
    with pytest.raises(ValueError,match='complete independent MC RNG'):
        load_training_checkpoint(broken_path,identity=identity,actor=actor,critic=critic,
            actor_optimizer=optimizer,critic_optimizer=value_optimizer,controller=controller)
    for row,saved in zip(rows,fixed):assert torch.equal(row.mc_record.old_loss,saved.old_loss)
    dp=copy.deepcopy(payload);dp['schema']=SCHEMA;dp['identity']['training']['actor_objective']='stepwise_coordinate_mean_clipped_ppo'
    dp_path=tmp_path/'old_dppo.pt';torch.save(dp,dp_path)
    with pytest.raises(ValueError,match='schema or actual'):
        load_training_checkpoint(dp_path,identity=identity,actor=actor,critic=critic,
            actor_optimizer=optimizer,critic_optimizer=value_optimizer,controller=controller)
    record=initialize_actor_weights_only(dp_path,actor,assets=config['stage10']['assets'])
    assert not record['optimizer_restored'] and not record['rollout_reused']
    print('ROBOFFT_FAKE_TIMING '+json.dumps(dict(rollout_seconds=rollout_seconds,actor_seconds=actor_seconds,
        critic_seconds=critic_seconds,total_seconds=time.perf_counter()-start,peak_cuda_memory_bytes=None,
        scope='CPU_small_network_fake_world_not_PhysX',reference_change=actor_report['reference_change'])))
    cache.close();buffer.clear();durable.clear()


def test_reference_audit_keeps_versions_and_separate_visibility_consumption_reward():
    ledger=ReferenceCreditLedger();ledger.reset('e')
    ledger.register('old',decision_tick=0,policy_version=0,parent_plan_id=None,horizon_ticks=1200)
    ledger.committed('old',0)
    ledger.register('new',decision_tick=0,policy_version=1,parent_plan_id='old',horizon_ticks=1200)
    ledger.committed('new',0)
    for begin in range(0,72,12):
        ticks=list(range(begin-120,begin+121,12))
        plans=['new' if t>=60 else 'old' for t in ticks]
        ledger.observe([dict(episode_id='e',control_tick_begin=begin,tick=begin+12,
            consumed_reference_ticks=ticks,consumed_plan_ids=plans,reference_tick=begin+12,
            reference_plan_id='new' if begin+12>=60 else 'old')])
    assert ledger.records['new']['first_window_observed_tick']==0
    assert ledger.records['new']['first_current_consumed_tick']==60
    assert ledger.records['new']['first_reward_reference_tick']==60
    assert ledger.records['old']['policy_version']==0
