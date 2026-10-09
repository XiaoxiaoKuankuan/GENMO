"""GPU多环境有限验收使用的原DPPO更新器桥接，不是长期训练入口。

本模块直接调用parallel_training._update，保留原Critic80步、BC、固定Actor学习率、
多minibatch PPO、完整KL验收和整轮回滚，不复制或简化强化学习公式。输入是多环境
真实采集的UpperTransition；先固定旧/下一价值，再按环境片段计算GAE，全局只归一化
一次。共享模型通过NCCL分桶SUM同步，旧rollout在更新期间不改写。

有限验收允许强制保存完整恢复点，包含八rank的环境计数/音乐游标、BC、RNG以及
两个Adam状态。恢复显式重建物理episode，不冒充保存了PhysX内部状态。checkpoint
身份明确为validation.v1，不能当作正式CPU旧run或未来生产多环境run的隐式恢复。
预算用于限制有限优化尝试，采集预算仍由所属环境持久记录；生产全局资源租约、
周期评估和稀疏归档必须在正式训练集成层另行完成并验证。
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .buffer import RolloutBuffer
from .checkpoint import VERSION_V2,capture_rank_state,save_checkpoint,load_checkpoint
from .critic import UpperCritic
from .dual_collector import fixed_fragment_targets
from .parallel_training import _update,_synchronize_models
from .parallel_support import local_call,root_call,build_global_manifest
from .returns import normalize_advantages_global
from .trainer import SupervisedAnchor,populate_values,trainable_actor_parameters
from .run_management import TrainingBudget


class FiniteVectorLearner:
    def __init__(self,actor,policy,train_config,config,distributed,output,*,max_iterations,num_envs):
        self.output=Path(output)
        c=self.c=SimpleNamespace(actor=actor,policy=policy,config=config,settings=config['stage9'],
            stage=config['stage10'],distributed=distributed,session=self.output,initial_iteration=0,
            profile=dict(microbatch=32,cfg_batch=True))
        c.critic=UpperCritic(qpos_mean=actor.endecoder.mean,qpos_std=actor.endecoder.std,
            proprio_scales=tuple(train_config.model.proprio_scales)).to(distributed.device)
        distributed.broadcast_module(c.actor);distributed.broadcast_module(c.critic)
        c.actor_optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=c.settings['actor_lr'],weight_decay=0.)
        c.critic_optimizer=torch.optim.AdamW(c.critic.parameters(),lr=c.settings['critic_lr'],weight_decay=0.)
        c.bc=local_call(distributed,lambda:SupervisedAnchor(config,actor,train_config) if distributed.rank==0 else None)
        c.generators=dict(actor=torch.Generator().manual_seed(config['stage10']['seed']+3003),
            critic=torch.Generator().manual_seed(config['stage10']['seed']+2002))
        c.state=dict(iteration=0,policy_version=0,actor_updates=0,critic_updates=0,buffer_size=0,pending_plan=False)
        limits=dict(accepted_iterations=max_iterations,optimizer_attempts=max_iterations*c.settings['max_actor_optimizer_steps'],
            generations=1,control_steps=1,physics_steps=4)
        c.budget=TrainingBudget(self.output/'learning_budget.json',limits) if distributed.rank==0 else None
        paths=config['paths']
        def digest(path):
            h=hashlib.sha256()
            with Path(path).open('rb') as f:
                for part in iter(lambda:f.read(1024**2),b''):h.update(part)
            return h.hexdigest()
        self.identity=root_call(distributed,lambda:dict(schema='genmo.gpu_vectorized.validation.v1',
            num_envs_per_rank=num_envs,world_size=8,transitions_per_rank=20,kernel=copy.deepcopy(policy.kernel_config),
            prefix_deadline_contract=config['runtime']['prefix_deadline_contract'],
            vector_audit_contract=config['runtime']['vector_audit_contract'],
            gmt_policy_sha256=digest(paths['gmt_policy']),configuration=copy.deepcopy(config['stage10']['training']),
            run_id=config['stage9']['run_id'],scope='finite_validation_not_production_resume'))
        self.collector=None
        distributed.enable_gradient_timing()

    @property
    def iteration(self):return self.c.state['iteration']

    @property
    def policy_version(self):return self.c.state['policy_version']

    def update(self,fragments):
        c=self.c
        buffer=RolloutBuffer(20)
        lengths=list(map(len,fragments))
        for fragment in fragments:
            for row in fragment:buffer.append(row)
        if len(buffer)!=20:raise ValueError('Finite vector validation keeps 20 transitions per rank')
        rows=buffer.transitions
        local_call(c.distributed,lambda:populate_values(rows,c.critic,c.distributed.device,
            critic_version=c.state['critic_updates'],batch_size=32))
        start=0;fixed=[]
        for count in lengths:fixed.append(rows[start:start+count]);start+=count
        _,local_targets=local_call(c.distributed,lambda:fixed_fragment_targets(fixed,c.critic,c.distributed.device,
            critic_version=c.state['critic_updates'],gamma_upper=c.settings['gamma_upper'],lambda_upper=c.settings['lambda_upper']))
        targets=normalize_advantages_global(local_targets,distributed=c.distributed)
        manifest=build_global_manifest(rows,c.distributed)
        if len(manifest)!=160:raise ValueError('Global vector rollout must remain 160')
        result=_update(c,buffer,targets,manifest,c.state['iteration']+1)
        c.state['iteration']+=1;c.state['policy_version']+=1
        c.state['actor_updates']+=result['actor']['optimizer_steps'];c.state['critic_updates']+=c.settings['critic_steps']
        root_call(c.distributed,lambda:c.budget.accept_iteration(identity=f'finite:{c.state["iteration"]}'))
        result['replicas']=_synchronize_models(c,full=True,phase='finite_vector_acceptance')
        return result

    def reject_for_validation(self, fragments):
        """仅有限测试：先实际完成更新和全量KL，再注入超限，核验整轮精确恢复。"""
        from . import parallel_training
        c=self.c
        if self.iteration<1 or not c.actor_optimizer.state or not c.critic_optimizer.state:
            raise ValueError('Fault acceptance requires a prior accepted round and nonempty Adam states')
        before=copy.deepcopy(c.state)
        spent=root_call(c.distributed,lambda:c.budget.state_dict()['used']['optimizer_attempts'])
        original=parallel_training.check_kl_limits
        observed={}
        def reject(kl, settings):
            observed['actual_full_kl']=copy.deepcopy(kl)
            changed=dict(kl,mean_joint_kl=settings['kl_stop_joint']+.001)
            observed['injected_mean_joint_kl']=changed['mean_joint_kl']
            original(changed,settings)
            raise AssertionError('Injected hard KL rejection failed')
        parallel_training.check_kl_limits=reject
        try:
            try:self.update(fragments)
            except RuntimeError as error:
                if not observed or 'Final whole-rollout KL rejected' not in str(error):raise
            else:raise AssertionError('Rejected iteration was incorrectly accepted')
        finally:parallel_training.check_kl_limits=original
        if c.state!=before:raise AssertionError('Rejected round changed accepted counters')
        rejected=root_call(c.distributed,lambda:json.loads((c.session/f'rejected_{self.iteration+1:06d}.json').read_text()))
        after=root_call(c.distributed,lambda:c.budget.state_dict()['used']['optimizer_attempts'])
        if not rejected['rolled_back'] or after<=spent:
            raise AssertionError('Rollback must restore all bytes without refunding optimization attempts')
        replicas=_synchronize_models(c,full=True,phase='finite_injected_KL_rollback')
        return dict(status='passed_expected_fault',injection=observed,
            accepted_counters_unchanged=True,charged_attempts=after-spent,
            exact_recovery_by_rank=rejected['rollback_audit_by_rank'],replicas=replicas,
            checkpoint_published=False)

    def save(self,collector,path):
        c=self.c
        local_state=local_call(c.distributed,lambda:dict(c.state,vector_collector=collector.state_dict()))
        samplers={} if c.bc is None else {'bc':c.bc}
        rank_state=capture_rank_state(c.distributed.rank,state=local_state,samplers=samplers,generators=c.generators)
        ranks=c.distributed.all_gather_object(rank_state)
        c.state['budget']=root_call(c.distributed,lambda:c.budget.state_dict())
        root_call(c.distributed,lambda:save_checkpoint(path,actor=c.actor,critic=c.critic,
            actor_optimizer=c.actor_optimizer,critic_optimizer=c.critic_optimizer,state=c.state,
            identity=self.identity,config=c.config,version=VERSION_V2,rank_states=ranks))
        return root_call(c.distributed,lambda:dict(path=str(path),size_bytes=Path(path).stat().st_size,
            sha256=self._sha(path),iteration=c.state['iteration']))

    @staticmethod
    def _sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for part in iter(lambda:f.read(1024**2),b''):h.update(part)
        return h.hexdigest()

    def restore(self,path,collector):
        c=self.c
        def verify_boundary():
            previous=json.loads((Path(path).parent/'report.json').read_text())
            if previous.get('status')!='passed':raise ValueError('Finite restore requires a normally completed validation session')
            record=previous['ranks'][0]['rounds'][-1]['checkpoint']
            if Path(record['path']).resolve()!=Path(path).resolve() or self._sha(path)!=record['sha256']:
                raise ValueError('Finite checkpoint path or exact bytes differ from completed report')
            return True
        root_call(c.distributed,verify_boundary)
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        expected=dict(self.identity,run_id=saved['identity'].get('run_id'))
        if saved['identity']!=expected:raise ValueError('Finite vector checkpoint contract mismatch')
        self.identity=expected
        config=c.config;config['stage9']['run_id']=expected['run_id']
        c.state=local_call(c.distributed,lambda:load_checkpoint(path,actor=c.actor,critic=c.critic,
            actor_optimizer=c.actor_optimizer,critic_optimizer=c.critic_optimizer,identity=self.identity,
            samplers={} if c.bc is None else {'bc':c.bc},generators=c.generators,rank=c.distributed.rank,world_size=8))
        record=c.state.pop('local_rank_state')
        local_call(c.distributed,lambda:collector.load_state_dict(record['vector_collector']))
        # 测试恢复只接受正常保存边界；真实故障尾部预算需要生产RunManager负责核验。
        def restore_budget():
            live=saved['state']['budget']
            if c.budget.limits!=live['limits']:raise ValueError('Finite resume cannot change budget limits')
            c.budget._save(copy.deepcopy(live));c.budget.state=copy.deepcopy(live)
        root_call(c.distributed,restore_budget)
        c.initial_iteration=c.state['iteration']
        return _synchronize_models(c,full=True,phase='finite_vector_resume')
