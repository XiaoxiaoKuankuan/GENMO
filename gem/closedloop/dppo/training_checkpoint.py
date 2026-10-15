"""Stage10 完整一致快照、单在途异步保存与严格新格式恢复。

调用方只在完成rollout验收后的边界冻结Actor、Critic、两个Adam、BC控制器、
调度状态、各rank RNG和采样游标为独立CPU快照。后台线程只处理这些不可变CPU
对象，不访问活动GPU模型、不修改训练RNG。最多一个在途保存，队列满显式等待
并报告背压；发布之前校验大小、SHA、fsync和原子文件，latest只指向完整文件。

当前 checkpoint具有独立schema与训练身份，旧Stage9/Stage10格式禁止透明恢复。
恢复重建PhysX并重新采样，不携带当前rollout或中途仿真状态。模型/优化器状态
在正式写入前固定，历史checkpoint不覆盖。best/重要评估点保存实际完整权重。
保存异常在下一个安全边界传播并受控停止，不能在后台失败后继续宣称已持久化。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import time
import uuid
import torch

from .budget import atomic_json

SCHEMA='genmo.closedloop.stage2.asymmetric_tracking.checkpoint.v1'


def immutable_cpu_snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:immutable_cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [immutable_cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(immutable_cpu_snapshot(v) for v in value)
    import copy
    return copy.deepcopy(value)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):digest.update(b)
    return digest.hexdigest()


class AsyncCheckpointWriter:
    def __init__(self,root,*,keep_last=2):
        self.root=Path(root);self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='checkpoint')
        self.pending=None;self.telemetry=[];self.closed=False
        if type(keep_last)is not int or keep_last<1:raise ValueError('Positive complete periodic retention required')
        self.keep_last=keep_last
        (self.root/'checkpoints').mkdir(parents=True,exist_ok=True)

    def poll(self,*,wait=False):
        if self.pending is None:return None
        if not wait and not self.pending.done():return None
        start=time.perf_counter();result=self.pending.result();self.pending=None
        result['foreground_wait_seconds']=time.perf_counter()-start
        self.telemetry.append(result)
        return result

    def submit(self,payload,*,reason):
        if self.closed or payload.get('schema')!=SCHEMA:raise ValueError('Invalid async checkpoint state')
        wait=self.poll(wait=True)
        if any(v.is_cuda for v in payload['actor'].values()):raise ValueError('Async writer accepts only immutable CPU model snapshots')
        self.pending=self.pool.submit(self._write,payload,reason)
        return dict(previous_backpressure=wait,queued=True,maximum_inflight=1)

    def _write(self,payload,reason):
        start=time.perf_counter();state=payload['state']
        # 恢复后可以在同一完成边界再次受控退出；每次发布独立身份，绝不覆盖旧权重。
        publication_id=f'{time.time_ns():020d}_{uuid.uuid4().hex[:12]}'
        name=f'stage10_r{state["rollouts_completed"]:06d}_a{state["actor_updates"]:06d}_{publication_id}_{reason}.pt'
        path=self.root/'checkpoints'/name;tmp=path.with_suffix('.pt.partial')
        if path.exists() or tmp.exists():raise FileExistsError(path)
        try:
            with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            sha=file_sha256(tmp);size=tmp.stat().st_size
            os.link(tmp,path);tmp.unlink()
            fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            descriptor=dict(schema='genmo.stage10.checkpoint_publication',iteration=state['iteration'],
                rollouts_completed=state['rollouts_completed'],path=str(path.relative_to(self.root)),
                sha256=sha,size_bytes=size,publication_id=publication_id,metadata=dict(reason=reason,actor_updates=state['actor_updates'],
                    critic_updates=state['critic_updates'],actor_model_fingerprint=payload['actor_fingerprint']))
            atomic_json(self.root/'latest.json',descriptor)
            if reason=='best':atomic_json(self.root/'best_checkpoint.json',descriptor)
            # 只清理当前运行产生的普通周期文件；初始、评估、best和终态不在此集合。
            candidates=sorted((self.root/'checkpoints').glob('stage10_r*_a*_periodic.pt'))
            for expired in candidates[:-self.keep_last]:
                if expired!=path:expired.unlink()
            return dict(**descriptor,background_write_seconds=time.perf_counter()-start)
        except BaseException:
            tmp.unlink(missing_ok=True);raise

    def close(self):
        try:return self.poll(wait=True)
        finally:self.pool.shutdown(wait=True);self.closed=True


def load_training_checkpoint(path,*,identity,actor,critic,actor_optimizer,critic_optimizer,controller):
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('schema')!=SCHEMA or payload.get('identity')!=identity:
        raise ValueError('Checkpoint schema/algorithm/kernel/asset/runtime identity differs; new experiment required')
    from .checkpoint import _validate_model_state,_validate_optimizer_state
    _validate_model_state(actor,payload['actor'],'当前 Actor')
    _validate_model_state(critic,payload['critic'],'当前 Critic')
    _validate_optimizer_state(actor_optimizer,payload['actor_optimizer'])
    _validate_optimizer_state(critic_optimizer,payload['critic_optimizer'])
    # 先验证控制器，再修改任何模型；拒绝损坏快照时保持调用方当前学习状态。
    import copy
    copy.deepcopy(controller).load_state_dict(payload['bc_controller'])
    actor.load_state_dict(payload['actor'],strict=True);critic.load_state_dict(payload['critic'],strict=True)
    actor_optimizer.load_state_dict(payload['actor_optimizer']);critic_optimizer.load_state_dict(payload['critic_optimizer'])
    controller.load_state_dict(payload['bc_controller'])
    if any(p.dtype!=torch.float32 for p in actor.parameters()):raise ValueError('Restored Actor master dtype changed')
    return payload


def load_evaluation_actor(path,*,actor,kernel,assets,execution_binding=None):
    """只读指定已完成checkpoint或固定数据候选的Actor，不恢复优化器/训练进度。

    正式checkpoint额外核对源码、奖励、时钟、数据和物理身份，允许改变纯评估任务
    数/长度和训练学习率以进行同条件P1/P2对照；绝不作为继续训练的恢复入口。
    """
    payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('schema')==SCHEMA:
        identity=payload.get('identity',{})
        if (not execution_binding or identity.get('assets')!=assets or identity.get('kernel')!=kernel or
                any(identity.get(k)!=v for k,v in execution_binding.items())):
            raise ValueError('Complete evaluation checkpoint execution identity differs')
        from .checkpoint import _validate_model_state
        _validate_model_state(actor,payload['actor'],'complete evaluation Actor')
        # 使用原Actor指纹算法的只读临时副本，先核验再修改调用者。
        import copy
        from .evaluation import _model_fingerprint
        candidate=copy.deepcopy(actor).cpu();candidate.load_state_dict(payload['actor'],strict=True)
        if _model_fingerprint(candidate)!=payload.get('actor_fingerprint'):
            raise ValueError('Complete evaluation Actor fingerprint differs')
        actor.load_state_dict(payload['actor'],strict=True)
        return dict(kind='complete_checkpoint_evaluation_only',path=str(Path(path).resolve()),sha256=file_sha256(path),
            source_iteration=payload['state']['iteration'],source_policy_version=payload['state']['policy_version'],
            training_resume=False)
    if (payload.get('schema')!='genmo.stage2.asymmetric_actor_evaluation.v1' or
            payload.get('evaluation_only') is not True or payload.get('kernel')!=kernel or
            payload.get('source_identity',{}).get('assets')!=assets or
            not execution_binding or any(payload.get('source_identity',{}).get(k)!=v
                                         for k,v in execution_binding.items()) or
            payload.get('hard_kl',{}).get('passed') is not True or
            payload.get('complete_update_and_soft_kl_passed') is not True):
        raise ValueError('Evaluation Actor asset/kernel/reward/physics/source/complete-update identity differs')
    update=payload.get('update',{})
    if (type(update.get('optimizer_steps')) is not int or update['optimizer_steps']<1 or
            update['optimizer_steps']!=update.get('planned_optimizer_steps') or
            not isinstance(payload.get('source_case_sha256'),str) or len(payload['source_case_sha256'])!=64):
        raise ValueError('Evaluation Actor is not a complete identified fixed-data update')
    descriptor=dict(kind='fixed_real_data_update_evaluation_only',path=str(Path(path).resolve()),
        sha256=file_sha256(path),source_case_sha256=payload['source_case_sha256'],update=update)
    from .checkpoint import _validate_model_state
    _validate_model_state(actor,payload['actor'],'当前 evaluation Actor')
    actor.load_state_dict(payload['actor'],strict=True)
    return descriptor
