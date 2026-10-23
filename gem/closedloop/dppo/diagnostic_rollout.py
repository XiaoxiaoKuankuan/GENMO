"""Stage10 内存rollout的显式诊断发布与固定数据重放读取。

周期诊断、异常或明确的固定数据验收需要保存时，将当前算法的真实学习记录
（Gaussian 完整链或 RoboFFT 最终 action_x0/固定 MC）、因果条件、GAE目标和物理证据发布
为独立当前格式；它不要求存在旧v2的raw_samples文件，不伪造磁盘路径或SHA。
同一CPU快照用共享memo保留列式大数组引用，不把控制字段展开成Python列表。
完整文件先排他写临时路径、fsync、原子发布，再以实际SHA发布manifest。

可选完整证据用于异常回放；固定Actor计算对照可以明确只保留学习张量与真实执行
身份/计数，此时manifest必须标明不是完整物理回放。原始当前轮的物理数据仍在
训练进程中保持到当前算法更新与诊断完成。读取检查schema、大小、SHA、行为核和逐条转移，
不改写old统计量，不重新采样，不透明兼容旧逐文件/归档格式。
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import torch

from .budget import atomic_json
from .buffer import UpperTransition,cpu_snapshot
from .training_checkpoint import file_sha256

SCHEMA='genmo.stage2.asymmetric_validation_rollout.v1'


def persist_rollout(path,rows,targets,manifest,*,kernel,identity,full_execution=True,guard=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    memo={};packed=[];origins=[]
    for row in rows:
        fields=vars(row).copy();metadata=row.metadata.copy()
        if not full_execution:
            for key in ('execution_trace','reward_details','generated','terminal_snapshot'):
                metadata.pop(key,None)
        fields['metadata']=metadata
        origins.append(fields)  # 共享memo的id键要求原容器保持存活，禁止循环中id复用。
        snapshot=UpperTransition(**cpu_snapshot(fields,memo));snapshot.validate();packed.append(snapshot)
    algorithms=sorted({row.algorithm for row in rows})
    payload=dict(schema=SCHEMA,algorithms=algorithms,identity=identity,kernel=kernel,transitions=packed,
        targets=cpu_snapshot(targets,memo),global_manifest=copy.deepcopy(manifest),
        full_execution_evidence=bool(full_execution))
    output=path/'rollout.pt';temporary=path/'rollout.pt.partial'
    if guard:
        learning_bytes=sum((row.chain if row.chain is not None else row.action_x0).numel()*4*3 for row in rows)
        mc_bytes=sum(v.numel()*v.element_size() for row in rows if row.mc_record is not None
            for v in (row.mc_record.eps,row.mc_record.old_loss,row.mc_record.sampling_indices,
                row.mc_record.original_timesteps,row.mc_record.mc_numbers))
        guard.check(learning_bytes+mc_bytes+65536)
    try:
        with temporary.open('xb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        size=temporary.stat().st_size;sha=file_sha256(temporary)
        os.link(temporary,output);temporary.unlink()
        fd=os.open(path,os.O_DIRECTORY|os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        record=dict(schema=SCHEMA,algorithms=algorithms,path=output.name,size_bytes=size,sha256=sha,
            real_upper_transitions=len(rows),real_control_steps=sum(r.executed_control_steps for r in rows),
            real_physics_steps=sum(r.executed_physics_steps for r in rows),
            full_execution_evidence=bool(full_execution))
        atomic_json(path/'manifest.json',record)
        if guard:guard.account_file(output);guard.account_file(path/'manifest.json')
        return record
    finally:temporary.unlink(missing_ok=True)


def load_rollout(path):
    import json
    root=Path(path).resolve(strict=True);record=json.loads((root/'manifest.json').read_text())
    if record.get('schema')!=SCHEMA or Path(record['path']).name!=record['path']:
        raise ValueError('Unsupported or escaping 当前 diagnostic rollout')
    source=root/record['path']
    if source.stat().st_size!=record['size_bytes'] or file_sha256(source)!=record['sha256']:
        raise ValueError('Diagnostic rollout size/SHA mismatch')
    payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    if payload.get('schema')!=SCHEMA or len(payload['transitions'])!=record['real_upper_transitions']:
        raise ValueError('Incomplete diagnostic rollout')
    if 'algorithms' in record and (record['algorithms']!=payload.get('algorithms') or
            record['algorithms']!=sorted({row.algorithm for row in payload['transitions']})):
        raise ValueError('Diagnostic rollout algorithm marker differs')
    for row in payload['transitions']:
        row.validate()
        if row.metadata['sampler_trace']['kernel_config']!=payload['kernel']:
            raise ValueError('Stored behavior kernel changed')
    return payload


def sampled_diagnostic(c, rows, targets, label, *, per_rank=8, keep_last=2):
    """周期采样只留少量真实链；完整训练与KL覆盖不变，不导出全体物理journal。"""
    indices=[i*len(rows)//min(per_rank,len(rows)) for i in range(min(per_rank,len(rows)))]
    selected=[rows[i] for i in indices]
    target={key:(value[indices] if torch.is_tensor(value) and value.ndim and len(value)==len(rows) else value)
            for key,value in targets.items()}
    local=[dict(owner_rank=c.distributed.rank,local_index=i,valid=True,has_free=bool(row.free_mask.any()))
           for i,row in enumerate(selected)]
    manifest=[item for part in c.distributed.all_gather_object(local) for item in part]
    base=c.session/'diagnostics'
    from .parallel_support import local_call,root_call
    result=local_call(c.distributed,lambda:persist_rollout(base/label/f'rank{c.distributed.rank:02d}',selected,target,manifest,
        kernel=c.policy.kernel_config,identity=c.identity,full_execution=False,guard=c.guard)
        if selected else dict(real_upper_transitions=0,full_execution_evidence=False))
    def prune():
        import shutil
        for expired in sorted(base.glob('train_*'))[:-keep_last]:
            if expired.is_dir() and expired.parent==base:
                shutil.rmtree(expired)
    root_call(c.distributed,prune)
    local_call(c.distributed,c.guard.refresh)
    return result
