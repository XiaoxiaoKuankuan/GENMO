"""Stage10 内存rollout的显式诊断发布与固定数据重放读取。

普通训练不调用本模块。异常或明确的固定数据验收需要保存时，将当前完整随机
链、旧均值/实际方差/联合概率、因果条件、GAE目标、真实控制奖励和物理证据发布
为独立当前格式；它不要求存在旧v2的raw_samples文件，不伪造磁盘路径或SHA。
同一CPU快照用共享memo保留列式大数组引用，不把控制字段展开成Python列表。
完整文件先排他写临时路径、fsync、原子发布，再以实际SHA发布manifest。

可选完整证据用于异常回放；固定Actor计算对照可以明确只保留学习张量与真实执行
身份/计数，此时manifest必须标明不是完整物理回放。原始当前轮的物理数据仍在
训练进程中保持到最终KL完成。读取检查schema、大小、SHA、行为核和逐条转移，
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

SCHEMA='genmo.stage10.validation_rollout'


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
    payload=dict(schema=SCHEMA,identity=identity,kernel=kernel,transitions=packed,
        targets=cpu_snapshot(targets,memo),global_manifest=copy.deepcopy(manifest),
        full_execution_evidence=bool(full_execution))
    output=path/'rollout.pt';temporary=path/'rollout.pt.partial'
    if guard:guard.check(sum(row.chain.numel()*row.chain.element_size()*3 for row in rows)+65536)
    try:
        with temporary.open('xb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        size=temporary.stat().st_size;sha=file_sha256(temporary)
        os.link(temporary,output);temporary.unlink()
        fd=os.open(path,os.O_DIRECTORY|os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        record=dict(schema=SCHEMA,path=output.name,size_bytes=size,sha256=sha,
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
    for row in payload['transitions']:
        row.validate()
        if row.metadata['sampler_trace']['kernel_config']!=payload['kernel']:
            raise ValueError('Stored behavior kernel changed')
    return payload
