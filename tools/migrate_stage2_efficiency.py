"""把已停止的Stage2完整恢复点显式迁移到本次纯工程优化，保留可审计父身份。

仅允许本次已列出的归档、诊断、KL复用和批量参考处理源码变化；训练参数、奖励、
实际随机核、物理配置与数据身份必须完全相同。源运行加排他锁，原checkpoint及SHA
不修改；新目录保存新身份、父checkpoint SHA、逐源码差异和验证证据。Actor、Critic、
Adam、BC控制器、各rank RNG/采样状态和累计轮数均计算内容摘要并在写后重新核验。

旧评估值仅作为同物理合同下的继承基线，明确附带原身份与“未重新评估”声明；best
实际权重用硬链接保留，不复制多份大模型。预算通过SQLite备份完整继承真实消耗。
此工具不启动训练、不放宽恢复加载器，也不支持跨算法、奖励或数值核迁移。
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.compact_evidence import summary_only
from gem.closedloop.dppo.training import _sources,_physical_configuration_identity
from gem.closedloop.dppo.training_checkpoint import AsyncCheckpointWriter,SCHEMA,file_sha256
from gem.closedloop.dppo.training_config import configuration

GENMO_FILES={
    'gem/closedloop/dppo/'+name+'.py' for name in (
        'actor_backtracking','full_batch_actor','kl_reuse','updater','training','training_config',
        'training_checkpoint','compact_evidence','evaluation','evaluation_runtime','evaluation_world',
        'evaluation_shards','periodic_monitor','vector_evaluation','mc_calibration','runtime',
        'diagnostic_rollout','vector_generation','memory_execution','tensor_cache','execution_checks',
        'frozen_prefix','env_adapter','columnar_reward')}
GENMO_FILES.add('tools/train_closedloop_stage10.py')
GMT_BASE='source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/'
GMT_FILES={GMT_BASE+name+'.py' for name in ('execution_journal','inprocess_service','vector_backend','vector_columns')}
GMT_FILES.add('scripts/motion/closedloop_reference.py')


def digest(value):
    """内容摘要不依赖Tensor存储地址、pickle容器共享或字典插入顺序。"""
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous();h.update(('tensor'+str(t.dtype)+str(tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy());return
        if isinstance(v,np.ndarray):
            h.update(('array'+v.dtype.str+str(v.shape)).encode());h.update(v.tobytes());return
        if isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda key:(type(key).__name__,repr(key))):visit(k);visit(v[k])
            return
        if isinstance(v,(list,tuple)):
            h.update((type(v).__name__+str(len(v))).encode())
            for item in v:visit(item)
            return
        h.update((type(v).__name__+repr(v)+';').encode())
    visit(value)
    return h.hexdigest()


def target_identity(parent,config):
    stage=config['stage10']
    expected=dict(training=stage['training'],critic=stage['critic'],bc_gradient=stage['bc_gradient'],
        kl=stage['kl'],scale=stage['scale'],reward=config['stage9']['reward'],runtime=config['runtime'],
        timing=config['timing'],dataset=stage['dataset'],evaluation=stage['evaluation'],
        run_control=stage['run_control'],resource_limits=stage['limits'],physical_episode=stage['physical_episode'],
        ppo_rollout=stage['ppo_rollout'],generation_horizon=stage['generation_horizon'],
        physical_configuration=_physical_configuration_identity(config),assets=stage['assets'])
    changed=[key for key,value in expected.items() if parent.get(key)!=value]
    if changed:raise ValueError('Engineering migration cannot change algorithm/physics/configuration: '+','.join(changed))
    sources=_sources(config);changes={}
    for repo,allowed in [('genmo',GENMO_FILES),('gmt',GMT_FILES)]:
        before=parent['source_files'][repo];after=sources[repo]
        diff={key:dict(before=before.get(key),after=after.get(key)) for key in before.keys()|after.keys()
              if before.get(key)!=after.get(key)}
        if set(diff)-allowed:raise ValueError(f'Unreviewed {repo} source changes: {sorted(set(diff)-allowed)}')
        changes[repo]=diff
    result=copy.deepcopy(parent);result['source_files']=sources
    return result,changes


def migrate(source,destination,config_path,proof_path):
    source=Path(source).resolve(strict=True);destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError('Migration destination must be new')
    proof_path=Path(proof_path).resolve(strict=True);proof=json.loads(proof_path.read_text())
    if proof.get('status')!='passed':raise ValueError('Passing numerical/reference verification report required')
    lock=(source/'.run.lock').open('a+b')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        pointer=json.loads((source/'latest.json').read_text());parent_path=source/pointer['path']
        if file_sha256(parent_path)!=pointer['sha256']:raise ValueError('Parent checkpoint SHA differs')
        payload=torch.load(parent_path,map_location='cpu',weights_only=False,mmap=True)
        if payload.get('schema')!=SCHEMA:raise ValueError('Exact current complete checkpoint schema required')
        if payload['identity']['torch_version']!=torch.__version__:
            raise ValueError('Engineering migration cannot change the actual Torch version')
        config=configuration(config_path);identity,changes=target_identity(payload['identity'],config)
        if proof.get('source_manifest_sha256')!=digest(identity['source_files']):
            raise ValueError('Verification report belongs to different source files')
        for name,expected in identity['assets'].items():
            if file_sha256(config['paths'][name])!=expected:raise ValueError('Actual asset differs: '+name)
        keys=('actor','critic','actor_optimizer','critic_optimizer','bc_controller','scheduler','state','rank_states','budget')
        learning={key:payload[key] for key in keys};before=digest(learning)
        receipt=dict(schema='genmo.stage2.verified_engineering_migration.v1',parent_run=str(source),
            parent_checkpoint=str(parent_path),parent_checkpoint_sha256=pointer['sha256'],
            parent_identity_sha256=digest(payload['identity']),target_identity_sha256=digest(identity),
            changes=changes,learning_state_sha256=before,proof=str(proof_path),proof_sha256=file_sha256(proof_path),
            no_algorithm_or_kernel_change=True,physical_recovery='new_PhysX_episode')
        destination.mkdir(parents=True)
        atomic_json(destination/'run_manifest.json',dict(schema='genmo.closedloop.stage10.run.v1',
            run_id=destination.name,engineering_parent=receipt))
        budget=json.loads((source/'budget.json').read_text());atomic_json(destination/'budget.json',budget)
        database=budget.get('database')
        if database:
            if Path(database).name!=database:raise ValueError('Budget database path escapes run')
            original=sqlite3.connect(f'file:{source/database}?mode=ro',uri=True)
            target=sqlite3.connect(destination/database)
            try:original.backup(target)
            finally:original.close();target.close()
        # 不改原始物理评估证据。继承汇总明确记录其实际来源、父身份和迁移原因。
        for name in ('evaluation_baseline.json','fixed_diagnostic_reference.json','best_observed.json','best_saved.json'):
            p=source/name
            if not p.exists():continue
            document=summary_only(json.loads(p.read_text()))
            original_identity=document.get('evaluation_identity')
            if original_identity is not None:
                document['evaluation_identity']=dict(original_identity,training_identity=identity)
                document['inherited_evidence']=dict(parent_file=str(p),parent_sha256=file_sha256(p),
                    parent_evaluation_identity_sha256=digest(original_identity),re_evaluated=False,
                    reason='verified_identical_algorithm_and_physics_engineering_migration')
            reference=document.get('checkpoint')
            if reference:
                old=source/reference['path'];new=destination/reference['path'];new.parent.mkdir(exist_ok=True)
                if file_sha256(old)!=reference['sha256']:raise ValueError('Best checkpoint SHA differs')
                if not new.exists():os.link(old,new)
            atomic_json(destination/name,document)
        payload=dict(payload,identity=identity,config=config,engineering_migration=receipt)
        writer=AsyncCheckpointWriter(destination)
        try:
            writer.submit(payload,reason='verified_engineering_migration');publication=writer.poll(wait=True)
        finally:writer.close()
        restored=torch.load(destination/publication['path'],map_location='cpu',weights_only=False,mmap=True)
        if digest({key:restored[key] for key in keys})!=before:raise RuntimeError('Learning state changed during migration')
        receipt.update(publication=publication,learning_state_verified=True)
        atomic_json(destination/'engineering_migration.json',receipt)
        atomic_json(destination/'training_identity.json',identity)
        if (source/'best_checkpoint.json').exists():
            best=json.loads((source/'best_checkpoint.json').read_text())
            old=source/best['path'];new=destination/best['path']
            if not new.exists():os.link(old,new)
            atomic_json(destination/'best_checkpoint.json',best)
        return receipt
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--verification-report',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(migrate(args.source_run,args.output_dir,args.config,args.verification_report),ensure_ascii=False,indent=2))
