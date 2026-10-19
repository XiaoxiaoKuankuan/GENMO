"""整理已停止Stage2运行的重复证据，保留模型、指标及可还原的原始物理数据库。

默认只清点。--apply取得运行排他锁后，把逐控制JSON转换为单份无损列式证据，
更新父报告的文件SHA并明确记载格式迁移；关键评估SQLite按原字节gzip冷存并核验。
其余阶段的重复底层journal在完整控制摘要或MC标签已保留后退役，清单标明证据范围。
严格训练rollout先提取每卡8条自包含真实链，再释放已完成更新的全量张量归档。
checkpoint只对字节SHA完全相同的文件合并硬链接，所有原路径及SHA保持有效，不把
不同RNG、优化器或计数的checkpoint冒充同一个恢复点。不访问模型GPU、不启动训练。
每步写入retention_manifest.json，异常保留已完成结果和原始未处理文件。
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.compact_evidence import read_episode,write_episode,summary_only
from gem.closedloop.dppo.evaluation import _publish_json
from gem.closedloop.dppo.training_checkpoint import file_sha256


def allocated_bytes(root):
    seen=set();total=0
    for parent,_,files in os.walk(root):
        for name in files:
            p=Path(parent)/name
            if p.is_symlink():continue
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:total+=s.st_blocks*512;seen.add(key)
    return total


def _replace_references(value,directory,root,changed):
    if isinstance(value,dict):
        raw=value.get('path')
        if isinstance(raw,str) and 'sha256' in value:
            path=Path(raw)
            for candidate in ([path] if path.is_absolute() else [directory/path,root/path]):
                replacement=changed.get(str(candidate.resolve()))
                if replacement is not None:value['sha256']=replacement;break
        for child in value.values():_replace_references(child,directory,root,changed)
    elif isinstance(value,list):
        for child in value:_replace_references(child,directory,root,changed)


def compact(root,*,apply=False):
    root=Path(root).resolve(strict=True)
    if not (root/'run_manifest.json').is_file():raise ValueError('Exact Stage2 run directory required')
    report_path=root/'retention_manifest.json'
    report=(json.loads(report_path.read_text()) if report_path.exists() else dict(
        schema='genmo.stage2.retention_compaction.v1',run=str(root),before_bytes=allocated_bytes(root),
        converted=[],cold_sqlite=[],retired_sqlite=[],sampled_rollouts=[],deduplicated=[]))
    if not apply:
        return dict(run=str(root),allocated_bytes=allocated_bytes(root),
            sqlite_count=len(list(root.glob('sessions/**/*.sqlite'))),
            episode_count=len(list(root.glob('sessions/**/report/episodes/*.json'))),apply=False)
    lock=(root/'.run.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def record():atomic_json(report_path,report)
    try:
        changed={entry['path']:entry['after_sha256'] for entry in report['converted']}
        episodes=sorted(root.glob('sessions/**/report/episodes/*.json'))
        for path in episodes:
            if '.compact.' in path.name:continue
            original=json.loads(path.read_text())
            if 'control_archive' in original:continue
            before=file_sha256(path)
            temporary=path.with_name(path.stem+'.compact.json')
            write_episode(temporary,original,publish_json=_publish_json)
            restored=read_episode(temporary)
            if restored['decisions']!=original['decisions']:
                raise RuntimeError('Lossless episode comparison failed: '+str(path))
            document=json.loads(temporary.read_text())
            document['format_migration']=dict(original_sha256=before,lossless_controls_verified=True)
            atomic_json(path,document);temporary.unlink()
            entry=dict(path=str(path),before_sha256=before,after_sha256=file_sha256(path))
            changed[str(path)]=entry['after_sha256'];report['converted'].append(entry);record()
        documents=set(root.glob('sessions/**/report/report.json'))
        documents.update(root.glob('sessions/*/evaluations/*.json'))
        documents.update(root.glob('sessions/*/session.json'))
        if (root/'evaluation_baseline.json').exists():documents.add(root/'evaluation_baseline.json')
        for path in sorted(documents,key=lambda p:len(p.parts),reverse=True):
            old=json.loads(path.read_text());document=summary_only(old)
            _replace_references(document,path.parent,root,changed)
            before=file_sha256(path)
            document['format_migration']=dict(original_sha256=before,
                evidence_scope='single_lossless_control_archive_per_episode_with_explicit_cold_SQLite')
            atomic_json(path,document)
            entry=dict(path=str(path),before_sha256=before,after_sha256=file_sha256(path))
            changed[str(path)]=entry['after_sha256'];report['converted'].append(entry);record()
        # 固定初始Actor/Critic诊断仍需原路径；只保护清单中真正引用的少量文件。
        protected=set()
        fixed=root/'fixed_diagnostic_reference.json'
        if fixed.exists():
            for rank in json.loads(fixed.read_text())['ranks']:
                protected.update(Path(row['path']).resolve() for row in rank['actor_records'])
                if rank.get('critic'):protected.add(Path(rank['critic']['path']).resolve())
        from gem.closedloop.dppo.rollout_storage import load_rollout_record
        from gem.closedloop.dppo.diagnostic_rollout import persist_rollout,load_rollout
        for collection in sorted(root.glob('sessions/*/collections/train_*')):
            ranks=sorted(collection.glob('rank*/rollout/manifest.json'))
            if not ranks:continue
            plans=[]
            for manifest in ranks:
                d=json.loads(manifest.read_text());count=d['transition_count'];n=min(8,count)
                plans.append((manifest,d,[i*count//n for i in range(n)]))
            global_manifest=[dict(owner_rank=int(path.parent.parent.name[-2:]),local_index=i,valid=True,has_free=True)
                for path,_,indices in plans for i in range(len(indices))]
            captured=[]
            for manifest,d,indices in plans:
                rank_dir=manifest.parent.parent;rows=[];cache={};offset=0
                for chunk in d['chunks']:
                    chunk_path=manifest.parent/chunk['path'];records=json.loads(chunk_path.read_text())['records']
                    for i in indices:
                        if offset<=i<offset+len(records):
                            row=records[i-offset]
                            rows.append(load_rollout_record(chunk_path.parent/row['path'],row,
                                rank_directory=rank_dir,cache=cache))
                    offset+=len(records)
                output=root/'retained_diagnostics'/collection.name/rank_dir.name
                if not output.exists():
                    persist_rollout(output,rows,{},global_manifest,kernel=rows[0].metadata['sampler_trace']['kernel_config'],
                        identity={'scope':'sampled_historical_behavior_not_full_update_replay'},full_execution=False)
                load_rollout(output)
                captured.append(str(output/'manifest.json'))
            report['sampled_rollouts'].append(dict(collection=str(collection),retained=captured,
                retired_scope='completed_update_full_rollout_arrays',per_rank=8));record()
            for manifest,_,_ in plans:
                rank_dir=manifest.parent.parent
                shutil.rmtree(manifest.parent)
                for path in rank_dir.glob('**/raw_samples/*.pt'):
                    if path.resolve() not in protected:path.unlink()
        # 周期评估与MC不再需要全量扩散链，固定诊断链和显式fixed_data_validation不在清理范围。
        for parent in ('phases','mc_calibration'):
            for path in root.glob(f'sessions/*/{parent}/**/raw_samples/*.pt'):
                if path.resolve() not in protected:path.unlink()
        databases=[p for parent in ('phases','collections','mc_calibration')
            for p in root.glob(f'sessions/*/{parent}/**/*.sqlite')]
        pinned=None
        if (root/'latest.json').exists():
            latest=json.loads((root/'latest.json').read_text())['iteration']
            pinned={0,latest,latest//100*100}
            if (root/'best_checkpoint.json').exists():
                pinned.add(json.loads((root/'best_checkpoint.json').read_text())['iteration'])
        report['complete_physics_evaluation_iterations']=None if pinned is None else sorted(pinned)
        for path in sorted(databases):
            wal=path.with_name(path.name+'-wal')
            if wal.exists() and wal.stat().st_size:continue
            keep=(pinned is None or 'phases' in path.parts and any(
                ('initial' in part if iteration==0 else f'r{iteration:06d}_' in part or f'final_{iteration:06d}' in part)
                for part in path.parts for iteration in pinned))
            if not keep:
                report.setdefault('retired_sqlite',[]).append(dict(path=str(path),bytes=path.stat().st_size,
                    sha256=file_sha256(path),retained='lossless_control_summary_or_complete_MC_labels_and_sampled_training_chains'))
                record();path.unlink();continue
            target=root/'cold_evidence'/path.relative_to(root).with_suffix('.sqlite.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.partial');h=hashlib.sha256();size=path.stat().st_size
            with path.open('rb') as stream,gzip.open(temporary,'wb',compresslevel=1) as destination:
                for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);destination.write(block)
            check=hashlib.sha256()
            with gzip.open(temporary,'rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):check.update(block)
            if check.hexdigest()!=h.hexdigest():raise RuntimeError('Cold SQLite byte restoration SHA differs')
            os.replace(temporary,target)
            report['cold_sqlite'].append(dict(original=str(path),original_bytes=size,original_sha256=h.hexdigest(),
                archive=str(target),archive_bytes=target.stat().st_size,archive_sha256=file_sha256(target)))
            record();path.unlink()
        groups=defaultdict(list)
        for path in (root/'checkpoints').glob('*.pt'):groups[path.stat().st_size].append(path)
        for candidates in groups.values():
            if len(candidates)<2:continue
            known={}
            for path in sorted(candidates):
                sha=file_sha256(path)
                if sha in known and path.stat().st_ino!=known[sha].stat().st_ino:
                    temporary=path.with_suffix('.dedup');os.link(known[sha],temporary);os.replace(temporary,path)
                    report['deduplicated'].append(dict(path=str(path),canonical=str(known[sha]),sha256=sha))
                else:known[sha]=path
        report.update(status='completed',after_bytes=allocated_bytes(root),
            fidelity='lossless_control_records;gzip_key_physics_SQLite;complete_MC_labels;sampled_training_chains;all_model_paths_retained')
        record();return report
    finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(compact(args.run,apply=args.apply),ensure_ascii=False,indent=2))
