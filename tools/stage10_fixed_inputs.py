"""本阶段真实固定证据的只读加载器，支持任意显式规模及世界共享原始链。

先核验归档总SHA，再按成员清单核验本rank实际需要的文件字节、大小和安全相对路径。
仅解包到调用者指定的全新临时目录，原证据不修改；转移使用正式load_rollout_record
恢复，避免诊断工具另造一个不兼容的raw_reference实现。没有固定每卡20条限制。
返回原始fixed_targets，不重算优势或old概率。调用者负责结束后清理临时解包文件。
"""
import hashlib
import json
from pathlib import Path
import tarfile
import torch
from gem.closedloop.dppo.rollout_storage import load_rollout_record
from gem.closedloop.dppo.run_management import file_sha256


def load_rank(iteration, rank, destination, *, journals=False, reuse_verified=False):
    iteration, destination = Path(iteration), Path(destination)
    manifest=json.loads((iteration/'archive_manifest.json').read_text())
    archive=iteration/manifest['archive']
    if file_sha256(archive)!=manifest['archive_sha256']:raise ValueError('Source archive SHA mismatch')
    prefix=f'rank{rank:02d}/'
    members={r['path']:r for r in manifest['members']}
    needed={name for name in members if name.startswith(prefix) and (
        '/raw_samples/' in name or '/rollout/' in name or name.endswith('/fixed_targets.pt') or
        journals and name.endswith('/world_journal.sqlite'))}
    if reuse_verified:
        for name in needed:
            relative=Path(name)
            path=destination/relative
            if (relative.is_absolute() or '..' in relative.parts or not path.is_file()
                    or path.is_symlink() or path.stat().st_size!=members[name]['size_bytes']
                    or file_sha256(path)!=members[name]['sha256']):
                raise ValueError('Reusable diagnostic input no longer matches immutable archive')
        return _read_rank(destination,rank,manifest,len(needed))
    destination.mkdir(parents=True,exist_ok=False)
    found=set()
    with tarfile.open(archive,'r|gz') as stream:
        for item in stream:
            if item.name not in needed:continue
            relative=Path(item.name)
            if (relative.is_absolute() or '..' in relative.parts or not item.isfile() or
                    item.name in found or item.size!=members[item.name]['size_bytes']):
                raise ValueError('Unsafe/duplicate/mismatched archive member')
            path=destination/relative;path.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            with stream.extractfile(item) as source,path.open('xb') as target:
                while block:=source.read(4*1024**2):digest.update(block);target.write(block)
            if digest.hexdigest()!=members[item.name]['sha256']:raise ValueError('Member SHA mismatch')
            found.add(item.name)
    if found!=needed:raise ValueError('Incomplete archive')
    return _read_rank(destination,rank,manifest,len(found))


def _read_rank(destination,rank,manifest,member_count):
    rank_dir=destination/f'rank{rank:02d}'
    roll=json.loads((rank_dir/'rollout/manifest.json').read_text());rows=[];cache={}
    for chunk in roll['chunks']:
        path=rank_dir/'rollout'/chunk['path']
        records=json.loads(path.read_text())['records']
        for record in records:
            row=load_rollout_record(path.parent/record['path'],record,rank_directory=rank_dir,cache=cache)
            rows.append(row)
    if len(rows)!=roll['transition_count']:raise ValueError('Real rollout count differs')
    targets=torch.load(rank_dir/'fixed_targets.pt',weights_only=False,map_location='cpu')
    return rows,targets,dict(archive_sha256=manifest['archive_sha256'],rank_directory=str(rank_dir),
                            real_local_chains=len(rows),members=member_count)
