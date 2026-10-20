"""为一次明确的GPU引用寿命修复生成可审计的完整续训快照。

输入同一实验的latest完整checkpoint与两个修复前Git提交。逐个核验全部绑定源码，
只允许GENMO bulk_cpu_copy及GMT to_host的原函数体原样包入try/finally，finally
只能清空捕获表和断开递归函数引用；任何其他算法、物理、资产或源码差异都拒绝。
旧Git内容还必须匹配checkpoint记录的SHA256，不能用另一份旧代码伪造等价关系。

保持Actor、Critic、全部Adam、BC、八rank RNG/采样游标、累计轮次和预算逐值不变，
仅更新已证明的两条源码哈希并加入维护来源。独占原训练锁，先写独立文件并完整
读回比对，再发布维护事件和latest。原文件、原指针副本及旧身份均留存；不放宽
训练加载器的严格身份校验，不导入旧rollout，不恢复中途PhysX，也不重新初始化。
该工具只适用于这一次结构明确的内存修复，不是任意源码变更的checkpoint兼容器。
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RULES = {
    'genmo': ('gem/closedloop/dppo/vector_generation.py', 'bulk_cpu_copy',
              'groups.clear()\nconverted.clear()\nvisit = rebuild = None'),
    'gmt': ('source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/vector_diagnostics.py',
            'to_host', 'groups.clear()\nlocations.clear()\ncollect = build = None'),
}


def verify_cleanup_only(before, after, name, cleanup):
    """AST级证明仅增加退出清理；不忽略任何原语句、默认值或模块其他成员。"""
    old, new = ast.parse(before), ast.parse(after)
    old_function = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
    new_function = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
    if not new_function.body or not isinstance(new_function.body[-1], ast.Try):
        raise ValueError('Expected original preamble and one cleanup try/finally')
    block = new_function.body[-1]
    dump = lambda value: ast.dump(value, include_attributes=False)
    if block.handlers or block.orelse or dump(ast.Module(body=block.finalbody, type_ignores=[])) != dump(ast.parse(cleanup)):
        raise ValueError('Unexpected exception handling or cleanup operations')
    new_function.body = new_function.body[:-1] + block.body
    if dump(old) != dump(new):
        raise ValueError('Source contains changes beyond exact host-copy lifetime cleanup')
    return dict(function=name, original_statements_unchanged=True,
                module_other_statements_unchanged=True, cleanup=cleanup.splitlines())


def same_state(before, after, path='payload'):
    """读回整个训练状态逐值比对；校验不限于模型权重或状态计数。"""
    if torch.is_tensor(before):
        if (not torch.is_tensor(after) or before.dtype != after.dtype or before.shape != after.shape
                or not torch.equal(before, after)):
            raise ValueError('Tensor state changed: '+path)
    elif isinstance(before, np.ndarray):
        if not isinstance(after, np.ndarray) or before.dtype != after.dtype or not np.array_equal(before, after):
            raise ValueError('Array state changed: '+path)
    elif isinstance(before, dict):
        if before.keys() != after.keys():
            raise ValueError('State keys changed: '+path)
        for key in before:
            same_state(before[key], after[key], path+'.'+str(key))
    elif isinstance(before, (list, tuple)):
        if type(before) is not type(after) or len(before) != len(after):
            raise ValueError('Sequence state changed: '+path)
        for i, (a, b) in enumerate(zip(before, after)):
            same_state(a, b, path+f'[{i}]')
    elif type(before) is not type(after) or before != after:
        raise ValueError('Scalar state changed: '+path)


def migrate(run, commits):
    from gem.closedloop.dppo.budget import atomic_json
    from gem.closedloop.dppo.training import _sources
    from gem.closedloop.dppo.training_checkpoint import SCHEMA, file_sha256

    run = run.resolve(strict=True)
    with (run/'.run.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pointer = json.loads((run/'latest.json').read_text())
        source = (run/pointer['path']).resolve(strict=True)
        if source.parent != run/'checkpoints' or file_sha256(source) != pointer['sha256']:
            raise ValueError('Source checkpoint path/SHA differs')
        payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        if payload.get('schema') != SCHEMA or 'host_copy_maintenance' in payload:
            raise ValueError('Unsupported checkpoint or maintenance already applied')
        original_identity = copy.deepcopy(payload['identity'])
        old_sources = original_identity['source_files']
        current = _sources(payload['config'])
        proofs = {}
        for repo, (allowed_path, name, cleanup) in RULES.items():
            old, new = old_sources[repo], current[repo]
            if old.keys() != new.keys() or [k for k in old if old[k] != new[k]] != [allowed_path]:
                raise ValueError('Exactly one permitted host-copy source must change in '+repo)
            root = Path(payload['config']['paths'][repo+'_repo'])
            branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
            required = 'feature/bumi-music-closedloop' if repo == 'genmo' else 'feature/bumi-frozen-gmt-backend'
            if branch != required or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
                raise ValueError('Expected fixed clean branch in '+repo)
            before = subprocess.check_output(['git', 'show', commits[repo]+':'+allowed_path], cwd=root)
            after = (root/allowed_path).read_bytes()
            if hashlib.sha256(before).hexdigest() != old[allowed_path] or hashlib.sha256(after).hexdigest() != new[allowed_path]:
                raise ValueError('Git/runtime content does not match bound source SHA')
            proofs[repo] = dict(path=allowed_path, old_sha256=old[allowed_path], new_sha256=new[allowed_path],
                previous_commit=commits[repo], current_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                proof=verify_cleanup_only(before, after, name, cleanup))
        if old_sources.keys() != current.keys():
            raise ValueError('Bound repositories changed')
        stamp = f'{time.time_ns():020d}_{uuid.uuid4().hex[:12]}'
        event = run/'maintenance'/stamp
        event.mkdir(parents=True, exist_ok=False)
        atomic_json(event/'previous_latest.json', pointer)
        atomic_json(event/'previous_identity.json', original_identity)
        new_identity = copy.deepcopy(original_identity)
        new_identity['source_files'] = current
        provenance = dict(schema='genmo.host_copy_lifetime_maintenance.v1', parent_checkpoint=pointer,
            proofs=proofs, old_identity_path=str(event/'previous_identity.json'),
            unchanged_learning_state=True, resume_physics='fresh_PhysX_episodes_no_partial_rollout_reuse')
        # 原payload本身不改写，读回后仍作为完整状态对照。
        updated = dict(payload, identity=new_identity, host_copy_maintenance=provenance)
        state = payload['state']
        name = f'stage10_r{state["rollouts_completed"]:06d}_a{state["actor_updates"]:06d}_{stamp}_host_copy_lifetime_maintenance.pt'
        final = run/'checkpoints'/name
        temporary = final.with_suffix('.pt.partial')
        try:
            with temporary.open('xb') as stream:
                torch.save(updated, stream)
                stream.flush()
                os.fsync(stream.fileno())
            check = torch.load(temporary, map_location='cpu', weights_only=False, mmap=True)
            same_state(updated, check)
            descriptor = dict(pointer, path=str(final.relative_to(run)), sha256=file_sha256(temporary),
                size_bytes=temporary.stat().st_size, publication_id=stamp,
                metadata=dict(pointer.get('metadata', {}), reason='host_copy_lifetime_maintenance',
                              reasons=['host_copy_lifetime_maintenance']))
            os.link(temporary, final)
            temporary.unlink()
            fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            atomic_json(event/'migration.json', dict(provenance, checkpoint=descriptor,
                entire_payload_roundtrip_identical=True))
            atomic_json(run/'latest.json', descriptor)
            return dict(checkpoint=descriptor, evidence=str(event/'migration.json'), state=state)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--genmo-before', required=True)
    parser.add_argument('--gmt-before', required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.run_dir, dict(genmo=args.genmo_before, gmt=args.gmt_before)), indent=2))


if __name__ == '__main__':
    main()
