"""固定正余弦位置表的规范比较、只迁移权重修复与来源记录。

旧 Stage2 的全局 requires_grad_(True) 可能已经改变位置表。重新冻结只能阻止继续改变，
不能恢复已污染的数值。本模块按实际 PositionalEncoding 类重新构建规范表，比较所有
state_dict 别名，并可在显式修复流程中一起恢复；其他学习参数保持原样。

修复输出使用 Stage1 已有可审计 weights-only 容器，不携带 Actor/Critic 优化器、RNG、
采样游标或训练步数。恢复训练须新建优化器，只给可训练参数分配 Adam 状态，不能把旧
完整断点改个标签继续 resume。输出保留源完整断点 SHA、原轮次、规范表差异及原资产
身份，原始断点不被覆盖。此迁移会改变策略输出，仍需重新建立执行验证基线。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_position_tables(actor):
    """使用当前架构的原位置编码构造器，保留共享模块在 state_dict 中的全部别名。"""
    result = {}
    for name, module in actor.named_modules(remove_duplicate=False):
        if module.__class__.__name__ != 'PositionalEncoding' or not isinstance(getattr(module, 'pe', None), torch.Tensor):
            continue
        table = module.pe
        if table.ndim != 3 or table.shape[1] != 1:
            raise ValueError(f'Unsupported fixed position table layout: {name}')
        canonical = type(module)(int(table.shape[-1]), dropout=0., max_len=int(table.shape[0]))
        result[f'{name}.pe' if name else 'pe'] = canonical.pe.detach().cpu().to(table.dtype).clone()
    return result


def audit_position_tables(actor, *, canonical=None, repair=False):
    """报告规范差异；repair=True 时只恢复固定表并冻结，共享别名必须一起正确。"""
    canonical = canonical_position_tables(actor) if canonical is None else canonical
    state = actor.state_dict()
    entries = []
    for name, expected in canonical.items():
        actual = state[name].detach().cpu()
        if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.isfinite(actual).all():
            raise ValueError(f'Invalid fixed position table: {name}')
        difference = actual.double() - expected.double()
        entries.append(dict(name=name, values=actual.numel(), changed_values=int((actual != expected).sum()),
                            max_abs_difference=float(difference.abs().max()), l2_difference=float(difference.norm())))
    changed = any(item['changed_values'] for item in entries)
    if repair:
        with torch.no_grad():
            for name, expected in canonical.items():
                state[name].copy_(expected.to(state[name].device))
        for name, module in actor.named_modules(remove_duplicate=False):
            key = f'{name}.pe' if name else 'pe'
            if key in canonical:
                module.pe.requires_grad_(False)
    return dict(canonical_algorithm='architecture_PositionalEncoding_constructor', tables=entries,
                modified_before_repair=changed, repaired=bool(repair and changed),
                table_aliases_count=len(entries), optimizer_migration='discard_all_old_optimizer_state')


def repair_stage2_weights(source, architecture_checkpoint, *, stats, kinematics, output=None):
    """审计或显式导出修复后的 weights-only Actor；资产与初始化模型身份必须完全一致。"""
    from gem.closedloop.checkpoint import save_stage1_checkpoint
    from gem.closedloop.dppo.checkpoint import FULL_STATE_VERSIONS
    from gem.closedloop.dppo.trainer import load_actor
    source, architecture_checkpoint = Path(source).resolve(), Path(architecture_checkpoint).resolve()
    if output is not None and Path(output).resolve() in (source, architecture_checkpoint):
        raise ValueError('Position repair must never overwrite a source checkpoint')
    if output is not None and Path(output).exists():
        raise FileExistsError(f'Refusing to overwrite repaired weights: {output}')
    payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('version') not in FULL_STATE_VERSIONS:
        raise ValueError('Position repair input must be a recognized Stage2 full-state checkpoint')
    identities = dict(checkpoint=file_sha256(architecture_checkpoint), stats=file_sha256(stats),
                      kinematics=file_sha256(kinematics))
    assets = payload.get('identity', {}).get('assets', {})
    if any(assets.get(key) != digest for key, digest in identities.items()):
        raise ValueError('Stage2 source assets differ from architecture checkpoint/stats/FK')
    config = dict(paths=dict(checkpoint=str(architecture_checkpoint), stats=str(stats), kinematics=str(kinematics)),
                  runtime=dict(genmo_device='cpu'))
    actor, train_config, _ = load_actor(config)
    if payload['identity'].get('actor_interface') != dict(actor.interface_config):
        raise ValueError('Stage2 source Actor interface differs from the architecture checkpoint')
    canonical = canonical_position_tables(actor)
    actor.load_state_dict(payload['actor'], strict=True)
    for name, value in actor.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite source Actor value: {name}')
    audit = audit_position_tables(actor, canonical=canonical, repair=output is not None)
    report = dict(mode='stage2_fixed_position_weights_only_repair' if output else 'stage2_fixed_position_audit',
                  source=str(source), source_sha256=file_sha256(source), source_iteration=payload.get('state', {}).get('iteration'),
                  architecture_checkpoint=str(architecture_checkpoint), assets=identities, position_tables=audit,
                  source_asset_identity=payload['identity'], optimizer_restored=False, global_step_restored=False,
                  policy_baseline_must_be_reestablished=True)
    if output is not None:
        from omegaconf import OmegaConf
        destination = Path(output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 同文件系统完成、fsync 后用排他硬链接发布，失败不会留下伪完整目标文件。
        with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.repair_', dir=destination.parent) as temporary:
            staged = Path(temporary) / 'weights.pt'
            save_stage1_checkpoint(actor, staged, config=OmegaConf.to_container(train_config, resolve=True),
                                   global_step=0, warm_start_report=report)
            with staged.open('rb') as stream:
                os.fsync(stream.fileno())
            os.link(staged, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        report.update(output=str(Path(output).resolve()), output_sha256=file_sha256(output))
    return report
