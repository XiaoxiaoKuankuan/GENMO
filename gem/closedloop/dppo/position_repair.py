"""Stage1固定正余弦位置表的规范身份检查。

按当前架构构造规范表，核对共享别名及有限值，装配Actor时拒绝被修改的位置表。
不再提供历史Stage2修复权重、旧checkpoint转换或优化器迁移入口。
"""
from __future__ import annotations

import torch


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
