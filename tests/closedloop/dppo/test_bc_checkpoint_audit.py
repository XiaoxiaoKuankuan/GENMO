"""服务器1八rank运行的BC断点分布独立审计回归。

旧实现只允许rank0保存BC状态，新大规模合同让八卡各自承担全局128中的16条监督样本。
本测试直接检查审计函数的归属、批量、累计更新以及独立采样随机流，并保留旧rank0兼容。
负例覆盖丢失分片、错误归一化、更新数偏差和rank0快照覆盖全部rank，不能靠放宽断点
检查解决真实训练产物的误拒绝。此处不运行模型或物理，不代替真实八卡完整恢复验收。
"""
import copy

import pytest
import torch

from tools.eval.audit_closedloop_stage10_v2 import _audit_bc_rank_states


def states(sharded=True):
    contract = dict(bc_distribution='all_ranks' if sharded else 'rank0',
                    bc_batch=128 if sharded else 2, bc_weight=.1)
    ranks = [dict(samplers={}) for _ in range(8)]
    for rank in range(8 if sharded else 1):
        ranks[rank]['samplers']['bc'] = dict(batch_size=16 if sharded else 2,
            bc_update_steps=8, generator=torch.Generator().manual_seed(42+1000003*rank).get_state())
    return ranks, contract


@pytest.mark.parametrize('sharded', [False, True])
def test_bc_checkpoint_distribution_matches_contract(sharded):
    ranks, contract = states(sharded)
    _audit_bc_rank_states(ranks, contract, 8)


def test_legacy_absent_distribution_defaults_to_rank_zero():
    ranks, contract = states(False)
    del contract['bc_distribution']
    _audit_bc_rank_states(ranks, contract, 8)


@pytest.mark.parametrize('fault', ['missing', 'batch', 'updates', 'rng', 'rng_missing', 'divisibility', 'distribution'])
def test_sharded_bc_checkpoint_tampering_is_rejected(fault):
    ranks, contract = states()
    if fault == 'missing': ranks[7]['samplers'].pop('bc')
    if fault == 'batch': ranks[3]['samplers']['bc']['batch_size'] = 128
    if fault == 'updates': ranks[5]['samplers']['bc']['bc_update_steps'] = 7
    if fault == 'rng': ranks[6]['samplers']['bc'] = copy.deepcopy(ranks[0]['samplers']['bc'])
    if fault == 'rng_missing': ranks[1]['samplers']['bc'].pop('generator')
    if fault == 'divisibility': contract['bc_batch'] = 127
    if fault == 'distribution': contract['bc_distribution'] = 'unknown'
    with pytest.raises(ValueError): _audit_bc_rank_states(ranks, contract, 8)


def test_rank_zero_contract_rejects_nonowner_state():
    ranks, contract = states(False)
    ranks[2]['samplers']['bc'] = copy.deepcopy(ranks[0]['samplers']['bc'])
    with pytest.raises(ValueError): _audit_bc_rank_states(ranks, contract, 8)


def test_inactive_bc_has_zero_updates_or_no_state():
    ranks, contract = states()
    contract['bc_weight'] = 0.
    for item in ranks: item['samplers']['bc']['bc_update_steps'] = 0
    _audit_bc_rank_states(ranks, contract, 8)
    ranks[7]['samplers'].pop('bc')
    _audit_bc_rank_states(ranks, contract, 8)


def test_retired_metadata_does_not_claim_rng_byte_validation():
    ranks, contract = states()
    for item in ranks: item['samplers']['bc'].pop('generator')
    report = _audit_bc_rank_states(ranks, contract, 8, complete=False)
    assert report['rng_scope'] == 'unavailable_in_retired_metadata'
    with pytest.raises(ValueError): _audit_bc_rank_states(ranks, contract, 8)


def test_retired_metadata_still_requires_shard_counts():
    ranks, contract = states()
    ranks[2]['samplers']['bc']['bc_update_steps'] = 7
    with pytest.raises(ValueError): _audit_bc_rank_states(ranks, contract, 8, complete=False)
