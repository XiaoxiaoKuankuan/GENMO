"""汇总阶段的权重身份与GPU计量合同回归。

相同配置名称不能代替实际模型指纹、文件SHA及真实rollout来源；显存只按八张卡
的原始监控读数统计。小夹具不代表真实物理结果或GPU性能。
"""
import copy
import pytest

from gem.closedloop.dppo.training_checkpoint import file_sha256
from tools.eval.summarize_stage2_actor_lr import verify_actor, gpu_peak


def test_summary_rejects_replaced_actor_or_wrong_case(tmp_path):
    path = tmp_path / 'actor.pt'
    path.write_bytes(b'weight fixture')
    expected = dict(path=str(path), sha256=file_sha256(path), actor_fingerprint='fingerprint')
    report = dict(actor_identity=dict(model_fingerprint='fingerprint'), evaluation_identity=dict(
        training_identity=dict(actor_initialization=dict(path=str(path), sha256=expected['sha256'], source_case_sha256='case'))))
    verify_actor(dict(actor=expected), report, 'case')
    wrong = copy.deepcopy(report)
    wrong['actor_identity']['model_fingerprint'] = 'other'
    with pytest.raises(ValueError, match='differs'):
        verify_actor(dict(actor=expected), wrong, 'case')
    with pytest.raises(ValueError, match='differs'):
        verify_actor(dict(actor=expected), report, 'other case')
    path.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='weights changed'):
        verify_actor(dict(actor=expected), report, 'case')


def test_gpu_summary_keeps_polled_peak_scope(tmp_path):
    path = tmp_path / 'gpu.csv'
    path.write_text(''.join(f'timestamp, {i}, {1024 * (i + 1)}, 50\n' for i in range(8)))
    result = gpu_peak(path)
    assert result['max_memory_GiB'] == 8
    assert result['per_gpu_GiB']['0'] == 1
    assert result['sampling_interval_seconds'] == 5
    assert result['includes_external_PhysX_allocations']
    path.write_text('incomplete\n')
    with pytest.raises(ValueError, match='Incomplete'):
        gpu_peak(path)
