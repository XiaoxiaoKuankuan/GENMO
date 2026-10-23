"""纯 CPU 概率取证工具的统计、身份、实际 minibatch 映射与写入边界回归。

使用现有 ProbabilityRecordWriter 在 pytest 临时目录生成真实格式的轻量证据，
通过稀疏原 manifest 索引和不同参数快照发现“把存储行号当全局索引”“拿更新后
系数解释刚执行梯度”等错误。极端 log-ratio 验证稳定 ESS 与生产概率失败边界，
正负/零优势和裁剪等号验证分支语义；来源及符号分解应恢复总体代理目标。

另覆盖 SHA/列/顺序/固定优势篡改、无候选时不归因实际 minibatch、候选订单和
记录绑定、来源目录及符号链接写入保护、原子发布拒绝覆盖与合成文件 CLI。
所有文件均为临时测试证据；不加载训练模型、不连接服务器、不初始化 CUDA，
这些回归不替代停训后的真实文件 CLI 或八卡/物理收益验收。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools.validate_stage2_full_chain import ProbabilityRecordWriter
from tools.summarize_stage2_probability_records import (
    COLUMNS, mass_statistics, minibatch_plan, probability_vectors,
    publish_json, summarize,
)
from gem.closedloop.dppo.gaussian_probability import chain_statistics, chain_diagnostics


def records(log_ratios):
    ell = torch.as_tensor(log_ratios, dtype=torch.float64)
    values = torch.zeros(len(ell), 5, 8, dtype=torch.float64)
    values[:, :, 3] = torch.arange(2, len(ell)+2, dtype=torch.float64)[:, None]
    values[:, 0, 2] = ell
    values[:, :, 0] = ell.abs()[:, None] / 50
    values[:, :, 1] = values[:, :, 0] / values[:, :, 3]
    return values


def diagnostics(values, advantage):
    statistics = chain_statistics(values[:, :, 2], values[:, :, 0], values[:, :, 3])
    return chain_diagnostics(statistics, advantage, clip=.1)


@pytest.fixture
def evidence(tmp_path):
    source, fixed = tmp_path/'source', tmp_path/'fixed_case'
    source.mkdir(); fixed.mkdir()
    case = fixed/'case.json'
    case.write_text('{"fixture": "immutable_case"}\n')
    sha = hashlib.sha256(case.read_bytes()).hexdigest()
    selected = [1, 3, 4, 7, 8, 10]
    manifest = [dict(owner_rank=i % 2, local_index=i//2, valid=i in selected, has_free=True)
                for i in range(11)]
    advantage = torch.tensor([1., -2., 0., .7, -1., 2.], dtype=torch.float64)
    sources = ['AIST', 'FineDance', 'AIOZ', 'FineDance', 'AIST', 'Mine']
    before = records(torch.zeros(6, dtype=torch.float64))
    after1 = records(torch.tensor([1.5, 2., .5, .2, .5, .95], dtype=torch.float64).log())
    after2 = records(torch.tensor([1.2, 1.8, .7, .4, .6, 1.05], dtype=torch.float64).log())
    writer = ProbabilityRecordWriter(source/'probability', case_sha256=sha,
                                     manifest=manifest, trainable_steps=range(4, 9))
    for stage, value in [('before_update', before), ('accepted_Adam', after1),
                         ('accepted_Adam', after2), ('final', after2)]:
        receipt = writer.observe(stage, value, advantage, sources)
    index_path = writer.directory/'index.json'
    index = json.loads(index_path.read_text())
    order = [7, 1, 10, 8, 3, 4]
    row1, row2 = [3, 0, 5, 4], [1, 2]
    steps = [dict(optimizer_step=1, epoch=0, global_chain_count=4,
                  pre_step_chain=diagnostics(before[row1], advantage[row1]),
                  full_rollout_before=diagnostics(before, advantage),
                  full_rollout_after=diagnostics(after1, advantage)),
             dict(optimizer_step=2, epoch=0, global_chain_count=2,
                  pre_step_chain=diagnostics(after1[row2], advantage[row2]),
                  full_rollout_before=diagnostics(after1, advantage),
                  full_rollout_after=diagnostics(after2, advantage))]
    candidate = dict(probability_records=receipt, acceptance=dict(passed=True),
                     hard_kl_acceptance=dict(passed=True),
                     actor=dict(optimizer_steps=2, planned_optimizer_steps=2,
                                epoch_orders=[order], steps=steps))
    candidate_path = source/'candidate.json'
    candidate_path.write_text(json.dumps(candidate))
    return dict(source=source, case=case, sha=sha, probability=writer.directory,
                index_path=index_path, index=index, candidate_path=candidate_path,
                candidate=candidate, output=tmp_path/'analysis'/'summary.json',
                advantage=advantage, selected=selected, before=before, after1=after1)


def run_summary(evidence, *, candidate=True, **overrides):
    kwargs = dict(clip=.1, output=evidence['output'], expected_case_sha256=evidence['sha'],
                  candidate_json=evidence['candidate_path'] if candidate else None)
    kwargs.update(overrides)
    return summarize(evidence['probability'], evidence['case'], **kwargs)


def rewrite_payload(evidence, position, mutate):
    index = json.loads(evidence['index_path'].read_text())
    path = evidence['probability']/index['entries'][position]['path']
    payload = torch.load(path, map_location='cpu', weights_only=True)
    mutate(payload)
    torch.save(payload, path)
    data = path.read_bytes()
    index['entries'][position].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    evidence['index_path'].write_text(json.dumps(index))


def test_log_mass_is_stable_scale_invariant_and_zero_mass_is_null():
    ell = torch.tensor([1000., 0., -1000.], dtype=torch.float64)
    shares, result = mass_statistics(ell)
    shifted_shares, shifted = mass_statistics(ell + 123.)
    assert result['ess'] == shifted['ess'] == 1.
    assert torch.equal(shares, shifted_shares) and float(shares.sum()) == 1.
    _, uniform = mass_statistics(torch.zeros(17, dtype=torch.float64))
    assert uniform['ess'] == pytest.approx(17.)
    empty, missing = mass_statistics(torch.full((5,), -float('inf'), dtype=torch.float64))
    assert not empty.any() and missing['nonzero_count'] == 0
    assert missing['ess'] is missing['ess_fraction'] is missing['log_total_mass'] is missing['top_mass'] is None
    for invalid in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='NaN'):
            mass_statistics(torch.tensor([invalid], dtype=torch.float64))


def test_sign_clip_equalities_and_full_chain_sum_match_actual_coefficients():
    low, high = math.log1p(-.1), math.log1p(.1)
    ell = torch.tensor([low, high, 2., -2., 2., -2., 0.], dtype=torch.float64)
    advantage = torch.tensor([-1., 1., 1., -1., -1., 1., 0.], dtype=torch.float64)
    value = records(ell)
    value[:, 0, 2] = ell / 2
    value[:, 4, 2] = ell / 2
    vectors = probability_vectors(dict(records=value, advantages=advantage, trainable_steps=list(range(4, 9))), .1)
    assert torch.equal(vectors['log_ratio'], ell)
    assert vectors['active_clip'].tolist() == [False, False, True, True, False, False, False]
    expected = torch.tensor([.9, 1.1, 0., 0., math.exp(2), math.exp(-2), 0.], dtype=torch.float64)
    torch.testing.assert_close(vectors['log_coefficient'].exp(), expected, atol=1e-15, rtol=1e-15)


def test_real_writer_summary_maps_sparse_indices_to_pre_update_minibatches(evidence, monkeypatch):
    def forbid_cuda(*args, **kwargs):
        raise AssertionError('CPU summary must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbid_cuda)
    report = run_summary(evidence)
    assert report['semantics']['parameter_gradient_attribution'] is False
    assert report['semantics']['physical_effectiveness_evidence'] is False
    assert len(report['snapshots']) == 4
    assert report['snapshots'][0]['ratio']['ess'] == pytest.approx(6.)
    steps = report['actual_minibatches']
    assert steps[0]['global_manifest_indices'] == [7, 1, 10, 8]
    assert steps[0]['record_rows'] == [3, 0, 5, 4]
    assert steps[1]['global_manifest_indices'] == [3, 4]
    assert steps[1]['record_rows'] == [1, 2]
    assert steps[1]['parameter_snapshot_entry'] == '001_accepted_Adam.pt'
    assert steps[1]['ratio']['ess'] == pytest.approx(2.5**2 / (4.+.25))
    assert steps[1]['coefficient']['ess'] == 1.
    assert steps[1]['groups']['source:FineDance']['coefficient_mass_share'] == 1.
    assert steps[1]['groups']['source:AIOZ']['coefficient_mass_share'] == 0.
    groups = report['snapshots'][-1]['groups']
    assert groups['source:FineDance']['advantage_mean'] == pytest.approx(-.65)
    assert groups['source:FineDance']['advantage_population_variance'] == pytest.approx(1.8225)
    assert groups['source:FineDance']['chain_joint_kl']['max'] == pytest.approx(abs(math.log(.4)) / 10)
    tail = report['snapshots'][-1]['highest_joint_kl_chains']
    assert tail[0]['global_manifest_index'] == 7 and tail[0]['source'] == 'FineDance'
    assert tail[0]['advantage'] == .7 and tail[0]['joint_log_ratio'] == pytest.approx(math.log(.4))
    assert len(tail) == 6 and all(tail[i]['chain_joint_kl'] >= tail[i+1]['chain_joint_kl'] for i in range(5))
    assert sum(groups['sign:'+s]['gain_from_initial_global_mean'] for s in ('positive', 'negative', 'zero')) == pytest.approx(groups['all']['gain_from_initial_global_mean'])
    assert json.loads(evidence['output'].read_text()) == report


def test_without_candidate_does_not_claim_actual_minibatch(evidence):
    report = run_summary(evidence, candidate=False)
    assert report['actual_minibatches'] is None
    assert report['actual_minibatch_mapping_reason'] == 'candidate_JSON_not_supplied'
    assert all('not_executed_minibatch' in row['scope'] for row in report['snapshots'])


def test_extreme_final_ratio_keeps_stable_ess_but_no_surrogate(evidence):
    def extreme(payload):
        payload['records'][:, 0, 2] = torch.tensor([-1000., 1000., 0., 0., 0., 0.])
    rewrite_payload(evidence, -1, extreme)
    result = run_summary(evidence, candidate=False)['snapshots'][-1]
    assert not result['production_probability_valid'] and result['probability_failure']
    assert result['ratio']['ess'] == result['coefficient']['ess'] == 1.
    assert result['groups']['all']['surrogate_contribution_global_mean'] is None
    assert result['groups']['all']['gain_from_initial_global_mean'] is None


def test_all_zero_advantages_have_null_coefficient_ess_but_measured_zero_gain(evidence):
    for i in range(4):
        rewrite_payload(evidence, i, lambda payload: payload['advantages'].zero_())
    report = run_summary(evidence, candidate=False)
    for snapshot in report['snapshots']:
        assert snapshot['coefficient']['ess'] is None
        assert snapshot['groups']['all']['coefficient_mass_share'] is None
        assert snapshot['groups']['all']['gain_from_initial_global_mean'] == 0.


@pytest.mark.parametrize('fault', ['columns', 'advantage', 'source', 'count', 'negative_kl',
                                  'average_kl', 'nonfinite', 'physical', 'manifest', 'stage'])
def test_changed_evidence_contract_fails_without_output(evidence, fault):
    def mutate(p):
        if fault == 'columns': p['columns'] = list(reversed(COLUMNS))
        if fault == 'advantage': p['advantages'][0] += 1
        if fault == 'source': p['sources'][0] = 'different'
        if fault == 'count': p['records'][0, :, 3] += 1
        if fault == 'negative_kl': p['records'][0, 0, 0] = -.1
        if fault == 'average_kl': p['records'][0, 0, 1] += 1
        if fault == 'nonfinite': p['records'][0, 0, 2] = float('nan')
        if fault == 'physical': p['records'][0, 0, 4] = 1
        if fault == 'manifest': p['global_manifest'].reverse()
        if fault == 'stage': p['stage'] = 'final'
    rewrite_payload(evidence, 1, mutate)
    with pytest.raises(ValueError):
        run_summary(evidence, candidate=False)
    assert not evidence['output'].exists()


@pytest.mark.parametrize('fault', ['sha', 'path', 'sequence', 'case', 'external_case'])
def test_identity_and_index_faults_fail_closed(evidence, fault):
    index = evidence['index']
    if fault == 'sha': index['entries'][1]['sha256'] = '0'*64
    if fault == 'path': index['entries'][0]['path'] = '../escape.pt'
    if fault == 'sequence': index['entries'][1]['stage'] = 'before_update'
    if fault == 'case': index['source_case_sha256'] = '0'*64
    evidence['index_path'].write_text(json.dumps(index))
    with pytest.raises(ValueError):
        run_summary(evidence, candidate=False, expected_case_sha256='0'*64 if fault == 'external_case' else evidence['sha'])
    assert not evidence['output'].exists()


@pytest.mark.parametrize('fault', ['receipt', 'duplicate', 'omission', 'count', 'order', 'epoch',
                                  'pre_probability', 'after_probability'])
def test_candidate_binding_or_actual_minibatch_mapping_mismatch_fails(evidence, fault):
    candidate = evidence['candidate']
    actor = candidate['actor']
    if fault == 'receipt': candidate['probability_records']['entries'][1]['sha256'] = '0'*64
    if fault == 'duplicate': actor['epoch_orders'][0][0] = actor['epoch_orders'][0][1]
    if fault == 'omission': actor['epoch_orders'][0].pop()
    if fault == 'count': actor['steps'][1]['global_chain_count'] = 3
    if fault == 'order': actor['epoch_orders'][0] = list(reversed(actor['epoch_orders'][0]))
    if fault == 'epoch': actor['steps'][1]['epoch'] = 1
    if fault == 'pre_probability': actor['steps'][1]['pre_step_chain']['objective_mean'] += 1
    if fault == 'after_probability': actor['steps'][1]['full_rollout_after']['objective_mean'] += 1
    evidence['candidate_path'].write_text(json.dumps(candidate))
    with pytest.raises(ValueError):
        run_summary(evidence)
    assert not evidence['output'].exists()


def test_multiple_epochs_map_each_full_order_from_its_own_start(evidence):
    candidate = copy.deepcopy(evidence['candidate'])
    candidate['actor']['epoch_orders'] = [evidence['selected'], list(reversed(evidence['selected']))]
    for i, step in enumerate(candidate['actor']['steps']):
        step.update(epoch=i, global_chain_count=6)
    baseline_path = evidence['probability']/evidence['index']['entries'][0]['path']
    manifest = torch.load(baseline_path, weights_only=True)['global_manifest']
    plan = minibatch_plan(candidate, evidence['index'], manifest)
    assert plan[0]['record_rows'] == list(range(6))
    assert plan[1]['record_rows'] == list(reversed(range(6)))


@pytest.mark.parametrize('location', ['probability', 'case', 'candidate', 'symlink', 'existing'])
def test_output_cannot_modify_source_or_existing_evidence(evidence, tmp_path, location):
    before = {p: p.read_bytes() for root in (evidence['source'], evidence['case'].parent)
              for p in root.rglob('*') if p.is_file()}
    if location == 'probability': output = evidence['probability']/'analysis.json'
    if location == 'case': output = evidence['case'].parent/'analysis.json'
    if location == 'candidate': output = evidence['source']/'analysis.json'
    if location == 'symlink':
        alias = tmp_path/'alias'; alias.symlink_to(evidence['probability'], target_is_directory=True)
        output = alias/'analysis.json'
    if location == 'existing':
        output = tmp_path/'existing.json'; output.write_bytes(b'preserve this output')
    with pytest.raises(ValueError, match='输出'):
        run_summary(evidence, output=output)
    assert all(p.read_bytes() == value for p, value in before.items())
    if location == 'existing': assert output.read_bytes() == b'preserve this output'
    else: assert not output.exists()


def test_atomic_publication_refuses_late_collision_and_cleans_only_own_temporary(tmp_path, monkeypatch):
    output = tmp_path/'new.json'
    def racing_writer(source, target):
        Path(target).write_text('preserve competing evidence')
        raise FileExistsError(target)
    monkeypatch.setattr(os, 'link', racing_writer)
    with pytest.raises(FileExistsError):
        publish_json(output, {'valid': True})
    assert output.read_text() == 'preserve competing evidence'
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize('kind', ['index', 'case'])
def test_input_file_symlink_protects_its_actual_source_directory(evidence, tmp_path, kind):
    alias = tmp_path/'aliases'; alias.mkdir()
    probability, case = evidence['probability'], evidence['case']
    if kind == 'index':
        (alias/'index.json').symlink_to(evidence['index_path'])
        probability, output = alias, evidence['probability']/'forbidden.json'
    else:
        (alias/'case.json').symlink_to(evidence['case'])
        case, output = alias, evidence['case'].parent/'forbidden.json'
    with pytest.raises(ValueError, match='输出'):
        summarize(probability, case, clip=.1, output=output)
    assert not output.exists()


def test_cpu_cli_on_synthetic_writer_files(evidence):
    tool = Path(__file__).resolve().parents[3]/'tools'/'summarize_stage2_probability_records.py'
    process = subprocess.run([sys.executable, '-B', str(tool), '--probability-records', str(evidence['index_path']),
        '--case', str(evidence['case'].parent), '--candidate-json', str(evidence['candidate_path']),
        '--clip', '.1', '--expected-case-sha256', evidence['sha'], '--output', str(evidence['output'])],
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, check=True)
    stdout = json.loads(process.stdout)
    assert stdout['snapshots'] == 4 and stdout['actual_minibatches'] == 2
    assert json.loads(evidence['output'].read_text())['inputs']['case']['sha256'] == evidence['sha']
