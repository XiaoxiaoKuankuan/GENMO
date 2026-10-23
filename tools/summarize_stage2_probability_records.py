"""只读汇总 Stage2 固定行为数据的完整链概率、PPO 系数与来源分解。

输入是 validate_stage2_full_chain 生成的 probability 目录或 index.json、原固定
case 目录或 case.json，以及可选候选 JSON。逐项校验 SHA、字节数、列合同、步骤、
链顺序、自由坐标与固定优势；所有张量只加载到 CPU，不加载 Actor、不启动分布式。
联合 log-ratio 明确为 log p_new - log p_old，沿全部可训练随机步骤求和后组成
一条链的比值，不按坐标或去噪步平均。ESS 在 log 域平移后计算，不截断真实 ratio。

输出按全量快照、来源、优势符号和 ratio 桶报告有限样本 ratio ESS，以及未被
PPO active clipping 置零的 |A*r| 系数集中度、质量份额与 surrogate 增益。
各组同时保留逐链联合 KL 的 mean/p95/p99/max，全局最高十条链只列原 manifest
索引、source、优势、log-ratio 与 KL，不导出原状态；KL 尾部大不直接等于模型差。
优势沿用原全局标准化结果，不重新按 source 标准化；零质量的 ESS/份额为 null，
不冒充零误差。极端有限 log-ratio 仍可做稳定质量分析，但生产概率验收失败时
surrogate 留空，不能把这种最终拒绝快照当成有效更新。

只有提供与 index 逐项 SHA 绑定的候选 JSON，才使用 epoch_orders 的原 manifest
索引恢复每个已接受 Adam 步的实际 minibatch，并从前一个参数快照计算更新前系数。
这些统计不包含拒绝试探的独立梯度、不包括 BC，也不是完整参数梯度归因；没有逐链
score 向量就不能推断梯度范数、抵消或 Adam 贡献。全量快照统计不称为实际小批次。
所有代理目标与 ESS 均不证明真实物理收益、独立样本数量或部署安全性。

仅向显式独立的新 JSON 路径原子发布结果，拒绝已有文件、输入目录内写入、路径
逃逸或证据不一致；不修改概率证据、case、候选、checkpoint 或训练算法。CPU 合同
测试可以验证计算与索引，但真实文件的 CLI 端到端取证应由训练停止后的独立流程执行。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from gem.closedloop.dppo.denoising_contract import validate_trainable_steps
from gem.closedloop.dppo.gaussian_probability import chain_statistics, ppo_surrogate


SCHEMA = 'genmo.stage2.full_chain_probability_records.v1'
COLUMNS = ['joint_kl', 'mean_coordinate_kl', 'joint_log_ratio', 'free_count',
           'normalized_mean_change_rms', 'root_mean_change_rms',
           'orientation_mean_change_rms', 'joint_mean_change_rms']
SCOPE = 'fixed_behavior_full_rollout_conditional_Gaussian_statistics_not_physical_return'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_bytes(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return data, dict(path=str(path), size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def json_file(path):
    data, identity = file_bytes(path)
    def invalid(value):
        raise ValueError('输入 JSON 含非有限常量：' + value)
    return json.loads(data, parse_constant=invalid), identity


def independent_output(path, protected_roots):
    """解析符号链接后保护来源；输出必须是独立且尚不存在的 JSON 文件。"""
    original = Path(path)
    require(not original.exists() and not original.is_symlink(), '输出已存在，拒绝覆盖')
    output = original.resolve()
    require(output.suffix == '.json', '输出必须显式指定 .json 文件')
    for root in protected_roots:
        root = Path(root).resolve(strict=True)
        require(not output.is_relative_to(root), '输出不得位于输入证据目录内：' + str(root))
    return output


def publish_json(path, value):
    """先完成有限 JSON 序列化，再用独占硬链接发布；失败只清理自身临时文件。"""
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def mass_statistics(log_weights):
    """有限总体的稳定 ESS；-inf 是零质量，未对输入概率做任何裁剪。"""
    require(log_weights.device.type == 'cpu' and log_weights.dtype == torch.float64,
            '质量计算只接受 CPU FP64 向量')
    require(log_weights.ndim == 1 and not torch.isnan(log_weights).any()
            and not torch.isposinf(log_weights).any(), '质量向量包含 NaN/+inf 或形状错误')
    positive = torch.isfinite(log_weights)
    shares = torch.zeros_like(log_weights)
    count, population = int(positive.sum()), len(log_weights)
    if count == 0:
        return shares, dict(ess=None, ess_fraction=None, ess_fraction_nonzero=None,
                            nonzero_count=0, log_total_mass=None, maximum_mass=None, top_mass=None)
    maximum = log_weights[positive].max()
    weights = (log_weights[positive] - maximum).exp()
    total = weights.sum()
    weights /= total
    shares[positive] = weights
    ess = float(weights.square().sum().reciprocal())
    require(1. - 1e-10 <= ess <= count + 1e-8, 'ESS 超出有限样本范围')
    ranked = weights.sort(descending=True).values
    top = {str(fraction): float(ranked[:max(1, math.ceil(population * fraction))].sum())
           for fraction in (.001, .01, .05, .10)}
    return shares, dict(ess=ess, ess_fraction=ess/population, ess_fraction_nonzero=ess/count,
                        nonzero_count=count, log_total_mass=float(maximum + total.log()),
                        maximum_mass=float(ranked[0]), top_mass=top)


def validate_payload(payload, *, case_sha256, steps, chains, baseline=None):
    require(payload.get('schema') == SCHEMA and payload.get('scope') == SCOPE,
            '概率记录 schema/scope 不符')
    require(payload.get('source_case_sha256') == case_sha256 and payload.get('columns') == COLUMNS,
            'case SHA 或概率列合同不符')
    require(payload.get('trainable_steps') == steps, '随机步骤顺序不符')
    validate_trainable_steps(steps)
    records, advantage = payload['records'], payload['advantages']
    require(records.device.type == advantage.device.type == 'cpu'
            and records.dtype == advantage.dtype == torch.float64
            and records.shape == (chains, len(steps), 8) and advantage.shape == (chains,)
            and torch.isfinite(records).all() and torch.isfinite(advantage).all(),
            '概率记录必须是对齐且有限的 CPU FP64 张量')
    count = records[:, :, 3]
    require((count >= 0).all() and torch.equal(count, count.round())
            and (count.sum(1) > 0).all() and (records[:, :, 0] >= 0).all(), '自由坐标数或 KL 非法')
    require(torch.allclose(records[:, :, 1] * count, records[:, :, 0], atol=1e-12, rtol=1e-10),
            '坐标平均 KL 与联合 KL 不一致')
    require((records[:, :, :3][count == 0] == 0).all(), '零自由度步骤贡献了概率')
    require(payload.get('physical_change_columns_measured') is False
            and (records[:, :, 4:] == 0).all(), '当前记录合同的物理变化槽应明确未测量')
    manifest, sources = payload['global_manifest'], payload['sources']
    require(len(manifest) == len(sources) == chains and all(isinstance(s, str) for s in sources),
            'manifest/source 链数不符')
    for row in manifest:
        require(row.get('valid') is True and row.get('has_free') is True
                and all(type(row.get(k)) is int and row[k] >= 0
                        for k in ('global_manifest_index', 'owner_rank', 'local_index')),
                'manifest 包含非 eligible 链或非法索引')
    ids = [row['global_manifest_index'] for row in manifest]
    owners = [(row['owner_rank'], row['local_index']) for row in manifest]
    require(ids == sorted(set(ids)) and len(set(owners)) == chains, 'manifest 顺序或身份重复')
    if baseline is not None:
        require(manifest == baseline['global_manifest'] and sources == baseline['sources']
                and torch.equal(advantage, baseline['advantages'])
                and torch.equal(count, baseline['records'][:, :, 3]),
                '快照间优势、manifest、source 或自由坐标发生变化')


def probability_vectors(payload, clip):
    """复用生产 PPO 计算精确 surrogate；非法指数范围只保留 log 域诊断。"""
    records, advantage = payload['records'], payload['advantages']
    statistics = chain_statistics(records[:, :, 2], records[:, :, 0], records[:, :, 3],
                                  steps=payload['trainable_steps'])
    ell = statistics['joint_log_ratio']
    require(torch.isfinite(ell).all() and torch.isfinite(statistics['joint_kl']).all(),
            '全链 log-ratio/KL 求和非有限')
    low, high = math.log1p(-clip), math.log1p(clip)
    clipped = (ell < low) | (ell > high)
    active_clip = ((advantage > 0) & (ell > high)) | ((advantage < 0) & (ell < low))
    active = advantage.ne(0) & ~active_clip
    coefficient = torch.full_like(ell, -float('inf'))
    coefficient[active] = advantage[active].abs().log() + ell[active]
    require(torch.isfinite(coefficient[active]).all(), '活跃系数 log 质量非有限')
    try:
        surrogate = ppo_surrogate(statistics, advantage, clip=clip)
        objective, error = surrogate['objective'], None
        require(torch.equal(surrogate['objective_clipped'], active_clip), '生产 clipping 分支不一致')
    except FloatingPointError as failure:
        objective, error = None, str(failure)
    return dict(log_ratio=ell, log_coefficient=coefficient, advantage=advantage,
                kl=statistics['joint_kl'], clipped=clipped, active_clip=active_clip,
                objective=objective, production_probability_valid=error is None, error=error)


def summarize_vectors(vectors, sources, manifest, *, baseline_objective, previous_objective, clip):
    """所有组的目标贡献以本次总体 N 为分母，可严格相加；不重新归一化优势。"""
    ell, advantage = vectors['log_ratio'], vectors['advantage']
    coefficient, objective = vectors['log_coefficient'], vectors['objective']
    n = len(advantage)
    ratio_mass, ratio_summary = mass_statistics(ell)
    coefficient_mass, coefficient_summary = mass_statistics(coefficient)
    def group(mask):
        count = int(mask.sum())
        if not count:
            return dict(chains=0, statistics=None)
        a = advantage[mask]
        kl = vectors['kl'][mask]
        quantiles = torch.quantile(kl, kl.new_tensor([.95, .99])).tolist()
        _, ratio = mass_statistics(ell[mask])
        _, coeff = mass_statistics(coefficient[mask])
        gains = dict(surrogate_contribution_global_mean=None, gain_from_initial_global_mean=None,
                     gain_from_previous_global_mean=None, gain_from_behavior_global_mean=None)
        if objective is not None:
            gains['surrogate_contribution_global_mean'] = float(objective[mask].sum()/n)
            gains['gain_from_behavior_global_mean'] = float((objective[mask]-a).sum()/n)
            for key, reference in (('gain_from_initial_global_mean', baseline_objective),
                                   ('gain_from_previous_global_mean', previous_objective)):
                if reference is not None:
                    gains[key] = float((objective[mask]-reference[mask]).sum()/n)
        return dict(chains=count, chain_fraction=count/n, advantage_mean=float(a.mean()),
                    advantage_population_variance=float(a.var(unbiased=False)),
                    advantage_positive_fraction=float(a.gt(0).double().mean()),
                    advantage_negative_fraction=float(a.lt(0).double().mean()),
                    advantage_zero_fraction=float(a.eq(0).double().mean()),
                    ratio_mass_share=float(ratio_mass[mask].sum()),
                    coefficient_mass_share=float(coefficient_mass[mask].sum())
                        if coefficient_summary['nonzero_count'] else None,
                    ratio=ratio, coefficient=coeff,
                    clip_fraction=float(vectors['clipped'][mask].double().mean()),
                    active_clip_fraction=float(vectors['active_clip'][mask].double().mean()),
                    chain_joint_kl=dict(mean=float(kl.mean()), p95=quantiles[0], p99=quantiles[1],
                                       max=float(kl.max())),
                    **gains)
    signs = dict(positive=advantage > 0, negative=advantage < 0, zero=advantage == 0)
    low, high = math.log1p(-clip), math.log1p(clip)
    # lower/2、upper*2 等相对阈值兼容任意合法 clip，不暗中固定 0.1。
    l01, l05, l2, l10 = low + math.log(.1), low + math.log(.5), high + math.log(2), high + math.log(10)
    buckets = {'r<lower/10': ell < l01, 'lower/10<=r<lower/2': (ell >= l01) & (ell < l05),
               'lower/2<=r<lower': (ell >= l05) & (ell < low),
               'lower<=r<=upper': (ell >= low) & (ell <= high),
               'upper<r<=2upper': (ell > high) & (ell <= l2),
               '2upper<r<=10upper': (ell > l2) & (ell <= l10), 'r>10upper': ell > l10}
    source_masks = {source: torch.tensor([s == source for s in sources], dtype=torch.bool, device='cpu')
                    for source in sorted(set(sources))}
    require(torch.stack(list(buckets.values())).sum(0).eq(1).all(), 'ratio 桶没有完整互斥覆盖')
    partitions = dict(sign=signs, source=source_masks, bucket=buckets)
    groups = {'all': group(torch.ones(n, dtype=torch.bool, device='cpu'))}
    for label, masks in partitions.items():
        require(torch.stack(list(masks.values())).sum(0).eq(1).all(), label + ' 分组没有完整覆盖')
        for name, mask in masks.items():
            groups[f'{label}:{name}'] = group(mask)
        for metric in ('ratio_mass_share', 'coefficient_mass_share', 'surrogate_contribution_global_mean',
                       'gain_from_initial_global_mean', 'gain_from_previous_global_mean', 'gain_from_behavior_global_mean'):
            expected = groups['all'][metric]
            values = [groups[f'{label}:{name}'].get(metric) for name in masks]
            if expected is not None:
                actual = math.fsum(value for value in values if value is not None)
                require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
                        label + ' 分组分解不守恒：' + metric)
    for sign, sign_mask in signs.items():
        for label, masks in (('source', source_masks), ('bucket', buckets)):
            for name, mask in masks.items():
                groups[f'{label}:{name}/sign:{sign}'] = group(mask & sign_mask)
    highest = torch.argsort(vectors['kl'], descending=True, stable=True)[:10].tolist()
    tail = [dict(global_manifest_index=manifest[i]['global_manifest_index'], source=sources[i],
                 advantage=float(advantage[i]), joint_log_ratio=float(ell[i]),
                 chain_joint_kl=float(vectors['kl'][i])) for i in highest]
    return dict(chains=n, production_probability_valid=vectors['production_probability_valid'],
                probability_failure=vectors['error'], ratio=ratio_summary, coefficient=coefficient_summary,
                highest_joint_kl_chains=tail,
                ratio_bucket_edges=dict(lower=1-clip, upper=1+clip, comparison='log_domain_strict_clip_boundaries'),
                groups=groups)


def minibatch_plan(candidate, index, manifest):
    """从受 SHA 绑定的订单和每步链数恢复原索引，不用快照行号冒充全局索引。"""
    receipt = candidate.get('probability_records') or {}
    require(all(receipt.get(key) == index[key] for key in
                ('source_case_sha256', 'trainable_steps', 'chains', 'entries', 'complete')),
            '候选 JSON 未逐项绑定当前概率 index')
    actor = candidate.get('actor')
    if actor is None:
        return None
    reports, orders = actor.get('steps'), actor.get('epoch_orders')
    require(isinstance(reports, list) and isinstance(orders, list), '候选缺少步骤或 epoch_orders')
    require(type(actor.get('optimizer_steps')) is int and actor['optimizer_steps'] == len(reports)
            == len(index['entries'])-2, '候选已接受步数与概率快照不符')
    ids = [m['global_manifest_index'] for m in manifest]
    mapping = {value: i for i, value in enumerate(ids)}
    for order in orders:
        require(isinstance(order, list) and all(type(i) is int for i in order)
                and sorted(order) == ids, 'epoch_orders 不是 eligible 原索引的完整排列')
    positions, current_epoch, cursor = [], 0, 0
    for ordinal, step in enumerate(reports, 1):
        epoch, count = step.get('epoch'), step.get('global_chain_count')
        require(type(step.get('optimizer_step')) is int and step['optimizer_step'] == ordinal and type(epoch) is int
                and current_epoch <= epoch <= current_epoch+1 and epoch < len(orders)
                and type(count) is int and count > 0, '候选步骤顺序、epoch 或链数非法')
        if epoch != current_epoch:
            require(cursor == len(ids), '前一个 epoch 尚未完整消费便跳到下一 epoch')
            current_epoch, cursor = epoch, 0
        require(cursor+count <= len(ids), 'minibatch 超出 epoch 订单')
        original = orders[epoch][cursor:cursor+count]
        positions.append(dict(optimizer_step=ordinal, epoch=epoch, global_manifest_indices=original,
                              record_rows=[mapping[i] for i in original], report=step))
        cursor += count
    return positions


def verify_logged_diagnostics(vectors, logged):
    require(isinstance(logged, dict) and vectors['production_probability_valid'],
            '实际 minibatch 缺少诊断或更新前概率非法')
    expected = dict(complete_chain_count=len(vectors['advantage']),
                    objective_mean=float(vectors['objective'].mean()),
                    clip_fraction=float(vectors['clipped'].double().mean()),
                    active_clip_fraction=float(vectors['active_clip'].double().mean()))
    for key, value in expected.items():
        actual = logged.get(key)
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, value, rel_tol=1e-8, abs_tol=1e-9),
                '候选更新前概率诊断与索引映射不符：' + key)
    measured = (logged.get('chain_joint_kl') or {}).get('mean')
    require(type(measured) in (int, float) and math.isfinite(measured)
            and math.isclose(measured, float(vectors['kl'].mean()), rel_tol=1e-8, abs_tol=1e-9),
            '候选更新前 KL 与索引映射不符')


def summarize(probability, case, *, clip, output, candidate_json=None, expected_case_sha256=None):
    require(math.isfinite(clip) and 0 < clip < 1, '必须显式提供实际配置的合法 PPO clip')
    probability, case = Path(probability).resolve(strict=True), Path(case).resolve(strict=True)
    index_path = (probability/'index.json' if probability.is_dir() else probability).resolve(strict=True)
    case_path = (case/'case.json' if case.is_dir() else case).resolve(strict=True)
    source = index_path.parent
    roots = [source, case_path.parent]
    if candidate_json is not None:
        candidate_json = Path(candidate_json).resolve(strict=True)
        roots.append(candidate_json.parent)
    output = independent_output(output, roots)
    index, index_identity = json_file(index_path)
    _, case_identity = file_bytes(case_path)
    case_sha = case_identity['sha256']
    require(expected_case_sha256 is None or expected_case_sha256 == case_sha, '固定 case SHA 不符')
    require(index.get('complete') is True and index.get('source_case_sha256') == case_sha,
            '概率 index 未完整结束或绑定了其他 case')
    entries, steps, chains = index['entries'], index['trainable_steps'], index['chains']
    require(type(chains) is int and chains > 0 and isinstance(entries, list) and len(entries) >= 2,
            'index 链数或快照数非法')
    require(isinstance(steps, list) and all(type(step) is int for step in steps), '随机步骤必须是整数有序列表')
    validate_trainable_steps(steps)
    expected_stages = ['before_update'] + ['accepted_Adam']*(len(entries)-2) + ['final']
    require([e.get('stage') for e in entries] == expected_stages, 'index 快照顺序非法')
    candidate = candidate_identity = None
    if candidate_json is not None:
        candidate, candidate_identity = json_file(candidate_json)
    snapshots, actual_minibatches, identities = [], [], []
    baseline = previous = baseline_vectors = previous_vectors = plan = None
    for ordinal, entry in enumerate(entries):
        name = entry['path']
        require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'),
                '概率文件路径逃逸')
        path = (source/name).resolve(strict=True)
        require(path.parent == source, '概率文件符号链接逃逸')
        data, identity = file_bytes(path)
        require(identity['sha256'] == entry['sha256'] and identity['size_bytes'] == entry['size_bytes'],
                '概率文件 SHA/字节数不符')
        payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        validate_payload(payload, case_sha256=case_sha, steps=steps, chains=chains, baseline=baseline)
        accepted_count = min(ordinal, len(entries)-2)
        require(payload.get('stage') == entry['stage']
                and type(entry.get('accepted_optimizer_steps_observed')) is int
                and entry['accepted_optimizer_steps_observed'] == accepted_count
                and type(payload.get('accepted_optimizer_steps_observed')) is int
                and payload.get('accepted_optimizer_steps_observed') == accepted_count,
                '快照阶段或接受步计数不符')
        vectors = probability_vectors(payload, clip)
        require(payload['stage'] == 'final' or vectors['production_probability_valid'],
                '更新前或已接受快照的生产概率非法')
        if baseline is None:
            baseline, baseline_vectors = payload, vectors
            if candidate is not None:
                plan = minibatch_plan(candidate, index, payload['global_manifest'])
        summary = summarize_vectors(vectors, payload['sources'], payload['global_manifest'],
                                    baseline_objective=baseline_vectors['objective'],
                                    previous_objective=(previous_vectors or baseline_vectors)['objective'], clip=clip)
        snapshots.append(dict(stage=entry['stage'], accepted_optimizer_steps_observed=accepted_count,
                              scope='whole_fixed_eligible_cohort_at_snapshot_not_executed_minibatch', **summary))
        identities.append(identity)
        if entry['stage'] == 'accepted_Adam' and plan is not None:
            step = plan[ordinal-1]
            verify_logged_diagnostics(previous_vectors, step['report'].get('full_rollout_before'))
            verify_logged_diagnostics(vectors, step['report'].get('full_rollout_after'))
            rows = torch.tensor(step['record_rows'], dtype=torch.long, device='cpu')
            before = {key: value[rows] if torch.is_tensor(value) else value
                      for key, value in previous_vectors.items()}
            verify_logged_diagnostics(before, step['report'].get('pre_step_chain'))
            subset_sources = [previous['sources'][i] for i in step['record_rows']]
            subset_manifest = [previous['global_manifest'][i] for i in step['record_rows']]
            subset = summarize_vectors(before, subset_sources, subset_manifest,
                                       baseline_objective=baseline_vectors['objective'][rows],
                                       previous_objective=before['objective'], clip=clip)
            actual_minibatches.append(dict(optimizer_step=step['optimizer_step'], epoch=step['epoch'],
                global_manifest_indices=step['global_manifest_indices'], record_rows=step['record_rows'],
                parameter_snapshot_entry=entries[ordinal-1]['path'],
                scope='accepted_Adam_pre_update_minibatch_PPO_coefficients_not_parameter_gradient_attribution',
                loss_coefficient_divisor=len(step['record_rows']), **subset))
        previous, previous_vectors = payload, vectors
    result = dict(schema='genmo.stage2.probability_records_summary.v1', clip=clip,
        inputs=dict(index=index_identity, case=case_identity, candidate=candidate_identity, records=identities),
        semantics=dict(log_ratio='log_p_new_minus_log_p_old_sum_all_trainable_steps',
            advantage='stored_global_valid_upper_transition_normalization_not_renormalized_by_source',
            coefficient='abs(A*exp(log_ratio))_unless_sign_dependent_active_clip',
            ess='finite_cohort_mass_concentration_not_independent_sample_count',
            parameter_gradient_attribution=False, physical_effectiveness_evidence=False,
            training_algorithm_modified=False, top_mass_population='fraction_of_all_group_chains_including_zero_coefficients'),
        snapshots=snapshots, actual_minibatches=actual_minibatches if plan is not None else None,
        actual_minibatch_mapping_reason='bound_epoch_orders_and_pre_update_snapshots' if plan is not None else
            'candidate_JSON_not_supplied' if candidate is None else 'candidate_actor_report_missing',
        candidate_acceptance=candidate.get('acceptance') if candidate is not None else None,
        candidate_hard_kl_acceptance=candidate.get('hard_kl_acceptance') if candidate is not None else None)
    publish_json(output, result)
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probability-records', required=True, type=Path, help='probability 目录或 index.json')
    parser.add_argument('--case', required=True, type=Path, help='原固定 case 目录或 case.json；实际 SHA 必须匹配')
    parser.add_argument('--expected-case-sha256', help='可选外部冻结的 case.json SHA256')
    parser.add_argument('--candidate-json', type=Path, help='可选候选 JSON；用于恢复实际更新前 minibatch')
    parser.add_argument('--clip', required=True, type=float, help='此次诊断实际配置的 PPO clip，不暗用默认值')
    parser.add_argument('--output', required=True, type=Path, help='来源目录之外、尚不存在的 JSON')
    return parser


def main():
    args = argument_parser().parse_args()
    result = summarize(args.probability_records, args.case, clip=args.clip, output=args.output,
                       candidate_json=args.candidate_json, expected_case_sha256=args.expected_case_sha256)
    print(json.dumps(dict(output=str(args.output.resolve()), snapshots=len(result['snapshots']),
                          actual_minibatches=None if result['actual_minibatches'] is None else
                          len(result['actual_minibatches'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
