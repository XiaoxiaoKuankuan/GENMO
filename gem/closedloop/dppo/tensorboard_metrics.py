"""第二阶段训练效果的精简 TensorBoard：一个训练任务对应一个 run。

本模块把真实采集、固定物理验证、优化器和完整 MC 报告转换成固定白名单指标。
只保留奖励、原始跟踪误差、动作保留、终止、策略、损失、采样份额和吞吐八组；
禁止递归展开预算、会话 UUID、网络结构、逐 minibatch 或逐去噪步内部字段。
训练、固定验证和完整 MC 共用一个事件目录及累计接受迭代横轴；训练与验证的
同名总回报在写入时分别命名，避免混成一条曲线。验证回报按来源等权，
配对区间按来源内音乐组重采样，两个种子不充当两首独立音乐。RMS 从平方误差
重算，身体项沿用 GMT 的水平位置/yaw 对齐定义，根部项保留世界坐标漂移。
所有聚合只读已有报告，不改变奖励、随机生成器、模型、优化器或仿真；缺测省略，
绝不补零。详细审计仍留在原 JSON/JSONL，曲线定义和精简记录另行持久化。
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np


SOURCES = ('AIOZ-GDANCE', 'AIST++', 'FineDance', 'Mine')
VERSION = 'genmo.stage10.compact_tensorboard.v2'
DEFINITIONS = {
    'Reward/total_per_second': '分/秒，越高越好。训练为各来源积分回报/实际采集秒数再四来源等权；验证为各任务每秒回报先来源内平均、再来源等权。',
    'Reward/episode_return': '分，固定验证任务积分回报的四来源等权平均；执行更久也会增加，必须一起看每秒回报和终止。',
    'Reward/tracking': '分/秒，已乘权重和活动度门控的跟踪奖励；越高越好，但不能替代原始误差。',
    'Reward/stable': '分/秒，根高和非 yaw 姿态稳定奖励，越高越好。',
    'Reward/alive': '分/秒，有效存活控制区间奖励，越高越好。',
    'Reward/command_penalty': '分/秒，控制指令变化惩罚，非正，越接近零越好。',
    'Reward/torque_penalty': '分/秒，力矩惩罚，非正，越接近零越好。',
    'Reward/contact_penalty': '分/秒，滑脚和异常接触惩罚，非正，越接近零越好。',
    'Reward/joint_limit_penalty': '分/秒，实际及参考关节限位惩罚；不能当作硬限位最大越界距离。',
    'Reward/music': '分/秒，仅音乐奖励启用时记录；奖励关闭时不画恒零曲线。',
    'Reward/paired_gain_per_second': '分/秒，同一固定任务相对 Stage1 的来源等权回报差，正值才是收益。',
    'Reward/paired_gain_ci95_low': '上述收益的音乐组配对 bootstrap 95%区间下界；下界大于零才支持该验证集的明确收益。',
    'Reward/paired_gain_ci95_high': '上述收益的音乐组配对 bootstrap 95%区间上界；区间跨零表示提升不明确。',
    'Tracking/root_position_rms_m': '米，世界坐标根位置误差 RMS，越低越好；来源内按控制时间平均平方误差，再来源等权开方。',
    'Tracking/root_orientation_rms_rad': '弧度，根部完整姿态角差 RMS，越低越好；包含 yaw。',
    'Tracking/body_position_rms_m': '米，GMT 水平位置/yaw 对齐后的身体位置 RMS；不等于蓝色影子的全局位移。',
    'Tracking/body_orientation_rms_rad': '弧度，GMT yaw 对齐后的身体姿态 RMS，越低越好。',
    'Tracking/joint_position_rms_rad': '弧度，关节角误差 RMS，越低越好。',
    'Tracking/joint_velocity_rms_rad_s': '弧度/秒，关节速度误差 RMS，越低越好；当前奖励权重为零，属于诊断。',
    'Tracking/root_position_p95_m': '米，来源等权的逐控制步根位置误差第95百分位，越低越好，用于观察尾部漂移。',
    'Motion/activity_rad_s': '弧度/秒，实际机器人0.5秒因果窗关节速度 RMS 的来源等权均值；不是越小越好。',
    'Motion/activity_retention': '相对同一固定任务 Stage1 实际活动度的比例；1为持平，持续下降提示动作收缩。',
    'Motion/music_alignment': '实际动作与音乐的节拍同步诊断，越高越好；不是当前训练奖励，也不是完整舞蹈质量评分。',
    'Termination/physical_failures_per_1000s': '次/1000机器人秒，普通 Actor 采集物理阈值失败次数/实际采集秒数；不把每次失败都称为摔倒。',
    'Termination/physical_failure_fraction': '固定验证物理阈值失败任务数/全部任务数，越低越好。',
    'Termination/window_limit_fraction': '固定验证因行政时间上限结束的任务比例；正常短窗结束，不是物理失败或整曲完成。',
    'Termination/music_end_fraction': '固定验证真实到达可用音乐终点的任务比例；10秒短窗中的该比例不能当作整曲成功率。',
    'Termination/new_failure_count': '与 Stage1 同任务配对后新增的失败数，越低越好；不能被另一些任务恢复成功抵消。',
    'Termination/reference_rejections_per_1000s': '次/1000机器人秒，固定验证参考拒绝次数/实际执行秒数，越低越好。',
    'Policy/learning_rate': '实际 Actor 优化器学习率，不是配置上限。',
    'Policy/trainable_kl': '本轮更新前后可训练去噪步骤的平均每自由坐标 KL；排除冻结步骤，大小本身不等于收益。',
    'Policy/max_step_kl': '完整去噪链中最大的单步平均每自由坐标 KL，用于检查更新是否过大。',
    'Policy/clipped_objective_fraction': '可训练去噪步骤最终检查中真正被 PPO 裁剪的目标比例；接近零不单独证明无效。',
    'Policy/gradient_norm': 'Actor 每次更新裁剪前 PPO+BC 总梯度范数的均值；持续异常增大需检查。',
    'Policy/bc_to_ppo_gradient_ratio': '每次更新加权 BC 梯度范数/PPO 梯度范数的均值；不是 Adam 参数更新贡献比例。',
    'Loss/ppo': 'Actor 本轮 PPO 代理损失按内部样本访问数加权；优势归一化后常在零附近，不能按单调下降验收效果。',
    'Loss/bc': 'Actor 本轮 BC 损失按实际 BC 样本数加权；衡量监督约束拟合，不等于物理跟踪。',
    'Loss/critic_mse_before': '本轮 Critic 更新前，对本轮自举目标的 MSE；不同轮目标会变化。',
    'Loss/critic_mse_after': '同批 Critic 更新后的 MSE；只说明拟合程度，不能代替独立 MC。',
    'Loss/critic_ev_before': '本轮更新前自举目标解释方差，1较好、0接近只预测均值、负值较差；不是无偏真实价值精度。',
    'Loss/critic_mc_ev': '独立完整轨迹 MC 上最终选定 Critic 的解释方差；该集合参与候选选择，不是完全未使用的最终测试。',
    'Loss/critic_mc_bias': '独立完整 MC 上预测均值减真实回报均值，分；接近零较好，负值表示低估。',
    'Loss/stage1_validation': '固定 Stage1 监督验证集来源等权损失，越低表示原监督任务拟合更好。',
    'Perf/iteration_seconds': '秒，本次接受迭代墙钟，不包含迭代之间另做的周期评估/MC/保存开销。',
    'Perf/controls_per_second': '实际 Actor 采集控制步总数/本轮墙钟；用于判断吞吐，不是策略质量。',
}
for _source in SOURCES:
    DEFINITIONS[f'Sampling/time_share_{_source}'] = '本轮普通 Actor 实际采集秒数中该来源的占比；不是选曲概率，也不是唯一时间覆盖率。'

SCOPES = ('training', 'validation', 'critic_mc')
EVENT_TAG_OVERRIDES = {
    ('training', 'Reward/total_per_second'): 'Reward/train_total_per_second',
    ('validation', 'Reward/total_per_second'): 'Reward/validation_total_per_second',
}
EVENT_DEFINITIONS = {tag: description for tag, description in DEFINITIONS.items()
                     if tag != 'Reward/total_per_second'}
EVENT_DEFINITIONS.update({
    'Reward/train_total_per_second': '分/秒，普通训练采集各来源积分回报/实际采集秒数再来源等权；任务不断变化，只作训练监视。',
    'Reward/validation_total_per_second': '分/秒，固定验证各任务每秒回报先来源内平均再来源等权；优先用此曲线判断训练效果。',
})

ERROR_TAGS = {
    'anchor_position_error_sq_m2': 'root_position_rms_m',
    'anchor_orientation_error_sq_rad2': 'root_orientation_rms_rad',
    'body_position_mean_error_sq_m2': 'body_position_rms_m',
    'body_orientation_mean_error_sq_rad2': 'body_orientation_rms_rad',
    'joint_position_mean_error_sq_rad2': 'joint_position_rms_rad',
    'joint_velocity_mean_error_sq_rad2_s2': 'joint_velocity_rms_rad_s',
}
COMPONENT_TAGS = dict(track='tracking', stable='stable', alive='alive', cmd='command_penalty',
                      torque='torque_penalty', contact='contact_penalty', joint_limit='joint_limit_penalty')


def checked_scalars(values):
    """未知标签、布尔值或非有限数值直接拒绝；缺测不生成虚假的零点。"""
    result = {}
    for tag, value in values.items():
        if tag not in DEFINITIONS:
            raise ValueError(f'Unknown compact TensorBoard tag: {tag}')
        if value is None:
            continue
        if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
            raise ValueError(f'Nonfinite or nonnumeric compact metric: {tag}')
        result[tag] = float(value)
    return result


def dashboard_scalars(scope, values):
    """只区分训练和验证的同名总回报；其余标签沿用八个功能分组，不创建子 run。"""
    if scope not in SCOPES:
        raise ValueError('Invalid compact curve scope')
    result = {EVENT_TAG_OVERRIDES.get((scope, tag), tag): value
              for tag, value in checked_scalars(values).items()}
    if not result.keys() <= EVENT_DEFINITIONS.keys():
        raise ValueError('Metric has no single-run event definition')
    return result


def _weighted(rows, key, weight):
    pairs = [(float(r[key]), float(r[weight])) for r in rows if r.get(key) is not None and r.get(weight, 0) > 0]
    return sum(v*w for v, w in pairs)/sum(w for _, w in pairs) if pairs else None


def training_scalars(record):
    """跨 rank 合并分子/分母；仅接受实际完成的 Actor 更新，避免预热/回退污染横轴。"""
    if record.get('mode') != 'dppo_bc':
        return {}
    collections = record['collection_by_rank']
    seconds = sum(c['executed_seconds'] for c in collections)
    sources = {}
    values = {}
    for source in SOURCES:
        rows = [c.get('by_source', {}).get(source, {}) for c in collections]
        duration = sum(r.get('executed_seconds', 0.) for r in rows)
        if duration > 0:
            sources[source] = sum(r.get('reward_sum', 0.) for r in rows)/duration
        if seconds > 0:
            values[f'Sampling/time_share_{source}'] = duration/seconds
    if len(sources) == len(SOURCES):
        values['Reward/total_per_second'] = sum(sources.values())/len(SOURCES)
    if seconds > 0:
        values['Termination/physical_failures_per_1000s'] = 1000*sum(c['physical_failures'] for c in collections)/seconds
    actor = record.get('actor') or {}
    steps = actor.get('steps', [])
    for tag, key, weight in [('Loss/ppo', 'ppo_loss', 'internal_sample_visits'),
                             ('Loss/bc', 'bc_loss', 'bc_global_samples')]:
        values[tag] = _weighted(steps, key, weight)
    if steps:
        values['Policy/learning_rate'] = steps[-1].get('learning_rate')
        values['Policy/gradient_norm'] = float(np.mean([s['total_gradient_norm'] for s in steps]))
        ratios = []
        for step in steps:
            gradients = step.get('gradient_contributions', {}).get('all', {})
            if gradients.get('ppo_norm', 0) > 0:
                ratios.append(gradients['weighted_bc_norm']/gradients['ppo_norm'])
        if ratios:
            values['Policy/bc_to_ppo_gradient_ratio'] = float(np.mean(ratios))
    kl = record.get('full20_kl') or {}
    frozen = kl.get('frozen_prefix', {}).get('steps', 0)
    active = [s for s in kl.get('per_denoising_step', []) if s['step'] >= frozen]
    if active:
        values['Policy/trainable_kl'] = float(np.mean([s['mean_coordinate_kl']['mean'] for s in active]))
        values['Policy/clipped_objective_fraction'] = float(np.mean([s['objective_clip_fraction'] for s in active]))
    values['Policy/max_step_kl'] = kl.get('maximum_step_mean_coordinate_kl')
    critic = record.get('critic') or {}
    values['Loss/critic_mse_before'] = critic.get('before', {}).get('mse')
    values['Loss/critic_mse_after'] = critic.get('after', {}).get('mse')
    values['Loss/critic_ev_before'] = critic.get('before', {}).get('explained_variance')
    wall = record.get('wall_seconds', 0.)
    if wall > 0:
        values['Perf/iteration_seconds'] = wall
        values['Perf/controls_per_second'] = sum(c['control_steps'] for c in collections)/wall
    return checked_scalars(values)


def evaluation_view(report):
    """压缩固定验证原始任务，保留配对身份、来源及音乐组；不持有完整轨迹副本。"""
    if report.get('status') != 'passed':
        raise ValueError('Cannot plot incomplete evaluation as policy quality')
    tasks = {t['task_id']: t for t in report['plan']['tasks']}
    rows = {}
    errors = {source: defaultdict(list) for source in SOURCES}
    for episode in report['episodes']:
        if not episode.get('transition_valid') or episode['task_id'] in rows:
            raise ValueError('Invalid or duplicate evaluation episode')
        task = tasks[episode['task_id']]
        source = episode['dataset']
        controls = [c for decision in episode['decisions'] for c in decision['controls']]
        duration = sum(c['dt_s'] for c in controls)
        components = {}
        for name in COMPONENT_TAGS:
            if duration > 0 and all(name in c['components'] for c in controls):
                components[name] = sum(c['components'][name]['integrated_reward'] for c in controls)/duration
        if any(c['components'].get('music', {}).get('weight', 0.) for c in controls):
            components['music'] = sum(c['components']['music']['integrated_reward'] for c in controls)/duration
        activities = [c for c in controls if c['activity'].get('valid')]
        activity = (sum(c['activity']['actual_activity_rad_s']*c['dt_s'] for c in activities)/duration
                    if duration > 0 and len(activities) == len(controls) else None)
        for key in ERROR_TAGS:
            for control in controls:
                if key in control['raw_errors']:
                    value = float(control['raw_errors'][key])
                    if value < 0 or not math.isfinite(value):
                        raise ValueError('Invalid raw tracking squared error')
                    errors[source][key].append((value, control['dt_s']))
        rows[episode['task_id']] = dict(source=source,
            music_group=task['sample']['group_id'], reward=episode['reward_sum'],
            rate=episode['reward_sum']/duration if duration > 0 else None, activity=activity, components=components,
            failed=bool(episode['physical_failure']), end_reason=episode['end_reason'],
            music_alignment=(episode.get('motion_diagnostic') or {}).get('music_alignment'))
    if set(rows) != set(tasks) or {r['source'] for r in rows.values()} != set(SOURCES):
        raise ValueError('Fixed evaluation task/source coverage differs')
    return dict(plan_sha256=report['plan_sha256'], evaluation_identity=report.get('evaluation_identity'),
                rows=rows, errors=errors)


def _balanced(rows, getter):
    groups = {source: [] for source in SOURCES}
    for row in rows:
        value = getter(row)
        if value is None:
            return None
        groups[row['source']].append(float(value))
    if any(not values for values in groups.values()):
        return None
    return float(np.mean([np.mean(values) for values in groups.values()]))


def evaluation_scalars(report, baseline=None, *, bootstrap_repeats=2000):
    """固定任务评价；来源内音乐簇配对区间只衡量该验证集，不证明跨训练种子收敛。"""
    view = evaluation_view(report)
    before = view if baseline is None else baseline
    if 'rows' not in before:
        before = evaluation_view(before)
    if (view['plan_sha256'] != before['plan_sha256'] or view['evaluation_identity'] != before['evaluation_identity']
            or view['rows'].keys() != before['rows'].keys()):
        raise ValueError('Baseline tasks or execution identity differ')
    rows = list(view['rows'].values())
    values = {'Reward/total_per_second': _balanced(rows, lambda r: r['rate']),
              'Reward/episode_return': _balanced(rows, lambda r: r['reward']),
              'Motion/activity_rad_s': _balanced(rows, lambda r: r['activity']),
              'Motion/music_alignment': _balanced(rows, lambda r: r['music_alignment'])}
    initial_activity = _balanced(before['rows'].values(), lambda r: r['activity'])
    if initial_activity is not None and initial_activity > 0 and values['Motion/activity_rad_s'] is not None:
        values['Motion/activity_retention'] = values['Motion/activity_rad_s']/initial_activity
    for name, tag in {**COMPONENT_TAGS, 'music': 'music'}.items():
        values['Reward/'+tag] = _balanced(rows, lambda r: r['components'].get(name))
    for key, tag in ERROR_TAGS.items():
        parts = [view['errors'][s].get(key, []) for s in SOURCES]
        if all(parts):
            values['Tracking/'+tag] = math.sqrt(float(np.mean([
                sum(v*w for v, w in part)/sum(w for _, w in part) for part in parts])))
    parts = [view['errors'][s].get('anchor_position_error_sq_m2', []) for s in SOURCES]
    if all(parts):
        weighted = []
        for part in parts:
            duration = sum(w for _, w in part)
            weighted.extend((math.sqrt(v), w/duration/len(SOURCES)) for v, w in part)
        weighted.sort()
        cumulative = 0.
        for value, weight in weighted:
            cumulative += weight
            if cumulative >= .95-1e-12:
                values['Tracking/root_position_p95_m'] = value
                break
    values['Termination/physical_failure_fraction'] = sum(r['failed'] for r in rows)/len(rows)
    values['Termination/window_limit_fraction'] = sum(r['end_reason'] == 'physical_safety_limit' for r in rows)/len(rows)
    values['Termination/music_end_fraction'] = sum(r['end_reason'] in ('music_end', 'task_complete') for r in rows)/len(rows)
    values['Termination/new_failure_count'] = sum(r['failed'] and not before['rows'][key]['failed'] for key, r in view['rows'].items())
    overall = report['aggregate']['overall']
    if overall['executed_seconds'] > 0:
        values['Termination/reference_rejections_per_1000s'] = 1000*overall['rejection_count']/overall['executed_seconds']
    pairs = defaultdict(lambda: defaultdict(list))
    deltas = []
    for key, row in view['rows'].items():
        if row['rate'] is None or before['rows'][key]['rate'] is None:
            pairs.clear(); deltas.clear()
            break
        delta = row['rate']-before['rows'][key]['rate']
        pairs[row['source']][row['music_group']].append(delta)
        deltas.append(dict(source=row['source'], delta=delta))
    values['Reward/paired_gain_per_second'] = _balanced(deltas, lambda r: r['delta'])
    if bootstrap_repeats <= 0:
        raise ValueError('bootstrap_repeats must be positive')
    if deltas and all(len(pairs[source]) >= 2 for source in SOURCES):
        rng = np.random.default_rng(181738)
        draws = []
        for source in SOURCES:
            groups = [np.asarray(values, dtype=float) for _, values in sorted(pairs[source].items())]
            sums = np.asarray([group.sum() for group in groups])
            counts = np.asarray([len(group) for group in groups])
            indices = rng.integers(0, len(groups), size=(bootstrap_repeats, len(groups)))
            draws.append(sums[indices].sum(1)/counts[indices].sum(1))
        low, high = np.quantile(np.mean(draws, axis=0), [.025, .975])
        values['Reward/paired_gain_ci95_low'], values['Reward/paired_gain_ci95_high'] = low, high
    values['Loss/stage1_validation'] = (report.get('stage1_retention') or {}).get('source_balanced_loss')
    return checked_scalars(values)


def mc_scalars(report):
    after = report['after']
    return checked_scalars({'Loss/critic_mc_ev': after.get('explained_variance'),
                            'Loss/critic_mc_bias': after['mean_prediction']-after['mean_target']})


class DashboardWriter:
    """全部指标写入一个事件目录；scope 只保留在审计记录中，不再创建三个 run。"""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.writer = None
        (self.directory/'metric_definitions.json').write_text(json.dumps(
            dict(schema=VERSION, layout='single_run', x_axis='accepted_iteration',
                 metrics=EVENT_DEFINITIONS,
                 source_tag_overrides=[dict(scope=scope, source_tag=tag, event_tag=event_tag)
                                       for (scope, tag), event_tag in EVENT_TAG_OVERRIDES.items()]),
            ensure_ascii=False, indent=2)+'\n')

    def write(self, scope, step, values, *, walltime=None):
        if scope not in SCOPES or type(step) is not int or step < 0:
            raise ValueError('Invalid compact curve scope or accepted iteration')
        values = checked_scalars(values)
        if not values:
            return
        event_values = dashboard_scalars(scope, values)
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.directory), max_queue=128, flush_secs=30)
        timestamp = time.time() if walltime is None else walltime
        for tag, value in event_values.items():
            self.writer.add_scalar(tag, value, step, walltime=timestamp)
        with (self.directory/'scalars.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(scope=scope, step=step, wall_time=timestamp, scalars=values),
                                    ensure_ascii=False, allow_nan=False)+'\n')

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()
