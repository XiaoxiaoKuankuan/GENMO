"""只读核验完整真实 episode 的无自举回报及 Stage10 v3 Critic 预测。

输入是已完成的独立物理评估报告、同次评估保存的因果条件参考，以及一个或多个
带明确 SHA 的 v3 完整 checkpoint。逐 episode 检查真实终止、50Hz 连续控制和
四个物理子步；行政截断、缺失条件、重复决策或不同 Actor/随机核一律拒绝。
从逐控制积分奖励独立反向计算 Monte Carlo 折扣回报，保留零步事件的原语义。
这些标签不读取 Critic 输出，不使用评估文件中已有的 GAE target，也不把奖励或
未来状态送入 Critic。当前条件与音乐剩余时间仍是唯一价值输入。

每个模型只加载 Actor 身份及自己的冻结预训练时序 Critic，不恢复优化器、不做
反向或更新。支持 CPU 分批评估，输出整体、来源和完整 episode 起点的 MSE/EV。
一个随机 episode 的回报是高方差样本；此诊断检验价值校准，不能证明无偏 V(s)
的精确误差、长期收敛或真实机器人安全。输入只读，输出必须是不存在的新文件。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def identity_key(identity):
    return json.dumps(identity, sort_keys=True, separators=(',', ':'))


def complete_returns(report, gamma_upper):
    """按真实控制时间求完整回报；没有 Critic、自举或可训练输出参与标签计算。"""
    if not 0 < gamma_upper <= 1 or not math.isfinite(gamma_upper):
        raise ValueError('Finite gamma in (0,1] required')
    if report.get('status') != 'passed' or report.get('task_count') != len(report['episodes']):
        raise ValueError('Complete successful evaluation required')
    gamma = gamma_upper ** (1 / 25)
    records = {}
    episodes = []
    for episode in report['episodes']:
        decisions = episode['decisions']
        if (not episode.get('transition_valid') or not decisions or episode.get('truncated')
                or not decisions[-1].get('terminated') or decisions[-1].get('truncated')):
            raise ValueError('Administrative truncation cannot provide complete no-bootstrap returns')
        previous_tick = None
        total_controls = 0
        episode_identity = None
        for index, decision in enumerate(decisions):
            identity = decision['identity']
            current = tuple(identity[k] for k in ('backend_session_id', 'env_id', 'episode_id', 'policy_version'))
            if episode_identity is not None and current != episode_identity:
                raise ValueError('Episode identity changed inside a complete trajectory')
            episode_identity = current
            if decision['truncated'] or (decision['terminated'] and index != len(decisions) - 1):
                raise ValueError('Terminal or truncation boundary occurs before the episode end')
            controls = decision['controls']
            count = decision['executed_control_steps']
            if len(controls) != count or decision['executed_physics_steps'] != 4 * count:
                raise ValueError('Actual controls and four physical substeps differ')
            for control in controls:
                if (not math.isclose(control['dt_s'], .02, abs_tol=1e-12)
                        or not math.isfinite(control['reward'])
                        or (previous_tick is not None and control['tick'] != previous_tick + 12)):
                    raise ValueError('Invalid reward, control time or noncontiguous episode')
                previous_tick = control['tick']
            event = decision['zero_step_event_reward']
            if (not math.isfinite(event) or not math.isclose(decision['reward_sum'],
                    sum(c['reward'] for c in controls) + event, rel_tol=1e-9, abs_tol=1e-8)):
                raise ValueError('Event reward is nonfinite, missing or counted twice')
            total_controls += count
        value = 0.
        for index in range(len(decisions) - 1, -1, -1):
            decision = decisions[index]
            controls = decision['controls']
            immediate = sum(gamma ** j * c['reward'] for j, c in enumerate(controls))
            immediate += gamma ** max(len(controls) - 1, 0) * decision['zero_step_event_reward']
            value = immediate + gamma ** len(controls) * value
            key = identity_key(decision['identity'])
            if key in records or not math.isfinite(value):
                raise ValueError('Duplicate decision or nonfinite complete return')
            records[key] = dict(target=value, dataset=episode['dataset'], task_id=episode['task_id'],
                episode_start=index == 0, executed_control_steps=len(controls))
        episodes.append(dict(task_id=episode['task_id'], dataset=episode['dataset'],
            physical_failure=episode['physical_failure'], end_reason=episode['end_reason'],
            decisions=len(decisions), control_steps=total_controls, initial_discounted_return=value))
    return records, episodes


def load_conditions(reference, expected, report):
    import torch
    from gem.closedloop.dppo.checkpoint_v3 import file_sha256
    manifest = json.loads(reference.read_text())
    if (manifest['plan_sha256'] != report['plan_sha256']
            or manifest['evaluation_identity'] != report['evaluation_identity']):
        raise ValueError('Condition reference is from a different physical evaluation')
    result = {}
    for rank in manifest['ranks']:
        descriptor = rank.get('critic')
        if descriptor is None:
            continue
        path = Path(descriptor['path'])
        if file_sha256(path) != descriptor['sha256']:
            raise ValueError('Saved causal conditions SHA differs')
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value['reference_sha256'] != descriptor['reference_sha256']:
            raise ValueError('Saved reference identity differs')
        for record in value['records']:
            key = identity_key(record['identity'])
            if key not in expected or key in result:
                raise ValueError('Missing or repeated causal decision identity')
            result[key] = record
    if set(result) != set(expected):
        raise ValueError('Complete episode decisions and causal conditions do not align')
    return result


def evaluate_checkpoint(path, sha, report, conditions, targets, batch_size):
    import torch
    from gem.closedloop.dppo.checkpoint_v3 import SCHEMA, file_sha256
    from gem.closedloop.dppo.checkpoint import _validate_model_state
    from gem.closedloop.dppo.evaluation import _model_fingerprint
    from gem.closedloop.dppo.periodic_monitor import summarize_critic_predictions
    from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
    from gem.closedloop.dppo.trainer import load_actor
    if file_sha256(path) != sha:
        raise ValueError('Explicit complete checkpoint SHA differs')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    identity = report['evaluation_identity']['training_identity']
    if payload.get('schema') != SCHEMA:
        raise ValueError('Only explicit complete v3 checkpoints can be evaluated')
    for key in ('assets', 'kernel', 'reward'):
        if payload['identity'][key] != identity[key]:
            raise ValueError('Evaluation and Critic model contract differs: ' + key)
    if payload['config']['stage10']['training']['gamma_upper'] != identity['training']['gamma_upper']:
        raise ValueError('Value discount differs from the actual return target')
    config = copy.deepcopy(payload['config'])
    config['runtime']['genmo_device'] = 'cpu'
    actor, _, _ = load_actor(config)
    _validate_model_state(actor, payload['actor'], 'complete-return Actor')
    actor.load_state_dict(payload['actor'])
    if _model_fingerprint(actor) != report['actor_identity']['model_fingerprint']:
        raise ValueError('Complete return policy differs from the Critic checkpoint Actor')
    settings = config['stage10']['critic']
    if settings['type'] != 'pretrained_temporal' or not settings['freeze_encoders']:
        raise ValueError('This audit targets independent frozen pretrained temporal Critics')
    critic = PretrainedConditionCritic(actor, dimension=settings['dimension'], layers=settings['layers'],
        heads=settings['heads'], freeze_encoders=True)
    _validate_model_state(critic, payload['critic'], 'complete-return Critic')
    critic.load_state_dict(payload['critic']); critic.eval()
    fingerprint = _model_fingerprint(critic)
    keys = list(targets)
    predictions = []
    with torch.no_grad():
        for begin in range(0, len(keys), batch_size):
            batch = [conditions[k] for k in keys[begin:begin + batch_size]]
            context = {key: torch.cat([r['context'][key] for r in batch]) for key in batch[0]['context']}
            remaining = torch.tensor([r['remaining_music_seconds'] for r in batch])
            predictions.extend(critic(context, remaining).tolist())
    if _model_fingerprint(critic) != fingerprint or any(p.grad is not None for p in critic.parameters()):
        raise RuntimeError('Read-only complete return audit modified the Critic')
    rows = [dict(targets[k], prediction=p) for k, p in zip(keys, predictions)]
    def summarize(selected):
        return summarize_critic_predictions([r['prediction'] for r in selected], [r['target'] for r in selected])
    return dict(checkpoint=str(path.resolve()), sha256=sha, state=payload['state'],
        overall=summarize(rows), episode_starts=summarize([r for r in rows if r['episode_start']]),
        by_source={source: summarize([r for r in rows if r['dataset'] == source])
            for source in sorted({r['dataset'] for r in rows})}, predictions=rows,
        critic_unchanged=True, actor_matches_evaluated_behavior=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', nargs=3, action='append', metavar=('LABEL', 'CHECKPOINT', 'SHA256'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1 or len({r[0] for r in args.model}) != len(args.model):
        parser.error('New output, positive batch size and unique model labels required')
    import torch
    from gem.closedloop.dppo.checkpoint_v3 import file_sha256
    torch.set_num_threads(4)
    report = json.loads(args.report.read_text())
    gamma = report['evaluation_identity']['training_identity']['training']['gamma_upper']
    targets, episodes = complete_returns(report, gamma)
    conditions = load_conditions(args.reference, targets, report)
    models = {label: evaluate_checkpoint(Path(path), sha, report, conditions, targets, args.batch_size)
              for label, path, sha in args.model}
    result = dict(schema='genmo.stage10.complete_episode_value_audit.v3', gamma_upper=gamma,
        target_scope='actual_discounted_complete_episode_returns_without_critic_bootstrap',
        statistical_scope='finite_sample_MC_returns_not_exact_expected_value_error',
        episodes=episodes, models=models, inputs=dict(report=str(args.report.resolve()),
            report_sha256=file_sha256(args.report), reference=str(args.reference.resolve()),
            reference_sha256=file_sha256(args.reference)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v['overall'] for k, v in models.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
