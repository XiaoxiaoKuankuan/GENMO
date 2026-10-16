"""用同一已验收Critic只读重评各Actor整曲轨迹，补齐真实MC价值偏差。

纯Actor物理评估没有加载预热Critic，但保存了生成动作之前的完整因果条件。本工具
从已核验的固定Critic初始化文件加载独立网络，在CPU上重新预测这些状态，并与该
Actor实际执行到真实终止的50Hz积分奖励所构成的MC回报比较。绝不使用保存文件中
随机Value Head的prediction或GAE target，绝不更新网络、优化器或产生新的物理步。

复用正式Critic初始化校验以绑定Stage1、奖励、随机核、物理源码及价值结构；校验
需要构造Adam容器，但从不调用step。不同Actor的数据只用于对应策略的离线诊断，
不会进入监督训练。原始上下文、模型和报告不可改写；额外输出CPU执行身份与SHA，
不把CPU诊断冒充线上GPU预测。支持只读跟随既有物理队列，不占用任何GPU。
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.critic_preheat import initialize_critic, MATCHED, ARCHITECTURE, _value_targets
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.periodic_monitor import summarize_critic_predictions
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import file_sha256
from gem.closedloop.dppo.training_config import configuration
from tools.eval.analyze_stage2_actor_lr_physics import identity_key
from tools.eval.compare_closedloop_stage10 import _load_periodic
from tools.eval.stage10_group_comparison import control_mc_returns


def value_identity(identity):
    return dict(matched={k: identity[k] for k in MATCHED},
                architecture={k: identity['critic'][k] for k in ARCHITECTURE},
                value_targets=_value_targets(identity), physical_sources=identity['source_files']['gmt'])


def load_critic(config, identity):
    c = configuration(config)
    c['runtime']['genmo_device'] = 'cpu'
    actor, _, _ = load_actor(c)
    settings = c['stage10']['critic']
    critic = PretrainedConditionCritic(actor, **{k: settings[k] for k in ('dimension', 'layers', 'heads', 'history_scales')})
    history = {id(p) for p in critic.history_encoder.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=list(critic.history_encoder.parameters()), lr=settings['history_learning_rate'], name='tracking_history'),
        dict(params=[p for p in critic.parameters() if p.requires_grad and id(p) not in history],
             lr=settings['learning_rate'], name='temporal_value')], weight_decay=0.)
    initialization = initialize_critic(settings['preheat_artifact'], identity=identity, actor=actor,
        critic=critic, optimizer=optimizer, genmo_repo=ROOT, quality_gate=settings['complete_mc']['quality_gate'])
    critic.eval().requires_grad_(False)
    del optimizer, actor
    return critic, dict(artifact=initialization['artifact'], validation=initialization['validation'],
                        fingerprint=_model_fingerprint(critic), identity=value_identity(identity))


def join_causal_records(episodes, records, *, gamma_upper):
    """精确对应真实决策，不读取记录中的旧prediction/GAE标签；真终止之后不自举。"""
    by_key = {}
    for record in records:
        key = identity_key(record['identity'])
        if key in by_key:
            raise ValueError('Duplicate causal Critic decision')
        by_key[key] = record
    result = []
    for episode in episodes:
        decisions = episode['decisions']
        if (episode['truncated'] or not episode['transition_valid'] or not decisions
                or not decisions[-1]['terminated']):
            raise ValueError('Only complete nonadministrative real episodes have MC labels')
        targets = control_mc_returns([[c['reward'] for c in d['controls']] for d in decisions],
            gamma_upper=gamma_upper, event_rewards=[d['zero_step_event_reward'] for d in decisions])
        for decision, target in zip(decisions, targets):
            key = identity_key(decision['identity'])
            if key not in by_key:
                raise ValueError('Missing causal Critic decision')
            record = by_key.pop(key)
            # 两处记录必须属于同一个实际策略版本；张量内部时间合同由Critic逐批验证。
            if record['identity'] != decision['identity']:
                raise ValueError('Critic/physics decision identity differs')
            if record['control_tick_end'] - record['control_tick_begin'] != 12 * decision['executed_control_steps']:
                raise ValueError('Critic/physics execution interval differs')
            result.append(dict(record=record, target=float(target), task_id=episode['task_id'],
                               source=episode['dataset'], decision=decision['decision_index']))
    if by_key:
        raise ValueError('Unmatched causal Critic decisions')
    return result


def analyze(report_path, output, critic, initialization):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    report, plan, episodes, provenance = _load_periodic(report_path)
    identity = report['evaluation_identity']['training_identity']
    if value_identity(identity) != initialization['identity']:
        raise ValueError('Critic evaluation behavior, reward, physical or value identity changed')
    if plan['task_count'] != 32 or plan['episode_seconds'] is not None:
        raise ValueError('Expected the same 32 full available paired episodes')
    reference_path = report_path.parent / 'fixed_diagnostic_reference.json'
    reference = json.loads(reference_path.read_text())
    if (reference['evaluation_identity'] != report['evaluation_identity']
            or reference['initial_iteration'] != report['iteration']
            or reference['plan_sha256'] != plan['plan_sha256']):
        raise ValueError('Causal input snapshot belongs to another Actor evaluation')
    records, files = [], []
    for rank in reference['ranks']:
        info = rank['critic']
        if info is None or file_sha256(info['path']) != info['sha256']:
            raise ValueError('Missing or changed causal input snapshot')
        data = torch.load(info['path'], map_location='cpu', weights_only=False, mmap=True)
        records.extend(data['records'])
        files.append(copy.deepcopy(info))
    pairs = join_causal_records(episodes, records, gamma_upper=identity['training']['gamma_upper'])
    with torch.no_grad():
        for start in range(0, len(pairs), 64):
            part = pairs[start:start + 64]
            contexts = [r['record']['critic_context'] for r in part]
            context = {k: torch.cat([c[k] for c in contexts]) for k in contexts[0]}
            remaining = torch.tensor([r['record']['remaining_music_seconds'] for r in part])
            prediction = critic(context, remaining).tolist()
            for item, value in zip(part, prediction):
                item['prediction'] = float(value)
    if _model_fingerprint(critic) != initialization['fingerprint']:
        raise RuntimeError('Read-only Critic parameters or buffers changed')
    rows = [{k: v for k, v in r.items() if k != 'record'} for r in pairs]
    def metrics(values):
        result = summarize_critic_predictions([r['prediction'] for r in values], [r['target'] for r in values])
        result['bias'] = float(np.mean([r['prediction'] - r['target'] for r in values]))
        return result
    result = dict(schema='genmo.stage2.uniform_lr_posthoc_fixed_critic_mc.v1',
        source=provenance, source_report=str(report_path), source_report_sha256=file_sha256(report_path),
        actor=report['actor_identity'], initialization=initialization,
        reference=dict(path=str(reference_path), sha256=file_sha256(reference_path), files=files),
        implementation_sha256=file_sha256(__file__),
        execution=dict(device='cpu', dtype='float32', torch_version=str(torch.__version__), threads=torch.get_num_threads()),
        scope='read_only_fixed_preheated_Critic_on_current_Actor_complete_MC_not_training_or_saved_online_prediction',
        episodes=len(episodes), overall=metrics(rows),
        by_source={s: metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
        rows=rows, wall_seconds=time.perf_counter() - started)
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output), metrics=result['overall'], seconds=result['wall_seconds']), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    index = dict(schema='genmo.stage2.uniform_lr_critic_analysis_queue.v1', queue=str(args.queue),
                 config=str(args.config), config_sha256=file_sha256(args.config), status='running', attempts=[])
    atomic_json(args.output / 'index.json', index)
    critic = initialization = None
    try:
        while True:
            queue = json.loads(args.queue.read_text())
            for job in queue['attempts']:
                if job['status'] != 'passed' or any(r['name'] == job['name'] for r in index['attempts']):
                    continue
                path = Path(job['evidence']['report'])
                if file_sha256(path) != job['evidence']['report_sha256']:
                    raise ValueError('Published physical report changed')
                if critic is None:
                    identity = json.loads(path.read_text())['evaluation_identity']['training_identity']
                    critic, initialization = load_critic(args.config, identity)
                destination = args.output / (job['name'] + '.json')
                analyze(path, destination, critic, initialization)
                index['attempts'].append(dict(name=job['name'], aliases=job['aliases'], status='passed',
                    path=str(destination), sha256=file_sha256(destination)))
                atomic_json(args.output / 'index.json', index)
            if queue['status'] != 'running':
                index['status'] = 'completed' if queue['status'] == 'completed_all_distinct_survivors' else 'physics_incomplete'
                break
            time.sleep(30)
    except BaseException as exc:
        index.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(args.output / 'index.json', index)


if __name__ == '__main__':
    main()
