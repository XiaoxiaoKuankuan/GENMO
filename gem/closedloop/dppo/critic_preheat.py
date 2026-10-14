"""固定Stage1行为策略下的Critic预热成果导出和新实验初始化。

完整歌曲采集昂贵，已经通过独立MC验证的价值网络可以作为同一Stage1行为的
初始化。此模块只导出Critic及其Adam状态、验证结果与采集来源，不迁移Actor、
Actor优化器、RNG、BC游标或累计PPO轮数，因此不是训练checkpoint恢复。
导入前核验Actor实际权重、随机核、奖励、物理源码/配置、数据、Horizon和价值结构。
价值目标定义（上层gamma/lambda与MC/TD候选比例）同样绑定：MC回报按gamma折扣、
TD候选按lambda构造，二者不同时预热价值尺度不可复用。
冻结编码器必须逐元素一致；所有验证完成前不修改Critic。旧Actor版本数据不能
用于新Actor的MC监督，导入只允许尚未进行Actor更新的s595000初始化阶段。
来源MC交互单列记录并纳入累计训练交互，当前运行预算只结算本运行新增物理执行。
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import torch

from .budget import atomic_json
from .checkpoint import _validate_model_state, _validate_optimizer_state
from .evaluation import _model_fingerprint
from .training_checkpoint import SCHEMA, file_sha256, immutable_cpu_snapshot

SCHEMA_PREHEAT = 'genmo.stage10.fixed_actor_critic_preheat.v2'
MATCHED = ('assets', 'kernel', 'reward', 'timing', 'physical_configuration',
    'dataset', 'data_catalog', 'source_stage1_step', 'generation_horizon')
ARCHITECTURE = ('type', 'dimension', 'layers', 'heads', 'freeze_encoders', 'learning_rate')


def _value_targets(identity):
    """预热MC标签与TD候选实际使用的价值目标定义。"""
    return dict(gamma_upper=identity['training']['gamma_upper'],
        lambda_upper=identity['training']['lambda_upper'],
        mix_candidates=list(identity['critic']['complete_mc']['mix_candidates']))


def _critical_sources(root):
    """行为前向及表示源码绑定；纯checkpoint校验辅助函数不改变随机行为。"""
    root = Path(root)
    files = list((root/'gem/network').rglob('*.py')) + list((root/'gem/robots').rglob('*.py'))
    files += [root/('gem/closedloop/'+n) for n in
        ('actor.py', 'contracts.py', 'online_conditions.py', 'training.py')]
    files += [p for p in (root/'gem/closedloop/dppo').glob('*.py') if p.name in
        ('policy.py', 'dense_policy.py', 'batch_execution.py', 'fixed_tile_linear.py',
         'compensated_gemm.py', 'compiled_forward.py', 'dense_attention.py',
         'frozen_prefix.py','run_control.py','env_adapter.py','vector_environment.py',
         'evaluation.py','evaluation_world.py','vector_evaluation.py',
         'world_collector.py','mc_calibration.py','complete_episode.py','trainer.py',
         'pretrained_critic.py','rewards.py','columnar_reward.py','vector_reward_adapter.py')
        or p.name.startswith('stable_')]
    signatures = {}
    import hashlib
    for path in sorted(set(files)):
        tree = ast.parse(path.read_text())
        if path.name == 'feature_codec.py':
            tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef)
                or n.name != 'validate_native_checkpoint']
        signatures[str(path.relative_to(root))] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    return signatures


def export_preheat(checkpoint, report, output):
    import json
    checkpoint, report, output = map(Path, (checkpoint, report, output))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    validation = json.loads(report.read_text())
    state, identity = payload['state'], payload['identity']
    if (payload.get('schema') != SCHEMA or not state.get('mc_preheat_complete')
            or state['actor_updates'] or state['policy_version'] or not validation.get('quality_gate_passed')
            or identity['actor_initialization']['kind'] != 'source_stage1'):
        raise ValueError('Only validated complete MC under unchanged Stage1 Actor can initialize Critic')
    record = dict(schema=SCHEMA_PREHEAT, actor_fingerprint=payload['actor_fingerprint'],
        matched_identity={k:identity[k] for k in MATCHED},
        architecture={k:identity['critic'][k] for k in ARCHITECTURE},
        value_targets=_value_targets(identity),
        physical_sources=identity['source_files']['gmt'],
        behavior_sources=_critical_sources(payload['config']['paths']['genmo_repo']),
        critic=payload['critic'], critic_optimizer=payload['critic_optimizer'],
        selected_updates=state['critic_updates'], validation=validation,
        prior_mc_control_steps=state['mc_training_control_steps'],
        prior_mc_physics_steps=state['mc_training_physics_steps'],
        source=dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=file_sha256(checkpoint),
            validation=str(report.resolve()), validation_sha256=file_sha256(report),
            budget=payload['budget'], initialization_only=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    # 不覆盖已经发布的模型；partial与目标均要求不存在。
    partial = output.with_suffix(output.suffix+'.partial')
    if output.exists() or partial.exists():raise FileExistsError(output)
    import os
    try:
        with partial.open('xb') as f:
            torch.save(immutable_cpu_snapshot(record), f); f.flush(); os.fsync(f.fileno())
        os.link(partial, output)
    finally:partial.unlink(missing_ok=True)
    result = dict(path=str(output.resolve()), sha256=file_sha256(output), source=record['source'])
    atomic_json(output.with_suffix(output.suffix+'.json'), result)
    return result


def initialize_critic(artifact, *, identity, actor, critic, optimizer, genmo_repo, quality_gate):
    if file_sha256(artifact['path']) != artifact['sha256']:
        raise ValueError('Critic initialization artifact SHA mismatch')
    data = torch.load(artifact['path'], map_location='cpu', weights_only=False, mmap=True)
    if (data.get('schema') != SCHEMA_PREHEAT
            or data['actor_fingerprint'] != _model_fingerprint(actor)
            or data['matched_identity'] != {k:identity[k] for k in MATCHED}
            or data['architecture'] != {k:identity['critic'][k] for k in ARCHITECTURE}
            or data.get('value_targets') != _value_targets(identity)
            or data['physical_sources'] != identity['source_files']['gmt']
            or data['behavior_sources'] != _critical_sources(genmo_repo)):
        raise ValueError('Fixed Actor, kernel, physics, dataset, value target or Critic initialization identity differs')
    report = data['validation']
    if (not report['quality_gate_passed']
            or report['relative_validation_mse_reduction'] < quality_gate['minimum_relative_mse_reduction']
            or report['after']['explained_variance'] < quality_gate['minimum_explained_variance']):
        raise ValueError('Independent MC quality is insufficient for Actor learning')
    _validate_model_state(critic, data['critic'], 'preheated Critic')
    _validate_optimizer_state(optimizer, data['critic_optimizer'])
    for name, value in critic.named_parameters():
        if not value.requires_grad and not torch.equal(value.detach().cpu(), data['critic'][name]):
            raise ValueError('Preheated frozen encoder differs from Stage1 independent copy')
    critic.load_state_dict(data['critic'], strict=True)
    optimizer.load_state_dict(data['critic_optimizer'])
    return dict(artifact=copy.deepcopy(artifact), source=data['source'],
        selected_updates=data['selected_updates'], prior_mc_control_steps=data['prior_mc_control_steps'],
        prior_mc_physics_steps=data['prior_mc_physics_steps'], validation=report,
        actor_unchanged=True, fresh_PhysX=True,
        scope='Critic_initialization_only_not_checkpoint_resume_or_new_policy_MC_labels')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(export_preheat(args.checkpoint, args.report, args.output))
