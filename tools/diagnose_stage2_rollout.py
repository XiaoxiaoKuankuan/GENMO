"""对封存的 Stage2 rollout 做有限离线诊断，不启动 GENMO→GMT 采样或长期训练。

必需输入：原运行 YAML、只读完整 Stage2 checkpoint、每个 rank 的 rollout/manifest.json
及对应 fixed_targets.pt。支持 kernel（各步噪声/KL/均值变化）、gradients（逐步反向，
绝不 Actor.step）和 critic_compare（独立内存副本 20/40/80 次拟合），可显式选择 all。
归档必须先在独立目录解包；工具校验路径、SHA、模型资产、Actor接口及旧采样核，不更改
原数据、配置、权重或优化器文件。默认 CPU、最多8条已存在链，--max-chains 最大160。

只把 JSON 写到一个不存在的新 --output 文件；输出区分当前批拟合与可选独立固定价值
参考。--critic-reference 指向已保存的独立验证参考 .pt。该工具不自动标定或修改正式
学习率，噪声/概率替代目标必须由显式训练配置及另行验证管理，不能冒称等价提速。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gem.closedloop.dppo.checkpoint import load_weights_checkpoint
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.offline_diagnostics import (
    critic_step_comparison, denoising_gradient_diagnostic, load_immutable_rollouts,
)
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.position_repair import audit_position_tables, file_sha256
from gem.closedloop.dppo.trainer import analytic_kl_local, load_actor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--rollout-manifest', type=Path, action='append', required=True)
    parser.add_argument('--targets', type=Path, action='append', required=True)
    parser.add_argument('--mode', choices=('kernel', 'gradients', 'critic_compare', 'all'), default='kernel')
    parser.add_argument('--max-chains', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--critic-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must not already exist')
    if args.cpu_threads < 1 or args.microbatch < 1:
        parser.error('Thread and microbatch counts must be positive')
    torch.set_num_threads(args.cpu_threads)
    from tools.train_closedloop_stage10 import configuration
    config = configuration(args.config)
    config['runtime']['genmo_device'] = args.device
    rows, targets, inputs = load_immutable_rollouts(args.rollout_manifest, args.targets, max_chains=args.max_chains)
    actor, train, _ = load_actor(config)
    critic = UpperCritic(qpos_mean=actor.endecoder.mean, qpos_std=actor.endecoder.std,
                        proprio_scales=tuple(train.model.proprio_scales)).to(args.device)
    source_sha = file_sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    identity = payload.get('identity', {})
    for name in ('checkpoint', 'stats', 'kinematics'):
        if identity.get('assets', {}).get(name) != file_sha256(config['paths'][name]):
            raise ValueError(f'Diagnostic checkpoint asset mismatch: {name}')
    if identity.get('actor_interface') != dict(actor.interface_config):
        raise ValueError('Diagnostic checkpoint Actor interface mismatch')
    loading = load_weights_checkpoint(args.checkpoint, actor=actor, critic=critic, identity=identity)
    kernel = rows[0].metadata['sampler_trace']['kernel_config']
    policy = DPPODiffusionPolicy(actor, steps=kernel['steps'], eta=kernel['eta'], std_floor=kernel['std_floor'],
        guidance_scale=kernel['guidance_scale'], cfg_batch=kernel.get('cfg_forward') == 'batched',
        std_schedule=kernel.get('effective_std_floors'))
    reference = None
    if args.critic_reference:
        reference = torch.load(args.critic_reference, map_location='cpu', weights_only=False)
        if not str(reference.get('schema', '')).endswith('.fixed_critic_reference'):
            raise ValueError('Unknown independent Critic reference schema')
        inputs['inputs'].append(dict(path=str(args.critic_reference.resolve()), sha256=file_sha256(args.critic_reference)))
    settings = config['stage9']
    report = dict(schema='genmo.closedloop.stage2.offline_diagnostics.v1', inputs=inputs,
        checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=source_sha,
                        iteration=loading['state'].get('iteration'), policy_version=loading['state'].get('policy_version')),
        fixed_position_tables=audit_position_tables(actor), mode=args.mode, no_environment_sampling=True,
        actor_optimizer_steps=0, production_files_modified=False)
    if args.mode in ('kernel', 'all'):
        report['kernel'] = analytic_kl_local(policy, rows, denoising_microbatch=args.microbatch)
    if args.mode in ('gradients', 'all'):
        report['gradients'] = denoising_gradient_diagnostic(policy, rows, targets, denoising_microbatch=args.microbatch,
            clip=settings['ppo_clip'], gamma_denoising=settings['gamma_denoising'],
            objective_logprob_reduction=settings.get('objective_logprob_reduction', 'joint_sum'))
    if args.mode in ('critic_compare', 'all'):
        report['critic_compare'] = critic_step_comparison(critic, payload['critic_optimizer'], rows, targets,
            batch_size=settings['critic_batch'], learning_rate=settings['critic_lr'], seed=config['stage10']['seed'], reference=reference)
    if file_sha256(args.checkpoint) != source_sha or any(file_sha256(entry['path']) != entry['sha256'] for entry in inputs['inputs']):
        raise RuntimeError('Diagnostic inputs changed during analysis; refusing to publish a mixed report')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output.resolve()), mode=args.mode, selected_chains=len(rows)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
