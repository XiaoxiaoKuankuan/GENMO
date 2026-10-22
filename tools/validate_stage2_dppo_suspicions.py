"""八卡固定真实rollout验证原DPPO目标方向、裁剪有效性和学习率/KL限制。

数据流：校验完整采集case和非空Adam来源，按原生产updater在同一数据、GAE、
Actor、Adam及随机起点运行clip0.01/0.005/0.003。每次Adam前后复算整个rollout的
原坐标平均PPO目标，记录整批梯度与小批梯度、实际Adam增量方向；在相同非零策略
偏移上比较三种clip梯度，避免只在ratio=1的平凡起点比较。另固定首批梯度与Adam
状态做单步LR扫描，记录原KL门槛及生成参考变化，不把越界候选写回正式训练。

生产gem源码和配置不改，原GAE及旧高斯链只读。联合链PPO仅作为独立诊断量。
生成输出比较是相同条件/旧创新噪声的反事实探针，真实GMT消费需结合保存的账本
及新鲜物理评价；不把参数变化、KL通过或代理目标上涨冒充物理收益。
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from gem.closedloop.dppo.actor_step_diagnostics import output_probe, compare_probes
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.continuation import DisabledBC
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.gaussian_probability import gaussian_statistics, ppo_surrogate
from gem.closedloop.dppo.gradient_merge import BCGradientController, take_gradients, merge_global_gradients
from gem.closedloop.dppo.optimization_support import _parameters, _old_kernel, _owned, probability_check_local
from gem.closedloop.dppo.parallel_support import capture_local_rng, restore_local_rng, local_call, root_call
from gem.closedloop.dppo.rollback_audit import exact_state_digest
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache, ConditionGraphCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import file_sha256, immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import actor_update, final_kl, check_kl, _eligible
from tools.stage2_dppo_diagnostic_math import summarize_records, vector_relation, actual_update_direction


def gather_statistics(policy, rows, cache, manifest, distributed, *, microbatch, clip, gamma, backward=False,
                      optimizer=None, controller=None):
    selected = _eligible(manifest)
    owned = _owned(selected, manifest, distributed)
    local = [rows[index] for _, index in owned]
    records = torch.zeros(len(local), 4, 3, dtype=torch.float64, device=distributed.device)
    if backward:
        optimizer.zero_grad(set_to_none=True)
    pairs = [(index, step) for index in range(len(local)) for step in policy.trainable_steps]
    with torch.set_grad_enabled(backward), policy_phase(policy):
        conditions = ConditionGraphCache(policy, cache, include_frozen=False)
        conditions.prime(local)
        for start in range(0, len(pairs), microbatch):
            chunk = pairs[start:start+microbatch]
            batch = [local[index] for index, _ in chunk];steps = [step for _, step in chunk]
            params, mask = _parameters(policy, batch, steps, distributed.device, cache, conditions)
            old_mean, old_std = _old_kernel(batch, steps, distributed.device, cache)
            stats = gaussian_statistics(cache.get('chain', batch, [step+1 for step in steps]),
                                        old_mean, old_std, params['mean'], params['std'], mask)
            indices = torch.tensor([index for index, _ in chunk], device=distributed.device)
            records[indices, cache.step_indices(steps)-15] = torch.stack(
                (stats['joint_log_ratio'].detach(), stats['joint_kl'].detach(), stats['free_count']), 1)
            if backward:
                advantage = cache.get('advantages', batch)*cache.denoising_discounts(steps, 20, gamma)
                proxy = ppo_surrogate(stats, advantage, clip=clip, reduction='free_coordinate_mean')
                (-proxy['objective'].sum()/(len(selected)*4)).backward()
        if backward:
            conditions.backward()
        else:
            conditions.validate_inputs()
    advantages = cache.get('advantages', local).double() if local else records.new_empty(0)
    packed = torch.cat((records.reshape(len(local), 12), advantages[:, None]), 1)
    full = distributed.gather_rows(packed, [position for position, _ in owned], len(selected))
    full_records, full_advantages = full[:, :12].reshape(-1, 4, 3), full[:, 12]
    metrics = summarize_records(full_records, full_advantages, clip=clip, gamma_denoising=gamma)
    grads = None
    if backward:
        ppo = take_gradients(policy.actor)
        merge_global_gradients(policy.actor, ppo, {}, controller, distributed=distributed, bc_enabled=False)
        grads = take_gradients(policy.actor)
    return metrics, full_records, full_advantages, grads


def restore_case(actor, optimizer, controller, anchor, learning, rank_state, generator, base_lr):
    actor.load_state_dict(learning['actor'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(learning['actor_optimizer']))
    controller.load_state_dict(copy.deepcopy(learning['controller']))
    anchor.load_state_dict(copy.deepcopy(rank_state['bc']))
    restore_local_rng(rank_state['rng'], {'actor': generator})
    optimizer.param_groups[0]['lr'] = base_lr
    optimizer.zero_grad(set_to_none=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clips', nargs='+', type=float, default=[.01, .005, .003])
    parser.add_argument('--microbatch', type=int, default=128)
    parser.add_argument('--skip-lr-sweep', action='store_true')
    parser.add_argument('--lr-multiplier', type=float, default=1., help='独立有限对照的基础学习率倍率，正式配置不改')
    args = parser.parse_args()
    rank, world, local = (int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or rank != local:
        raise ValueError('此验证只支持单机八卡')
    dist.init_process_group('gloo', timeout=timedelta(minutes=35))
    group = cache = None
    started = time.perf_counter()
    try:
        control = DistributedCollectives(rank, world)

        def reserve():
            busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if busy:
                raise RuntimeError('真实采集尚未释放八卡: '+busy)
            args.output_dir.mkdir(parents=True, exist_ok=False)
        root_call(control, reserve)
        torch.cuda.set_device(local);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        group = dist.new_group(backend='nccl', timeout=timedelta(minutes=30))
        collective = DistributedCollectives(rank, world, tensor_group=group, device=f'cuda:{local}')
        config = configuration(args.config, genmo_repo=ROOT)
        config['runtime']['genmo_device'] = str(collective.device)
        settings = config['stage9'];stage = config['stage10']
        if settings.get('bc_enabled', True):
            raise ValueError('本次验证必须保持原正式任务BC关闭')
        actor, _, loading = local_call(collective, lambda: load_actor(config))
        policy = build_policy(actor, settings, stage['precision'])
        header = json.loads((args.case/'case.json').read_text())
        if not header['nonempty_adam'] or file_sha256(args.case/'learning_state.pt') != header['learning_state_sha256']:
            raise ValueError('必须使用已核SHA的非空Adam完整真实状态')
        learning = torch.load(args.case/'learning_state.pt', map_location='cpu', weights_only=False, mmap=True)
        rank_path = args.case/f'rank{rank:02d}'
        for name, key in (('manifest.json', 'rollout_manifest_sha256'), ('rank_state.pt', 'rank_state_sha256')):
            if file_sha256(rank_path/name) != header['ranks'][rank][key]:
                raise ValueError('rank来源SHA不匹配')
        payload = local_call(collective, lambda: load_rollout(rank_path))
        source = torch.load(args.source_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if payload['kernel'] != policy.kernel_config or learning['identity'] != source['identity']:
            raise ValueError('真实行为核或父checkpoint身份不一致')
        if exact_state_digest(learning['actor']) != exact_state_digest(source['actor']):
            raise ValueError('采集Actor与完整来源模型不一致')
        if exact_state_digest(learning['actor_optimizer']) != exact_state_digest(source['actor_optimizer']):
            raise ValueError('采集Adam与完整来源状态不一致')
        source_base_lr = float(source['scheduler']['base_lr'])
        base_lr = source_base_lr*args.lr_multiplier
        if not math.isfinite(base_lr) or not settings['actor_lr_adaptation']['minimum']<=base_lr<=settings['actor_lr_adaptation']['maximum']:
            raise ValueError('独立LR对照超出原配置候选范围')
        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=base_lr, weight_decay=0.)
        controller = BCGradientController(**stage['bc_gradient']);anchor = DisabledBC();generator = torch.Generator()
        rank_state = torch.load(rank_path/'rank_state.pt', map_location='cpu', weights_only=False)
        restore_case(actor, optimizer, controller, anchor, learning, rank_state, generator, base_lr)
        initial_digest = exact_state_digest(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
                                                rng=capture_local_rng({'actor': generator})))
        rows, targets, manifest = payload['transitions'], payload['targets'], payload['global_manifest']
        cache = RolloutTensorCache(rows, targets, collective.device, max_device_bytes=4*1024**3)
        zero = probability_check_local(policy, rows, global_manifest=manifest, distributed=collective,
                                       denoising_microbatch=args.microbatch, tensor_cache=cache, include_frozen=False)
        root_call(collective, lambda: atomic_json(args.output_dir/'zero_update.json', zero))
        selected = _eligible(manifest);owned = _owned(selected, manifest, collective)
        probe_rows = [rows[index] for _, index in owned][:16]
        baseline_probe = output_probe(policy, probe_rows, cache)
        outcomes = [];sweep_gradient = None
        for clip in args.clips:
            restore_case(actor, optimizer, controller, anchor, learning, rank_state, generator, base_lr)
            tag = 'clip_'+str(clip).replace('.', '_')
            folder = args.output_dir/tag
            root_call(collective, lambda: folder.mkdir(exist_ok=False))
            baseline, baseline_records, advantages, _ = gather_statistics(policy, rows, cache, manifest, collective,
                microbatch=args.microbatch, clip=clip, gamma=settings['gamma_denoising'])
            history = [];pending = {};saved_records = []

            def before_step(current_actor, index):
                nonlocal sweep_gradient
                rng = capture_local_rng({'actor': generator})
                original = {name:p.grad for name,p in actor.named_parameters() if p.requires_grad and p.grad is not None}
                if clip == args.clips[0] and index == 0:
                    sweep_gradient = {name:g.clone() for name,g in original.items()}
                before, records, _, full_gradient = gather_statistics(policy, rows, cache, manifest, collective,
                    microbatch=args.microbatch, clip=clip, gamma=settings['gamma_denoising'],
                    backward=True, optimizer=optimizer, controller=controller)
                pending.clear();pending.update(index=index, before=before, full_gradient=full_gradient,
                    parameter_before={name:p.detach().clone() for name,p in actor.named_parameters() if p.requires_grad},
                    minibatch_vs_full=vector_relation(full_gradient, original))
                if clip == args.clips[0] and index == math.ceil(len(selected)/stage['scale']['actor_minibatch_chains'])-1:
                    comparisons=[]
                    for other_clip in (.01, .005, .003):
                        if other_clip == clip:
                            other_metrics, other_grad = before, full_gradient
                        else:
                            other_metrics, _, _, other_grad = gather_statistics(policy, rows, cache, manifest, collective,
                                microbatch=args.microbatch, clip=other_clip, gamma=settings['gamma_denoising'],
                                backward=True, optimizer=optimizer, controller=controller)
                        comparisons.append(dict(clip=other_clip, metrics=other_metrics,
                                                gradient_vs_production_clip=vector_relation(full_gradient, other_grad)))
                    root_call(collective, lambda: atomic_json(folder/'same_shifted_model_clip_gradients.json', comparisons))
                for name,p in actor.named_parameters():
                    if p.requires_grad:p.grad=original.get(name)
                restore_local_rng(rng, {'actor': generator})

            def after_step(step):
                rng = capture_local_rng({'actor': generator})
                after, records, _, _ = gather_statistics(policy, rows, cache, manifest, collective,
                    microbatch=args.microbatch, clip=clip, gamma=settings['gamma_denoising'])
                direction = actual_update_direction(actor, pending['parameter_before'], pending['full_gradient'])
                item = dict(optimizer_step=step['optimizer_step'], learning_rate=step['learning_rate'],
                    gradient_clip_factor=step['gradient_clip_factor'], before=pending['before'], after=after,
                    actual_full_objective_gain=after['production_objective']-pending['before']['production_objective'],
                    minibatch_gradient_vs_full=pending['minibatch_vs_full'], **direction)
                history.append(item);saved_records.append(records.cpu())
                root_call(collective, lambda: (atomic_json(folder/f'step_{len(history):02d}.json', item),
                    print(f'[DPPO CHECK] clip={clip} step={len(history)} full_gain={item["actual_full_objective_gain"]:.12g} active_clip={after["active_clip_fraction"]:.6g}', flush=True)))
                pending.clear();restore_local_rng(rng, {'actor': generator})

            update = actor_update(policy, optimizer, rows, targets, cache, manifest, controller,
                distributed=collective, anchor=anchor, generator=generator,
                ppo_epochs=settings['ppo_epochs'], minibatch_chains=stage['scale']['actor_minibatch_chains'],
                microbatch=args.microbatch, sampled_steps=4, clip=clip,
                gamma_denoising=settings['gamma_denoising'], kl_reduction='free_coordinate_mean',
                objective_reduction='free_coordinate_mean', grad_clip_norm=settings['grad_clip_norm'],
                gradient_observer=before_step, step_observer=after_step,
                bc_samples_per_rollout=0, step_strategy='intermediate4', kl_limits=stage['kl'],
                backtracking=settings['optimizer_backtracking'], kl_microbatch=args.microbatch,
                minimum_lr=settings['actor_lr_adaptation']['minimum'], bc_enabled=False)
            final, records, advantages, _ = gather_statistics(policy, rows, cache, manifest, collective,
                microbatch=args.microbatch, clip=clip, gamma=settings['gamma_denoising'])
            kl = final_kl(policy, rows, cache, manifest, distributed=collective, microbatch=args.microbatch,
                          clip=clip, objective_reduction='free_coordinate_mean', include_reference=False)
            hard = check_kl(kl, stage['kl'])
            probe = output_probe(policy, probe_rows, cache)
            changes = compare_probes(baseline_probe, probe, collective)
            fingerprints = collective.all_gather_object(_model_fingerprint(actor))
            if len(set(fingerprints)) != 1:raise RuntimeError('候选八卡参数不一致')
            result = dict(clip=clip, initial=baseline, final=final, optimizer=update, step_history=history,
                hard_kl=hard, full_kl=kl, same_condition_output=changes, actor_fingerprint=fingerprints[0],
                full_objective_gain=final['production_objective']-baseline['production_objective'])
            def publish_candidate():
                atomic_json(folder/'summary.json', result)
                torch.save(dict(initial=baseline_records.cpu(),after_steps=saved_records,final=records.cpu(),
                                advantages=advantages.cpu()), folder/'probability_records.pt')
                if hard['passed'] and update['optimizer_steps']==update['planned_optimizer_steps'] and update['optimizer_steps']>0:
                    torch.save(dict(schema='genmo.stage2.asymmetric_actor_evaluation.v1', evaluation_only=True,
                        actor=immutable_cpu_snapshot(actor.state_dict()), kernel=policy.kernel_config,
                        source_identity=learning['identity'], source_case_sha256=file_sha256(args.case/'case.json'),
                        hard_kl=hard,complete_update_and_soft_kl_passed=True,
                        update=dict(optimizer_steps=update['optimizer_steps'],planned_optimizer_steps=update['planned_optimizer_steps'],
                                    clip=clip, actor_fingerprint=fingerprints[0], source='real_fixed_DPPO_validation_only')),
                        folder/'evaluation_actor.pt')
            root_call(collective, publish_candidate);outcomes.append(result)
        if not args.skip_lr_sweep and sweep_gradient is not None:
            sweep=[]
            for lr in sorted({base_lr/2,base_lr,base_lr*2,min(base_lr*4,1e-6),1e-6}):
                restore_case(actor, optimizer, controller, anchor, learning, rank_state, generator, base_lr)
                for name,p in actor.named_parameters():
                    if p.requires_grad:p.grad=None if name not in sweep_gradient else sweep_gradient[name].clone()
                norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],settings['grad_clip_norm']))
                optimizer.param_groups[0]['lr']=lr;optimizer.step()
                metrics, _, _, _ = gather_statistics(policy, rows, cache, manifest, collective,
                    microbatch=args.microbatch,clip=.01,gamma=settings['gamma_denoising'])
                changes=compare_probes(baseline_probe,output_probe(policy,probe_rows,cache),collective)
                item=dict(learning_rate=lr, metrics=metrics,hard_kl=check_kl(metrics,stage['kl']),
                          gradient_norm=norm,same_condition_output=changes,
                          scope='one_same_first_minibatch_gradient_and_same_nonempty_Adam_no_acceptance_filter')
                sweep.append(item)
                root_call(collective,lambda:(atomic_json(args.output_dir/'lr_sweep.json',sweep),
                    print(f'[DPPO LR] lr={lr:.6g} objective={metrics["production_objective"]:.12g} hard_pass={item["hard_kl"]["passed"]}',flush=True)))
        restore_case(actor, optimizer, controller, anchor, learning, rank_state, generator, base_lr)
        restored=exact_state_digest(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),rng=capture_local_rng({'actor':generator})))
        if restored!=initial_digest:raise RuntimeError('独立候选结束后无法精确恢复起点')
        root_call(collective,lambda:atomic_json(args.output_dir/'summary.json',dict(status='passed',
            case=str(args.case),case_sha256=file_sha256(args.case/'case.json'),source_checkpoint=str(args.source_checkpoint),
            source_checkpoint_sha256=file_sha256(args.source_checkpoint),base_lr=base_lr,
            source_base_lr=source_base_lr,lr_multiplier=args.lr_multiplier,nonempty_adam_states=len(optimizer.state),
            clips=[dict(clip=x['clip'],full_objective_gain=x['full_objective_gain'],actor_fingerprint=x['actor_fingerprint'],
                        optimizer_steps=x['optimizer']['optimizer_steps']) for x in outcomes],
            starting_state_restored_exactly=True,wall_seconds=time.perf_counter()-started)))
    finally:
        if cache is not None:cache.close()
        if group is not None:dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
