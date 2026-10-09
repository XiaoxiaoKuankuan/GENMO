"""服务器1八卡真实旧rollout上的反向归约与微批累计候选验收。

只读加载同一封存rollout、优势、Actor和非空Adam状态，保留采集old概率。第一项使用
原逐样本权重梯度/FP64累计及B=1；所有候选重置相同状态，比较全量未裁剪梯度、各
模块方向、Adam状态、参数、末端输出和最终全量KL。失败候选保留报告并继续比较，
不发布模型，不推进仿真，不放宽原概率/梯度门槛。此小数据诊断不能代替8192条验收。

另外独立计时全模型梯度累计，报告P50/P95、临时显存及按张量读写推导的逻辑字节量；
逻辑字节量不冒充硬件DRAM计数。所有计时先预热并等待GPU完成，八卡同时运行同一
候选；任何时候不混跑其他基准。输出须是不存在的独立目录，原证据保持只读。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist

from tools.verify_stage10_saved_learning import read_saved_rank, compare_named
from tools.train_closedloop_stage10 import configuration
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.trainer import load_actor, trainable_actor_parameters
from gem.closedloop.dppo.batch_execution import configure_gradients, MicrobatchGradientAccumulator
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import actor_update_v2, analytic_kl_local, probability_check_local
from gem.closedloop.dppo.parallel_support import cpu_snapshot, root_call, local_call


def module_directions(current, reference):
    groups = {}
    for name, value in current.items():
        other = reference[name]
        if value is None or other is None:
            continue
        key = '.'.join(name.split('.')[:3])
        a, b = value.double().reshape(-1), other.double().reshape(-1)
        row = groups.setdefault(key, [0., 0., 0., 0.])
        for i, scalar in enumerate((a.dot(a), b.dot(b), a.dot(b), (a-b).dot(a-b))):
            row[i] += float(scalar)
    return {key: dict(cosine=None if a*b == 0 else dot/(a*b)**.5,
                     relative_l2=None if b == 0 else (delta/b)**.5)
            for key, (a, b, dot, delta) in groups.items()}


@torch.no_grad()
def terminal_outputs(policy, rows, device):
    context = {key: torch.cat([row.context[key] for row in rows[:4]]).to(device)
               for key in rows[0].context}
    prepared = policy.prepare_conditions(context)
    result = []
    for step in (policy.steps-2, policy.steps-1):
        state = torch.stack([row.chain[step] for row in rows[:4]]).to(device)
        value = policy.transition_parameters(context, state, step, prepared=prepared)
        result.append(value['mean'].cpu())
    return torch.stack(result)


def benchmark_accumulator(actor, collective, repeats):
    gradients = [(p, p.grad.detach().clone()) for p in actor.parameters() if p.grad is not None]
    output = []
    for mode in ('fp64_reference', 'fp32', 'selective_fp64'):
        times, traffic, peaks = [], [], []
        for repeat in range(repeats+2):
            actor.zero_grad(set_to_none=True)
            for parameter, value in gradients:
                parameter.grad = value.clone()
            accumulator = MicrobatchGradientAccumulator(actor, mode)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); accumulator.add(); accumulator.finish(); end.record(); end.synchronize()
            if repeat >= 2:
                times.append(begin.elapsed_time(end)/1000.)
                traffic.append(accumulator.traffic_bytes)
                peaks.append(torch.cuda.max_memory_allocated()-base)
        shards = collective.all_gather_object(dict(seconds=times, logical_bytes=traffic,
                                                    temporary_peak_bytes=peaks))
        slowest = torch.tensor([max(s['seconds'][i] for s in shards) for i in range(repeats)])
        output.append(dict(mode=mode, p50_seconds=float(slowest.quantile(.5)),
            p95_seconds=float(slowest.quantile(.95)), ranks=shards,
            memory_scope='logical_tensor_read_write_bytes_not_hardware_DRAM_counter'))
    actor.zero_grad(set_to_none=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('iteration', 'weights', 'config', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--microbatches', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--accumulator-repeats', type=int, default=5)
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or rank != int(os.environ['LOCAL_RANK']):
        raise ValueError('Server1 eight GPU ranks are required')
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    control = DistributedCollectives(rank, world, device='cpu')
    devices = root_call(control, _available_gpus)
    root_call(control, lambda: args.output.mkdir(parents=True, exist_ok=False))
    torch.cuda.set_device(rank); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    group = dist.new_group(backend='nccl', timeout=timedelta(minutes=20))
    collective = DistributedCollectives(rank, world, device=f'cuda:{rank}', tensor_group=group)
    config = configuration(args.config)
    actor, _, _ = local_call(collective, lambda: load_actor(config))
    actor = actor.to(f'cuda:{rank}')
    payload = torch.load(args.weights, map_location='cpu', mmap=True, weights_only=False)
    actor.load_state_dict(payload['actor'])
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1', defer_checks=True)
    rows, targets, archive_sha = local_call(collective, lambda: read_saved_rank(args.iteration, rank))
    if any(row.metadata['sampler_trace']['kernel_config'] != policy.kernel_config for row in rows):
        raise ValueError('This equivalent-backward comparison cannot change the saved behavior policy')
    cache = RolloutTensorCache(rows, targets, f'cuda:{rank}', max_device_bytes=4*1024**3)
    manifest = [item for shard in collective.all_gather_object([
        dict(owner_rank=rank, local_index=i, valid=row.transition_valid, has_free=bool(row.free_mask.any()))
        for i, row in enumerate(rows)]) for item in shard]
    selected = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
    sensitive = [name for name, _ in actor.named_parameters() if any(
        word in name for word in ('gate_', 'norm', 'history_encoder', 'prefix_encoder', 'cond_embed'))]
    variants = [('sample_bmm', 'fp64_reference', 1)]
    variants += [(reduction, accumulation, micro)
        for reduction, accumulation in [('joint_gemm', 'fp64_reference'), ('chunked_gemm', 'fp64_reference'),
            ('bounded_sample_bmm', 'fp64_reference'), ('joint_gemm', 'fp32'), ('joint_gemm', 'selective_fp64')]
        for micro in args.microbatches]
    report = dict(schema='stage10.gradient_reduction_audit.v1', devices=devices,
        source_archive_sha256=archive_sha, real_global_chains=len(manifest),
        preserved_old_probabilities=True, restored_nonempty_adam=bool(payload['actor_optimizer']['state']),
        sensitive_parameter_names=sensitive, numerical_reference='sample_bmm_fp64_accumulator_B1', results=[])
    reference = None
    for reduction, accumulation, micro in variants:
        actor.load_state_dict(payload['actor']); actor.zero_grad(set_to_none=True)
        configure_gradients(actor, weight_reduction=reduction, accumulation=accumulation, sensitive_names=sensitive)
        optimizer = torch.optim.AdamW(trainable_actor_parameters(actor), lr=5e-9, weight_decay=0.)
        optimizer.load_state_dict(copy.deepcopy(payload['actor_optimizer']))
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-9
        check = probability_check_local(policy, rows, denoising_microbatch=micro,
            tensor_cache=cache, distributed=collective, global_manifest=manifest)
        gradients = {}
        def observe(model, step):
            if rank == 0:
                gradients.update({name: None if p.grad is None else p.grad.detach().cpu().clone()
                                  for name, p in model.named_parameters()})
        collective.barrier(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated(); started = time.perf_counter()
        update = actor_update_v2(policy, optimizer, rows, targets, global_manifest=manifest, distributed=collective,
            ppo_epochs=1, epoch_orders=[selected], actor_minibatch_internal_transitions=len(selected)*policy.steps,
            max_optimizer_steps=1, denoising_microbatch=micro, soft_kl_limit=None, gradient_diagnostics=False,
            tensor_cache=cache, kl_check_mode='pre_step_plus_final', gradient_observer=observe)
        torch.cuda.synchronize(); updated = time.perf_counter()
        kl = analytic_kl_local(policy, rows, distributed=collective, global_manifest=manifest,
            denoising_microbatch=256, tensor_cache=cache)
        torch.cuda.synchronize(); finished = time.perf_counter()
        timing = collective.all_gather_object(dict(actor_seconds=updated-started, final_kl_seconds=finished-updated,
            peak_extra_allocated_bytes=torch.cuda.max_memory_allocated()-allocated,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.memory_reserved()))
        terminal = terminal_outputs(policy, rows, f'cuda:{rank}')
        item = dict(reduction=reduction, accumulation=accumulation, microbatch=micro,
            probability=check, update_count=update['optimizer_steps'], final_kl=kl,
            timing_ranks=timing, timing_scope='diagnostic_includes_unclipped_gradient_capture_not_performance_P50')
        if rank == 0:
            weights = cpu_snapshot(actor.state_dict())
            adam = {f'{i}/{key}':value.cpu().clone() for i, state in optimizer.state_dict()['state'].items()
                    for key, value in state.items() if torch.is_tensor(value)}
            if reference is None:
                reference = gradients, weights, adam, terminal
                item['reference'] = True
            else:
                item['gradients'] = compare_named(gradients, reference[0], atol=3e-5, rtol=2e-4)
                item['weights'] = compare_named(weights, reference[1], atol=1e-7, rtol=0.)
                item['adam'] = compare_named(adam, reference[2], atol=1e-7, rtol=2e-4)
                item['module_directions'] = module_directions(gradients, reference[0])
                item['terminal_mean_max_abs_difference'] = float((terminal-reference[3]).abs().max())
                item['passed'] = all(item[key]['passed'] for key in ('gradients', 'weights', 'adam'))
            report['results'].append(item)
            (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(dict(reduction=reduction, accumulation=accumulation, microbatch=micro,
                passed=item.get('passed', True), actor_seconds=max(t['actor_seconds'] for t in timing)), ensure_ascii=False), flush=True)
        if reduction == 'sample_bmm':
            cost = benchmark_accumulator(actor, collective, args.accumulator_repeats)
            if rank == 0: report['accumulator_cost'] = cost
        del optimizer, gradients
    if rank == 0:
        report.update(status='completed_candidates_reported', scope='finite_numerical_diagnostic_no_training_publication')
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    cache.close(); dist.destroy_process_group(group); dist.destroy_process_group()


if __name__ == '__main__':
    main()
