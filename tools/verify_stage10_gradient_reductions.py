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
import math
import os
from pathlib import Path
import sys
import time
from datetime import timedelta
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist

from tools.verify_stage10_saved_learning import read_saved_rank, compare_named
from tools.train_closedloop_stage10 import configuration
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.trainer import load_actor, trainable_actor_parameters
from gem.closedloop.dppo.batch_execution import configure_gradients, MicrobatchGradientAccumulator, WEIGHT_REDUCTIONS
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import actor_update_v2, analytic_kl_local, probability_check_local
from gem.closedloop.dppo.parallel_support import cpu_snapshot, root_call, local_call
from gem.closedloop.dppo.numerical_execution import MODES


@torch.no_grad()
def sample_diagnostic_chains(policy, source, device, rank):
    """新数值合同实际生成新链，绝不覆盖原文件或为旧链重新定义 old 概率。

    只复用真实条件与固定优势作算子诊断。新动作没有进入物理环境，因此这些对象
    仅供本工具的数学对照，不能写成真实闭环 rollout 或用于策略效果验收。
    """
    context = {key: torch.cat([row.context[key] for row in source]).to(device) for key in source[0].context}
    trace = policy.sample_rollout(context, generator=[torch.Generator(device=device).manual_seed(
        9200+rank*1000+i) for i in range(len(source))])
    trace = cpu_snapshot(trace)
    result = []
    for index, old in enumerate(source):
        part = {key: ({name:value[index:index+1] for name,value in value.items()} if key=='conditions' else
            value[index:index+1] if torch.is_tensor(value) and key!='timestep_map' else copy.deepcopy(value))
            for key,value in trace.items()}
        result.append(replace(old, chain=part['chain'][0], old_log_prob=part['old_log_probs'][0],
            free_mask=part['free_mask'][0], metadata=dict(old.metadata, sampler_trace=part,
                numerical_diagnostic_only=True, diagnostic_behavior_contract=policy.kernel_config)))
    return result


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
    parser.add_argument('--warmstart-adam', action='store_true',
                        help='先用原真实rollout执行一次参考PPO更新，固定得到的非空Adam供所有候选比较')
    parser.add_argument('--precision-mode', choices=MODES,
                        help='显式新数值合同：实际生成新诊断链；原封存 rollout 和 old 概率保持不变')
    parser.add_argument('--compact', action='store_true', help='只比较各微批 joint GEMM与指定累计精度，保留完整梯度门槛')
    parser.add_argument('--weight-reductions',nargs='+',choices=WEIGHT_REDUCTIONS,default=['joint_gemm'],
                        help='compact模式的显式反向算子；原B1参考不变，采样和old概率不变')
    parser.add_argument('--accumulation-modes', nargs='+', default=['fp64_reference'],
                        choices=('fp64_reference','fp32','selective_fp64'),
                        help='compact模式独立选择累计精度，原B1 FP64仍保留基准')
    parser.add_argument('--fp32-blocked-gemm', action='store_true')
    parser.add_argument('--pipelined-tensorcore', action='store_true')
    parser.add_argument('--compensated-bf16',type=int,choices=(3,6))
    parser.add_argument('--compile-fixed-denoiser',action='store_true')
    parser.add_argument('--actor-lr', type=float, default=5e-9,
                        help='显式固定数据校准值，默认5e-9；不改正式配置，不自动搜索或降级')
    args = parser.parse_args()
    if not math.isfinite(args.actor_lr) or args.actor_lr <= 0:
        raise ValueError('Diagnostic learning rate must be finite and positive')
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
    # 必须在load_actor内第一次.to()之前指定本rank设备，避免七个额外GPU0上下文。
    config['runtime']['genmo_device'] = f'cuda:{rank}'
    actor, _, _ = local_call(collective, lambda: load_actor(config))
    actor = actor.to(f'cuda:{rank}')
    payload = torch.load(args.weights, map_location='cpu', mmap=True, weights_only=False)
    actor.load_state_dict(payload['actor'])
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1', defer_checks=True,
                                 precision_mode=args.precision_mode)
    if args.fp32_blocked_gemm:
        from gem.closedloop.dppo.numerical_execution import configure_blocked_fp32
        configure_blocked_fp32(policy)
    if args.pipelined_tensorcore:
        from gem.closedloop.dppo.numerical_execution import configure_pipelined_tensorcore
        configure_pipelined_tensorcore(policy)
    if args.compensated_bf16:
        from gem.closedloop.dppo.numerical_execution import configure_compensated_bf16
        configure_compensated_bf16(policy,args.compensated_bf16)
    if args.compile_fixed_denoiser:
        from gem.closedloop.dppo.numerical_execution import compile_fixed_denoiser
        compile_fixed_denoiser(policy)
    rows, targets, archive_sha = local_call(collective, lambda: read_saved_rank(args.iteration, rank))
    if args.precision_mode is not None:
        rows = local_call(collective, lambda: sample_diagnostic_chains(policy, rows, f'cuda:{rank}', rank))
    if any(row.metadata['sampler_trace']['kernel_config'] != policy.kernel_config for row in rows):
        raise ValueError('This equivalent-backward comparison cannot change the saved behavior policy')
    cache = RolloutTensorCache(rows, targets, f'cuda:{rank}', max_device_bytes=4*1024**3)
    manifest = [item for shard in collective.all_gather_object([
        dict(owner_rank=rank, local_index=i, valid=row.transition_valid, has_free=bool(row.free_mask.any()))
        for i, row in enumerate(rows)]) for item in shard]
    selected = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
    source_actor = payload['actor']
    if args.warmstart_adam:
        probability_check_local(policy, rows, denoising_microbatch=32, tensor_cache=cache,
                                distributed=collective, global_manifest=manifest)
        warm_lr = config['stage10']['training']['actor_lr']
        warm_optimizer = torch.optim.AdamW(trainable_actor_parameters(actor), lr=warm_lr, weight_decay=0.)
        warm_optimizer.load_state_dict(copy.deepcopy(payload['actor_optimizer']))
        for parameter_group in warm_optimizer.param_groups: parameter_group['lr'] = warm_lr
        actor_update_v2(policy, warm_optimizer, rows, targets, global_manifest=manifest, distributed=collective,
            ppo_epochs=1, epoch_orders=[selected], actor_minibatch_internal_transitions=len(selected)*policy.steps,
            max_optimizer_steps=1, denoising_microbatch=32, soft_kl_limit=None, gradient_diagnostics=False,
            tensor_cache=cache, kl_check_mode='pre_step_plus_final')
        payload = dict(actor=cpu_snapshot(actor.state_dict()), actor_optimizer=cpu_snapshot(warm_optimizer.state_dict()))
        del warm_optimizer
    sensitive = [name for name, _ in actor.named_parameters() if any(
        word in name for word in ('gate_', 'norm', 'history_encoder', 'prefix_encoder', 'cond_embed'))]
    variants = [('joint_gemm' if args.precision_mode else 'sample_bmm', 'fp64_reference', 1)]
    variants += [(reduction, accumulation, micro)
        for reduction, accumulation in ([(red, mode) for red in args.weight_reductions for mode in args.accumulation_modes] if args.compact else
            [('joint_gemm', 'fp64_reference'), ('chunked_gemm', 'fp64_reference'),
            ('bounded_sample_bmm', 'fp64_reference'), ('joint_gemm', 'fp32'), ('joint_gemm', 'selective_fp64')]
        )
        for micro in args.microbatches]
    report = dict(schema='stage10.gradient_reduction_audit.v1', devices=devices,
        source_archive_sha256=archive_sha, real_global_chains=len(manifest),
        preserved_source_archive=True, preserved_old_probabilities=True,
        precision_mode=args.precision_mode, numerical_contract=policy.kernel_config,
        chain_origin='actual_new_behavior_sampling_on_real_conditions_no_physics' if args.precision_mode else 'immutable_real_rollout',
        nonempty_adam=bool(payload['actor_optimizer']['state']),
        actor_lr=args.actor_lr, hard_kl_limit=config['stage10']['training']['kl_stop_joint'],
        warmstart_lr=config['stage10']['training']['actor_lr'] if args.warmstart_adam else None,
        adam_origin='one_fixed_real_PPO_warmup_step' if args.warmstart_adam else 'source_checkpoint',
        sensitive_parameter_names=sensitive, numerical_reference=variants[0], results=[])
    reference = None
    for reduction, accumulation, micro in variants:
        actor.load_state_dict(source_actor); actor.zero_grad(set_to_none=True)
        configure_gradients(actor, weight_reduction=reduction, accumulation=accumulation, sensitive_names=sensitive)
        optimizer = torch.optim.AdamW(trainable_actor_parameters(actor), lr=args.actor_lr, weight_decay=0.)
        optimizer.load_state_dict(copy.deepcopy(payload['actor_optimizer']))
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.actor_lr
        check = probability_check_local(policy, rows, denoising_microbatch=micro,
            tensor_cache=cache, distributed=collective, global_manifest=manifest)
        # warmstart后的参数相对采集旧策略已改变；零更新检查必须在原采集参数上执行。
        actor.load_state_dict(payload['actor'])
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
            hard_kl_accepted=bool(math.isfinite(kl['mean_joint_kl']) and kl['mean_joint_kl'] <= report['hard_kl_limit']),
            closedloop_validation_passed=False,
            timing_ranks=timing, timing_scope='diagnostic_includes_unclipped_gradient_capture_not_performance_P50')
        if rank == 0:
            weights = cpu_snapshot(actor.state_dict())
            adam = {f'{i}/{key}':value.cpu().clone() for i, state in optimizer.state_dict()['state'].items()
                    for key, value in state.items() if torch.is_tensor(value)}
            if reference is None:
                reference = gradients, weights, adam, terminal, kl
                item['reference'] = True
            else:
                item['gradients'] = compare_named(gradients, reference[0], atol=3e-5, rtol=2e-4)
                item['weights'] = compare_named(weights, reference[1], atol=1e-7, rtol=0.)
                item['adam'] = compare_named(adam, reference[2], atol=1e-7, rtol=2e-4)
                item['module_directions'] = module_directions(gradients, reference[0])
                item['terminal_mean_max_abs_difference'] = float((terminal-reference[3]).abs().max())
                item['final_kl_abs_difference_from_reference'] = abs(kl['mean_joint_kl']-reference[4]['mean_joint_kl'])
                item['gradient_weight_adam_parity_passed'] = all(item[key]['passed'] for key in ('gradients', 'weights', 'adam'))
                # 算子回归和训练接受是两件事，禁止将梯度三项通过写成笼统 passed。
                item['numerical_update_and_hard_kl_passed'] = bool(item['gradient_weight_adam_parity_passed'] and item['hard_kl_accepted'])
            report['results'].append(item)
            (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(dict(reduction=reduction, accumulation=accumulation, microbatch=micro,
                parity_passed=item.get('gradient_weight_adam_parity_passed'), hard_kl_accepted=item['hard_kl_accepted'],
                actor_seconds=max(t['actor_seconds'] for t in timing)), ensure_ascii=False), flush=True)
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
