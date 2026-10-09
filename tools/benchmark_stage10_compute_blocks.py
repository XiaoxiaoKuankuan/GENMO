"""服务器1八卡Stage10三大真实形状计算块与候选精度的独立有限验收。

从已核验真实rollout选取64个不同物理环境的条件，八卡运行相同数据和算子候选，
用于隔离硬件/数值成本；此处不推进物理，不冒充8192条闭环吞吐或策略效果验收。
每个候选自行生成实际行为链及old概率，保持64个独立可恢复CUDA Generator，随后
检查不同batch、置换、尾批和末端低方差步骤，门槛仍为1e-4/1e-3/1e-8。

分别计时完整20步CFG生成64条链、128内部转移PPO前反向、256内部转移KL前向。
预热后记录八卡最慢墙钟P50/P95、设备事件和实际显存；嵌套条件编码不重复求和。
每个候选顺序执行，不混跑基准，强制注意力后端；失败记录且不启用生产。所有结果
写入新的独立目录，不覆盖来源参数或证据。完整梯度等价回归由专用入口另行执行。
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tools.verify_stage10_saved_learning import read_saved_rank
from tools.train_closedloop_stage10 import configuration
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy, masked_joint_log_prob
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.numerical_execution import MODES
from gem.closedloop.dppo.parallel_support import root_call, local_call
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.updater_v2 import _joint_kl


def measure(function, collective, repeats):
    times, events, peaks = [], [], []
    for iteration in range(repeats+2):
        collective.barrier(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter(); begin.record()
        error = None
        try:
            result = function()
        except Exception as failure:
            error = dict(type=type(failure).__name__, message=str(failure))
            result = None
        end.record(); end.synchronize()
        wall = time.perf_counter()-started
        errors = collective.all_gather_object(error)
        if any(errors): raise RuntimeError(f'Cooperative compute-block failure: {errors}')
        if iteration >= 2:
            times.append(wall); events.append(begin.elapsed_time(end)/1000.)
            peaks.append(dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved(),
                              board_used=torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0]))
        del result
    ranks = collective.all_gather_object(dict(wall_seconds=times, cuda_seconds=events, peak_bytes=peaks))
    slowest = torch.tensor([max(row['wall_seconds'][i] for row in ranks) for i in range(repeats)])
    return dict(p50_seconds=slowest.quantile(.5).item(), p95_seconds=slowest.quantile(.95).item(), ranks=ranks)


def parameters(policy, trace, indices, steps, *, prepared=None):
    context = {key: value[indices] for key, value in trace['conditions'].items()}
    if prepared is not None:
        prepared = dict(prepared, adapted={key: value[indices] for key, value in prepared['adapted'].items()},
            conditional=prepared['conditional'][indices],
            unconditional=None if prepared['unconditional'] is None else prepared['unconditional'][indices],
            inputs=policy._input_signature(context))
    return policy.transition_parameters(context, trace['chain'][indices, steps], steps, prepared=prepared)


@torch.no_grad()
def consistency(policy, trace):
    device = trace['chain'].device
    count = trace['chain'].shape[0]*policy.steps
    order = torch.randperm(count, device=device, generator=torch.Generator(device=device).manual_seed(932))
    reports = []
    for batch in (1, 2, 7, 8, 32, 64, 128, 256, 512):
        # B1专门覆盖全部64条链最后两步；其他batch全链置换，包含不同末尾不足批。
        selected = (torch.stack((torch.arange(64, device=device)*20+18,
                                torch.arange(64, device=device)*20+19), 1).flatten() if batch < 64 else order)
        errors = []
        with policy_phase(policy):
            for begin in range(0, len(selected), batch):
                flat = selected[begin:begin+batch]; index, step = flat//20, flat%20
                result = parameters(policy, trace, index, step)
                logp = masked_joint_log_prob(trace['chain'][index, step+1], result['mean'], result['std'], result['free_mask'])
                delta = logp-trace['old_log_probs'][index, step]
                reference = torch.distributions.Normal(result['mean'].double(), result['std'].double()).log_prob(
                    trace['chain'][index, step+1].double()).masked_fill(~result['free_mask'], 0.).sum((-2, -1))
                errors.append(torch.stack((delta.abs().max(), delta.expm1().abs().max(), (logp-reference).abs().max())))
        maximum = torch.stack(errors).amax(0).cpu().tolist()
        reports.append(dict(batch=batch, checked=len(selected), log_prob_error=maximum[0], ratio_error=maximum[1],
            independent_gaussian_error=maximum[2], passed=maximum[0]<=1e-4 and maximum[1]<=1e-3 and maximum[2]<=1e-8))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'iteration', 'weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--operator-profile', action='store_true')
    parser.add_argument('--compile-denoiser', action='store_true')
    parser.add_argument('--fp32-blocked-gemm', action='store_true')
    parser.add_argument('--pipelined-tensorcore', action='store_true')
    args = parser.parse_args()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or rank != int(os.environ['LOCAL_RANK']): raise ValueError('Requires Server1 eight GPUs')
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    control = DistributedCollectives(rank, world, device='cpu')
    devices = root_call(control, _available_gpus)
    root_call(control, lambda: args.output.mkdir(parents=True, exist_ok=False))
    torch.cuda.set_device(rank); torch.set_num_threads(1)
    group = dist.new_group(backend='nccl', timeout=timedelta(minutes=30))
    collective = DistributedCollectives(rank, world, device=f'cuda:{rank}', tensor_group=group)
    rows, targets, archive_sha = local_call(collective, lambda: read_saved_rank(args.iteration, rank))
    unique = {}
    for i, row in enumerate(rows):
        unique.setdefault(row.identity['env_id'], (row.context, float(targets['advantages'][i])))
    sources = [pair for shard in collective.all_gather_object(list(unique.items())) for pair in shard]
    if len(sources) != 64: raise ValueError(f'Require 64 actual distinct source environments, got {len(sources)}')
    conditions = {key: torch.cat([pair[1][0][key] for pair in sources]).to(f'cuda:{rank}') for key in rows[0].context}
    advantages = torch.tensor([pair[1][1] for pair in sources], device=f'cuda:{rank}')
    config = configuration(args.config); config['runtime']['genmo_device'] = f'cuda:{rank}'
    actor, _, _ = local_call(collective, lambda: load_actor(config))
    payload = torch.load(args.weights, map_location='cpu', mmap=True, weights_only=False)
    report = dict(schema='stage10.compute_blocks.v1', devices=devices, source_archive_sha256=archive_sha,
        source='64_distinct_real_environment_conditions_replayed_for_compute_diagnosis', results=[])
    # 即使本次只计时一个候选，也用相同真实条件和独立噪声现场生成FP32参考。
    # 此一次额外对照不进入候选P50/P95，不用旧报告的未知输出冒充逐值比较。
    actor.load_state_dict(payload['actor'])
    reference_policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1',
        defer_checks=True, precision_mode='fp32_reference')
    def reference_generation():
        return reference_policy.sample_rollout(conditions, generator=[torch.Generator(device=f'cuda:{rank}').manual_seed(
            713+rank*1000+i) for i in range(64)])
    reference_trace = local_call(collective, reference_generation)
    reference_output = reference_trace['normalized'].cpu()
    del reference_trace, reference_policy
    for mode in args.modes:
        item = dict(mode=mode)
        actor.load_state_dict(payload['actor']); actor.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        try:
            policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1',
                defer_checks=True, precision_mode=mode)
            if args.fp32_blocked_gemm:
                from gem.closedloop.dppo.numerical_execution import configure_blocked_fp32
                configure_blocked_fp32(policy)
            if args.compile_denoiser:
                from gem.closedloop.dppo.numerical_execution import compile_denoiser
                compile_denoiser(policy)
            if args.pipelined_tensorcore:
                from gem.closedloop.dppo.numerical_execution import configure_pipelined_tensorcore
                configure_pipelined_tensorcore(policy)
            item['contract'] = policy.kernel_config
            def generate():
                generators = [torch.Generator(device=f'cuda:{rank}').manual_seed(713+rank*1000+i) for i in range(64)]
                return policy.sample_rollout(conditions, generator=generators)
            trace = local_call(collective, generate)
            checks = local_call(collective, lambda: consistency(policy, trace))
            item['consistency_ranks'] = collective.all_gather_object(checks)
            item['self_consistency_passed'] = all(row['passed'] for shard in item['consistency_ranks'] for row in shard)
            current = trace['normalized'].cpu()
            if reference_output is None and mode == 'fp32_reference': reference_output = current
            if reference_output is not None:
                delta = current-reference_output
                mask = trace['free_mask'].cpu()
                item['fp32_output_difference'] = dict(max_abs=float(delta.abs().max()),
                    rms=float(delta.square().mean().sqrt()), free_coordinate_rms=float(delta[mask].square().mean().sqrt()),
                    scope='same_parameters_conditions_noise_and_full_20step_sampling_different_explicit_execution_contracts')
            item['t_gen64'] = measure(generate, collective, args.repeats)
            index128 = torch.arange(128, device=f'cuda:{rank}')%64
            step128 = (torch.arange(128, device=f'cuda:{rank}')//64)*19
            def forward_backward():
                actor.zero_grad(set_to_none=True)
                with policy_phase(policy):
                    prepared = policy.prepare_conditions(conditions)
                    result = parameters(policy, trace, index128, step128, prepared=prepared)
                    probability = masked_joint_log_prob(trace['chain'][index128, step128+1], result['mean'], result['std'], result['free_mask'])
                    ratio = (probability-trace['old_log_probs'][index128, step128]).exp()
                    loss = -torch.minimum(ratio*advantages[index128], ratio.clamp(.99, 1.01)*advantages[index128]).sum()/(2048*20)
                    loss.backward()
                return loss.detach()
            item['t_fb128'] = measure(forward_backward, collective, args.repeats)
            actor.zero_grad(set_to_none=True)
            index256 = torch.arange(256, device=f'cuda:{rank}')%64
            step256 = (torch.arange(256, device=f'cuda:{rank}')//64)*6
            @torch.no_grad()
            def kl_forward():
                with policy_phase(policy):
                    prepared = policy.prepare_conditions(conditions)
                    result = parameters(policy, trace, index256, step256, prepared=prepared)
                    return _joint_kl(result, result['free_mask'], trace['old_means'][index256, step256].double(),
                                     trace['old_stds'][index256, step256].double())
            item['t_kl256'] = measure(kl_forward, collective, args.repeats)
            if args.operator_profile:
                collective.barrier()
                def profile_root():
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as profile:
                        forward_backward();torch.cuda.synchronize()
                    (args.output/(mode+'_operators.txt')).write_text(profile.key_averages().table(
                        sort_by='self_cuda_time_total', row_limit=45))
                    profile.export_chrome_trace(str(args.output/(mode+'_trace.json')))
                root_call(collective, profile_root)
            item['status'] = 'candidate_measured' if item['self_consistency_passed'] else 'rejected_self_consistency'
            del trace, policy
        except Exception as error:
            item.update(status='candidate_failed', error_type=type(error).__name__, error=str(error))
        finally:
            actor.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        report['results'].append(item)
        if rank == 0:
            (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(dict(mode=mode, status=item['status'], error=item.get('error'),
                times={name:item[name]['p50_seconds'] for name in ('t_gen64','t_fb128','t_kl256') if name in item}), ensure_ascii=False), flush=True)
    dist.destroy_process_group(group); dist.destroy_process_group()


if __name__ == '__main__': main()
