"""真实 Stage1 权重下 RoboFFT MC 微批 1/2/4 的有限数值、梯度与资源对照。

从显式 RoboFFT 配置装配原 Actor、冻结采样网络和四来源真实音乐/历史/前缀条件，
只采一次 compact 最终 action_x0，并一次固定16个 MC 及旧损失。三种执行微批共享
完全相同的动作、旧损失和人工非零优势；使用生产两遍流式反传，分别预热并报告
重复测量的墙钟、CUDA allocated/reserved 峰值、代理值/梯度误差。不会 optimizer.step，
不会加载或覆盖正式 Adam、checkpoint、MC RNG，不启动 Critic 或物理世界。

条件来自真实数据，但动作与优势仅用于计算验收，不能称为 PhysX rollout/真实 GAE，
也不能把该吞吐当成完整训练轮耗时或运动收益。测量不含采集、MC准备、Adam、逐步
诊断及全量最终诊断；显存包含本工具驻留 Actor/冻结网络/输入，不含 Critic/PhysX。
要求所选 GPU 空闲，输出文件必须不存在；使用者须将输出放入明确的临时测试目录，
把摘要记录到文档后精确清理。任何数值/梯度不等价会报错，不自动改变 LR 或温度。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.rollout_helpers import split_trace
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.robofft import ALGORITHM, OBJECTIVE
from gem.closedloop.robofft.updater import METRICS, _losses, objective_batch, prepare_mc
from tools.validate_stage10_compute import load_real_conditions


def gradient_comparison(actor, reference):
    squared_error = squared_reference = maximum = reference_maximum = 0.
    for name, parameter in actor.named_parameters():
        if not parameter.requires_grad:
            continue
        value = torch.zeros_like(reference[name]) if parameter.grad is None else parameter.grad.detach().cpu()
        expected = reference[name].double()
        difference = value.double() - expected
        squared_error += float(difference.square().sum())
        squared_reference += float(expected.square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        reference_maximum = max(reference_maximum, float(expected.abs().max()))
    if squared_reference == 0:
        raise AssertionError('Nonzero surrogate gradient required')
    relative = (squared_error / squared_reference)**.5
    if relative > 2e-4 or maximum > 2e-5 + 2e-4*reference_maximum:
        raise AssertionError(f'MC microbatch gradient mismatch: relative={relative}, max={maximum}')
    return dict(relative_l2_error=relative, maximum_absolute_error=maximum,
                reference_l2_norm=squared_reference**.5)


def measure_backward(policy, rows, cache, clip):
    policy.actor.zero_grad(set_to_none=True)
    torch.cuda.synchronize(cache.device)
    torch.cuda.reset_peak_memory_stats(cache.device)
    started = time.perf_counter()
    with policy_phase(policy):
        metrics = objective_batch(policy, rows, cache, clip=clip, global_count=len(rows), backward=True)
    torch.cuda.synchronize(cache.device)
    return dict(seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(cache.device)/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved(cache.device)/1024**3,
                metrics=dict(zip(METRICS, (metrics/len(rows)).cpu().tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gmt-repo', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--actions', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.actions < 2 or args.repeats < 1:
        parser.error('At least two actions and one measurement are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    # 在创建任何 CUDA 上下文前核对资源；此工具不停止训练来抢占卡。
    device = torch.device(args.device)
    if device.type != 'cuda' or device.index is None:
        parser.error('--device must be explicit cuda:N')
    busy = subprocess.check_output(['nvidia-smi', '-i', str(device.index),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('Benchmark GPU is occupied: '+busy)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    random.seed(426); np.random.seed(426); torch.manual_seed(426)
    config = configuration(args.config, genmo_repo=ROOT, gmt_repo=args.gmt_repo)
    if config['stage9']['actor_objective'] != OBJECTIVE:
        parser.error('Explicit RoboFFT configuration is required')
    config['runtime']['genmo_device'] = str(device)
    actor, _, loading = load_actor(config)
    policy = build_policy(actor, config['stage9'], config['stage10']['precision'])
    context = load_real_conditions(actor, config, device, args.actions, frames=60)
    with policy_phase(policy), torch.no_grad():
        trace = policy.sample_rollout(context, generator=[
            torch.Generator(device=device).manual_seed(59000+i) for i in range(args.actions)])
    rows = [SimpleNamespace(context={k:v[i:i+1] for k,v in context.items()}, identity={},
        algorithm=ALGORITHM, chain=None, old_log_prob=None, action_x0=trace['action_x0'][i],
        free_mask=trace['free_mask'][i], mc_record=None,
        metadata=dict(critic_context={}, remaining_music_seconds=0.,
                      sampler_trace=split_trace(trace, i, args.actions))) for i in range(args.actions)]
    advantages = torch.linspace(-1., 1., args.actions, dtype=torch.float64)
    advantages /= advantages.std(unbiased=False)
    cache = RolloutTensorCache(rows, dict(advantages=advantages), device)
    original_chunk = policy.robofft['mc_microbatch']
    generator = torch.Generator().manual_seed(72001)
    report = dict(schema='genmo.robofft.mc_microbatch_benchmark.v1',
        scope='real_Stage1_weights_conditions_compact_actions_synthetic_advantages_no_physics_or_Adam',
        measured_phase='production_two_pass_objective_backward_only',
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__, actions=args.actions,
        mc_samples=16, repeats=args.repeats, source_global_step=loading['source_global_step'],
        precision=config['stage10']['precision'], variants=[])
    try:
        policy.robofft['mc_microbatch'] = 1
        with policy_phase(policy):
            prepare_mc(policy, rows, cache, generator, microbatch=args.actions)
        old = torch.stack([r.mc_record.old_loss for r in rows]).clone()
        rng = generator.get_state().clone()
        versions = [(id(p), p._version) for p in actor.parameters()]
        reference_gradient = None
        reference_loss = None
        reference_metrics = None
        for chunk in (1, 2, 4):
            policy.robofft['mc_microbatch'] = chunk
            actor.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            # 每个执行形状独立预热，含真实反向；预热/JIT不计入稳态时间。
            measure_backward(policy, rows, cache, config['stage9']['ppo_clip'])
            samples = [measure_backward(policy, rows, cache, config['stage9']['ppo_clip'])
                       for _ in range(args.repeats)]
            if reference_gradient is None:
                reference_gradient = {n:(torch.zeros_like(p, device='cpu') if p.grad is None
                    else p.grad.detach().cpu().clone()) for n,p in actor.named_parameters() if p.requires_grad}
                reference_metrics = samples[-1]['metrics']
            gradient = gradient_comparison(actor, reference_gradient)
            with policy_phase(policy), torch.no_grad():
                current = torch.cat([_losses(policy, rows, cache, k, k+chunk)
                                     for k in range(0, 16, chunk)], dim=1).cpu()
            if reference_loss is None:
                reference_loss = current.clone()
            torch.testing.assert_close(current, reference_loss, atol=2e-5, rtol=2e-5)
            ratio_error = abs(samples[-1]['metrics']['ratio']-reference_metrics['ratio'])
            if ratio_error > 2e-6:
                raise AssertionError('MC microbatch action ratio differs')
            assert versions == [(id(p), p._version) for p in actor.parameters()]
            assert torch.equal(rng, generator.get_state())
            assert torch.equal(old, torch.stack([r.mc_record.old_loss for r in rows]))
            for row in rows:
                row.mc_record.validate()
            item = dict(mc_microbatch=chunk, samples=samples,
                median_seconds=statistics.median(s['seconds'] for s in samples),
                peak_allocated_gib=max(s['peak_allocated_gib'] for s in samples),
                peak_reserved_gib=max(s['peak_reserved_gib'] for s in samples),
                loss_max_absolute_error=float((current-reference_loss).abs().max()),
                mean_ratio_absolute_error=ratio_error, gradient=gradient)
            report['variants'].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
        report.update(passed=True, old_data_unchanged=True, mc_rng_unchanged=True, parameters_unchanged=True)
    finally:
        policy.robofft['mc_microbatch'] = original_chunk
        cache.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            json.dump(report, output, indent=2, ensure_ascii=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
