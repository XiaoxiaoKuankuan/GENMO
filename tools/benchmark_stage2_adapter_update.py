"""固定真实checkpoint/起点的Adapter完整更新性能与梯度对照工具。

从只读checkpoint提取模型、优化器和真实采集状态，保存一次固定1024样本夹具；
各microbatch/时域/投影实现都恢复同一参数与Adam状态，测量真正的前向、反向、
裁剪及optimizer.step。输出同步墙钟、CUDA分段、显存、GPU利用率、损失/梯度/
参数更新一致性、投影拒样和最终证书。原始运行目录从不写入；只在显式输出目录
保留审计结果和夹具，不产生正式策略训练计数，不把计算一致性当物理跟踪收益。
数值旧实现从显式baseline-source加载，避免把两次新代码测量冒充前后对照。
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import types


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline-source', required=True)
    p.add_argument('--fixture')
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--horizons', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--microbatches', type=int, nargs='+', default=[32, 64, 128, 256])
    p.add_argument('--implementations', nargs='+', default=['baseline', 'optimized'])
    p.add_argument('--no-compile-derivatives', action='store_true')
    p.add_argument('--compile-components', action='store_true')
    p.add_argument('--aux-weight', type=float, default=0.)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    # 配置只读提取：尚未反序列化InputSpec时先加入冻结SM模块。
    import yaml
    baseline = Path(args.baseline_source)
    sm = baseline.parent/'source_supermimic'
    sys.path.insert(0, str(sm))
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = ck['config']
    sys.path.insert(0, cfg['supermimic']['repo'])
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl, select_state
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.alternating_learning import policy_update
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    engine = build_engine(cfg, 'cuda:0')
    engine.step.adapter.load_state_dict(ck['adapter'])
    engine.dynamics.load_state_dict(ck['dynamics'])
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    control.disable_root_orientation = False
    control.soft_joint_limit_cost = False
    control.sm_soft_termination_only = True
    control.proposal_feasibility_weight = args.aux_weight
    from supermimic.networks import Critic
    critic = Critic(2835).cuda().requires_grad_(False)
    critic.load_state_dict(ck['critic'])
    scales = json.loads(Path(cfg['alternating']['calibration']).read_text())['scales']
    control.precision = PrecisionObjective(scales, **cfg['alternating']['precision'])
    compile_kinematics(engine, bank)
    if args.compile_components:
        from gem.closedloop.reference_adapter.compute import compile_components
        compile_components(control, engine.dynamics)
    if args.fixture:
        fixture = torch.load(args.fixture, map_location='cpu', weights_only=False)
        z = {k: v[:args.batch].to('cuda:0') for k, v in fixture['starts'].items()}
    else:
        state = ck['collector_state']
        ok = state['frame']+max(args.horizons) < bank.ends.cpu()[state['song']]
        ids = ok.nonzero().flatten()[:args.batch]
        if len(ids) != args.batch:
            raise ValueError('checkpoint没有足够的真实长时域起点，不复制起点补齐batch')
        z = {k: v[ids].to('cuda:0') for k, v in state.items()}
        torch.save(dict(checkpoint=args.checkpoint, indices=ids,
                        starts={k: v.cpu() for k,v in z.items()}), out/'fixed_starts.pt')
    if len(z['state']) != args.batch:
        raise ValueError('固定夹具不足实际batch')
    old_projection = load_module('gem.closedloop.reference_adapter.benchmark_old_projection',
        baseline/'gem/closedloop/reference_adapter/reference_constraints.py')
    old_so3 = load_module('gem.closedloop.reference_adapter.benchmark_old_so3',
        baseline/'gem/closedloop/reference_adapter/so3_projection.py')
    initial_adapter = copy.deepcopy(control.adapter.state_dict())
    class RecordingAdam(torch.optim.Adam):
        def step(self, *pos, **kw):
            self.gradient = torch.cat([v.grad.detach().flatten() for group in self.param_groups
                                      for v in group['params'] if v.grad is not None]).clone()
            return super().step(*pos, **kw)
    results = []
    vectors = {}
    for implementation in args.implementations:
        cls = old_projection.ReferenceProjection if implementation == 'baseline' else ReferenceProjection
        control.projection = cls(control.bundle.spec, control.adapter.bounds, **cfg['alternating']['projection']).cuda()
        control.projection.compile_derivatives = not args.no_compile_derivatives
        if implementation == 'baseline':
            def old_project(self, *inputs):
                return old_so3.project_rotation_implicit(self, *inputs)
            control.projection.project_rotation = types.MethodType(old_project, control.projection)
        control.adapter.load_state_dict(initial_adapter)
        warm_optimizer = RecordingAdam(control.adapter.parameters(), lr=cfg['training']['adapter_lr'], betas=(.9, .95))
        warm_begin = time.perf_counter()
        policy_update(control, engine.dynamics, critic, warm_optimizer, select_state(z, slice(0, min(32, args.batch))),
                      2, terminal_value=False, microbatch=32, diagnostics=False)
        torch.cuda.synchronize()
        (out/f'warmup_{implementation}.json').write_text(json.dumps(dict(seconds=time.perf_counter()-warm_begin,
            excluded_from_steady_measurement=True)))
        del warm_optimizer
        for horizon in args.horizons:
            for micro in args.microbatches:
                control.adapter.load_state_dict(initial_adapter)
                optimizer = RecordingAdam(control.adapter.parameters(), lr=cfg['training']['adapter_lr'], betas=(.9, .95))
                optimizer.load_state_dict(copy.deepcopy(ck['optimizers']['adapter']))
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                usage, stop = [], threading.Event()
                def sample_gpu():
                    import os
                    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
                    while not stop.wait(.25):
                        line = subprocess.check_output(['nvidia-smi', '-i', gpu,
                            '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True)
                        usage.append([float(v.strip()) for v in line.strip().split(',')])
                thread = threading.Thread(target=sample_gpu, daemon=True); thread.start()
                torch.cuda.synchronize(); begin = time.perf_counter()
                row = dict(implementation=implementation, horizon=horizon, batch=args.batch, microbatch=micro)
                try:
                    metrics, _ = policy_update(control, engine.dynamics, critic, optimizer, z, horizon,
                        terminal_value=False, microbatch=micro, diagnostics=True, timing=True)
                    torch.cuda.synchronize()
                    gradient = optimizer.gradient.cpu()
                    parameters = torch.cat([v.detach().flatten() for v in control.adapter.parameters()]).cpu()
                    row.update(seconds=time.perf_counter()-begin, metrics=metrics,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                    key = (implementation, horizon)
                    ref = vectors.get(key)
                    if ref is None:
                        vectors[key] = (gradient, parameters, metrics)
                        row['comparison'] = 'first_microbatch_reference'
                    else:
                        g, params, m = ref
                        row['comparison'] = dict(gradient_relative_l2=float((gradient-g).norm()/g.norm().clamp_min(1e-20)),
                            gradient_cosine=float(torch.nn.functional.cosine_similarity(gradient, g, dim=0)),
                            parameter_max_abs=float((parameters-params).abs().max()),
                            objective_abs=abs(metrics['objective']-m['objective']),
                            same_accepted_samples=metrics['accepted_samples']==m['accepted_samples'])
                    if implementation == 'optimized' and ('baseline', horizon) in vectors:
                        g, params, m = vectors[('baseline', horizon)]
                        row['baseline_comparison'] = dict(gradient_relative_l2=float((gradient-g).norm()/g.norm().clamp_min(1e-20)),
                            gradient_cosine=float(torch.nn.functional.cosine_similarity(gradient, g, dim=0)),
                            parameter_max_abs=float((parameters-params).abs().max()), objective_abs=abs(metrics['objective']-m['objective']))
                except Exception as error:
                    import traceback
                    row.update(error=repr(error), traceback=traceback.format_exc(), seconds=time.perf_counter()-begin)
                    torch.cuda.empty_cache()
                finally:
                    stop.set(); thread.join()
                row['gpu_usage_samples'] = usage
                results.append(row)
                (out/'benchmark.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ('metrics','gpu_usage_samples','traceback')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
