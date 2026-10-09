"""服务器1八卡直接使用已封存真实rollout的学习批量验收。

本入口不生成新扩散链，不修改任何old概率、优势、动作或历史。每个rank只读取原来
属于自己的20条真实转移，校验归档和内部成员SHA，再用采集时初始权重复算全部20步。
按B=1、8、16、32顺序执行相同全局1600内部转移的一次Adam更新，逐参数比较同步后
未裁剪梯度、参数以及Adam状态，并执行完整3200内部转移KL。另行对4次更新定额计时，
该离线性能试验不发布权重、不声称通过正式KL、不推进物理环境。原始文件只读，读取
结束再核验归档SHA。所有报告写入显式的新路径，禁止覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from gem.closedloop.training import build_stage1_actor
from gem.closedloop.dppo.buffer import UpperTransition
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import (
    actor_update_v2, analytic_kl_local, probability_check_local, balanced_epoch_order,
)
from gem.closedloop.dppo.parallel_support import cpu_snapshot
from gem.closedloop.dppo.run_management import file_sha256
from tools.train_closedloop_stage10_8gpu import _available_gpus


def read_saved_rank(directory, rank):
    manifest = json.loads((directory/'archive_manifest.json').read_text())
    archive = directory/manifest['archive']
    if file_sha256(archive) != manifest['archive_sha256']:
        raise ValueError('Immutable execution archive SHA mismatch')
    members = {r['path']: r for r in manifest['members']}
    prefix = f'rank{rank:02d}/'
    needed = {k for k in members if k.startswith(prefix) and (
        '/raw_samples/' in k or '/rollout/' in k or k.endswith('/fixed_targets.pt'))}
    content = {}
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            if member.name in needed:
                if not member.isfile() or member.size != members[member.name]['size_bytes']:
                    raise ValueError('Invalid archived file size/type')
                data = stream.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != members[member.name]['sha256']:
                    raise ValueError('Archived member SHA mismatch')
                content[member.name] = data
    if set(content) != needed:
        raise ValueError('Incomplete immutable archive')
    def load(name):
        return torch.load(io.BytesIO(content[name]), map_location='cpu', weights_only=False)
    rollout = json.loads(content[prefix+'rollout/manifest.json'])
    rows = []
    for chunk in rollout['chunks']:
        chunk_path = prefix+'rollout/'+chunk['path']
        record_manifest = json.loads(content[chunk_path])
        loaded = {}
        for record in record_manifest['records']:
            path = str(Path(chunk_path).parent/record['path'])
            if path not in loaded:
                loaded[path] = load(path)
            value = loaded[path]
            if record.get('storage') == 'block_raw_reference.v2':
                compact = value[record['index']]
                raw = load(prefix+compact['raw']['path'])
                trace = raw['trace']
                fields = compact['fields'].copy()
                fields['metadata'] = dict(fields['metadata'], sampler_trace=trace, generated=raw['generated'])
                value = UpperTransition(**fields, chain=trace['chain'][0],
                    old_log_prob=trace['old_log_probs'][0], free_mask=trace['free_mask'][0])
            value.validate()
            rows.append(value)
    if len(rows) != 20 or rollout['transition_count'] != 20:
        raise ValueError('Expected exactly 20 saved actual transitions per rank')
    return rows, load(prefix+'fixed_targets.pt'), manifest['archive_sha256']


def compare_named(current, reference, *, atol, rtol):
    if current.keys() != reference.keys():
        raise AssertionError('Named tensor coverage changed')
    rows = {}
    for name, a in current.items():
        b = reference[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise AssertionError(f'Gradient participation changed: {name}')
            rows[name] = dict(gradient_present=False)
            continue
        diff, base = (a.double()-b.double()), b.double()
        rows[name] = dict(max_abs=float(diff.abs().max()),
            relative_l2=float(diff.norm()/base.norm().clamp_min(1e-30)),
            passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)))
    return dict(passed=all(r.get('passed', True) for r in rows.values()),
                atol=atol, rtol=rtol, parameter_tensors=len(rows), tensors=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('iteration', 'weights', 'stage1-config', 'assets', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--timing-repeats', type=int, default=2)
    args = parser.parse_args()
    rank, world, local = [int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    if world != 8 or rank != local:
        raise ValueError('Requires Server1 single node eight GPUs')
    dist.init_process_group('gloo', timeout=timedelta(minutes=15))
    startup = [None]
    if rank == 0:
        try:
            if args.output.exists(): raise FileExistsError(args.output)
            startup[0] = dict(devices=_available_gpus())
        except Exception as error:
            startup[0] = dict(error=str(error))
    dist.broadcast_object_list(startup, 0)
    if 'error' in startup[0]: raise RuntimeError(startup[0]['error'])
    torch.cuda.set_device(local)
    device = f'cuda:{local}'
    group = dist.new_group(backend='nccl', timeout=timedelta(minutes=10))
    collective = DistributedCollectives(rank, world, tensor_group=group, device=device,
                                        measure_gradient_communication=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.create(json.loads(args.stage1_config.read_text()))
    config.endecoder.stats_path = str(args.assets/'qpos30_train_stats.json')
    config.endecoder.kinematics_path = str(args.assets/'bumi_kinematics_robot_retargeter_fe934_v1.json')
    data = OmegaConf.create(dict(qpos30_stats=dict(path=config.endecoder.stats_path),
        dataset_defaults=dict(kinematics_path=config.endecoder.kinematics_path),
        sample_contract=dict(history_steps=50), datasets={}))
    actor = build_stage1_actor(config, data).float().eval().to(device)
    initial = torch.load(args.weights, map_location='cpu', mmap=True, weights_only=False)['actor']
    actor.load_state_dict(initial, strict=True)
    policy = DPPODiffusionPolicy(actor, cfg_batch=True, numerical_layout='sample_matrix_bmm_fp32.v1', defer_checks=True)
    rows, targets, archive_sha = read_saved_rank(args.iteration, rank)
    if any(row.metadata['sampler_trace']['kernel_config'] != policy.kernel_config for row in rows):
        raise ValueError('Saved numerical identity differs; resampling forbidden')
    cache = RolloutTensorCache(rows, targets, device)
    manifest = [item for shard in collective.all_gather_object([
        dict(owner_rank=rank, local_index=i, valid=row.transition_valid, has_free=bool(row.free_mask.any()))
        for i,row in enumerate(rows)]) for item in shard]
    orders = [balanced_epoch_order(list(range(160)), manifest, torch.Generator().manual_seed(61+e)) for e in range(2)]
    report = dict(schema='stage10.saved_learning.v1', source_archive_sha256=archive_sha,
        preserved_old_data=True, resampled_chains=0, global_actual_transitions=160,
        scope='PPO_and_KL_fixed_actual_data_no_BC_no_physics_no_policy_publication', results=[])
    reference = None
    candidates = [(b,1,'numerical') for b in (1,8,16,32)]
    candidates += [(b,4,f'timing{repeat}') for repeat in range(args.timing_repeats) for b in (8,16,32)]
    for micro, steps, scope in candidates:
        actor.load_state_dict(initial); actor.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=5e-9, weight_decay=0.)
        check = probability_check_local(policy, rows, denoising_microbatch=micro, tensor_cache=cache,
                                        distributed=collective, global_manifest=manifest)
        gradients = {}
        def observe(model, step):
            if rank == 0 and scope == 'numerical':
                gradients.update({n:None if p.grad is None else p.grad.detach().cpu().clone()
                                  for n,p in model.named_parameters()})
        collective.barrier(); torch.cuda.synchronize(); start = time.perf_counter()
        sink = {}
        update = actor_update_v2(policy, optimizer, rows, targets, denoising_microbatch=micro,
            epoch_orders=orders, max_optimizer_steps=steps, soft_kl_limit=None, gradient_diagnostics=False,
            distributed=collective, global_manifest=manifest, tensor_cache=cache, kl_cache_sink=sink,
            gradient_observer=observe if scope == 'numerical' else None)
        torch.cuda.synchronize(); middle = time.perf_counter()
        kl = analytic_kl_local(policy, rows, denoising_microbatch=micro, tensor_cache=cache,
            distributed=collective, global_manifest=manifest, reuse_cache=sink.get('cache'))
        torch.cuda.synchronize(); end = time.perf_counter()
        times = collective.all_gather_object(dict(actor=middle-start, kl=end-middle, total=end-start,
            peak_memory=torch.cuda.max_memory_allocated(device)))
        item = dict(batch=micro, scope=scope, probability=check, kl=kl,
            actual_optimizer_steps=update['optimizer_steps'],
            seconds={key:max(t[key] for t in times) for key in times[0]},
            hard_kl_would_accept=kl['mean_joint_kl']<=.03)
        if rank == 0 and scope == 'numerical':
            weights = cpu_snapshot(actor.state_dict())
            state = {f'{i}/{key}':v.cpu().clone() for i,s in optimizer.state_dict()['state'].items()
                     for key,v in s.items() if torch.is_tensor(v)}
            if reference is None:
                reference = gradients, weights, state
            else:
                # 使用现有梯度回归容限，不放宽概率门槛。每个参数单独留证。
                item['gradients'] = compare_named(gradients,reference[0],atol=3e-5,rtol=2e-4)
                item['weights'] = compare_named(weights,reference[1],atol=1e-7,rtol=0.)
                item['adam'] = compare_named(state,reference[2],atol=1e-7,rtol=2e-4)
        passed = all(item.get(k,{}).get('passed',True) for k in ('gradients','weights','adam'))
        passed = all(collective.all_gather_object(passed))
        report['results'].append(item)
        if rank == 0:
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(dict(batch=micro,scope=scope,passed=passed,seconds=item['seconds'])),flush=True)
        if not passed:
            raise AssertionError('Full saved-rollout gradient/Adam comparison failed; inspect named report')
        del optimizer
    unchanged = file_sha256(args.iteration/'execution_evidence.tar.gz') == archive_sha
    if not all(collective.all_gather_object(unchanged)):
        raise AssertionError('Source archive changed')
    if rank == 0:
        report.update(status='passed',source_unchanged=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    actor.load_state_dict(initial); cache.close()
    dist.destroy_process_group(group); dist.destroy_process_group()


if __name__ == '__main__':
    main()
