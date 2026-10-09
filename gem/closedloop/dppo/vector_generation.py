"""真实多环境的批量生成外围：集中条件编码、传输、坐标转换和随机链回传。

环境线程仅保留自己预算、参考事务与持久化日志的所有权，提交不可变快照和前缀。
协调线程把真正不同环境的条件合批，一次传入GPU、执行20步DPPO扩散与合批CFG，
一次批量转换世界坐标，再按dtype合并完整随机链和动作的GPU到CPU传输。原先每个
环境分别同步GPU、复制20步链和转换输出的路径不再进入新协议。各环境仍使用原
独立噪声种子，完整old概率/均值/标准差/前缀及输出证据全部保留。

不跨线程操作SQLite或预算；准备参考和落盘仍由所属线程完成，证据可靠保存后才
允许物理推进。单个环境/episode因果顺序不变。批次仅消除重复数值工作，不把填充
行计为真实样本，也不把共享批次耗时多次加成GPU时间。部署时钟独立建模。
"""
from __future__ import annotations

from concurrent.futures import Future
import time
import torch
from .dual_collector import split_trace
from .env_adapter import cpu_copy
from gem.closedloop.frozen_actor import stable_noise_seed
from gem.runtime.closedloop_protocol import RemoteError

GENERATION_CONTRACT = 'genmo.vector_generation_pipeline.v1'


def bulk_cpu_copy(tree):
    """每个device/dtype只传输一个连续块；返回只读用途的CPU张量视图。"""
    groups, converted = {}, {}
    def visit(value):
        if torch.is_tensor(value):
            if id(value) not in converted:
                converted[id(value)] = None
                groups.setdefault((value.device, value.dtype), []).append(value)
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value: visit(item)
    visit(tree)
    for _, values in groups.items():
        host = torch.cat([v.detach().reshape(-1) for v in values]).cpu()
        offset = 0
        for value in values:
            size = value.numel()
            converted[id(value)] = host[offset:offset+size].reshape(value.shape)
            offset += size
    def rebuild(value):
        if torch.is_tensor(value): return converted[id(value)]
        if isinstance(value, dict): return {k:rebuild(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return type(value)(rebuild(v) for v in value)
        return value
    return rebuild(tree)


@torch.no_grad()
def generate_batch(policy, packets):
    started = time.perf_counter()
    builder = packets[0]['builder']
    contexts, metadata = builder.build_many(
        [p['snapshot'] for p in packets], [p['reservation'] for p in packets],
        [p['music'] for p in packets])
    conditions = {key:torch.cat([c[key] for c in contexts]) for key in contexts[0]}
    condition_end = time.perf_counter()
    device = next(policy.actor.parameters()).device
    device_conditions = {k:v.to(device) for k,v in conditions.items()}
    generators = [torch.Generator(device=device).manual_seed(p['seed']) for p in packets]
    trace = policy.sample_rollout(device_conditions, generator=generators)
    anchors = torch.tensor([m['world_anchor'] for m in metadata], device=device, dtype=torch.float32)
    world = policy.actor.endecoder.codec.apply_world_anchor(trace['qpos'], anchors)
    finite = torch.stack([torch.isfinite(v).all() for v in (world,trace['qpos30'],trace['contact'])]).all()
    if not bool(finite): raise FloatingPointError('Nonfinite vector generated outputs')
    generated_end = time.perf_counter()
    copied = bulk_cpu_copy(dict(trace=trace, qpos_world=world))
    finished = time.perf_counter()
    outputs = []
    for i,p in enumerate(packets):
        meta = metadata[i]
        meta.update(seed=p['seed'], decision_id=p['decision'], plan_id=f"{p['reservation']['request_id']}:plan")
        # 此处clone只发生在CPU，避免单行torch.save意外保存整批底层storage。
        local_trace = cpu_copy(split_trace(copied['trace'], i, len(packets)))
        generated = dict(meta, qpos_world=copied['qpos_world'][i].numpy().copy(),
                         qpos30=local_trace['qpos30'][0].numpy().copy(),
                         contact=local_trace['contact'][0].numpy().copy())
        outputs.append(dict(context=contexts[i], meta=meta, generated=generated, trace=local_trace))
    return outputs, dict(condition_batch_seconds=condition_end-started,
        generate_and_world_seconds=generated_end-condition_end, bulk_trace_transfer_seconds=finished-generated_end,
        cpu_split_seconds=time.perf_counter()-finished, components=dict(policy.last_sample_timing),
        generation_contract=GENERATION_CONTRACT, effective_rows=len(packets))


def generate_for_environment(env):
    """只在环境所属线程进行参考请求、预算预占、journal和独立原链保存。"""
    env.budget.reserve(env.phase, generations=1)
    started = time.perf_counter()
    request = env._request()
    reservation = env.backend.call('reserve_prefix', request=request)
    journal_seconds = env.backend.last_call_timing.get('journal_seconds', 0.)
    prefix_end = time.perf_counter()
    env.attempt += 1
    key = f"{env.config['stage9']['run_id']}:{env.iteration}:{env.episode_count}:{env.decision}:{env.attempt}"
    if env.rank is not None: key += f':rank:{env.rank}'
    key += f':environment:{env.collector_env_slot}'
    seed = stable_noise_seed(env.config['stage9']['seed'], key)
    if env.comparison_noise_index is not None:
        seed = stable_noise_seed(1729, str(env.comparison_noise_index))
        env.comparison_noise_index += 1
    packet = dict(_vector_generation=True, builder=env.builder, snapshot=env.snapshot,
        reservation=reservation, music=env.music, seed=seed, decision=env.decision)
    owner = env.policy.owner
    future = Future()
    owner.requests.put((env.collector_env_slot, packet, None, future, time.perf_counter()))
    result, shared_timing = future.result(timeout=owner.timeout_seconds)
    sample_ready = time.perf_counter()
    rejection, prepared = None, None
    try:
        prepared = env.backend.call('prepare_plan', generated_plan=result['generated'])
    except RemoteError as exc:
        if exc.code not in {'invalid_qpos','invalid_quaternion','invalid_plan_output','known_source_changed'}: raise
        rejection = dict(code=exc.code, message=str(exc), policy_penalty=True, category='finite_invalid_reference')
    journal_seconds += env.backend.last_call_timing.get('journal_seconds', 0.)
    prepared_at = time.perf_counter()
    path = env.output/'raw_samples'/f'{env.phase}_{env.attempt:06d}.pt'
    identity = env._save_evidence(dict(trace=result['trace'], generated=result['generated'],
        policy_version=env.policy_version), path)
    elapsed = time.perf_counter()-started
    timing = dict(timing_contract=env.timing_contract, generation_contract=GENERATION_CONTRACT,
        journal_seconds=journal_seconds, prefix_rpc_seconds=prefix_end-started,
        batched_pipeline_wait_seconds=sample_ready-prefix_end, prepare_rpc_seconds=prepared_at-sample_ready,
        critical_ready_seconds=max(0., prepared_at-started-journal_seconds), total_wall_seconds=elapsed,
        raw_evidence_seconds=time.perf_counter()-prepared_at, trace_copy_seconds=0.,
        raw_evidence_identity=identity, shared_batch_timing=shared_timing,
        excluded_audit_seconds=journal_seconds+time.perf_counter()-prepared_at)
    result.update(prepared=prepared, rejection=rejection, timing=timing, seed=seed,
        critical_ready_seconds=timing['critical_ready_seconds'], elapsed=elapsed, raw_path=str(path))
    return result
