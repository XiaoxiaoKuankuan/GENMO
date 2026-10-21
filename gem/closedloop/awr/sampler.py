"""完整Stage1 DDIM采样与世界批量生成接口。

同一Actor负责全部20个确定性DDIM步骤，随机性只来自逐任务独立初始高斯噪声。
直接复用Stage1.sample的CFG、已知前缀和解码行为，不重写采样公式、不保存概率链。
世界接口批量构建真实条件，转换Root Anchor并提供提交所需输出；最小trace只用于
既有物理事务外围的条件/采样器身份传递，不包含PPO概率或中间去噪轨迹。
"""
from __future__ import annotations

import time
import torch

from gem.closedloop.common.vector_generation import bulk_cpu_copy
from gem.closedloop.common.tracking_context import build_critic_context_batch
from gem.closedloop.common.joint_limits import generated_boundaries_tensors, generated_boundary_reports


class AWRSampler:
    def __init__(self, actor, *, steps=20, guidance_scale=2.5):
        if steps != 20 or guidance_scale != 2.5:
            raise ValueError("AWR v1 sampler is Stage1 DDIM20 eta=0 CFG=2.5")
        self.actor, self.steps, self.guidance_scale = actor, steps, guidance_scale
        self.last_sample_timing = {}
        self.kernel_config = dict(schema="genmo.awr.stage1_ddim.v1", steps=20, eta=0.,
                                  guidance_scale=2.5, noise="initial_gaussian",
                                  actor="same_network_all_steps")

    def _parameter_signature(self):
        return tuple((id(p), p._version) for p in self.actor.parameters())

    @torch.no_grad()
    def sample(self, conditions, *, seeds=None, noise=None):
        if self.actor.training:
            raise RuntimeError("AWR behavior Actor must be in eval mode")
        if noise is not None and seeds is not None:
            raise ValueError("Choose explicit initial noise or seeds")
        device = next(self.actor.parameters()).device
        c = {k: v.to(device) for k, v in conditions.items()}
        if seeds is not None:
            if len(seeds) != c["known_qpos30"].shape[0]:
                raise ValueError("One initial seed per actual request")
            noise = torch.stack([
                torch.randn(c["known_qpos30"].shape[1:], device=device,
                            generator=torch.Generator(device=device).manual_seed(int(seed)))
                for seed in seeds
            ])
        started = time.perf_counter()
        result = self.actor.sample(c, steps=self.steps, guidance_scale=self.guidance_scale, noise=noise)
        self.last_sample_timing = dict(sample_seconds=time.perf_counter() - started)
        return result


@torch.no_grad()
def generate_batch(policy, packets, *, shared_storage=False, device_trace=False):
    started = time.perf_counter()
    contexts, metadata = packets[0]["builder"].build_many(
        [p["snapshot"] for p in packets], [p["reservation"] for p in packets],
        [p["music"] for p in packets])
    conditions = {k: torch.cat([c[k] for c in contexts]) for k in contexts[0]}
    from .candidates import initial_state
    initial_states = [initial_state(p, contexts[i], metadata[i])
                      if p.get("candidate_task") and p["candidate_decision_index"] == 0 else None
                      for i, p in enumerate(packets)]
    condition_end = time.perf_counter()
    result = policy.sample(conditions, seeds=[p["seed"] for p in packets])
    device = result["qpos"].device
    anchors = torch.tensor([m["world_anchor"] for m in metadata], device=device, dtype=torch.float32)
    world = policy.actor.endecoder.codec.apply_world_anchor(result["qpos"], anchors)
    free = conditions["future_valid"][..., None] & ~conditions["known_qpos30_mask"]
    boundary = generated_boundaries_tensors(
        world, free[:, :, 9:].to(device), [p["snapshot"] for p in packets],
        policy.actor.endecoder.kinematics.joint_order, check_finite=True)
    critical = bulk_cpu_copy(dict(qpos_world=world, qpos30=result["qpos30"],
                                 contact=result["contact"], boundary=boundary))
    reports = generated_boundary_reports(critical["boundary"])
    critic = build_critic_context_batch(conditions, [p["snapshot"] for p in packets])
    outputs = []
    for index, packet in enumerate(packets):
        meta = dict(metadata[index], seed=packet["seed"], decision_id=packet["decision"],
                    plan_id=f"{packet['reservation']['request_id']}:plan")
        if "task_identity" in packet:
            meta["task_identity"] = packet["task_identity"]
        generated = dict(meta, qpos_world=critical["qpos_world"][index].numpy().copy(),
                         qpos30=critical["qpos30"][index].numpy().copy(),
                         contact=critical["contact"][index].numpy().copy(),
                         joint_limit_diagnostics=reports[index], awr_reference_audit=True)
        trace = dict(conditions=contexts[index], free_mask=free[index:index+1],
                     kernel_config=policy.kernel_config)
        outputs.append(dict(context=contexts[index], candidate_initial_state=initial_states[index],
                            critic_context={k: v[index:index+1] for k, v in critic.items()},
                            meta=meta, generated=generated, trace=trace))
    if shared_storage:
        block = dict(schema="genmo.awr.generated_batch.v1", generated=[v["generated"] for v in outputs])
        for index, output in enumerate(outputs):
            output["_shared_raw_block"] = (block, index)
    elapsed = time.perf_counter() - condition_end
    return outputs, dict(condition_batch_seconds=condition_end-started,
                         generate_and_world_seconds=elapsed, bulk_trace_transfer_seconds=0.,
                         critical_output_transfer_seconds=0., cpu_split_seconds=0.,
                         effective_rows=len(packets), generation_contract="genmo.awr.stage1_ddim.v1",
                         components=policy.last_sample_timing)
