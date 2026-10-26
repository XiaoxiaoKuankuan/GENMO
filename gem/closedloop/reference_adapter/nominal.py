"""冻结GENMO名义缓存的导入、导出与现有生成流程采集。

缓存是原始GENMO世界qpos28及关节名称/时钟/音乐/初态/种子来源，不接受执行状态冒充
名义参考。准备入口复用已有Stage1生成、调度和SM真实采集，显式禁用ForecastContext，
所有GENMO参数冻结；汇集真正接管的原计划，保持过去源点不变，不把Adapter写回前缀。
准备本身需要物理环境，只在显式调用时运行；训练直接消费固定缓存以严格配对。
"""

from pathlib import Path
import json

import torch

from supermimic.bridge.bundle import atomic_save, file_digest
from .reference import make_nominal

NOMINAL_SCHEMA = "genmo.reference_adapter.nominal.v1"


def required_support_tick(music_end_tick):
    """最后合法控制时刻仍需10帧+右halo；不要求执行越过音乐尾部。"""
    return (music_end_tick//12)*12 + 132


def check_support(nominal, music_end_tick):
    required = required_support_tick(music_end_tick)
    if nominal.upper < required:
        raise ValueError(f"真实GENMO名义尾部不足: upper={nominal.upper}, required={required}; 重新生成缓存，禁止外推")
    return dict(music_end_tick=music_end_tick, required_support_tick=required, actual_support_tick=nominal.upper)


def load_nominal(path, bundle, *, device, music_path, initial_path, seed, music_start_tick=0):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != NOMINAL_SCHEMA or payload.get("source") != "frozen_genmo_accepted_reference":
        raise ValueError("必须使用有来源的冻结GENMO名义缓存，不能以执行状态/动作标签代替")
    expected = dict(music_sha256=file_digest(music_path), initial_sha256=file_digest(initial_path), seed=seed,
                    music_start_tick=music_start_tick)
    if any(payload["contract"].get(k) != v for k, v in expected.items()):
        raise ValueError("缓存与音乐/初态/噪声种子不匹配")
    if payload["contract"].get("forecast_enabled") is not False:
        raise ValueError("首版缓存要求关闭未训练的GENMO在线Forecast分支")
    nominal = make_nominal(payload["world_qpos"].to(device), payload["joint_names"], bundle.spec,
                           start_tick=payload["start_tick"], contact=payload.get("contact"))
    return nominal, payload["contract"] | {"nominal_sha256": file_digest(path)}


def load_frozen_genmo(config, device):
    from omegaconf import OmegaConf
    from gem.closedloop.training import build_stage1_actor
    from gem.closedloop.checkpoint import load_stage1_checkpoint
    from gem.closedloop.mb_control.actor import ForecastActor
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    cfg.endecoder.stats_path = str(Path(config["stats"]).resolve())
    cfg.endecoder.kinematics_path = str(Path(config["kinematics"]).resolve())
    data = OmegaConf.create(dict(datasets={}, sample_contract=dict(history_steps=50),
                                qpos30_stats=dict(path=cfg.endecoder.stats_path),
                                dataset_defaults=dict(kinematics_path=cfg.endecoder.kinematics_path)))
    actor = build_stage1_actor(cfg, data)
    load_stage1_checkpoint(actor, checkpoint)
    actor = ForecastActor(actor, activation_checkpointing=False).to(device).eval().requires_grad_(False)
    return actor


@torch.no_grad()
def prepare_cache(env, bundle, genmo_cfg, task, music, initial, destination, *, bootstrap_latency_ticks):
    from gem.closedloop.mb_control.sm_adapter import SMAdapter
    from gem.closedloop.mb_control.scheduler import Scheduler
    from gem.closedloop.mb_control.collector import PhysicalCollector as GenmoCollector
    actor = load_frozen_genmo(genmo_cfg, initial.device)
    adapter = SMAdapter(bundle, actor.stage1.endecoder.kinematics.joint_order, str(initial.device))
    scheduler = Scheduler(bootstrap_latency_ticks=bootstrap_latency_ticks,
                          latency_margin_ticks=int(genmo_cfg.get("latency_margin_ticks", 0)))
    collector = GenmoCollector(env, actor, adapter, scheduler, adapter.runtime.reward,
                              forecast_enabled=False, record_trajectory=True)
    initial_z = collector.reset(music, initial, music_start_tick=task.get("music_start_tick", 0))
    archive = initial_z.reference
    generation_cursor = 0
    reason = "not_started"
    try:
        while not collector.terminal:
            intervals = collector.collect(1)
            reason = intervals[-1].reason
            for key in sorted(collector.generations):
                if key < generation_cursor:
                    continue
                generation = collector.generations[key]
                if generation.accepted:
                    # 使用真实接管时刻的保护规则，archive保留完整过去，独立于运行时trim。
                    protection = int(archive.dependency_support(generation.commit_tick-12).max()) if generation.commit_tick > generation.request.ticket.request_tick else generation.commit_tick-20
                    archive = archive.merge_plan(generation.plan, generation.commit_tick, protection)
                generation_cursor = key+1
    finally:
        collector.close()
    end = task.get("music_start_tick", 0)+(music.shape[1]-1)*20
    diagnostic = dict(task=task["id"], reason=reason, actual_control_end_tick=collector.state.tick,
                      music_end_tick=end, actual_support_tick=archive.upper,
                      required_support_tick=required_support_tick(end), events=collector.events,
                      cancelled_requests=collector.cancelled_requests, control_steps=len(collector.metrics),
                      average_sm_terms={k: sum(m[k] for m in collector.metrics)/len(collector.metrics)
                                        for k in collector.metrics[0]} if collector.metrics else {},
                      evidence="real_frozen_genmo_cache_preparation_not_adapter_training")
    # 即便尾部不足也保留真实失败轨迹，不能只留异常字符串或用成功片段覆盖失败。
    with Path(destination).with_suffix(".diagnostics.json").open("x") as stream:
        json.dump(diagnostic, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if collector.trajectory:
        atomic_save(dict(origin="real", task=task, trajectory=collector.trajectory,
                         metrics=collector.metrics, diagnostic=diagnostic), Path(destination).with_suffix(".execution.pt"))
    support = check_support(archive, end)
    # 保留真正已生成且接管的源点，包括末尾超过音乐结束的导数/Goal支持。
    # 不插值外推不存在的GENMO点，不以物理早结束放宽所声明音乐长度。
    keep = archive.ticks <= ((required_support_tick(end)+19)//20)*20
    if int(keep.sum()) < 8:
        raise ValueError("实际GENMO缓存过短，不创建可训练替代动作")
    contract = dict(music_sha256=file_digest(task["music_features"]), initial_sha256=file_digest(task["initial_state"]),
                    genmo_checkpoint_sha256=file_digest(genmo_cfg["checkpoint"]), seed=int(task["seed"]),
                    music_start_tick=task.get("music_start_tick", 0),
                    forecast_enabled=False, generation_reason=reason, actual_control_end_tick=collector.state.tick, **support,
                    generation_scope="real_frozen_genmo_accepted_plans_without_adapter", bootstrap_latency_ticks=bootstrap_latency_ticks,
                    latency_margin_ticks=scheduler.latency_margin_ticks)
    atomic_save(dict(schema=NOMINAL_SCHEMA, source="frozen_genmo_accepted_reference", contract=contract,
                     world_qpos=archive.world_qpos[:, keep].cpu(), contact=archive.contact[:, keep].cpu(),
                     start_tick=int(archive.ticks[0]), joint_names=tuple(actor.stage1.endecoder.kinematics.joint_order),
                     source_versions=archive.source_versions[keep].cpu(), events=collector.events), destination)
    return contract
