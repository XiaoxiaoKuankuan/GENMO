"""AWR专属物理世界的创建、校准与可靠关闭。

每rank独立启动本分支GMT服务，用原确认协议和PhysX配置，核对实际GPU UUID。
每个采集/评价阶段新建世界；只管理当前子进程，退出校验冻结GMT和全部ACK。
部署时钟通过单请求真实生成/准备耗时重新校准，原DPPO延迟曲线不可复用。
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from gem.closedloop.dppo.vector_evaluation import IsolatedWorld, close_isolated_world
from gem.closedloop.dppo.vector_devices import bind_vector_device
from gem.closedloop.dppo.memory_execution import MemoryReceiveJournal, PrepaidMemoryBudget
from gem.closedloop.common.vector_collector import VectorWorldClient
from gem.closedloop.common.world_flow import backend_operation
from tools.eval.run_closedloop_baseline import Workers
from .collection import CompleteCollector
from .io import atomic_json, file_sha256


def open_world(config, path, distributed, *, num_envs):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    c = copy.deepcopy(config)
    environment = bind_vector_device(c, distributed.rank)
    c["runtime"].update(num_envs=num_envs, generation_batch=num_envs, asset_conversion_dir=str(path/"usd_assets"),
                        columnar_first_audit=True)
    config_path = path/"physical.yaml"
    config_path.write_text(yaml.safe_dump(c, allow_unicode=True))
    handle = IsolatedWorld()
    handle.num_envs = num_envs
    handle.workers = Workers(c, path)
    started = time.perf_counter()
    try:
        socket = Path(handle.workers.temp.name)/"awr.sock"
        client = handle.workers.start("gmt", [c["paths"]["isaac_python"], "-B",
            str(Path(c["paths"]["gmt_repo"])/"scripts/rsl_rl/serve_frozen_gmt_vector.py"),
            "--config", str(config_path), "--socket", str(socket), "--headless"],
            c["paths"]["gmt_repo"], socket, strip_distributed=True, environment=environment)
        actual = handle.workers.entries[0]["identity"]
        if actual.get("gpu_uuid") != str(torch.cuda.get_device_properties(distributed.device).uuid):
            raise ValueError("AWR physical world is on a different GPU")
        handle.journal = MemoryReceiveJournal(path/"world_receipts")
        handle.world = VectorWorldClient(client, handle.journal, socket_path=socket)
        handle.start_seconds = time.perf_counter()-started
        return handle
    except BaseException as error:
        try:
            close_isolated_world(handle)
        except BaseException as shutdown_error:
            error.add_note(f"AWR world shutdown also failed: {shutdown_error!r}; see {path}/gmt_worker.log")
        raise


def budget_for_tasks(tasks, slots):
    # 包含预热及未使用槽物理子步的有限上界；实际各项继续逐控制回执计量。
    controls = sum(math_ceil_controls(t["remaining_music_seconds"]+2.) for t in tasks)
    return PrepaidMemoryBudget(dict(control_steps=max(1000, controls*slots),
                                   physics_steps=max(4000, controls*slots*4),
                                   generations=max(100, controls)))


def math_ceil_controls(seconds):
    import math
    return math.ceil(seconds*50)


def calibrate(config, actor_policy, builder, catalog, tasks, path, distributed, fingerprint):
    c = copy.deepcopy(config)
    c["runtime"].update(timing_contract="deployment_calibration.no_execution.v1",
                        phase_contract="single_request_calibration.v1")
    local_path = Path(path)/f"rank{distributed.rank:02d}"
    handle = open_world(c, local_path/"world", distributed, num_envs=1)
    budget = budget_for_tasks(tasks[:1], 1)
    driver = CompleteCollector(c, actor_policy, builder, handle.world, budget, local_path/"calibration",
                              num_envs=1, actor_fingerprint=fingerprint)
    measured = []
    def flow():
        env = driver.collector.states[0].resource.env
        task = tasks[0]
        yield from env.reset_task_flow(task["sample"], catalog.load_music(task["sample"]),
                                      seed=task["seed"], phase="evaluation", music_start_frame=0)
        for i in range(c["awr"]["calibration_warmup"]+c["awr"]["calibration_samples"]):
            torch.cuda.synchronize(distributed.device)
            result = yield from env.generate_flow()
            torch.cuda.synchronize(distributed.device)
            if result["prepared"] is None:
                raise ValueError("Stage1 calibration reference was rejected")
            if i >= c["awr"]["calibration_warmup"]:
                measured.append(result["critical_ready_seconds"])
            yield from backend_operation("discard_plan", prepared_plan_id=result["prepared"]["prepared_plan_id"])
            env.decision += 1
        yield from backend_operation("retire")
    try:
        with torch.no_grad():
            driver.collector._drive({0: flow()})
    finally:
        driver.close(); close_isolated_world(handle)
    all_samples = [x for rank in distributed.gather(measured) for x in rank]
    evidence = dict(schema="genmo.awr.deployment_calibration.v1", samples_seconds=all_samples,
                    actor_fingerprint=fingerprint, sampler=actor_policy.kernel_config,
                    source_commits=c["awr"]["source_commits"], assets=c["assets"],
                    scope="single_request_condition_generation_prepare_wallclock_excluding_audit")
    evidence_path = Path(path)/"deployment_calibration.json"
    def publish():
        atomic_json(evidence_path, evidence)
        return dict(schema="genmo.deployment_delay_profile.v1", target="single_robot_single_request",
                    samples_seconds=all_samples,
                    prefix_budget_seconds=float(np.quantile(all_samples, .99))+c["timing"]["latency_guard_s"],
                    provenance=[dict(path=str(evidence_path), sha256=file_sha256(evidence_path))],
                    sampler=actor_policy.kernel_config, measured_assets=c["assets"])
    return distributed.root(publish)
