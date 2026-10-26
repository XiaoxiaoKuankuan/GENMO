"""冻结音乐教师的真实资产及真实接管梯度验收。

使用原s595000、同一SM checkpoint和Newton真实采集请求，核验同请求教师/学生
初始一致、教师存储独立、已知前缀零梯度、20步DDIM主干和未来分支的各损失梯度。
只在独立审计进程进行一次优化，报告参数/输出变化并保存原始请求，不把该更新
或模型目标变化称为物理舞蹈收益。H=32的物理梯度方向另由short_audit验证。
"""

from dataclasses import replace
import json
from pathlib import Path
import torch

from .application import task_inputs, seed_all, SupervisionStream, build_stage1_losses
from .checkpoint import pack_value
from .dance_losses import reference_prior,reference_objectives,consumed_music_ticks,consumed_music_support
from .trainer import Stage2Trainer


def audit_teacher(collector, actor, critic, engine, original, cfg, tasks, output, device):
    output = Path(output)
    teacher = engine.teacher
    seed_all(cfg["training"]["seed"])
    music, initial = task_inputs(tasks[0], collector.adapter, device)
    collector.anchor_horizon = engine.horizon_steps
    collector.record_controls = collector.record_contacts = True
    collector.reset(music, initial)
    controls = []
    while len(collector.ready_anchors) < 3 and not collector.terminal:
        collector.collect(1)
        controls.extend(collector.take_controls())
    samples = list(collector.ready_anchors)[-2:]
    if not samples:
        raise RuntimeError("没有真实接管样本，不能执行教师梯度验收")
    sample = samples[-1]; record = sample.generation; request = record.request
    before = {n:v.detach().clone() for n,v in teacher.stage1.state_dict().items()}
    stores = {p.untyped_storage().data_ptr() for p in actor.parameters()}
    independent = not stores & {p.untyped_storage().data_ptr() for p in teacher.parameters()}
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone()
    local = teacher.local(request)
    teacher_rng_unchanged = torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    with torch.no_grad():
        candidate = actor.generate(request.snapshot, request.ticket, record.forecast, request.noise)
    parity = float((candidate.world_qpos-local.world_qpos).abs().max())
    variable = local.world_qpos.clone().requires_grad_(True)
    p = request.ticket.prefix_frames
    changed = variable.clone()
    changed[:,p:,7] = changed[:,p:,7]+.03
    prior, _ = reference_prior(replace(local,world_qpos=changed),local,engine.adapter)
    gradient, = torch.autograd.grad(prior, variable)
    known_gradient = float(gradient[:,:p].abs().max())
    mutable_gradient = float(gradient[:,p:].norm())
    global_known_gradient=global_mutable_gradient=None
    if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
        variable_global=local.world_qpos.clone().requires_grad_(True)
        changed_global=variable_global.clone();changed_global[:,p:,7]+= .03
        _,global_prior,_=reference_objectives(replace(local,world_qpos=changed_global),local,
            sample.state.teacher_reference,engine.adapter,anchor_state=sample.state.state,
            anchor_tick=sample.state.tick,scales=cfg["rewards"].get("dance_scales"),
            global_support_ticks=consumed_music_support(sample,engine.horizon_steps),
            global_control_ticks=consumed_music_ticks(sample,engine.horizon_steps),
            global_reference_records=sample.transitions[:engine.horizon_steps],global_reference_version=sample.generation.reference_version)
        global_gradient,=torch.autograd.grad(global_prior,variable_global)
        global_known_gradient=float(global_gradient[:,:p].abs().max())
        global_mutable_gradient=float(global_gradient[:,p:].norm())
    zero_prior, _ = reference_prior(local,local,engine.adapter)
    static = local.world_qpos.clone();static[:,p:] = static[:,p:p+1]
    static_prior, _ = reference_prior(replace(local,world_qpos=static),local,engine.adapter)
    identity = teacher.timeline(music,initial)
    memo = {}
    torch.save({"samples":pack_value(tuple(samples),memo),"controls":pack_value(tuple(controls),memo),
        "teacher_identity":teacher.identity}, output/"real_requests.pt")
    losses = build_stage1_losses(actor.stage1,original).to(device)
    trainer = Stage2Trainer(actor,critic,engine,losses,cfg["training"])
    supervision = SupervisionStream(cfg["stage1"]["data_config"],cfg["training"]["supervision_batch"],
        cfg["training"]["seed"],device)
    row = trainer.update_anchors(samples,supervision.next(),physical_intervals=controls)
    after = actor.generate(request.snapshot,request.ticket,record.forecast,request.noise)
    after_prior, _ = reference_prior(after,local,engine.adapter)
    prior_future = torch.autograd.grad(after_prior,actor.future_encoder.output.weight)[0]
    prior_future_norm = float(prior_future.norm())
    teacher.assert_frozen()
    with torch.no_grad():
        local_after = teacher.local(request)
    frozen = all(torch.equal(v,before[n]) for n,v in teacher.stage1.state_dict().items())
    fixed_timeline = teacher.timeline(music,initial) is identity
    unchanged_local = torch.equal(local_after.world_qpos,local.world_qpos)
    passed = (independent and teacher_rng_unchanged and parity <= 1e-5 and known_gradient == 0 and mutable_gradient > 0 and
        float(zero_prior) == 0 and float(static_prior) > 0 and frozen and fixed_timeline and unchanged_local and
        row["accepted_execution_feedback_update"] and row["future_encoder_gradient_norms"]["dance"] > 0 and
        prior_future_norm > 0 and row["future_encoder_stage1_bc_gradient"] == 0)
    if global_known_gradient is not None:
        passed=passed and global_known_gradient==0. and global_mutable_gradient>0.
    from .short_audit import teacher_target_contract
    report = {"method":"physical_anchor_short_bptt","horizon":engine.horizon_steps,"passed":bool(passed),
        "reward_contract":cfg["rewards"],"teacher_identity":teacher.identity,
        "teacher_target_contract":teacher_target_contract(cfg),
        "frozen_sha256":engine.adapter.bundle.metadata["sha256"],
        "zero_forecast_stage1_max_abs":parity,"teacher_storage_independent":independent,
        "teacher_preserves_student_rng":teacher_rng_unchanged,
        "after_update_prior_future_gradient_norm":prior_future_norm,
        "known_prefix_prior_gradient_max_abs":known_gradient,"mutable_prior_gradient_norm":mutable_gradient,
        "known_prefix_global_gradient_max_abs":global_known_gradient,"mutable_global_gradient_norm":global_mutable_gradient,
        "matched_prior":float(zero_prior),"static_candidate_prior":float(static_prior),
        "teacher_parameters_statistics_unchanged":frozen,"fixed_timeline_unchanged_after_actor_update":fixed_timeline,
        "same_request_teacher_unchanged_after_actor_update":unchanged_local,"actual_update":row,
        "frozen_parameters_and_buffers_unchanged":row["frozen_parameters_and_buffers_unchanged"],
        "physical_direction_evidence":False,"scope":"real_asset_real_anchor_differentiability_not_quality"}
    (output/"teacher_gradients.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({"event":"teacher_gradient_audit","passed":passed,"update":row},ensure_ascii=False),flush=True)
    if not passed:
        raise RuntimeError("冻结音乐教师或实际执行梯度验收失败，保留报告")
