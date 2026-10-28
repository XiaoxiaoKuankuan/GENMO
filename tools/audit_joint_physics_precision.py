"""正式联合训练启动前的单卡真实数据BF16数值核验。

从配置中的train离线真实请求选择四个不同音乐来源，复用原数据、真实历史和合法预测
FutureContext；静态构造同一随机初始化1024维16层GENMO，绝不加载GENMO checkpoint。
固定k=[50,100,250,999]及epsilon，关闭随机dropout，分别计算FP32和bf16_mixed去噪，
并在原Stage1 v5辅助项全部warmup完成的位置核验完整监督Loss及代表参数梯度。
这里没有optimizer、参数更新、Newton仿真或WM rollout；加载冻结SM/WM仅为核对sidecar
资产和复用既有参考/历史装配。结果不表示训练质量、真实物理收益或整曲可执行性。
网络可用BF16，x0、contact logits、codec/FK/Loss、参数和梯度必须仍为FP32。
输出目录独占新建；失败保存原因并非零退出，不覆盖已有报告，不放宽阈值以制造通过。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gem.closedloop.joint_physics.config import build_actor, build_adapter, read_config, seed_all
from gem.closedloop.joint_physics.contracts import digest
from gem.closedloop.joint_physics.data import SidecarStore, SourceCatalog


THRESHOLDS = {"output_relative_rmse_max": .03, "loss_relative_error_max": .03,
              "nonzero_gradient_cosine_min": .95, "output_rms_scale_floor": .1}


def _representative_parameters(actor):
    """含主干入口/首块/输出以及各条件末层；零末层上游的首步零梯度单独记录。"""
    named = dict(actor.named_parameters())
    selected = ["stage1.denoiser.add_cond_linear.weight",
                "stage1.denoiser.final_layer.fc2.weight",
                "stage1.history_encoder.out_proj.weight",
                "stage1.prefix_encoder.out_proj.weight",
                "future_encoder.output.weight", "future_encoder.input.weight",
                "stage1.music_embedder.fc1.weight"]
    selected.append(next(name for name, p in named.items()
                         if name.startswith("stage1.denoiser.blocks.0.") and p.ndim == 2))
    condition_output = "stage1.cond_exists_embedder.2.weight"
    if condition_output in named:
        selected.append(condition_output)
    return {name: named[name] for name in selected}


def _evaluate(actor, losses, batch, contexts, timesteps, noise, parameters, precision, loss_step):
    actor.precision = precision
    actor.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = actor.denoise_supervised(batch, contexts, timesteps, noise,
                                   torch.zeros(len(noise), dtype=torch.bool, device=noise.device))
    with torch.autocast(device_type="cuda", enabled=False):
        total, components = losses(batch, out["pred_x_start"], out["static_conf_logits"], loss_step)
    gradients = torch.autograd.grad(total, list(parameters.values()), allow_unused=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    values = {name: out[name].detach().cpu() for name in
              ("pred_x_start", "static_conf_logits", "target_x_start", "xt", "noise")}
    finite = (all(bool(x.isfinite().all()) for x in values.values()) and bool(total.isfinite())
              and all(bool(value.isfinite().all()) for value in components.values()))
    grad_values = {}
    for (name, parameter), gradient in zip(parameters.items(), gradients):
        if gradient is None:
            raise RuntimeError(f"代表参数没有进入监督图: {name}")
        finite = finite and bool(gradient.isfinite().all())
        if gradient.dtype != torch.float32 or parameter.dtype != torch.float32:
            raise RuntimeError(f"代表参数/梯度不是FP32: {name}")
        grad_values[name] = gradient.detach().float().cpu()
    if not finite or any(value.dtype != torch.float32 for value in values.values()):
        raise RuntimeError(f"{precision}出现非有限数值或非FP32物理输出")
    report = {"loss": float(total.detach()), "seconds": seconds,
              "components": {name: float(value.detach()) for name, value in components.items()},
              "output_dtypes": {name: str(value.dtype) for name, value in values.items()},
              "all_selected_gradients_finite_fp32": True}
    return report, values, grad_values


def run(args):
    cfg = read_config(args.config)
    if cfg["initialization"].get("resume_checkpoint") is not None:
        raise ValueError("数值检查只允许静态随机初始化，不能带resume_checkpoint")
    if cfg["stage1"]["model"]["latent_dim"] != 1024 or cfg["stage1"]["model"]["num_layers"] != 16:
        raise ValueError("必须检查正式1024维16层Transformer")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("数值检查要求支持BF16的目标CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(int(cfg["training"]["seed"]))
    actor, losses = build_actor(cfg, device)
    if any(p.dtype != torch.float32 for p in actor.parameters()):
        raise RuntimeError("GENMO参数必须为FP32")
    adapter = build_adapter(cfg, device)
    frozen = adapter.bundle.frozen_snapshot()
    # build_adapter先注册只读SM源码路径，之后才能导入其资产合同及请求装配。
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    from gem.closedloop.joint_physics.reference import attach_sidecars
    from gem.closedloop.joint_physics.requests import PhysicsRequestStream
    assets = asset_contract(cfg, adapter)
    catalog = SourceCatalog(cfg, "train")
    store = SidecarStore(cfg["data"]["sidecar"], "train", assets,
                         sample_contract=cfg["data"]["sample_contract"])
    requests = PhysicsRequestStream(catalog, store, adapter.bundle.spec.joint_names, 1,
                                    int(cfg["training"]["seed"]), historical_mode="diagnostic")
    if len(requests.keys) < 4:
        raise RuntimeError("真实BF16检查要求至少四个具有合法离线请求的音乐来源")
    batch, metadata = requests.next(4, device)
    batch, contexts, _, reasons = attach_sidecars(batch, store, adapter,
                                                 actor.stage1.endecoder.codec, 1)
    if any(reason != "available" for reason in reasons) or not all(bool(c.valid.all()) for c in contexts):
        raise RuntimeError(f"请求必须具有当时合法可用的FutureContext: {reasons}")
    timesteps = torch.tensor([50, 100, 250, 999], dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]) + 9031)
    noise = torch.randn(4, 60, 30, generator=generator).to(device)
    # 这是完整v5数值检查的Loss位置，不修改模型/课程global_step，也没有训练更新。
    options = cfg["stage1"]["loss"].get("options", {})
    loss_step = max(int(options.get("auxiliary_warmup_steps", 0)),
                    int(options.get("robust_joint_limit_start_step", 0)) + int(options.get("robust_joint_limit_warmup_steps", 0)),
                    int(options.get("advanced_physics_start_step", 0)) + int(options.get("advanced_physics_warmup_steps", 0)))
    parameters = _representative_parameters(actor)
    reference, fp, fp_grad = _evaluate(actor, losses, batch, contexts, timesteps, noise, parameters, "fp32", loss_step)
    mixed, bf, bf_grad = _evaluate(actor, losses, batch, contexts, timesteps, noise, parameters, "bf16_mixed", loss_step)
    output_errors = {}
    for name, mask in (("pred_x_start", (batch["target_qpos30_valid"] & ~batch["known_qpos30_mask"]).cpu()),
                       ("static_conf_logits", batch["future_valid"].cpu()[..., None].expand(-1, -1, 2))):
        expected, actual = fp[name][mask].double(), bf[name][mask].double()
        rmse = float((expected - actual).square().mean().sqrt())
        rms = float(expected.square().mean().sqrt())
        output_errors[name] = {"rmse": rmse, "reference_rms": rms,
            "relative_rmse": rmse / max(rms, THRESHOLDS["output_rms_scale_floor"])}
    gradient_comparison = {}
    for name in parameters:
        a, b = fp_grad[name].double().flatten(), bf_grad[name].double().flatten()
        an, bn = float(a.norm()), float(b.norm())
        both_zero = an == 0. and bn == 0.
        cosine = None if both_zero else float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
        gradient_comparison[name] = {"fp32_norm": an, "bf16_norm": bn, "cosine": cosine,
            "both_exact_zero_at_initialization": both_zero,
            "relative_difference": float((a - b).norm()) / max(an, 1e-12)}
    known = batch["known_qpos30_mask"].cpu()
    # sidecar条件是请求时已经提交的旧计划，不能把GT目标当作该承诺的精确值。
    # 两种精度都必须逐坐标严格保持真实known_x；条件与GT的差异只记录，不作精度门禁。
    with torch.autocast(device_type="cuda", enabled=False):
        known_condition = actor.stage1.adapt_conditions(batch)["known_x"].detach().cpu()[known]
    prefix_report = {"reference": "normalized_known_condition_from_committed_plan_not_target",
                     "known_coordinates": int(known.sum())}
    for precision, values in (("fp32", fp), ("bf16_mixed", bf)):
        actual = values["pred_x_start"][known]
        error = actual.double() - known_condition.double()
        prefix_report[precision] = {"exact": torch.equal(actual, known_condition),
                                    "max_absolute_error": float(error.abs().max()),
                                    "rmse": float(error.square().mean().sqrt())}
    condition_target_error = known_condition.double() - fp["target_x_start"][known].double()
    prefix_report["condition_vs_ground_truth"] = {
        "exact": torch.equal(known_condition, fp["target_x_start"][known]),
        "max_absolute_error": float(condition_target_error.abs().max()),
        "rmse": float(condition_target_error.square().mean().sqrt()),
        "role": "diagnostic_only_committed_prefix_and_ground_truth_have_distinct_contracts"}
    exact_prefix = prefix_report["fp32"]["exact"] and prefix_report["bf16_mixed"]["exact"]
    exact_noise = all(torch.equal(fp[name], bf[name]) for name in ("noise", "xt", "target_x_start"))
    loss_relative = abs(mixed["loss"] - reference["loss"]) / max(abs(reference["loss"]), 1e-8)
    nonzero_groups = [value for value in gradient_comparison.values() if not value["both_exact_zero_at_initialization"]]
    checks = {"output_error": all(value["relative_rmse"] <= THRESHOLDS["output_relative_rmse_max"] for value in output_errors.values()),
              "full_v5_loss_error": loss_relative <= THRESHOLDS["loss_relative_error_max"],
              "nonzero_gradient_cosine": bool(nonzero_groups) and all(value["cosine"] >= THRESHOLDS["nonzero_gradient_cosine_min"] for value in nonzero_groups),
              "future_projection_nonzero_gradient": fp_grad["future_encoder.output.weight"].norm().item() > 0 and bf_grad["future_encoder.output.weight"].norm().item() > 0,
              "known_prefix_exact": exact_prefix, "noise_target_xt_exact": exact_noise,
              "all_parameters_fp32": all(p.dtype == torch.float32 for p in actor.parameters()),
              "all_outputs_and_selected_gradients_finite_fp32": True}
    adapter.bundle.assert_unchanged(frozen)
    return {"schema": "genmo.joint_precision_audit.v1", "passed": all(checks.values()),
            "scope": "random_model_real_training_samples_numeric_only_no_newton_no_optimizer",
            "config": str(Path(args.config).resolve()), "config_sha256": digest(args.config),
            "initialization": actor.initialization, "architecture": {"latent_dim": 1024, "num_layers": 16},
            "device": torch.cuda.get_device_name(device), "physical_device_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "assets": assets, "source_requests": metadata, "timesteps": timesteps.cpu().tolist(),
            "forecast_valid": [bool(c.valid.all()) for c in contexts], "loss_global_step_for_full_warmup": loss_step,
            "noise_sha256": hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
            "thresholds": THRESHOLDS, "fp32": reference, "bf16_mixed": mixed,
            "output_errors": output_errors, "loss_relative_error": loss_relative,
            "gradient_comparison": gradient_comparison, "known_prefix": prefix_report, "checks": checks,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="独占新建的数值检查报告目录")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args)
    except Exception as error:
        report = {"passed": False, "error": str(error), "traceback": traceback.format_exc()}
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        raise
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "report": str(output / "result.json")}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
