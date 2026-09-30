"""同一批固定梯度上的有界学习率探测与可回滚优化器更新。

调用方已经完成 Actor 的 backward 和梯度裁剪。本模块最多尝试三个严格递增、
不超过 1e-6 的学习率；每次都恢复同一份初始模型参数、全部缓冲区、完整优化器
状态和固定梯度，再实际执行一次 optimizer.step。预算回调在每次实际尝试之前
调用，因此失败候选也能占用外部累计预算。联合 KL 由调用方在同一执行数据和
去噪链上计算，本模块不改变噪声、概率定义、奖励、GMT 或物理环境。

最终选择联合 KL 不超过门槛且参数确有改变的最大候选，只恢复该候选已经保存的
模型和优化器状态，不再多执行一步。全部候选拒绝、预算失败、诊断异常或任意
回调异常都会恢复初始状态并抛错；尝试过的外部预算不会回滚。只保留初始快照和
当前最大合格候选的 CPU 快照，不同时保存全部候选的大模型状态。

报告包含候选 KL、实际参数变化数量/比例/L2/最大值及顶层网络模块统计。ULP
诊断使用实际 FP32 参数差除以原权重沿变化方向的 nextafter 间距，明确属于已经
舍入的实际更新量，不能解释成未舍入的理论 Adam 步长。梯度、缓冲区、模型参数
与回调只读边界均做核验；统计只输出聚合数值，不输出完整权重或梯度数组。
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import math
from numbers import Real
import random

import numpy as np
import torch


class LearningRateCalibrationRejected(RuntimeError):
    """没有合格候选时保留结构化报告；模型和优化器已恢复到初始状态。"""

    def __init__(self, report):
        self.report = copy.deepcopy(report)
        super().__init__("No learning-rate candidate changed parameters within the joint KL limit")


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        result = copy.copy(value)
        result.clear()
        result.update((copy.deepcopy(key), _cpu_clone(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = copy.deepcopy(value._metadata)
        return result
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _model_snapshot(actor):
    buffers = {}
    modes = {}
    for module_name, module in actor.named_modules():
        modes[module_name] = bool(module.training)
        for name, value in module._buffers.items():
            buffers[(module_name, name)] = {
                "value": _cpu_clone(value), "device": None if value is None else value.device,
                "persistent": name not in module._non_persistent_buffers_set,
            }
    return {"state": _cpu_clone(actor.state_dict()), "buffers": buffers, "modes": modes}


def _restore_model(actor, snapshot):
    modules = dict(actor.named_modules())
    with torch.no_grad():
        for module_name, module in modules.items():
            for name in list(module._buffers):
                if (module_name, name) not in snapshot["buffers"]:
                    del module._buffers[name]
                    module._non_persistent_buffers_set.discard(name)
        for (module_name, name), saved in snapshot["buffers"].items():
            module, value = modules[module_name], saved["value"]
            current = module._buffers.get(name)
            if value is None:
                module._buffers[name] = None
            elif (isinstance(current, torch.Tensor) and current.shape == value.shape
                    and current.dtype == value.dtype and current.device == saved["device"]):
                current.copy_(value)
            else:
                module._buffers[name] = value.to(device=saved["device"]).clone()
            if saved["persistent"]:
                module._non_persistent_buffers_set.discard(name)
            else:
                module._non_persistent_buffers_set.add(name)
        actor.load_state_dict(snapshot["state"], strict=True)
        for module_name, training in snapshot["modes"].items():
            modules[module_name].training = training


def _restore_gradients(actor, gradients):
    for name, parameter in actor.named_parameters():
        value = gradients[name]
        if value is None:
            parameter.grad = None
        elif parameter.grad is not None and parameter.grad.shape == value.shape and parameter.grad.dtype == value.dtype:
            parameter.grad.copy_(value)
        else:
            parameter.grad = value.to(parameter.device).clone()


def _restore(actor, optimizer, model_state, optimizer_state, gradients):
    _restore_model(actor, model_state)
    # load_state_dict 在 CPU 上可能直接复用传入 state 的 tensor；必须复制，避免后续
    # optimizer.step 污染用于所有候选的不可变初始快照。
    optimizer.load_state_dict(_cpu_clone(optimizer_state))
    _restore_gradients(actor, gradients)


def _check_gradients(actor, gradients):
    for name, parameter in actor.named_parameters():
        saved, current = gradients[name], parameter.grad
        if (saved is None) != (current is None):
            raise RuntimeError(f"Calibration callback or optimizer changed gradient presence: {name}")
        if saved is not None and not torch.equal(saved, current.detach().cpu()):
            raise RuntimeError(f"Calibration callback or optimizer changed the fixed gradient: {name}")


def _check_buffers(actor, snapshot):
    current = {(module_name, name): value for module_name, module in actor.named_modules()
               for name, value in module._buffers.items()}
    if set(current) != set(snapshot["buffers"]):
        raise RuntimeError("Calibration callback changed the model buffer registry")
    for key, saved in snapshot["buffers"].items():
        value, original = current[key], saved["value"]
        if (value is None) != (original is None):
            raise RuntimeError(f"Calibration callback changed model buffer: {key}")
        if value is not None and (value.dtype != original.dtype or value.shape != original.shape
                                  or not torch.equal(value.detach().cpu(), original)):
            raise RuntimeError(f"Calibration callback changed model buffer: {key}")


def _update_hash(digest, name, value):
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())


def _fingerprint(actor):
    digest = hashlib.sha256()
    for name, parameter in actor.named_parameters():
        _update_hash(digest, name, parameter)
    return digest.hexdigest()


def _empty_stats():
    return {"parameter_count": 0, "changed_count": 0, "delta_squared_sum": 0., "max_abs": 0.,
            "gradient_present_count": 0, "nonzero_gradient_count": 0,
            "changed_with_nonzero_gradient_count": 0, "unchanged_with_nonzero_gradient_count": 0,
            "changed_with_zero_gradient_count": 0,
            "fp32_ulp": {"parameter_count": 0, "eligible_count": 0, "zero_change_count": 0,
                         "positive_change_count": 0, "delta_ulp_sum": 0., "max_delta_ulp": 0.,
                         "initial_zero_count": 0, "initial_nonzero_eligible_count": 0,
                         "initial_nonzero_delta_ulp_sum": 0.,
                         "min_positive_delta_ulp": None, "changed_at_least_one_ulp_count": 0}}


def _add_stats(destination, source):
    for key in ("parameter_count", "changed_count", "delta_squared_sum", "gradient_present_count",
                "nonzero_gradient_count", "changed_with_nonzero_gradient_count",
                "unchanged_with_nonzero_gradient_count", "changed_with_zero_gradient_count"):
        destination[key] += source[key]
    destination["max_abs"] = max(destination["max_abs"], source["max_abs"])
    left, right = destination["fp32_ulp"], source["fp32_ulp"]
    for key in ("parameter_count", "eligible_count", "zero_change_count", "positive_change_count",
                "delta_ulp_sum", "changed_at_least_one_ulp_count", "initial_zero_count",
                "initial_nonzero_eligible_count", "initial_nonzero_delta_ulp_sum"):
        left[key] += right[key]
    left["max_delta_ulp"] = max(left["max_delta_ulp"], right["max_delta_ulp"])
    if right["min_positive_delta_ulp"] is not None:
        left["min_positive_delta_ulp"] = right["min_positive_delta_ulp"] if left["min_positive_delta_ulp"] is None else min(left["min_positive_delta_ulp"], right["min_positive_delta_ulp"])


def _finalize_stats(stats):
    result = copy.deepcopy(stats)
    count, changed = result["parameter_count"], result["changed_count"]
    result["changed_fraction"] = changed / count if count else 0.
    result["l2"] = math.sqrt(result.pop("delta_squared_sum"))
    ulp = result["fp32_ulp"]
    ulp["excluded_nonfinite_or_zero_spacing_count"] = ulp["parameter_count"] - ulp["eligible_count"]
    ulp["mean_delta_ulp_excluding_initial_zero"] = (ulp.pop("initial_nonzero_delta_ulp_sum") / ulp["initial_nonzero_eligible_count"]
                                                      if ulp["initial_nonzero_eligible_count"] else None)
    ulp["used_as_acceptance_gate"] = False
    ulp["mean_delta_ulp"] = ulp.pop("delta_ulp_sum") / ulp["eligible_count"] if ulp["eligible_count"] else None
    ulp["zero_change_fraction"] = ulp["zero_change_count"] / ulp["parameter_count"] if ulp["parameter_count"] else None
    ulp["semantics"] = "actual_rounded_parameter_delta_over_initial_fp32_nextafter_spacing_in_update_direction"
    ulp["theoretical_unrounded_optimizer_update_available"] = False
    return result


def _parameter_change(actor, original, gradients):
    total, per_module = _empty_stats(), {}
    digest = hashlib.sha256()
    for name, parameter in actor.named_parameters():
        before = original["state"][name]
        after = parameter.detach().cpu()
        if not before.is_floating_point() or not after.is_floating_point():
            raise TypeError("Calibration requires real floating-point Actor parameters")
        if not torch.isfinite(after).all():
            raise FloatingPointError(f"Nonfinite candidate parameter: {name}")
        _update_hash(digest, name, after)
        difference = after.double() - before.double()
        absolute = difference.abs()
        changed = difference != 0
        current = _empty_stats()
        current.update(parameter_count=before.numel(), changed_count=int(changed.sum()),
                       delta_squared_sum=float(difference.square().sum()),
                       max_abs=float(absolute.max()) if before.numel() else 0.)
        gradient = gradients[name]
        nonzero_gradient = torch.zeros_like(changed) if gradient is None else gradient != 0
        current.update(gradient_present_count=0 if gradient is None else before.numel(),
            nonzero_gradient_count=int(nonzero_gradient.sum()),
            changed_with_nonzero_gradient_count=int((changed & nonzero_gradient).sum()),
            unchanged_with_nonzero_gradient_count=int((~changed & nonzero_gradient).sum()),
            changed_with_zero_gradient_count=int((changed & ~nonzero_gradient).sum()))
        if before.dtype == torch.float32:
            direction = torch.where(difference < 0, torch.full_like(before, -float("inf")), torch.full_like(before, float("inf")))
            spacing = (torch.nextafter(before, direction).double() - before.double()).abs()
            eligible = torch.isfinite(spacing) & (spacing > 0)
            relative = absolute[eligible] / spacing[eligible]
            positive = relative[relative > 0]
            nonzero_initial = eligible & (before != 0)
            nonzero_relative = absolute[nonzero_initial] / spacing[nonzero_initial]
            current["fp32_ulp"].update(parameter_count=before.numel(), eligible_count=int(eligible.sum()),
                zero_change_count=int((~changed).sum()), positive_change_count=int(changed.sum()),
                delta_ulp_sum=float(relative.sum()), max_delta_ulp=float(relative.max()) if relative.numel() else 0.,
                min_positive_delta_ulp=float(positive.min()) if positive.numel() else None,
                changed_at_least_one_ulp_count=int((relative >= 1.).sum()),
                initial_zero_count=int((before == 0).sum()), initial_nonzero_eligible_count=int(nonzero_initial.sum()),
                initial_nonzero_delta_ulp_sum=float(nonzero_relative.sum()))
        _add_stats(total, current)
        module_name = name.split(".", 1)[0] if "." in name else "<root>"
        _add_stats(per_module.setdefault(module_name, _empty_stats()), current)
    result = _finalize_stats(total)
    result["per_module"] = {name: _finalize_stats(value) for name, value in per_module.items()}
    return result, digest.hexdigest()


def _report_value(value):
    if isinstance(value, torch.Tensor):
        return _report_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("KL report requires string mapping keys")
        return {key: _report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite KL diagnostic")
        return int(value) if isinstance(value, int) else float(value)
    raise TypeError(f"KL report contains unsupported value type: {type(value).__name__}")


def _validate(actor, optimizer, candidates, kl_limit, evaluate_kl, reserve_attempt, progress):
    if not isinstance(actor, torch.nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("Calibration requires a torch module and optimizer")
    if not callable(evaluate_kl) or (reserve_attempt is not None and not callable(reserve_attempt)) or (progress is not None and not callable(progress)):
        raise TypeError("KL, budget and progress callbacks must be callable")
    if isinstance(kl_limit, bool) or not isinstance(kl_limit, Real) or not math.isfinite(kl_limit) or kl_limit <= 0:
        raise ValueError("Joint KL limit must be positive and finite")
    candidates = list(candidates)
    if not 1 <= len(candidates) <= 3:
        raise ValueError("Calibration requires one to three bounded learning-rate candidates")
    if any(isinstance(lr, bool) or not isinstance(lr, Real) or not math.isfinite(lr) or not 0 < lr <= 1e-6 for lr in candidates):
        raise ValueError("Every candidate learning rate must be positive, finite and <=1e-6")
    candidates = [float(lr) for lr in candidates]
    if any(right <= left for left, right in zip(candidates, candidates[1:])):
        raise ValueError("Learning-rate candidates must be strictly increasing")
    groups = optimizer.param_groups
    if not groups or any(not isinstance(group["lr"], Real) or isinstance(group["lr"], bool) or not math.isfinite(group["lr"]) for group in groups):
        raise ValueError("Optimizer must use finite scalar learning rates")
    if any(group["lr"] != groups[0]["lr"] for group in groups):
        raise ValueError("Optimizer groups must start with the same learning rate")
    parameters = dict(actor.named_parameters())
    actor_ids = {id(parameter) for parameter in parameters.values()}
    optimized = [id(parameter) for group in groups for parameter in group["params"]]
    if not optimized or len(set(optimized)) != len(optimized) or not set(optimized).issubset(actor_ids):
        raise ValueError("Optimizer must contain unique parameters belonging to this Actor")
    gradients = {}
    for name, parameter in parameters.items():
        if not parameter.is_floating_point() or not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Invalid initial Actor parameter: {name}")
        gradient = parameter.grad
        if gradient is not None and (gradient.is_sparse or not torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Calibration requires finite dense gradients: {name}")
        gradients[name] = _cpu_clone(gradient)
    if not any(gradients[name] is not None and id(parameter) in optimized for name, parameter in parameters.items()):
        raise ValueError("Calibration requires already-computed optimizer gradients")
    return candidates, gradients


def _rng_snapshot():
    return {"python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state()),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def calibrated_optimizer_step(actor, optimizer, evaluate_kl, candidates, kl_limit,
                              reserve_attempt=None, progress=None):
    """返回候选报告并只保留最大合格候选的一步状态；失败时完整恢复初始状态。

    evaluate_kl() 返回包含 mean_joint_kl 的字典。reserve_attempt() 在每次真正
    optimizer.step 前调用，可持久化预算并在不足时抛错。progress(event) 接收
    candidate_start/candidate_end/selected 字典；回调必须只读模型和梯度。
    """
    candidates, gradients = _validate(actor, optimizer, candidates, kl_limit, evaluate_kl, reserve_attempt, progress)
    initial_model = _model_snapshot(actor)
    initial_optimizer = _cpu_clone(optimizer.state_dict())
    initial_rng = _rng_snapshot()
    initial_fingerprint = _fingerprint(actor)
    report = {"candidates": [], "selected_lr": None, "accepted_lr": None, "attempt_count": 0,
              "accepted_updates": 0, "selected_candidate_index": None,
              "selection_rule": "largest_candidate_below_joint_kl_limit", "kl_limit": float(kl_limit),
              "base_state_restored_per_candidate": True, "gradients_reused": True}
    best = None
    try:
        for index, learning_rate in enumerate(candidates):
            _restore(actor, optimizer, initial_model, initial_optimizer, gradients)
            _restore_rng(initial_rng)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            candidate = {"lr": learning_rate, "kl": None, "parameter_change": None, "accepted": False,
                         "candidate_index": index, "attempted": False, "status": "pending"}
            report["candidates"].append(candidate)
            if progress is not None:
                progress({"event": "candidate_start", "candidate_index": index, "lr": learning_rate})
            if reserve_attempt is not None:
                reserve_attempt()
            _check_gradients(actor, gradients)
            _check_buffers(actor, initial_model)
            if _fingerprint(actor) != initial_fingerprint:
                raise RuntimeError("Budget or progress callback changed initial Actor parameters")
            report["attempt_count"] += 1
            candidate.update(attempted=True, status="optimizer_step")
            optimizer.step()
            _check_gradients(actor, gradients)
            _check_buffers(actor, initial_model)
            change, before_kl_fingerprint = _parameter_change(actor, initial_model, gradients)
            candidate["parameter_change"] = change
            with torch.no_grad():
                kl = _report_value(evaluate_kl())
            if not isinstance(kl, dict) or "mean_joint_kl" not in kl:
                raise ValueError("KL callback must return a dictionary containing mean_joint_kl")
            mean = kl["mean_joint_kl"]
            if isinstance(mean, bool) or not isinstance(mean, Real) or mean < 0:
                raise ValueError("mean_joint_kl must be a finite nonnegative scalar")
            _check_gradients(actor, gradients)
            _check_buffers(actor, initial_model)
            if _fingerprint(actor) != before_kl_fingerprint:
                raise RuntimeError("KL callback changed Actor parameters")
            candidate["kl"] = kl
            candidate["accepted"] = bool(mean <= kl_limit and change["changed_count"] > 0)
            candidate["status"] = "accepted" if candidate["accepted"] else "rejected_kl" if mean > kl_limit else "rejected_zero_parameter_change"
            if candidate["accepted"]:
                # 当前学习率严格递增；先释放之前的合格大快照，避免同时保存三份候选。
                best = None
                best = (_model_snapshot(actor), _cpu_clone(optimizer.state_dict()))
                report.update(selected_lr=learning_rate, accepted_lr=learning_rate, selected_candidate_index=index)
            if progress is not None:
                progress({"event": "candidate_end", "candidate_index": index, "lr": learning_rate,
                          "mean_joint_kl": float(mean), "changed_fraction": change["changed_fraction"],
                          "accepted": candidate["accepted"], "status": candidate["status"],
                          "candidate": copy.deepcopy(candidate)})
                _check_gradients(actor, gradients)
                _check_buffers(actor, initial_model)
                if _fingerprint(actor) != before_kl_fingerprint:
                    raise RuntimeError("Progress callback changed Actor parameters")
        if best is None:
            raise LearningRateCalibrationRejected(report)
        _restore(actor, optimizer, best[0], best[1], gradients)
        # KL 回调可能切换 eval 模式；最终模式使用初始调用者状态。
        for name, module in actor.named_modules():
            module.training = initial_model["modes"][name]
        _check_buffers(actor, initial_model)
        _check_gradients(actor, gradients)
        report["accepted_updates"] = 1
        report["buffers_unchanged"] = True
        report["rng_state_restored"] = True
        selected_fingerprint = _fingerprint(actor)
        if progress is not None:
            progress({"event": "selected", "selected_lr": report["selected_lr"], "attempt_count": report["attempt_count"]})
        _check_gradients(actor, gradients)
        _check_buffers(actor, initial_model)
        if _fingerprint(actor) != selected_fingerprint:
            raise RuntimeError("Selection progress callback changed Actor parameters")
        _restore_rng(initial_rng)
        return report
    except BaseException as error:
        _restore(actor, optimizer, initial_model, initial_optimizer, gradients)
        _restore_rng(initial_rng)
        report["accepted_updates"] = 0
        report["rolled_back_to_initial_state"] = True
        report["selected_lr"] = report["accepted_lr"] = report["selected_candidate_index"] = None
        if isinstance(error, LearningRateCalibrationRejected):
            error.report = copy.deepcopy(report)
        try:
            error.lr_calibration_report = copy.deepcopy(report)
        except (AttributeError, TypeError):
            pass
        raise
