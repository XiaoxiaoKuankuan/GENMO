"""真实Stage1与冻结SM的计算优化实验，不授权正式训练或代替部署延迟标定。

从真实音乐和名义启动张量构造一次合法请求，分别计时完整旧参考前瞻、当前GENMO
全20步DDIM和世界参考解码。随后仅编译相同网络计算，比较FP32前向一致性与稳定
计算耗时；不降低采样步数、不更新权重或normalizer、不改变目标/历史/前缀。
这是没有并行真实物理竞争的计算诊断；即使加速，也须另做输入梯度、完整异步物理
标定及WM方向验收后才允许接入正式路径。本工具不会修改训练配置或生成通过许可。
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config)
    sys.path.insert(0, str(Path(config.supermimic.repo).resolve()))
    import torch
    from gem.closedloop.mb_control.application import read_config, build_components, read_tasks, task_inputs, seed_all
    from gem.closedloop.mb_control.gradient_audit import tensor_fixture
    from gem.closedloop.mb_control.forecast import forecast_rollout
    from gem.closedloop.mb_control.online_conditions import build_conditions
    from gem.closedloop.mb_control.compilation import compile_networks
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = read_config(args.config)
    seed_all(42)
    actor, adapter, scheduler, _, _, _, _ = build_components(cfg, "cuda:0")
    before = adapter.bundle.frozen_snapshot()
    music, initial = task_inputs(read_tasks(cfg["tasks"])[0], adapter, "cuda:0")
    z = tensor_fixture(adapter, music, initial)
    ticket = scheduler.reserve(z)
    conditions, anchor = build_conditions(z, ticket, actor.stage1.endecoder.codec)
    noise = torch.randn(1, 60, 30, device="cuda:0")
    def measure():
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            context = forecast_rollout(z, adapter, ticket.first_influence_tick).context
            torch.cuda.synchronize()
            forecast_end = time.perf_counter()
            result = actor.sample_differentiable(conditions, context, noise)
            actor.stage1.endecoder.codec.apply_world_anchor(result["qpos28"], anchor)
            torch.cuda.synchronize()
            end = time.perf_counter()
        return result, context, {"forecast_seconds": forecast_end-start,
                                 "generation_seconds": end-forecast_end, "total_seconds": end-start}
    rows = {}
    for _ in range(2):
        measure()
    original, original_context, _ = measure()
    rows["eager"] = [measure()[2] for _ in range(args.samples)]
    compile_start = time.perf_counter()
    compile_networks(actor, adapter)
    for _ in range(2):
        measure()
    compile_seconds = time.perf_counter() - compile_start
    compiled, compiled_context, _ = measure()
    rows["compiled"] = [measure()[2] for _ in range(args.samples)]
    adapter.bundle.assert_unchanged(before)
    comparison = {key: float((compiled[key] - original[key]).abs().max()) for key in ("qpos30", "qpos28", "contact")}
    comparison["forecast_features"] = float((compiled_context.features - original_context.features).abs().max())
    result = {"scope": "compute_benchmark_only_not_deployment_or_training_acceptance", "rows": rows,
              "compile_seconds": compile_seconds, "fp32_max_absolute_difference": comparison,
              "same_sampling_forward_passed": comparison["qpos30"] <= 1e-5,
              "frozen_unchanged": True, "gradient_validation": "not_run", "supported_control_steps": 0}
    (output / "compute_benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
