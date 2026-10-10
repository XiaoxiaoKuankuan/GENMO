#!/usr/bin/env python3
"""用未改写的Stage1 ONNX独立验收正式部署包中的TensorRT FP32引擎。

先核验清单及全部资产哈希，再以ONNX Runtime CUDA（关闭TF32）为参考，比较空历史、
部分/完整50步历史、12帧动作前缀、短尾窗和全padding的单步motion/contact输出。
额外把无效槽填入NaN/Inf，验证掩码确实隔离未选分支；同时比较CUDA Graph和普通执行。
完整DDIM使用相同CPU噪声和20步eta=0采样，每个参考轨迹步额外用相同输入核验两个后端，
区分单步算子差异和独立采样的累计误差。physical qpos30/canonical qpos28沿用单步容差；
接触logits按1e-3绝对误差核验，并另外要求sigmoid接触概率差异不超过1e-4，避免只用
中间logits尺度判断物理影响。严格检查前缀逐位回填和padding补零，再运行120/12/108生成器，
核验两窗口连续输出、站姿前缀、归一化四元数及ONNX/TensorRT的世界轨迹与FK位置误差。

--audio可重复提供音乐文件；只读取开头指定帧数对应的片段，使用控制台同一35维特征
提取器。未提供音乐时用固定种子的合成特征。测试不连接GMT、ROS、Redis或机器人，
不构建引擎，不写训练目录，不修改源ONNX/部署清单，也不将数值对齐当作控制安全验收。
--output明确指定JSON结果路径，按仓库约定应使用系统临时目录或独立测试目录；无论
成功或失败均写入已有进度、资产身份、版本、误差和阈值，并以非零退出码报告失败。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from gem.robots.bumi.endecoder import BumiEndecoder  # noqa: E402
from gem.runtime.bumi_deployment_bundle import load_bumi_deployment_manifest  # noqa: E402
from gem.runtime.bumi_music_contract import PREFIX_FRAMES, SOURCE_FPS, WINDOW_FRAMES  # noqa: E402
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BumiOrtStepRunner, BumiStage1QposGenerator, BumiTensorRTStepRunner,
    Stage1DdimSampler, stage1_warmup_inputs,
)
from gem.runtime.bumi_stage1_history import CausalDemoProprio48Builder  # noqa: E402
from gem.utils.music_features import align_features_to_length, extract_edge_baseline35  # noqa: E402


def compare(actual, reference, *, atol=2e-4, rtol=1e-4):
    """分别保留绝对误差和阈值结论；不在失败后自动放宽阈值。"""
    actual, reference = actual.detach().cpu().double(), reference.detach().cpu().double()
    if actual.shape != reference.shape:
        raise AssertionError(f"输出形状不一致: {actual.shape} != {reference.shape}")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    difference = (actual - reference).abs()
    return {"pass": finite and bool(torch.all(difference <= atol + rtol * reference.abs())),
            "finite": finite, "shape": list(actual.shape), "atol": atol, "rtol": rtol,
            "max_abs": float(difference.max()) if finite else None,
            "mean_abs": float(difference.mean()) if finite else None}


def require(record, message):
    if not record["pass"]:
        raise AssertionError(f"{message}: {record}")


def make_inputs(endecoder, *, decision, valid_frames, timestep, prefix=True):
    values = stage1_warmup_inputs("cpu")
    random = torch.Generator().manual_seed(1000 + timestep)
    values["noisy_motion"] = torch.randn((1, 120, 30), generator=random)
    values["music_features"] = torch.randn((1, 120, 35), generator=random)
    values["diffusion_timestep"].fill_(timestep)
    values["future_valid"][:, valid_frames:] = False
    values["music_valid"][:, valid_frames:] = False
    values["guidance_scale"].fill_(1.0 if decision == 9 else 2.5)
    trajectory = endecoder.kinematics.make_standing_qpos().cpu().repeat(decision + 12, 1)
    clock = torch.arange(len(trajectory), dtype=torch.float32) / SOURCE_FPS
    trajectory[:, 7:] += 0.025 * torch.sin(clock[:, None] * 2.0 + torch.arange(21)[None])
    trajectory[:, 0] += 0.01 * clock
    builder = CausalDemoProprio48Builder(endecoder.kinematics)
    history, valid, times = builder.build_history(trajectory[:decision + 1], decision_frame=decision)
    values["proprio_history"] = history[None]
    values["proprio_history_valid"] = valid[None]
    values["history_relative_times"] = (times - decision / SOURCE_FPS).float()[None]
    if prefix:
        count = min(PREFIX_FRAMES, valid_frames)
        encoded = endecoder.codec.encode(trajectory[decision:decision + max(count, 1)])
        values["known_qpos30"][0, :count] = encoded.physical_features[:count]
        values["known_qpos30_mask"][0, :count, 2:] = True
        values["known_qpos30_mask"][0, :max(count - 1, 0), :2] = True
        values["known_qpos30"][~values["known_qpos30_mask"]] = 0.0
    return values


def snapshot(runner, values):
    # TensorRT返回持久缓冲视图；下一次运行前必须复制，避免把同一缓冲误当对齐证据。
    with torch.inference_mode():
        return tuple(value.detach().cpu().clone() for value in runner(values))


def check_generation(runner, endecoder, music, args):
    generator = BumiStage1QposGenerator(runner, endecoder, device=args.device,
                                       steps=args.steps, guidance_scale=2.5)
    chunks = list(generator.generate(music, seed=args.seed))
    cursor, records = 0, []
    for index, chunk in enumerate(chunks):
        if chunk.absolute_start_frame != cursor or chunk.total_frames != len(music):
            raise AssertionError("生成块帧号不连续或总帧数错误")
        if chunk.is_last != (index == len(chunks) - 1):
            raise AssertionError("生成块结束标记错误")
        cursor += len(chunk.qpos)
        records.append({"start": chunk.absolute_start_frame, "frames": len(chunk.qpos),
                        "is_last": chunk.is_last})
    qpos = torch.cat([chunk.qpos for chunk in chunks])
    if cursor != len(music) or len(chunks) != 2 or not bool(torch.isfinite(qpos).all()):
        raise AssertionError("完整轨迹帧数/窗口数/有限性错误")
    standing = endecoder.kinematics.make_standing_qpos().cpu().repeat(PREFIX_FRAMES, 1)
    if not torch.equal(qpos[:PREFIX_FRAMES], standing):
        raise AssertionError("初始12帧站姿被改变")
    if not torch.allclose(torch.linalg.vector_norm(qpos[:, 3:7], dim=-1),
                          torch.ones(len(qpos)), atol=2e-5, rtol=0):
        raise AssertionError("生成四元数未归一化")
    return qpos, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path,
                        default=ROOT / "models/bumi_stage1_s595000/deployment.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=167)
    parser.add_argument("--audio", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not WINDOW_FRAMES < args.frames <= 2 * WINDOW_FRAMES - PREFIX_FRAMES:
        parser.error("--frames必须为121..228，覆盖恰好两个窗口及短尾窗")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"pass": False, "single_step": [], "ddim": [], "generation": []}
    started = time.perf_counter()
    try:
        bundle = load_bumi_deployment_manifest(args.deployment_manifest)
        report.update(manifest=str(bundle.manifest_path), asset_sha256=bundle.hashes,
                      source_checkpoint_sha256=bundle.source_checkpoint_sha256,
                      steps=args.steps, seed=args.seed, frames=args.frames)
        report["versions"] = {name: metadata.version(name) for name in
                              ("torch", "numpy", "onnxruntime-gpu")}
        endecoder = BumiEndecoder(bundle.paths["kinematics"], bundle.paths["stats"],
                                 enable_contact_targets=False)
        trt = BumiTensorRTStepRunner(bundle.paths["engine"], device=args.device)
        # 部署环境可能仅安装tensorrt-cu13-bindings，不存在名为tensorrt的发行包。
        report["versions"]["tensorrt"] = trt.trt.__version__
        ort = BumiOrtStepRunner(bundle.paths["onnx"], device=args.device, provider="cuda")
        report.update(gpu=torch.cuda.get_device_name(args.device),
                      ort_providers=ort.session.get_providers(), ort_use_tf32=False,
                      cuda_graph=trt.cuda_graph is not None,
                      libnvinfer=trt.linked_tensorrt_version)
        cases = [("empty_history", 0, 120, 999, False),
                 ("partial_history_prefix12", 9, 120, 500, True),
                 ("history50_prefix12", 60, 120, 250, True),
                 ("short_tail", 9, 59, 0, True),
                 ("all_padding", 0, 0, 999, False)]
        inputs, results = {}, {}
        for name, decision, length, timestep, prefix in cases:
            values = make_inputs(endecoder, decision=decision, valid_frames=length,
                                 timestep=timestep, prefix=prefix)
            inputs[name] = values
            expected, actual = snapshot(ort, values), snapshot(trt, values)
            results[name] = {"onnx": expected, "tensorrt": actual}
            record = {"case": name, "history_valid": int(values["proprio_history_valid"].sum()),
                      "motion": compare(actual[0], expected[0]),
                      "contact": compare(actual[1], expected[1])}
            report["single_step"].append(record)
            require(record["motion"], f"{name} motion")
            require(record["contact"], f"{name} contact")
            for output in actual + expected:
                if not torch.equal(output[:, length:], torch.zeros_like(output[:, length:])):
                    raise AssertionError(f"{name}无效未来槽没有严格补零")
            print(f"单步通过 {name}: motion={record['motion']['max_abs']:.8g}, "
                  f"contact={record['contact']['max_abs']:.8g}", flush=True)
        poisoned = {key: value.clone() for key, value in inputs["short_tail"].items()}
        for field, mask in (("music_features", poisoned["music_valid"][..., None]),
                            ("proprio_history", poisoned["proprio_history_valid"][..., None]),
                            ("history_relative_times", poisoned["proprio_history_valid"]),
                            ("known_qpos30", poisoned["known_qpos30_mask"]),
                            ("noisy_motion", poisoned["future_valid"][..., None])):
            poisoned[field].masked_fill_(~mask, float("nan"))
        poisoned["music_features"][:, 100:] = float("inf")
        isolation = {"case": "invalid_nan_inf_isolation"}
        report["single_step"].append(isolation)
        for backend, runner in (("onnx", ort), ("tensorrt", trt)):
            outputs = snapshot(runner, poisoned)
            isolation[backend] = [compare(value, ref, atol=0, rtol=0) for value, ref in
                                  zip(outputs, results["short_tail"][backend])]
            for item in isolation[backend]:
                require(item, f"{backend}无效NaN/Inf槽隔离")
        captured = trt.cuda_graph
        trt.cuda_graph = None
        try:
            outputs = snapshot(trt, inputs["history50_prefix12"])
        finally:
            trt.cuda_graph = captured
        report["ordinary_vs_cuda_graph"] = [compare(value, ref, atol=0, rtol=0) for value, ref in
                                              zip(outputs, results["history50_prefix12"]["tensorrt"])]
        for item in report["ordinary_vs_cuda_graph"]:
            require(item, "普通执行与CUDA Graph输出")
        noise = torch.randn((1, 120, 30), generator=torch.Generator().manual_seed(args.seed))
        for name in ("history50_prefix12", "short_tail"):
            conditions = {key: value for key, value in inputs[name].items()
                          if key not in ("noisy_motion", "diffusion_timestep", "guidance_scale")}
            outputs, aligned_steps = {}, []

            def paired_reference(feed):
                reference, actual = ort(feed), trt(feed)
                item = {"timestep": int(feed["diffusion_timestep"].item()),
                        "motion": compare(actual[0], reference[0]),
                        "contact": compare(actual[1], reference[1])}
                aligned_steps.append(item)
                require(item["motion"], f"{name}同输入DDIM单步motion")
                require(item["contact"], f"{name}同输入DDIM单步contact")
                return reference

            for backend, runner in (("onnx", paired_reference), ("tensorrt", trt)):
                sampler = Stage1DdimSampler(runner, endecoder, device=args.device,
                                           steps=args.steps, guidance_scale=2.5)
                result = sampler.sample(conditions, noise)
                outputs[backend] = {key: value.cpu().clone() for key, value in result.items()}
                mask = conditions["known_qpos30_mask"]
                if not torch.equal(outputs[backend]["qpos30"][mask], conditions["known_qpos30"][mask]):
                    raise AssertionError(f"{backend} DDIM改变已知前缀")
                valid = conditions["future_valid"]
                for key, value in outputs[backend].items():
                    if not torch.equal(value[~valid], torch.zeros_like(value[~valid])):
                        raise AssertionError(f"{backend} DDIM {key}无效槽没有严格补零")
            record = {"case": name, "prefix_exact": True, "padding_exact": True,
                      "same_input_steps": aligned_steps}
            report["ddim"].append(record)
            for key in outputs["onnx"]:
                # 多步独立采样会累积FP32差异；接触head另加更严格的实际概率误差界。
                record[key] = (compare(outputs["tensorrt"][key], outputs["onnx"][key], atol=1e-3, rtol=0)
                               if key == "contact_logits" else
                               compare(outputs["tensorrt"][key], outputs["onnx"][key]))
                require(record[key], f"{name}完整DDIM {key}")
            probabilities = {backend: torch.where(conditions["future_valid"][..., None],
                                                   output["contact_logits"].sigmoid(), 0.0)
                             for backend, output in outputs.items()}
            record["contact_probability"] = compare(probabilities["tensorrt"], probabilities["onnx"],
                                                      atol=1e-4, rtol=0)
            require(record["contact_probability"], f"{name}完整DDIM接触概率")
            record["contact_label_mismatch"] = int(((probabilities["tensorrt"] >= 0.5) !=
                                                     (probabilities["onnx"] >= 0.5)).sum())
            print(f"完整DDIM通过 {name}: qpos30={record['qpos30']['max_abs']:.8g}", flush=True)
        selections = args.audio or [None]
        for source in selections:
            if source is None:
                music = torch.randn((args.frames, 35), generator=torch.Generator().manual_seed(args.seed))
                music_metadata = {"kind": "seeded_synthetic"}
            else:
                music, music_metadata = extract_edge_baseline35(
                    source, duration_sec=args.frames / SOURCE_FPS, target_fps=SOURCE_FPS)
                music = align_features_to_length(music, args.frames, policy="trim_or_pad_last")
            outputs, blocks = {}, {}
            durations = {}
            for backend, runner in (("onnx", ort), ("tensorrt", trt)):
                begin = time.perf_counter()
                outputs[backend], blocks[backend] = check_generation(runner, endecoder, music, args)
                durations[backend] = time.perf_counter() - begin
            if blocks["onnx"] != blocks["tensorrt"]:
                raise AssertionError("两后端生成块规划不同")
            record = {"audio": str(source) if source else None, "feature_metadata": music_metadata,
                      "chunks": blocks["tensorrt"], "seconds": durations,
                      "qpos28": compare(outputs["tensorrt"], outputs["onnx"], atol=1e-3, rtol=1e-4),
                      "root_xyz": compare(outputs["tensorrt"][:, :3], outputs["onnx"][:, :3],
                                          atol=1e-3, rtol=0),
                      "joint_radians": compare(outputs["tensorrt"][:, 7:], outputs["onnx"][:, 7:],
                                               atol=1e-3, rtol=0),
                      "body_position_meters": compare(
                          endecoder.kinematics.forward_body_positions(outputs["tensorrt"]),
                          endecoder.kinematics.forward_body_positions(outputs["onnx"]), atol=1e-3, rtol=0)}
            report["generation"].append(record)
            for key in ("qpos28", "root_xyz", "joint_radians", "body_position_meters"):
                require(record[key], f"{source}两窗口续接 {key}")
            print(f"两窗口生成通过 {source or 'synthetic'}: qpos28={record['qpos28']['max_abs']:.8g}", flush=True)
        report["pass"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(f"验收结果: {args.output}; pass={report['pass']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
