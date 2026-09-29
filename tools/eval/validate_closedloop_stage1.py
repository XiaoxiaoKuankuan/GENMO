#!/usr/bin/env python3
"""导出并验证已完成的 Stage 1 模型，生成四库完整舞蹈对比产物。

使用checkpoint内嵌架构、严格权重加载及同SHA归一化/FK资产，拒绝替换成旧纯音乐模型。
先导出固定120帧、50帧历史的条件ONNX，再比较真实PyTorch与ONNX的完整20步DDIM。
固定窗口按同一音乐、种子和4秒区间分别测P=0/P=12；完整舞蹈采用12帧重叠的自主滚动
生成，仅首个0.4秒动作来自示范，后续历史和前缀均来自自身生成，不逐窗口喂入GT未来。
视频由现有render_bumi_motion.py渲染并使用相同音乐同步封装；原动作和生成动作保持
相同帧数，原始数值产物保留世界坐标，显示时只共同平移地面。该流程是运动学验证，
没有执行GMT控制器或动力学仿真，也不证明真实机器人追踪能力。
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import load_stage1_checkpoint
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.stage1_dataset import (
    BumiClosedLoopStage1Dataset,
    collate_stage1_training_samples,
)
from gem.closedloop.training import (
    batch_to_device,
    build_stage1_actor,
    build_stage1_losses,
    load_stage1_data_config,
)
from gem.robots.bumi.kinematics import sha256_file
from gem.robots.bumi.metrics import compute_bumi_kinematic_metrics
from gem.runtime.closedloop_stage1_onnx import (
    CONTRACT_VERSION,
    INPUT_NAMES,
    OUTPUT_NAMES,
    Stage1GuidedDenoiser,
    Stage1OnnxSampler,
    make_inputs,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def load_actor(checkpoint, stats, kinematics, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = OmegaConf.create(payload["config"])
    config.endecoder.stats_path = str(stats.resolve())
    config.endecoder.kinematics_path = str(kinematics.resolve())
    data = load_stage1_data_config(config)
    for entries in data.datasets.values():
        for entry in entries.values():
            entry.kinematics_path = str(kinematics.resolve())
    actor = build_stage1_actor(config, data)
    loading = load_stage1_checkpoint(actor, payload)
    nonfinite = [k for k, v in payload["state_dict"].items() if not torch.isfinite(v).all()]
    if nonfinite:
        raise FloatingPointError(nonfinite)
    summary = {
        "global_step": payload["global_step"],
        "checkpoint_version": payload["checkpoint_version"],
        "asset_identity": payload["asset_identity"],
        "interface": payload["actor_interface_config"],
        "state_dict_keys": len(payload["state_dict"]),
        "nonfinite_tensors": len(nonfinite),
        "loaded_keys": len(loading["loaded"]),
        "python_import": str(sys.modules["gem"].__file__),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    del payload
    gc.collect()
    actor = actor.eval().to(device)
    return actor, config, summary


def dataset_for(data_root, entry, kinematics, prefix=12):
    return BumiClosedLoopStage1Dataset(
        root=data_root / entry["source"],
        dataset_name=entry["row"]["dataset"],
        split=entry["split"],
        kinematics_path=kinematics,
        prefix_min_frames=prefix,
        prefix_max_frames=prefix,
        prefix_zero_probability=0,
        duration_aware_sampling=False,
        random_decision=False,
        validate_source_hashes_on_init=True,
        joint_limit_tolerance=0.0001,
    )


@torch.inference_mode()
def export_model(actor, sample, path, checkpoint_summary):
    import onnx

    metadata_path = path.with_suffix(".onnx.json")
    if path.exists():
        old = json.loads(metadata_path.read_text())
        if old["checkpoint_sha256"] != checkpoint_summary["checkpoint_sha256"]:
            raise RuntimeError("拒绝覆盖不同checkpoint的ONNX")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    # CUDA GRUCell 会追踪成 ONNX 未支持的 fused 算子；CPU 使用同权重的可导出基础运算。
    original_device = next(actor.parameters()).device
    actor.cpu()
    wrapper = Stage1GuidedDenoiser(actor).eval()
    noise = torch.randn((1, 120, 30))
    conditions = {key: sample[key].cpu() for key in STAGE1_CONDITION_KEYS}
    inputs = make_inputs(conditions, noise, torch.tensor([999], device=noise.device), 2.5)
    torch.onnx.export(
        wrapper,
        inputs,
        str(path),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path))
    metadata = {
        "contract_version": CONTRACT_VERSION,
        **checkpoint_summary,
        "scope": "condition_encoders_and_cfg_denoiser_step_ddim_and_fk_external",
        "input_contract": {key: list(value.shape) for key, value in zip(INPUT_NAMES, inputs)},
        "output_contract": {"pred_motion": [1, 120, 30], "pred_foot_contact_logits": [1, 120, 2]},
    }
    onnx.helper.set_model_props(
        graph,
        {
            "contract_version": CONTRACT_VERSION,
            "checkpoint_sha256": checkpoint_summary["checkpoint_sha256"],
            "global_step": str(checkpoint_summary["global_step"]),
            "interface": json.dumps(checkpoint_summary["interface"]),
            "asset_identity": json.dumps(checkpoint_summary["asset_identity"]),
            "scope": metadata["scope"],
        },
    )
    onnx.save(graph, str(path))
    del graph
    gc.collect()
    onnx.checker.check_model(str(path))
    metadata["onnx_sha256"] = sha256_file(path)
    metadata["onnx_size_bytes"] = path.stat().st_size
    write_json(metadata_path, metadata)
    actor.to(original_device).eval()
    print("ONNX_EXPORTED", path, flush=True)
    return metadata


@torch.inference_mode()
def parity(actor, runtime, batch, output):
    records = []
    for prefix, cfg, empty_history in ((12, 2.5, False), (0, 2.5, False), (12, 1.0, True)):
        conditions = {k: batch[k].clone() for k in STAGE1_CONDITION_KEYS}
        if prefix == 0:
            conditions["known_qpos30"].zero_()
            conditions["known_qpos30_mask"].zero_()
        if empty_history:
            conditions["proprio_history"].zero_()
            conditions["proprio_history_valid"].zero_()
        noise = torch.randn((1, 120, 30), generator=torch.Generator().manual_seed(42)).to(
            batch["known_qpos30"]
        )
        reference = actor.sample(
            conditions, noise=noise, steps=20, guidance_scale=cfg, return_trace=True
        )
        actual = runtime.sample(
            conditions, noise=noise, steps=20, guidance_scale=cfg, return_trace=True
        )
        record = {
            "prefix": prefix,
            "cfg": cfg,
            "empty_history": empty_history,
            **{
                key + "_max_abs": float((actual[key] - reference[key]).abs().max())
                for key in ("qpos30", "qpos", "normalized", "contact")
            },
        }
        mask = conditions["known_qpos30_mask"]
        record["known_max_abs"] = (
            float((actual["qpos30"][mask] - conditions["known_qpos30"][mask]).abs().max())
            if mask.any()
            else 0.0
        )
        expected = actor.endecoder.normalize(conditions["known_qpos30"])
        record["trace_known_max_abs"] = max(
            float((v[mask] - expected[mask]).abs().max()) if mask.any() else 0.0
            for v in actual["trace"]
        )
        if (
            record["qpos_max_abs"] > 0.005
            or record["qpos30_max_abs"] > 0.005
            or record["contact_max_abs"] > 0.005
        ):
            raise AssertionError(f"完整DDIM数值对齐失败：{record}")
        assert record["known_max_abs"] == record["trace_known_max_abs"] == 0
        records.append(record)
        print("PARITY", json.dumps(record), flush=True)
    write_json(
        output,
        {
            "passed": True,
            "tolerance": 0.005,
            "cases": records,
            "providers": runtime.session.get_providers(),
        },
    )


def world_qpos(codec, qpos, anchor):
    return codec.apply_world_anchor(
        qpos,
        {
            "root_xy": anchor.position_w[..., :2],
            "yaw": anchor.yaw,
            "anchor_z": anchor.default_root_height,
        },
    )


def metric_values(qpos, kinematics, **kwargs):
    return {
        k: float(v) for k, v in compute_bumi_kinematic_metrics(qpos, kinematics, **kwargs).items()
    }


@torch.inference_mode()
def fixed_evaluation(actor, runtime, losses, dataset, index, seed):
    records = []
    device = next(actor.parameters()).device
    for prefix in (0, 12):
        dataset.prefix_min_frames = dataset.prefix_max_frames = prefix
        # 选取完整120帧中央窗口；Dataset默认中心是决策点，短序列可能留下尾部padding。
        decision = max(0, (int(dataset.rows[index]["num_frames"]) - 120) // 2)
        source = dataset.get_window(index, start_frame=decision)
        batch = batch_to_device(collate_stage1_training_samples([source]), device)
        conditions = {k: batch[k] for k in STAGE1_CONDITION_KEYS}
        noise = torch.randn((1, 120, 30), generator=torch.Generator().manual_seed(seed)).to(device)
        result = runtime.sample(conditions, noise=noise)
        loss, components = losses(
            batch, result["normalized"], result["contact_logits"], global_step=350000
        )
        known = conditions["known_qpos30_mask"]
        known_error = (
            float((result["qpos30"][known] - conditions["known_qpos30"][known]).abs().max())
            if known.any()
            else 0.0
        )
        assert known_error == 0.0
        gt = actor.endecoder.codec.decode_to_canonical_qpos(batch["target_qpos30"])
        valid = batch["future_valid"].clone()
        valid[:, :prefix] = False
        ground = (
            source["meta"]["ground_supervision"]["ground_height_world_m"]
            - actor.endecoder.codec.default_root_height
        )
        metrics = metric_values(
            result["qpos"],
            actor.endecoder.kinematics,
            target_qpos=gt,
            valid_mask=valid,
            pred_contact_logits=result["contact_logits"],
            target_contact=batch["target_contact"],
            music_beats=batch["music_features"][..., 34],
            ground_height=ground,
        )
        records.append(
            {
                "prefix_frames": prefix,
                "loss": float(loss),
                "known_max_abs_error": known_error,
                "decision_frame": source["meta"]["decision_frame"],
                "seed": seed,
                "components": {
                    k: float(v)
                    for k, v in components.items()
                    if isinstance(v, torch.Tensor) and v.numel() == 1
                },
                "metrics": metrics,
            }
        )
    return records


@torch.inference_mode()
def generate_full(actor, runtime, dataset, index, seed):
    """只首12帧读取示范，之后用自身输出作为历史及重叠前缀，保留全部源帧。"""
    dataset.prefix_min_frames = dataset.prefix_max_frames = 12
    source = dataset.get_window(index, start_frame=0)
    sequence = dataset.reader.load_aligned_sequence(dataset.rows[index])
    original, music = sequence["qpos"], sequence["music"]
    total = len(original)
    device = next(actor.parameters()).device
    codec = actor.endecoder.codec
    generated = original[:12].clone()
    contacts = torch.zeros((12, 2))
    decision, window = 0, 0
    prefix_error = 0.0
    while len(generated) < total:
        count = min(120, total - decision)
        prefix = generated[decision : decision + 12]
        encoded = codec.encode(prefix)
        known = torch.zeros(120, 30)
        known_mask = torch.zeros(120, 30, dtype=torch.bool)
        known_mask[: len(prefix), 2:] = True
        known_mask[: len(prefix) - 1, :2] = True
        known[: len(prefix)] = encoded.physical_features
        known[~known_mask] = 0.0
        hist, hist_valid, hist_times = dataset.proprio_builder.build_history(
            generated[: decision + 1], decision_frame=decision, history_steps=50
        )
        music_window = torch.zeros(120, 35)
        music_window[:count] = music[decision : decision + count]
        valid = torch.arange(120) < count
        cond = {
            "music_features": music_window[None],
            "music_valid": valid[None],
            "proprio_history": hist[None],
            "proprio_history_valid": hist_valid[None],
            "proprio_history_times": hist_times[None],
            "known_qpos30": known[None],
            "known_qpos30_mask": known_mask[None],
            "future_valid": valid[None],
            "future_times": ((decision + torch.arange(120, dtype=torch.float64)) / 30)[None],
            "decision_time": torch.tensor([decision / 30], dtype=torch.float64),
        }
        cond = batch_to_device(cond, device)
        noise = torch.randn(
            (1, 120, 30), generator=torch.Generator().manual_seed(seed + window)
        ).to(device)
        result = runtime.sample(cond, noise=noise)
        error = float(
            (
                result["qpos30"][cond["known_qpos30_mask"]]
                - cond["known_qpos30"][cond["known_qpos30_mask"]]
            )
            .abs()
            .max()
        )
        prefix_error = max(prefix_error, error)
        segment = world_qpos(codec, result["qpos"][0, :count].cpu(), encoded.anchor)
        generated = torch.cat((generated, segment[len(prefix) :]), 0)
        contacts = torch.cat((contacts, result["contact_logits"][0, len(prefix) : count].cpu()), 0)
        decision = len(generated) - 12
        window += 1
    assert len(generated) == total and prefix_error == 0.0
    return (
        original,
        generated,
        contacts,
        {
            "windows": window,
            "known_max_abs_error": prefix_error,
            "ground_height_world_m": source["meta"]["ground_supervision"]["ground_height_world_m"],
            "initial_demo_frames": 12,
            "future_demo_after_initial_prefix": False,
            "history_source": "generated_kinematic_proxy_no_physics",
            "num_frames": total,
        },
    )


def render_video(motion, audio, mjcf, output, log):
    silent = output.with_suffix(".silent.mp4")
    temporary = output.with_suffix(".mux.mp4")
    env = {**os.environ, "MUJOCO_GL": "egl", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        with log.open("a") as handle:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "tools/eval/render_bumi_motion.py"),
                    "--motion",
                    str(motion),
                    "--mjcf",
                    str(mjcf),
                    "--output",
                    str(silent),
                    "--width",
                    "960",
                    "--height",
                    "540",
                    "--follow-root",
                ],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(silent),
                    "-i",
                    str(audio),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-af",
                    "apad",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        temporary.replace(output)
    finally:
        silent.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def render_pair(paths, audio, mjcf, directory):
    for kind, path in paths.items():
        output = directory / f"{kind}.mp4"
        if not output.exists():
            render_video(path, audio, mjcf, output, directory / "render.log")
    print("RENDERED", directory.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="复用已生成动作，仅补齐视频")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    entries = json.loads(args.selection.read_text())["samples"]
    if args.render_only:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = []
            for number, entry in enumerate(entries):
                directory = args.output / "samples" / f"{number + 1:02d}"
                paths = {
                    kind: directory / f"{kind}_display.pt" for kind in ("original", "generated")
                }
                audio = args.data / entry["source"] / entry["row"]["audio_path"]
                futures.append(pool.submit(render_pair, paths, audio, args.mjcf, directory))
            for future in futures:
                future.result()
        write_json(
            args.output / "completion.json",
            {
                "status": "complete",
                "samples": len(entries),
                "videos": len(list((args.output / "samples").glob("*/*.mp4"))),
                "total_source_seconds": sum(r["row"]["num_frames"] / 30 for r in entries),
            },
        )
        return
    actor, config, summary = load_actor(args.checkpoint, args.stats, args.kinematics, "cuda")
    write_json(args.output / "checkpoint_capture.json", summary)
    first_dataset = dataset_for(args.data, entries[0], args.kinematics)
    batch = batch_to_device(collate_stage1_training_samples([first_dataset.get_window(0)]), "cuda")
    export_model(actor, batch, args.onnx, summary)
    runtime = Stage1OnnxSampler(args.onnx, actor)
    parity(actor, runtime, batch, args.output / "onnx_parity.json")
    if args.export_only:
        return
    losses = build_stage1_losses(actor, config).eval().cuda()
    write_json(
        args.output / "protocol.json",
        {
            "checkpoint": summary,
            "onnx": str(args.onnx),
            "backend": "ONNX Runtime CUDA FP32",
            "ddim_steps": 20,
            "cfg": 2.5,
            "seed": 42,
            "fixed_window": {
                "frames": 120,
                "prefix_frames": [0, 12],
                "history": "causal_demo_proxy",
            },
            "full_sequence": {
                "initial_demo_frames": 12,
                "overlap": 12,
                "future_gt": False,
                "history": "own_generated_kinematic_proxy",
                "full_source_length": True,
            },
            "scope": "offline_kinematic_no_gmt_no_dynamics",
            "selection_count": len(entries),
        },
    )
    datasets, records, futures = {}, [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for number, entry in enumerate(entries):
            start = time.monotonic()
            key = (entry["source"], entry["split"])
            if key not in datasets:
                datasets[key] = dataset_for(args.data, entry, args.kinematics)
            dataset = datasets[key]
            index = next(
                i
                for i, row in enumerate(dataset.rows)
                if row["sample_id"] == entry["row"]["sample_id"]
            )
            directory = args.output / "samples" / f"{number + 1:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            report_path = directory / "report.json"
            paths = {kind: directory / f"{kind}_display.pt" for kind in ("original", "generated")}
            if report_path.exists():
                record = json.loads(report_path.read_text())
            else:
                fixed = fixed_evaluation(actor, runtime, losses, dataset, index, 42)
                original, generated, contact, rollout = generate_full(
                    actor, runtime, dataset, index, 42
                )
                ground = rollout["ground_height_world_m"]
                original_metrics = metric_values(
                    original, actor.endecoder.kinematics, ground_height=ground
                )
                generated_metrics = metric_values(
                    generated,
                    actor.endecoder.kinematics,
                    pred_contact_logits=contact,
                    ground_height=ground,
                )
                for kind, qpos in (("original", original), ("generated", generated)):
                    payload = {
                        "qpos": qpos,
                        "fps": 30,
                        "robot_name": "bumi",
                        "quaternion_convention": "wxyz",
                        "qpos_order": "mujoco_native",
                        "joint_names": list(actor.endecoder.kinematics.joint_order),
                    }
                    torch.save(payload, directory / f"{kind}.pt")
                    display = qpos.clone()
                    display[:, 2] -= ground
                    torch.save(
                        {**payload, "qpos": display, "display_ground_translation_m": -ground},
                        paths[kind],
                    )
                record = {
                    "number": number + 1,
                    **entry,
                    "fixed_window": fixed,
                    "rollout": rollout,
                    "original_metrics": original_metrics,
                    "generated_metrics": generated_metrics,
                    "duration_seconds": len(original) / 30,
                    "backend": "ONNX Runtime CUDA FP32",
                }
                write_json(report_path, record)
            records.append(record)
            write_json(args.output / "results.json", records)
            audio = args.data / entry["source"] / entry["row"]["audio_path"]
            futures.append(pool.submit(render_pair, paths, audio, args.mjcf, directory))
            print(
                "GENERATED",
                number + 1,
                len(entries),
                entry["row"]["sample_id"],
                round(time.monotonic() - start, 2),
                "seconds",
                flush=True,
            )
        for future in futures:
            future.result()
    write_json(
        args.output / "completion.json",
        {
            "status": "complete",
            "samples": len(records),
            "videos": len(list((args.output / "samples").glob("*/*.mp4"))),
            "total_source_seconds": sum(r["duration_seconds"] for r in records),
        },
    )


if __name__ == "__main__":
    main()
