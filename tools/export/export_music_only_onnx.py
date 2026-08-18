#!/usr/bin/env python3
"""Export the AIST++ music-only GEM neural denoising step to ONNX.

This intentionally exports one CFG-guided diffusion network step, not an unrolled
50-step DDIM loop.  See ``docs/MUSIC_ONLY_ONNX.md`` for the exact boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.runtime.music_only_onnx import (  # noqa: E402
    MusicOnlyGuidedDenoiser,
    MusicOnlyTensorRTDenoiser,
    make_onnx_inputs,
    output_statistics,
)
from gem.utils.net_utils import load_pretrained_model  # noqa: E402

DEFAULT_CHECKPOINT = Path("inputs/checkpoints/music_only_aistpp/version_1/last.ckpt")
DEFAULT_OUTPUT = Path("outputs/onnx/music_only_aistpp_s260000/music_only_denoiser.onnx")


def _sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _load_export_music(path: Path | None, seq_len: int) -> torch.Tensor:
    if path is None:
        # A deterministic non-zero signal keeps every intended music operation in the trace.
        time = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        channel = torch.arange(35, dtype=torch.float32).unsqueeze(0)
        music = torch.sin(time * 0.07 + channel * 0.11) * 0.1
        music[:, 33:35] = 0
        return music
    music = load_music_feature_tensor(path)
    validate_musicfeat_v2(music, source=path)
    if music.shape[0] < seq_len:
        raise ValueError(
            f"--music-embed has {music.shape[0]} frames, fewer than --seq-len={seq_len}"
        )
    return music[:seq_len].contiguous()


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    try:
        state_dict = checkpoint.get("state_dict", {})
        music_weights = [
            [key, list(value.shape)]
            for key, value in state_dict.items()
            if key.endswith("music_embedder.fc1.weight")
        ]
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "global_step": checkpoint.get("global_step"),
            "epoch": checkpoint.get("epoch"),
            "state_dict_keys": len(state_dict),
            "music_weights": music_weights,
        }
    finally:
        del checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp", default="gem_smpl_music_only")
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--music-embed", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trt-deployment",
        action="store_true",
        help=(
            "Export the fixed-shape, batched-CFG, pred_motion-only graph used by "
            "the physical robot TensorRT runtime. The legacy export stays unchanged "
            "when this flag is absent."
        ),
    )
    return parser


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not 1 <= args.seq_len <= 120:
        raise ValueError("--seq-len must be in 1..120 for the trained specialist")
    if args.opset < 17:
        raise ValueError("--opset must be >= 17")
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package; install it in the GENMO venv"
        ) from exc

    output = args.output.resolve()
    output_data = output.with_name(output.name + ".data")
    metadata_path = output.with_suffix(output.suffix + ".json")
    existing = [path for path in (output, output_data, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"ONNX output already exists: {existing}; pass --overwrite to replace it"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    music = _load_export_music(args.music_embed, args.seq_len)
    checkpoint_summary = _checkpoint_summary(args.ckpt)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    if list(cfg.pipeline.args.in_attr) != ["encoded_music"]:
        raise RuntimeError(
            f"experiment '{args.exp}' is not strict music-only: {list(cfg.pipeline.args.in_attr)}"
        )
    print(f"[ONNX] 实例化 {args.exp}，加载 checkpoint: {args.ckpt}")
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.ckpt)
    model = model.eval().to(device)
    wrapper_cls = MusicOnlyTensorRTDenoiser if args.trt_deployment else MusicOnlyGuidedDenoiser
    wrapper = wrapper_cls(model).eval().to(device)
    inputs = make_onnx_inputs(
        music,
        seed=args.seed,
        timestep=args.timestep,
        guidance_scale=args.cfg_scale,
        device=device,
    )
    reference_raw = wrapper(*inputs)
    reference = (reference_raw,) if isinstance(reference_raw, torch.Tensor) else reference_raw
    if not all(torch.isfinite(value).all() for value in reference):
        raise RuntimeError("PyTorch reference forward contains NaN or Inf")
    print(f"[ONNX] PyTorch 单步前向通过: {output_statistics(reference)}")

    input_names = [
        "noisy_motion",
        "diffusion_timestep",
        "music",
        "length",
        "guidance_scale",
    ]
    output_names = (
        ["pred_motion"]
        if args.trt_deployment
        else ["pred_motion", "pred_camera", "static_conf_logits"]
    )
    dynamic_axes = (
        None
        if args.trt_deployment
        else {name: {0: "batch"} for name in input_names[:-1] + output_names}
    )
    batch_description = "固定 batch=1" if args.trt_deployment else "动态 batch"
    print(
        f"[ONNX] 开始导出固定 T={args.seq_len}、{batch_description}的 opset {args.opset} 图"
    )
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        external_data=True,
    )
    onnx.checker.check_model(str(output), full_check=False)

    artifacts = [path for path in (output, output_data) if path.is_file()]
    metadata = {
        "format": (
            "gem_music_only_trt_denoiser_step_v2"
            if args.trt_deployment
            else "gem_music_only_guided_denoiser_step_v1"
        ),
        "checkpoint": checkpoint_summary,
        "experiment": args.exp,
        "condition_list": list(cfg.pipeline.args.in_attr),
        "sequence_length": args.seq_len,
        "opset": args.opset,
        "fixed_sequence_length": True,
        "dynamic_batch": not args.trt_deployment,
        "input_contract": (
            {
                "noisy_motion": [1, args.seq_len, 151],
                "diffusion_timestep": [1],
                "music": [1, args.seq_len, 35],
                "length": [1],
                "guidance_scale": [1],
            }
            if args.trt_deployment
            else {
                "noisy_motion": ["B", args.seq_len, 151],
                "diffusion_timestep": ["B"],
                "music": ["B", args.seq_len, 35],
                "length": ["B"],
                "guidance_scale": [1],
            }
        ),
        "output_contract": (
            {"pred_motion": [1, args.seq_len, 151]}
            if args.trt_deployment
            else {
                "pred_motion": ["B", args.seq_len, 151],
                "pred_camera": ["B", args.seq_len, 3],
                "static_conf_logits": ["B", args.seq_len, 6],
            }
        ),
        "export_boundary": (
            "EDGE35 embedding + conditional/unconditional CFG + one x_start denoiser step; "
            "DDIM iteration and EnDecoder/SMPL decoding remain outside ONNX"
        ),
        "deployment": {
            "tensorrt": bool(args.trt_deployment),
            "cfg_internal_batch": 2 if args.trt_deployment else None,
            "returns_pred_motion_only": bool(args.trt_deployment),
        },
        "pytorch_reference": output_statistics(reference),
        "artifacts": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[ONNX] 导出完成: {output}")
    print(f"[ONNX] 合约与校验信息: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
