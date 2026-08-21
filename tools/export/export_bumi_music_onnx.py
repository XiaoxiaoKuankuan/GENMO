#!/usr/bin/env python3
"""Export one BUMI-native music CFG denoiser step to fixed-shape ONNX.

The graph is fixed to batch=1 and a selected sequence length (normally 120).
It outputs normalized motion93 only.  DDIM iteration and authoritative
motion93 -> qpos28 -> Torch FK decoding remain in the repository runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.runtime.bumi_music_onnx import (  # noqa: E402
    BUMI_ONNX_CONTRACT_VERSION,
    BumiMusicGuidedDenoiser,
    make_bumi_onnx_inputs,
    tensor_statistics,
    validate_bumi_checkpoint_state_dict,
)
from gem.utils.net_utils import load_pretrained_model  # noqa: E402

DEFAULT_OUTPUT = Path("outputs/onnx/bumi_music/bumi_music_denoiser_t120.onnx")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def configure_assets(kinematics: Path | None, stats: Path | None) -> tuple[Path, Path]:
    if kinematics is not None:
        os.environ["BUMI_KINEMATICS_PATH"] = str(kinematics.expanduser().resolve())
    if stats is not None:
        os.environ["BUMI_MUSIC_STATS_PATH"] = str(stats.expanduser().resolve())
    values = []
    for name in ("BUMI_KINEMATICS_PATH", "BUMI_MUSIC_STATS_PATH"):
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"{name} is required; export it or pass the matching CLI option")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        values.append(path)
    return values[0], values[1]


def load_export_music(path: Path | None, seq_len: int) -> torch.Tensor:
    if path is None:
        time = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        channel = torch.arange(35, dtype=torch.float32).unsqueeze(0)
        music = torch.sin(time * 0.07 + channel * 0.11) * 0.1
        music[:, 33:35] = 0.0
        return music
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"--music-embed must contain Tensor[T,35]: {path}")
    music = payload.detach().cpu().float()
    if music.ndim != 2 or music.shape[1] != 35 or music.shape[0] < seq_len:
        raise ValueError(
            f"--music-embed must be finite [T,35] with T >= {seq_len}; got {music.shape}"
        )
    if not bool(torch.isfinite(music).all()):
        raise ValueError(f"--music-embed contains NaN or Inf: {path}")
    return music[:seq_len].contiguous()


def checkpoint_summary(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    try:
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint has no state_dict: {path}")
        architecture = validate_bumi_checkpoint_state_dict(state)
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "global_step": checkpoint.get("global_step") if isinstance(checkpoint, dict) else None,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "state_dict_keys": len(state),
            **architecture,
        }
    finally:
        del checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exp", default="gem_bumi_music_only_4set_random_v1")
    parser.add_argument("--kinematics", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--music-embed", type=Path)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.ckpt.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not 1 <= args.seq_len <= 120:
        raise ValueError("--seq-len must be in 1..120 for the trained BUMI model")
    if args.opset < 17:
        raise ValueError("--opset must be >= 17")
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires the 'onnx' Python package") from exc

    kinematics_path, stats_path = configure_assets(args.kinematics, args.stats)
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    candidates = [output, output.with_name(output.name + ".data"), metadata_path]
    existing = [path for path in candidates if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist: {existing}; pass --overwrite")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    music = load_export_music(args.music_embed, args.seq_len)
    ckpt_info = checkpoint_summary(checkpoint)
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    print(f"[BUMI ONNX] instantiate {args.exp}; checkpoint={checkpoint}")
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, checkpoint)
    model = model.eval().to(device)
    wrapper = BumiMusicGuidedDenoiser(model).eval().to(device)
    inputs = make_bumi_onnx_inputs(
        music,
        seed=args.seed,
        timestep=args.timestep,
        guidance_scale=args.cfg_scale,
        device=device,
    )
    reference = wrapper(*inputs)
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError("PyTorch BUMI ONNX reference contains NaN or Inf")
    print(f"[BUMI ONNX] reference: {tensor_statistics(reference)}")

    input_names = [
        "noisy_motion",
        "diffusion_timestep",
        "music",
        "length",
        "guidance_scale",
    ]
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["pred_motion"],
        dynamo=False,
        external_data=True,
    )
    onnx.checker.check_model(str(output), full_check=False)

    artifacts = sorted(
        path
        for path in output.parent.glob(output.name + "*")
        if path.is_file() and path != metadata_path
    )
    metadata = {
        "contract_version": BUMI_ONNX_CONTRACT_VERSION,
        "representation_contract_version": model.endecoder.representation_contract_version,
        "checkpoint": ckpt_info,
        "experiment": args.exp,
        "sequence_length": args.seq_len,
        "fixed_batch": 1,
        "cfg_internal_batch": 2,
        "opset": args.opset,
        "input_contract": {
            "noisy_motion": [1, args.seq_len, 93],
            "diffusion_timestep": [1],
            "music": [1, args.seq_len, 35],
            "length": [1],
            "guidance_scale": [1],
        },
        "output_contract": {"pred_motion": [1, args.seq_len, 93]},
        "kinematics": {
            "path": str(kinematics_path),
            "sha256": sha256_file(kinematics_path),
        },
        "stats": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
        "export_boundary": (
            "EDGE35 embedding + batched conditional/unconditional CFG + one normalized "
            "93-D x-start denoiser step. Python runtime owns DDIM and qpos/FK decoding."
        ),
        "pytorch_reference": tensor_statistics(reference),
        "artifacts": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[BUMI ONNX] exported: {output}")
    print(f"[BUMI ONNX] metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
