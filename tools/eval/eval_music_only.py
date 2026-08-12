#!/usr/bin/env python3
"""Export deterministic AIST++ music-only generations without claiming paper FID/BAS."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datamodule.mocap_trainX_testY import collate_fn  # noqa: E402
from gem.datasets.aistpp.aistplusplus import AISTPlusPlusSmplDataset  # noqa: E402
from gem.utils.net_utils import load_pretrained_model, to_cuda  # noqa: E402

PAPER_METRIC_NOTICE = (
    "Official AIST++ FIDk/FIDm/BAS evaluator is not bundled; generation export "
    "is available but paper-equivalent metrics are not claimed."
)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    return value


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("inputs/AIST++"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("music-only evaluation requires a CUDA GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(config_name="train", overrides=["exp=gem_smpl_music_only"])
    if list(cfg.pipeline.args.in_attr) != ["encoded_music"]:
        raise RuntimeError(f"unexpected specialist conditions: {list(cfg.pipeline.args.in_attr)}")

    dataset = AISTPlusPlusSmplDataset(
        root=args.root,
        split=args.split,
        feat_version="v2",
        eval_gen_only=True,
        strict_music_alignment=True,
        max_music_motion_frame_mismatch=2,
        load_raw_music_audio=False,
        eval_motion_frames=120,
        eval_clip_mode="center",
        music_only_conditioning=True,
    )
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.ckpt)
    model = model.cuda().eval()
    # Calling ``validation`` outside Lightning would otherwise fall back to the
    # test stage and silently enable foot-lock post-processing.  Evaluation
    # exports the specialist's direct generation, matching validation.
    model._trainer = SimpleNamespace(
        state=SimpleNamespace(stage="validate"), global_step=0
    )
    if model.text_condition_enabled:
        raise RuntimeError("music-only evaluator unexpectedly enabled text conditioning")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_reports = []
    requested = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for index in range(requested):
            item = dataset[index]
            batch = collate_fn([item], mode="val", collate_cfg=cfg.data.collate_cfg)
            batch = to_cuda(batch)
            source_motion = model.endecoder.encode(batch).detach().cpu()[0]
            outputs = model.validation(batch, "default", index, 0)
            generated = outputs["model_output"]["pred_x"].detach().cpu()[0]
            finite = bool(torch.isfinite(generated).all())
            if generated.shape != source_motion.shape:
                raise RuntimeError(
                    f"generated/source shape mismatch for {item['meta']['vid']}: "
                    f"{tuple(generated.shape)} != {tuple(source_motion.shape)}"
                )
            if not finite:
                raise RuntimeError(f"non-finite generation for {item['meta']['vid']}")

            sequence = item["meta"]["vid"]
            sample_dir = args.output_dir / f"{index:04d}_{sequence}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            torch.save(generated, sample_dir / "generated_motion_151d.pt")
            torch.save(source_motion, sample_dir / "source_motion_151d.pt")
            torch.save(
                _cpu_tree(outputs.get("pred_body_params_global", {})),
                sample_dir / "pred_body_params_global.pt",
            )
            metadata = {
                "index": index,
                "source_sequence": sequence,
                "split": args.split,
                "music_feature_path": item["meta"]["music_feature_path"],
                "music_motion_alignment": item["meta"]["music_motion_alignment"],
                "source_frame_range": list(item["meta"]["start_end"]),
                "generated_length": int(generated.shape[0]),
                "motion_dim": int(generated.shape[1]),
                "finite": finite,
                "generated_statistics": _tensor_stats(generated),
                "source_statistics": _tensor_stats(source_motion),
                "seed": args.seed,
                "condition_list": list(cfg.pipeline.args.in_attr),
                "paper_equivalent_metrics_claimed": False,
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            sample_reports.append(metadata)
            print(f"[{index + 1}/{requested}] {sequence}: {tuple(generated.shape)}, finite={finite}")

    summary = {
        "status": "passed",
        "checkpoint": str(args.ckpt.resolve()),
        "split": args.split,
        "num_samples": requested,
        "seed": args.seed,
        "condition_list": list(cfg.pipeline.args.in_attr),
        "all_finite": all(item["finite"] for item in sample_reports),
        "paper_metric_notice": PAPER_METRIC_NOTICE,
        "samples": sample_reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(PAPER_METRIC_NOTICE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
