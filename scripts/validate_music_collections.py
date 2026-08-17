#!/usr/bin/env python3
"""Render a deterministic 20+20 full-length music-only validation suite."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
AIST_PREFIXES = ("mBR", "mHO", "mJB", "mJS", "mKR", "mLH", "mLO", "mMH", "mPO", "mWA")
GTZAN_GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def selections(aist_root: Path, gtzan_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for prefix in AIST_PREFIXES:
        for index in (0, 3):
            audio = aist_root / "wav" / f"{prefix}{index}.wav"
            items.append({"dataset": "AISTPP", "id": audio.stem, "audio": audio})
    for genre in GTZAN_GENRES:
        for index in (0, 50):
            audio = gtzan_root / genre / f"{genre}.{index:05d}.wav"
            items.append({"dataset": "GTZAN", "id": audio.stem, "audio": audio})
    missing = [str(item["audio"]) for item in items if not item["audio"].is_file()]
    if missing:
        raise FileNotFoundError("Selected audio files are missing:\n" + "\n".join(missing))
    return items


def completed_result(item: dict[str, Any], sample_dir: Path) -> dict[str, Any] | None:
    report_path = sample_dir / "demo_report.json"
    rendered = sample_dir / "motion_global.mp4"
    muxed = sample_dir / "motion_with_audio.mp4"
    if not (report_path.is_file() and rendered.is_file() and muxed.is_file()):
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("final_pass", False):
        return None
    return summarize(item, sample_dir, report, elapsed_seconds=0.0, reused=True)


def summarize(
    item: dict[str, Any],
    sample_dir: Path,
    report: dict[str, Any] | None,
    *,
    elapsed_seconds: float,
    reused: bool,
    returncode: int = 0,
) -> dict[str, Any]:
    audio_duration = media_duration(item["audio"])
    rendered = sample_dir / "motion_global.mp4"
    muxed = sample_dir / "motion_with_audio.mp4"
    rendered_duration = media_duration(rendered) if rendered.is_file() else None
    muxed_duration = media_duration(muxed) if muxed.is_file() else None
    frames = None
    final_pass = False
    sanity = None
    if report is not None:
        shape = report.get("generated_shape")
        frames = shape[1] if isinstance(shape, list) and len(shape) == 3 else None
        final_pass = bool(report.get("final_pass", False))
        sanity = report.get("motion_sanity")
    duration_error = (
        None if muxed_duration is None else abs(float(muxed_duration) - audio_duration)
    )
    status = (
        "passed"
        if returncode == 0
        and final_pass
        and rendered_duration is not None
        and muxed_duration is not None
        and duration_error is not None
        and duration_error <= 0.15
        else "failed"
    )
    return {
        "dataset": item["dataset"],
        "id": item["id"],
        "audio": str(item["audio"]),
        "output_dir": str(sample_dir),
        "status": status,
        "reused": reused,
        "returncode": returncode,
        "audio_duration_sec": audio_duration,
        "generated_frames": frames,
        "rendered_duration_sec": rendered_duration,
        "muxed_duration_sec": muxed_duration,
        "muxed_audio_duration_error_sec": duration_error,
        "elapsed_seconds": elapsed_seconds,
        "motion_sanity": sanity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aist-root", type=Path, default=Path("/home/weili/datasets/AISTPP_official/music")
    )
    parser.add_argument(
        "--gtzan-root", type=Path, default=Path("/home/weili/GTZAN/data/1/1/genres_original")
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/music_validation_s440000")
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.ckpt = args.ckpt.resolve()
    args.output_root = (REPO_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    items = selections(args.aist_root.resolve(), args.gtzan_root.resolve())
    for item in items:
        item["duration_sec"] = media_duration(item["audio"])
        item["max_frames"] = math.ceil(item["duration_sec"] * 30) + 2
    manifest = {
        "checkpoint": str(args.ckpt),
        "selection_policy": "AISTPP: indices 0 and 3 per prefix; GTZAN: indices 00000 and 00050 per genre",
        "render": {"width": args.width, "height": args.height, "fps": 30},
        "generation": {
            "full_audio": True,
            "ddim_steps": args.ddim_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "postproc": True,
        },
        "items": [
            {key: str(value) if isinstance(value, Path) else value for key, value in item.items()}
            for item in items
        ],
    }
    atomic_json(args.output_root / "selection.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.device
    results: list[dict[str, Any]] = []
    for number, item in enumerate(items, start=1):
        sample_dir = args.output_root / item["dataset"] / item["id"]
        existing = completed_result(item, sample_dir)
        if existing is not None:
            results.append(existing)
            print(f"[{number:02d}/40] reuse {item['dataset']}/{item['id']}", flush=True)
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/demo_music_only.py"),
            "--audio",
            str(item["audio"]),
            "--ckpt",
            str(args.ckpt),
            "--output-dir",
            str(sample_dir),
            "--max-frames",
            str(item["max_frames"]),
            "--cfg-scale",
            str(args.cfg_scale),
            "--ddim-steps",
            str(args.ddim_steps),
            "--seed",
            str(args.seed),
            "--postproc",
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]
        print(
            f"[{number:02d}/40] start {item['dataset']}/{item['id']} "
            f"({item['duration_sec']:.3f}s, max_frames={item['max_frames']})",
            flush=True,
        )
        started = time.monotonic()
        with (sample_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.monotonic() - started
        report_path = sample_dir / "demo_report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else None
        )
        try:
            result = summarize(
                item,
                sample_dir,
                report,
                elapsed_seconds=elapsed,
                reused=False,
                returncode=process.returncode,
            )
        except Exception as exc:
            result = {
                "dataset": item["dataset"],
                "id": item["id"],
                "audio": str(item["audio"]),
                "output_dir": str(sample_dir),
                "status": "failed",
                "reused": False,
                "returncode": process.returncode,
                "elapsed_seconds": elapsed,
                "summary_error": repr(exc),
            }
        results.append(result)
        atomic_json(args.output_root / "batch_summary.json", {"results": results})
        print(
            f"[{number:02d}/40] {result['status']} {item['dataset']}/{item['id']} "
            f"in {elapsed:.1f}s",
            flush=True,
        )

    passed = sum(result["status"] == "passed" for result in results)
    summary = {
        "checkpoint": str(args.ckpt),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    atomic_json(args.output_root / "batch_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("total", "passed", "failed")}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
