#!/usr/bin/env python3
"""Stage deterministic held-out SMPL/audio clips for four-set comparison."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


DATASET_ROOT = Path("/data0/user/liwei/datasets")
REVIEW_ROOT = DATASET_ROOT / "music_dance_review/music_only_4set_v1"


def read_master() -> list[dict[str, Any]]:
    path = REVIEW_ROOT / "index/master.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def one_per_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    selected = []
    for group_id in sorted(grouped):
        choices = sorted(
            grouped[group_id],
            key=lambda row: (
                row.get("person_id") not in (None, 0),
                row.get("role") not in (None, "leader"),
                row["sample_id"],
            ),
        )
        selected.append(choices[0])
    return selected


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aist_test = [
        row for row in rows if row["dataset"] == "aistpp" and row["split"] == "test"
    ]
    by_music: dict[str, list[dict[str, Any]]] = {}
    for row in aist_test:
        by_music.setdefault(row["music_id"], []).append(row)
    aist = [
        sorted(by_music[music_id], key=lambda row: (-row["num_frames"], row["sample_id"]))[0]
        for music_id in sorted(by_music)
    ]
    if len(aist) != 10:
        raise RuntimeError(f"expected 10 AIST++ test music IDs, got {len(aist)}")

    def shuffled(dataset: str, seed: int) -> list[dict[str, Any]]:
        candidates = one_per_group(
            [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["split"] == "test"
                and row["duration_sec"] >= 20
            ]
        )
        random.Random(seed).shuffle(candidates)
        if len(candidates) < 10:
            raise RuntimeError(f"{dataset}: fewer than 10 eligible test groups")
        return candidates[:10]

    aioz = shuffled("aioz_gdance", 42)
    finedance = shuffled("finedance", 43)

    compas_test = one_per_group(
        [
            row
            for row in rows
            if row["dataset"] == "compas3d"
            and row["split"] == "test"
            and row.get("role") == "leader"
        ]
    )
    compas_val = one_per_group(
        [
            row
            for row in rows
            if row["dataset"] == "compas3d"
            and row["split"] == "val"
            and row.get("role") == "leader"
        ]
    )
    random.Random(44).shuffle(compas_test)
    random.Random(45).shuffle(compas_val)
    if len(compas_test) != 8 or len(compas_val) < 2:
        raise RuntimeError(
            f"unexpected CoMPAS3D held-out groups: test={len(compas_test)}, val={len(compas_val)}"
        )
    return aist + aioz + finedance + compas_test + compas_val[:2]


def audio_source(row: dict[str, Any]) -> Path | None:
    dataset = row["dataset"]
    if dataset == "aistpp":
        return None
    if dataset == "aioz_gdance":
        return (
            DATASET_ROOT
            / "music_dance_raw/AIOZ-GDANCE/extracted/musics"
            / f"{row['group_id']}.wav"
        )
    if dataset == "finedance":
        return (
            DATASET_ROOT
            / "music_dance_raw/FineDance/raw/finedance/music_wav"
            / f"{row['sample_id']}.wav"
        )
    if dataset == "compas3d":
        return (
            DATASET_ROOT
            / "music_dance_genmo/CoMPAS3D/audio_cache"
            / f"{row['group_id']}.wav"
        )
    raise ValueError(dataset)


def stage(output: Path, *, finedance_full: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    selected = select_rows(read_master())
    manifest: list[dict[str, Any]] = []
    for number, row in enumerate(selected, start=1):
        dataset = row["dataset"]
        use_full_sequence = finedance_full and dataset == "finedance"
        frames = (
            int(row["num_frames"])
            if use_full_sequence
            else min(int(row["num_frames"]), 600)
        )
        sample_id = row["sample_id"]
        source_motion = REVIEW_ROOT / row["review_motion_path"]
        motion_relative = Path("motion") / dataset / f"{sample_id}.npz"
        motion_output = output / motion_relative
        motion_output.parent.mkdir(parents=True, exist_ok=True)
        with np.load(source_motion, allow_pickle=False) as source:
            np.savez_compressed(
                motion_output,
                pose=np.ascontiguousarray(source["pose"][:frames], dtype=np.float32),
                transl=np.ascontiguousarray(source["transl"][:frames], dtype=np.float32),
                betas=np.ascontiguousarray(source["betas"][:frames], dtype=np.float32),
                fps=np.asarray(30.0, dtype=np.float32),
            )

        source_audio = audio_source(row)
        audio_relative: str | None = None
        aist_audio_hint: str | None = None
        if source_audio is None:
            aist_audio_hint = (
                f"/home/weili/datasets/AISTPP_official/music/wav/{row['music_id']}.wav"
            )
        else:
            if not source_audio.is_file():
                raise FileNotFoundError(source_audio)
            audio_path = Path("audio") / dataset / f"{sample_id}.flac"
            audio_output = output / audio_path
            audio_output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source_audio),
                    "-t",
                    f"{frames / 30.0:.9f}",
                    "-vn",
                    "-c:a",
                    "flac",
                    str(audio_output),
                ],
                check=True,
            )
            audio_relative = str(audio_path)

        manifest.append(
            {
                "number": number,
                "dataset": dataset,
                "sample_id": sample_id,
                "group_id": row["group_id"],
                "split": row["split"],
                "music_id": row.get("music_id"),
                "clip_frames": frames,
                "clip_duration_sec": frames / 30.0,
                "clip_mode": "full_sequence" if use_full_sequence else "max_20_seconds",
                "source_num_frames": int(row["num_frames"]),
                "motion": str(motion_relative),
                "audio": audio_relative,
                "aist_audio_hint": aist_audio_hint,
                "review_sha256": row["review_sha256"],
            }
        )
        print(
            f"[{number:02d}/40] {dataset}/{sample_id} split={row['split']} "
            f"frames={frames}",
            flush=True,
        )
    (output / "manifest.json").write_text(
        json.dumps({"items": manifest}, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--finedance-full",
        action="store_true",
        help=(
            "Stage each selected FineDance sample at its full canonical motion/music "
            "length instead of the default 600-frame comparison clip. The selected "
            "test samples contain up to 5801 frames and are expensive to infer/render."
        ),
    )
    args = parser.parse_args()
    stage(args.output.resolve(), finedance_full=args.finedance_full)


if __name__ == "__main__":
    main()
