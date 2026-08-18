#!/usr/bin/env python3
"""Create deterministic selected WAV/EDGE35 file lists for BUMI conversion."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.bumi.build_bumi_music_dataset import (  # noqa: E402
    DATASET_SPECS,
    _mapping,
    _parse_mapping,
    _read_jsonl,
    _sample_basename,
    load_human_indices,
    pairing_fields,
)


def _audio_key_from_selected_id(dataset: str, sample_id: str) -> str:
    if dataset == "aistpp":
        fields = sample_id.split("_")
        if len(fields) < 5 or not fields[4].startswith("m"):
            raise ValueError(f"{sample_id}: invalid AIST++ music token")
        return fields[4]
    if dataset == "aioz_gdance":
        return re.sub(r"_dancer_\d+$", "", sample_id)
    if dataset == "finedance":
        return sample_id
    if dataset == "compas3d":
        return re.sub(r"_(leader|follower)$", "", sample_id)
    raise ValueError(dataset)


def build_filelists(
    selected_root: Path, human_roots: dict[str, Path] | None, output: Path
) -> dict:
    selected_root = selected_root.expanduser().resolve()
    output = output.expanduser().resolve()
    indices = None if human_roots is None else load_human_indices(human_roots)
    audio: dict[str, set[str]] = defaultdict(set)
    features: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, int] = defaultdict(int)
    for row in _read_jsonl(selected_root / "manifests" / "selected.jsonl"):
        dataset, sample_id = _sample_basename(row)
        if row.get("quality_accepted") is not True or row.get("quality_status") != "PASS":
            raise ValueError(f"{dataset}/{sample_id}: selected row is not PASS")
        audio[dataset].add(f"{_audio_key_from_selected_id(dataset, sample_id)}.wav")
        if indices is not None:
            human = indices[dataset].get(sample_id)
            if human is None:
                raise ValueError(f"{dataset}/{sample_id}: missing exact human source row")
            pair = pairing_fields(dataset, sample_id, human)
            if pair["audio_key"] != _audio_key_from_selected_id(dataset, sample_id):
                raise ValueError(f"{dataset}/{sample_id}: filename/human audio pairing differs")
            feature_relative = str(human.get("music_feature_path", ""))
            if not feature_relative:
                raise ValueError(f"{dataset}/{sample_id}: missing music_feature_path")
            features[dataset].add(feature_relative)
        samples[dataset] += 1
    output.mkdir(parents=True, exist_ok=True)
    report = {"contract_version": "genmo.bumi_transfer_plan.v1", "datasets": {}}
    for dataset in DATASET_SPECS:
        audio_values = sorted(audio[dataset])
        feature_values = sorted(features[dataset])
        (output / f"{dataset}_audio.txt").write_text(
            "".join(f"{value}\n" for value in audio_values), encoding="utf-8"
        )
        if indices is not None:
            (output / f"{dataset}_edge35.txt").write_text(
                "".join(f"{value}\n" for value in feature_values), encoding="utf-8"
            )
        report["datasets"][dataset] = {
            "selected_samples": samples[dataset],
            "unique_audio": len(audio_values),
            "unique_edge35": len(feature_values),
            "audio_filelist": f"{dataset}_audio.txt",
            "edge35_filelist": (
                f"{dataset}_edge35.txt" if indices is not None else None
            ),
        }
    (output / "transfer_plan.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-root", required=True, type=Path)
    parser.add_argument("--human-root", action="append", type=_parse_mapping)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_filelists(
        args.selected_root,
        None if not args.human_root else _mapping(args.human_root, "--human-root"),
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
