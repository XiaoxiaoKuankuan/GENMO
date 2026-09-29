#!/usr/bin/env python3
"""固定四库各八条训练与两条留出样本，并生成可审计的传输清单。

读取服务器原始完整 train/val/test manifests，以重划分关联组为单位去重，组内取最长
动作，避免短舞蹈与完整音乐混淆。按固定种子选样，不查看模型输出；test 优先选择旧划分
也未训练的样本，但不会把预训练见过的样本标成严格未见。完整清单原样保留，单独写出
验证子集 manifests 和 meta。文件列表供 rsync 从服务器按需复制音乐、特征与动作。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

SOURCES = ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine")


def prepare(metadata: Path, output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    selections, files, manifest_hashes = [], set(), {}
    for source in SOURCES:
        shutil.copytree(metadata / source / "meta", output / source / "meta", dirs_exist_ok=True)
        original = {}
        for split in ("train", "val", "test"):
            path = metadata / source / "manifests" / f"{split}.jsonl"
            original[split] = [json.loads(line) for line in path.read_text().splitlines()]
            manifest_hashes[f"{source}/{split}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = {
            s: {r["resplit_provenance"]["group_id"] for r in rows} for s, rows in original.items()
        }
        assert not groups["train"] & (groups["val"] | groups["test"])
        for split, count in (("train", 8), ("test", 2)):
            representatives = {}
            for index, row in enumerate(original[split]):
                group = row["resplit_provenance"]["group_id"]
                old = representatives.get(group)
                if old is None or row["num_frames"] > old[1]["num_frames"]:
                    representatives[group] = (index, row)
            candidates = sorted(representatives.values(), key=lambda pair: pair[1]["sample_id"])
            random.Random(f"{seed}/{source}/{split}").shuffle(candidates)
            if split == "test":
                candidates.sort(
                    key=lambda pair: pair[1]["resplit_provenance"]["original_split"] == "train"
                )
            if len(candidates) < count:
                raise ValueError(f"{source}/{split}: 独立组不足 {count}")
            chosen = candidates[:count]
            manifest = output / source / "manifests" / f"{split}.jsonl"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row in chosen)
            )
            for index, row in chosen:
                for key in ("motion_path", "music_feature_path", "audio_path"):
                    files.add(f"{source}/{row[key]}")
                selections.append(
                    {
                        "source": source,
                        "split": split,
                        "source_row_index": index,
                        "pretraining_split": row["resplit_provenance"]["original_split"],
                        "stage1_train_group_overlap": False if split == "test" else True,
                        "pretraining_unseen_claim": False,
                        "row": row,
                    }
                )
    payload = {
        "seed": seed,
        "policy": "group_unique_longest_then_seeded_no_model_cherry_picking",
        "full_manifest_sha256": manifest_hashes,
        "samples": selections,
    }
    (output.parent / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (output.parent / "transfer_files.txt").write_text("\n".join(sorted(files)) + "\n")
    print(
        json.dumps(
            {
                "selected": len(selections),
                "transfer_files": len(files),
                "heldout_old_splits": [
                    (r["source"], r["row"]["sample_id"], r["pretraining_split"])
                    for r in selections
                    if r["split"] == "test"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    prepare(args.metadata, args.output, args.seed)
