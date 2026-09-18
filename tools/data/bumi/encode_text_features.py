#!/usr/bin/env python3
"""为转换清单中尚无特征的caption生成本地T5-3B 150-token特征。

只处理embeddings为空的记录；已有MotionMillion引用不重算，旧50-token不能伪装为150。
输出新的清单和分片，不改原清单/数据。真实T5/GPU执行需另行授权；本轮仅实现入口。
原文本逐字保留，编码时沿用GENMO去除首尾空白规则，索引与caption SHA同时记录。
"""

from pathlib import Path
import argparse
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def encode(source, output, t5_model, device="cuda:0", records_per_shard=128):
    import torch
    from gem.runtime.bumi_text_runtime import ResidentBumiTextEngine
    from gem.runtime.bumi_text_contract import sha256_file
    from gem.datasets.pure_motion.bumi_text import caption_hash, resolve_reference
    from tools.data.bumi.prepare_bumi_text import write_json

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or records_per_shard < 1:
        raise ValueError("需要新的输出目录和正records_per_shard")
    payload = copy.deepcopy(json.loads(source.read_text()))
    if payload["schema"] != "genmo.bumi_text_conversion.v1":
        raise ValueError("需要conversion.v1")
    payload["kinematics"]["path"] = str(
        resolve_reference(payload["kinematics"]["path"], source.parent)
    )
    for record in payload["records"]:
        record["qpos_path"] = str(resolve_reference(record["qpos_path"], source.parent))
        for ref in record.get("embeddings", []):
            for key in ("path", "motion_manifest", "embedding_manifest"):
                if key in ref:
                    ref[key] = str(resolve_reference(ref[key], source.parent))
    output.mkdir(parents=True)
    engine = ResidentBumiTextEngine(t5_model=t5_model, device=device)
    pending, records = [], []
    shard_id = 0

    def flush():
        nonlocal shard_id
        if not pending:
            return
        path = output / f"t5_{shard_id:05d}.pt"
        torch.save(records, path)
        digest = sha256_file(path)
        for rid, record in enumerate(pending):
            record["embeddings"] = [
                dict(
                    format="bumi_text_t5_v1",
                    path=str(path),
                    sha256=digest,
                    record_index=rid,
                    text_index=tid,
                    motion_id=record.get("text_source_motion_id", record["motion_id"]),
                    caption_sha256=caption_hash(c),
                )
                for tid, c in enumerate(record["captions"])
            ]
        pending.clear()
        records.clear()
        shard_id += 1

    try:
        for record in payload["records"]:
            if record.get("embeddings"):
                continue
            if not record["captions"] or any(not c.strip() for c in record["captions"]):
                raise ValueError("caption为空")
            encoded = [engine.encode_prompt(c) for c in record["captions"]]
            records.append(
                dict(
                    motion_id=record.get("text_source_motion_id", record["motion_id"]),
                    captions=record["captions"],
                    encoder="t5-3b",
                    max_text_len=150,
                    t5_model=str(Path(t5_model).resolve()),
                    embeddings=torch.cat([e.cpu().half() for e, _ in encoded]),
                    attention_mask=torch.cat([m.cpu() for _, m in encoded]),
                )
            )
            pending.append(record)
            if len(pending) >= records_per_shard:
                flush()
        flush()
        payload["feature_source_conversion_sha256"] = sha256_file(source)
        write_json(output / "conversion.json", payload)
    finally:
        engine.close()
    return output / "conversion.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--t5-model", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--records-per-shard", type=int, default=128)
    a = p.parse_args()
    print(encode(a.source, a.output, a.t5_model, a.device, a.records_per_shard))
