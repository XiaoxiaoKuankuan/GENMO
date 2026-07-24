from __future__ import annotations

import argparse
import json

import pytest
import torch

from tools.data.motionxpp.common import atomic_write_jsonl
from tools.data.motionxpp.extract_t5_embeddings import (
    HIDDEN_DIM,
    MAX_TEXT_LEN,
    _load_caption_records,
    encode_caption_records,
    extract_embeddings,
)


def test_dummy_encoder_preserves_caption_ownership():
    captions = {
        "motion_a": ["First.", "Second."],
        "motion_b": ["Third."],
    }
    calls = []

    def encode_batch(values):
        calls.append(list(values))
        base = sum(len(item) for item in calls[:-1])
        output = torch.zeros(len(values), MAX_TEXT_LEN, HIDDEN_DIM)
        for index in range(len(values)):
            output[index].fill_(base + index + 1)
        return output

    result = encode_caption_records(captions, encode_batch=encode_batch, batch_size=2)
    assert result["motion_a"].shape == (2, 50, 1024)
    assert result["motion_b"].shape == (1, 50, 1024)
    assert result["motion_a"].dtype == torch.float16
    assert torch.all(result["motion_a"][0] == 1)
    assert torch.all(result["motion_a"][1] == 2)
    assert torch.all(result["motion_b"][0] == 3)


def _source_artifacts(tmp_path):
    source = tmp_path / "genmo_support"
    shard_rel = "shards/train/motion.pth"
    shard_path = source / shard_rel
    shard_path.parent.mkdir(parents=True)
    records = {
        "a": {
            "text_data": [{"caption": "A."}, {"caption": "A second."}],
        },
        "b": {"text_data": [{"caption": "B."}]},
    }
    torch.save(records, shard_path)
    manifest = source / "manifests/train.jsonl"
    atomic_write_jsonl(
        manifest,
        [
            {
                "motion_id": key,
                "shard_path": shard_rel,
                "record_key": key,
                "caption_count": len(value["text_data"]),
            }
            for key, value in records.items()
        ],
    )
    return source, manifest


def test_caption_loader_reads_one_motion_shard_and_checks_counts(tmp_path):
    source, manifest = _source_artifacts(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    captions, invalid = _load_caption_records(rows, source, strict=True)
    assert invalid == []
    assert captions == {"a": ["A.", "A second."], "b": ["B."]}
    rows[0]["caption_count"] = 99
    with pytest.raises(Exception, match="caption count mismatch"):
        _load_caption_records(rows, source, strict=True)


def test_full_sharded_extraction_resume_with_dummy_t5(tmp_path, monkeypatch):
    _, manifest = _source_artifacts(tmp_path)
    output = tmp_path / "embeddings"

    class DummyEncoder:
        pass

    monkeypatch.setattr(
        "tools.data.motionxpp.extract_t5_embeddings._load_t5",
        lambda *args, **kwargs: (DummyEncoder(), object()),
    )

    def fake_encode_text_batch(*, raw_text, **kwargs):
        return torch.ones(len(raw_text), 50, 1024, dtype=torch.float32)

    monkeypatch.setattr(
        "tools.data.motionxpp.extract_t5_embeddings.encode_text_batch",
        fake_encode_text_batch,
    )
    args = argparse.Namespace(
        manifest=manifest,
        output_root=output,
        batch_size=2,
        motions_per_shard=1,
        model_name_or_path="dummy-t5-3b",
        cache_dir=None,
        local_files_only=True,
        device="cpu",
        model_dtype="float32",
        resume=False,
        limit=None,
        strict=True,
    )
    first = extract_embeddings(args)
    assert first["motion_count"] == 2
    assert first["total_caption_count"] == 3
    assert len(first["shards"]) == 2
    assert set(first["motion_to_shard"]) == {"a", "b"}
    for item in first["shards"]:
        shard = torch.load(output / item["path"], map_location="cpu", weights_only=False)
        for tensor in shard.values():
            assert tensor.dtype == torch.float16
            assert tensor.shape[1:] == (50, 1024)
            assert torch.isfinite(tensor).all()

    # 模拟第二个分片已原子写完、但进程尚未来得及把它追加到 manifest。
    partial_manifest_path = output / "manifests/train.json"
    partial = json.loads(partial_manifest_path.read_text(encoding="utf-8"))
    partial["shards"] = partial["shards"][:1]
    partial["motion_to_shard"] = {"a": partial["motion_to_shard"]["a"]}
    partial_manifest_path.write_text(json.dumps(partial), encoding="utf-8")
    (output / first["shards"][1]["path"]).with_suffix(".meta.json").unlink()
    args.resume = True
    second = extract_embeddings(args)
    assert second["resumed_shards"] == 1
    assert second["encoded_caption_count_this_run"] == 1
    assert (output / first["shards"][1]["path"]).with_suffix(".meta.json").is_file()
