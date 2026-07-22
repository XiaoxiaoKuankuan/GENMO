"""CPU-only tests for sharded HumanML3D T5 token embedding extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tools.data.humanml3d.extract_t5_embeddings as extractor
from tools.data.humanml3d.extract_t5_embeddings import (
    HIDDEN_DIM,
    MAX_TEXT_LEN,
    T5EmbeddingBuildError,
    compute_fingerprint,
    encode_shard,
    extract_caption_metadata,
    finalize_shards,
    flatten_captions,
    make_manifest,
    prepare_manifest,
    publish_validated_shard,
    restore_owner_embeddings,
    shard_layout,
    validate_embedding_dict,
    validate_embedding_tensor,
    validate_shard_file,
    write_shard_tmp,
)


class FakeTokenizer:
    """Small tokenizer exposing the batch_encode_plus contract used by GENMO."""

    def batch_encode_plus(
        self,
        raw_text,
        *,
        return_tensors,
        padding,
        max_length,
        truncation,
    ):
        assert return_tensors == "pt"
        assert padding == "max_length" and max_length == MAX_TEXT_LEN and truncation
        input_ids = torch.zeros(len(raw_text), max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, caption in enumerate(raw_text):
            length = min(len(caption.split()) + 1, max_length)
            input_ids[index, :length] = torch.arange(1, length + 1)
            attention_mask[index, :length] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class FakeEncoder:
    def __init__(self, nonfinite: bool = False):
        self.nonfinite = nonfinite

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, HIDDEN_DIM).clone()
        hidden += torch.arange(HIDDEN_DIM, device=input_ids.device).float() / HIDDEN_DIM
        if self.nonfinite:
            hidden[0, 0, 0] = float("nan")
        return SimpleNamespace(last_hidden_state=hidden)


def _source_file(path: Path) -> dict[str, dict]:
    # Deliberately insert keys out of order and retain a duplicate caption.
    source = {
        "z_motion": {
            "text_data": [
                {"caption": "same caption", "tokens": ["ignored/NOUN"]},
                {"caption": "same caption", "tokens": ["also/ADV", "ignored/VERB"]},
            ]
        },
        "a_motion": {
            "text_data": [{"caption": "walk forward", "tokens": ["walk/VERB"]}]
        },
    }
    torch.save(source, path)
    return source


def _captions() -> dict[str, list[str]]:
    return {"a_motion": ["walk forward"], "z_motion": ["same caption", "same caption"]}


def _half_embeddings(captions: dict[str, list[str]], ids: list[str]) -> dict[str, torch.Tensor]:
    return {
        motion_id: torch.zeros(
            len(captions[motion_id]), MAX_TEXT_LEN, HIDDEN_DIM, dtype=torch.float16
        )
        for motion_id in ids
    }


def test_input_parsing_sorts_ids_and_preserves_caption_order_and_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "motions.pth"
    _source_file(path)
    captions, stats, invalid, retained = extract_caption_metadata(path)
    assert list(captions) == ["a_motion", "z_motion"]
    assert captions["z_motion"] == ["same caption", "same caption"]
    assert stats["motion_record_count"] == 2 and stats["total_caption_count"] == 3
    assert invalid == [] and retained is None


def test_limit_uses_first_deterministic_sorted_motion(tmp_path: Path) -> None:
    path = tmp_path / "motions.pth"
    _source_file(path)
    captions, stats, _, _ = extract_caption_metadata(path, limit=1)
    assert list(captions) == ["a_motion"]
    assert stats["total_caption_count"] == 1


def test_invalid_motion_key_is_reported_before_sorting(tmp_path: Path) -> None:
    path = tmp_path / "bad_key.pth"
    torch.save(
        {
            123: {"text_data": [{"caption": "invalid key"}]},
            "valid": {"text_data": [{"caption": "valid caption"}]},
        },
        path,
    )
    captions, _, invalid, _ = extract_caption_metadata(path, strict=False)
    assert captions == {"valid": ["valid caption"]}
    assert invalid == [
        {"motion_id": "123", "error": "motion key must be a non-empty string"}
    ]
    with pytest.raises(T5EmbeddingBuildError, match="Invalid record key"):
        extract_caption_metadata(path, strict=True)


@pytest.mark.parametrize(
    "text_data",
    [[], [{"tokens": ["missing"]}], [{"caption": "", "tokens": []}]],
)
def test_empty_or_invalid_text_data_is_reported(
    tmp_path: Path, text_data: list[dict]
) -> None:
    path = tmp_path / "bad.pth"
    torch.save({"bad": {"text_data": text_data}}, path)
    captions, _, invalid, _ = extract_caption_metadata(path, strict=False)
    assert captions == {} and len(invalid) == 1
    with pytest.raises(T5EmbeddingBuildError, match="Invalid record"):
        extract_caption_metadata(path, strict=True)


def test_flat_caption_owner_mapping_preserves_every_caption() -> None:
    captions = _captions()
    flat, owners = flatten_captions(captions, list(captions))
    assert flat == ["walk forward", "same caption", "same caption"]
    assert owners == [("a_motion", 0), ("z_motion", 0), ("z_motion", 1)]


def test_owner_restore_shape_dtype_device_and_order() -> None:
    captions = _captions()
    flat, owners = flatten_captions(captions, list(captions))
    embeddings = torch.arange(
        len(flat) * MAX_TEXT_LEN * HIDDEN_DIM, dtype=torch.float32
    ).reshape(len(flat), MAX_TEXT_LEN, HIDDEN_DIM)
    restored = restore_owner_embeddings(embeddings, owners, captions, list(captions))
    assert restored["a_motion"].shape == (1, MAX_TEXT_LEN, HIDDEN_DIM)
    assert restored["z_motion"].shape == (2, MAX_TEXT_LEN, HIDDEN_DIM)
    assert restored["z_motion"].dtype == torch.float16
    assert restored["z_motion"].device.type == "cpu"
    assert restored["z_motion"].is_contiguous()
    assert torch.equal(restored["z_motion"][0], embeddings[1].half())


def test_fake_encoding_zeros_padding_and_retains_duplicate_count() -> None:
    captions = _captions()
    shard, encoded_count, padding_error = encode_shard(
        captions,
        list(captions),
        text_encoder=FakeEncoder(),
        tokenizer=FakeTokenizer(),
        device="cpu",
        batch_size=2,
        strict=True,
    )
    assert encoded_count == 3 and padding_error == 0.0
    assert shard["z_motion"].shape[0] == 2
    assert torch.count_nonzero(shard["a_motion"][:, 3:]) == 0


def test_nonfinite_fake_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        encode_shard(
            {"motion": ["caption"]},
            ["motion"],
            text_encoder=FakeEncoder(nonfinite=True),
            tokenizer=FakeTokenizer(),
            device="cpu",
            batch_size=1,
            strict=False,
        )


def test_embedding_caption_count_and_nonfinite_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_embedding_tensor(
            torch.zeros(2, MAX_TEXT_LEN, HIDDEN_DIM, dtype=torch.float16), 1, "motion"
        )
    value = torch.zeros(1, MAX_TEXT_LEN, HIDDEN_DIM, dtype=torch.float16)
    value[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_embedding_tensor(value, 1, "motion")


def test_fingerprint_changes_with_caption_or_model() -> None:
    captions = _captions()
    baseline = compute_fingerprint(captions, "t5-3b")
    changed_caption = {**captions, "z_motion": ["same caption", "different"]}
    assert compute_fingerprint(changed_caption, "t5-3b") != baseline
    assert compute_fingerprint(captions, "local/t5-3b") != baseline


def test_shard_save_reload_and_corruption_detection(tmp_path: Path) -> None:
    captions = _captions()
    ids = list(captions)
    final = tmp_path / "shard_00000.pth"
    temporary = write_shard_tmp(_half_embeddings(captions, ids), final)
    size = publish_validated_shard(temporary, final, captions, ids)
    assert size > 0
    assert validate_shard_file(final, captions, ids)[:2] == (2, 3)
    torch.save({"a_motion": torch.zeros(1)}, final)
    with pytest.raises(ValueError, match="key mismatch|shape"):
        validate_shard_file(final, captions, ids)


def test_resume_rejects_incompatible_fingerprint(tmp_path: Path) -> None:
    captions = _captions()
    layout = shard_layout(list(captions), 1, tmp_path)
    first = make_manifest(
        captions,
        layout,
        fingerprint="first",
        model_name_or_path="t5-3b",
        model_dtype="float32",
        motions_per_shard=1,
    )
    extractor._atomic_write_json(tmp_path / "manifest.json", first)
    second = make_manifest(
        captions,
        layout,
        fingerprint="second",
        model_name_or_path="t5-3b",
        model_dtype="float32",
        motions_per_shard=1,
    )
    with pytest.raises(T5EmbeddingBuildError, match="incompatible"):
        prepare_manifest(tmp_path, second, resume=True, overwrite=False)


def _publish_test_shards(
    shard_dir: Path, captions: dict[str, list[str]], motions_per_shard: int = 1
) -> dict:
    layout = shard_layout(list(captions), motions_per_shard, shard_dir)
    manifest = make_manifest(
        captions,
        layout,
        fingerprint=compute_fingerprint(captions, "t5-3b"),
        model_name_or_path="t5-3b",
        model_dtype="float32",
        motions_per_shard=motions_per_shard,
    )
    for metadata in manifest["shards"]:
        ids = metadata["motion_ids"]
        final = Path(metadata["output_file"])
        temporary = write_shard_tmp(_half_embeddings(captions, ids), final)
        metadata["output_size_bytes"] = publish_validated_shard(
            temporary, final, captions, ids
        )
        metadata["status"] = "complete"
    extractor._atomic_write_json(shard_dir / "manifest.json", manifest)
    return manifest


def test_final_merge_exact_keys_no_metadata_and_atomic_output(tmp_path: Path) -> None:
    captions = _captions()
    shard_dir = tmp_path / "shards"
    manifest = _publish_test_shards(shard_dir, captions)
    output = tmp_path / "all_text_embed.pth"
    validation = finalize_shards(captions, manifest, output, overwrite=False)
    merged = extractor.safe_torch_load(output)
    assert set(merged) == set(captions)
    assert not any(key.startswith("__") for key in merged)
    assert validation["key_set_exact"] is True
    assert not output.with_name(output.name + ".tmp").exists()


def test_exact_key_validation_rejects_missing_or_extra() -> None:
    captions = {"motion": ["caption"]}
    with pytest.raises(ValueError, match="key mismatch"):
        validate_embedding_dict({}, captions)
    with pytest.raises(ValueError, match="key mismatch"):
        validate_embedding_dict(
            {"motion": _half_embeddings(captions, ["motion"])["motion"], "metadata": None},
            captions,
        )


def test_estimate_only_never_loads_model(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "motions.pth"
    _source_file(source)
    monkeypatch.setattr(
        extractor,
        "load_t5_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    output = tmp_path / "output.pth"
    assert (
        extractor.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--report-dir",
                str(tmp_path / "report"),
                "--estimate-only",
            ]
        )
        == 0
    )
    assert not output.exists()


def test_finalize_only_never_loads_model(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "motions.pth"
    _source_file(source)
    captions = _captions()
    shard_dir = tmp_path / "shards"
    _publish_test_shards(shard_dir, captions, motions_per_shard=1)
    monkeypatch.setattr(
        extractor,
        "load_t5_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    output = tmp_path / "output.pth"
    assert (
        extractor.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--shard-dir",
                str(shard_dir),
                "--report-dir",
                str(tmp_path / "report"),
                "--motions-per-shard",
                "1",
                "--finalize-only",
            ]
        )
        == 0
    )
    assert set(extractor.safe_torch_load(output)) == set(captions)
