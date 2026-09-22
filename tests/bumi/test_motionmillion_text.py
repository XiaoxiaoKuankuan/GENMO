"""原生UMR文本来源绑定与官方划分的合成回归测试。

所有归档、报告和SQLite仅生成于pytest临时目录。覆盖同名不同目录、镜像不借用
原文本、缺失文本明确计数、官方划分保留、原文本SHA和双输入SHA核验，并验证
重复文本失败时不发布半成品。这些测试只证明数据绑定契约，不证明语义质量。
"""

import hashlib
import io
import json
import tarfile

import pytest

from gem.datasets.text_source_contract import sha256_file
from tools.data.bumi.motionmillion_text import TextCatalog, bind_report, build_catalog


def archive(path, entries):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))


def fixture(tmp_path):
    report = tmp_path / "report"
    (report / "reports").mkdir(parents=True)
    ids = ["Mirror_MotionGV/folder0/1", "Mirror_MotionGV/folder1/1", "Mirror_MotionGV/folder2/1"]
    rows = [
        dict(
            source_motion_id=k,
            source_file=k + ".npy",
            source_sha256="a" * 64,
            human_sha256="b" * 64,
            training_eligible=i != 1,
        )
        for i, k in enumerate(ids)
    ]
    (report / "reports/folder.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (report / "quality_summary.json").write_text(
        json.dumps(dict(run_fingerprint="test", processed_records=3))
    )
    (report / "run.json").write_text(
        json.dumps(
            dict(
                state="complete",
                partial_scan=False,
                summary_sha256=sha256_file(report / "quality_summary.json"),
                fingerprint="test",
                indexed_records=3,
            )
        )
    )
    texts, splits = tmp_path / "texts.tar.gz", tmp_path / "split.tar.gz"
    archive(
        texts,
        [
            ("texts/" + ids[0] + ".txt", b" first caption \n\nsecond caption\n"),
            ("texts/" + ids[1] + ".txt", b"different folder\n"),
            ("texts/MotionGV/folder2/1.txt", b"must not borrow original text\n"),
        ],
    )
    archive(
        splits,
        [
            (f"split/version1/t2m_60_300/{split}.txt", data)
            for split, data in [("train", ids[0].encode()), ("val", ids[1].encode()), ("test", b"")]
        ],
    )
    return report, texts, splits, rows


def test_source_binding_captions_splits_and_missing(tmp_path):
    report, texts, splits, rows = fixture(tmp_path)
    output = tmp_path / "catalog"
    result = build_catalog(report, texts, splits, output)
    assert (result["matched_records"], result["missing_records"], result["captions"]) == (2, 1, 3)
    assert result["official_unlisted"] == 1
    catalog = TextCatalog(output)
    try:
        first = catalog.lookup(rows[0])
        assert first["captions"] == ["first caption", "second caption"]
        assert first["official_split"] == "train"
        assert (
            first["text_sha256"]
            == hashlib.sha256(b" first caption \n\nsecond caption\n").hexdigest()
        )
        assert catalog.lookup(rows[1])["captions"] == ["different folder"]
        assert catalog.lookup(rows[2]) is None
        with pytest.raises(ValueError, match="SHA"):
            catalog.lookup(dict(rows[0], human_sha256="changed"))
    finally:
        catalog.close()
    bound = bind_report(report, output)
    assert bound["run_fingerprint"] == "test"
    assert bound["counts"]["eligible_with_text"] == 1
    assert bound["counts"]["eligible_missing_text"] == 1
    assert bound["eligible_official_splits"] == {"train": 1}


def test_duplicate_text_is_rejected_without_publishing(tmp_path):
    report, texts, splits, rows = fixture(tmp_path)
    entry = ("texts/" + rows[0]["source_motion_id"] + ".txt", b"caption")
    archive(texts, [entry, entry])
    output = tmp_path / "catalog"
    with pytest.raises(ValueError, match="重复"):
        build_catalog(report, texts, splits, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".catalog.staging-*"))


def test_publish_all_pass_keeps_namespace_text_and_missing(tmp_path):
    from tools.data.bumi.umr_text_preprocess import publish_umr_text_pass

    report, texts, splits, rows = fixture(tmp_path)
    inputs, humans = tmp_path / "inputs", tmp_path / "humans"
    inputs.mkdir()
    humans.mkdir()
    for i, row in enumerate(rows):
        relative = f"folder{i}/same_basename.npz"
        motion = inputs / relative
        motion.parent.mkdir()
        motion.write_bytes(f"synthetic publisher payload {i}".encode())
        human = humans / f"{i}.npz"
        human.write_bytes(b"synthetic human payload")
        row.update(
            dataset="motionmillion",
            relative_path=relative,
            status="PASS",
            frames=301,
            source_sha256=sha256_file(motion),
            human_sha256=sha256_file(human),
            human_path=str(human),
        )
    (report / "reports/folder.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = dict(run_fingerprint="test", processed_records=3, status_counts={"PASS": 3})
    (report / "quality_summary.json").write_text(json.dumps(summary))
    run = json.loads((report / "run.json").read_text())
    run.update(
        summary_sha256=sha256_file(report / "quality_summary.json"),
        identity=dict(
            paths=dict(dataset="motionmillion", input_root=str(inputs), source_root=str(humans)),
            source_contract_hashes={},
        ),
    )
    (report / "run.json").write_text(json.dumps(run))
    catalog = tmp_path / "catalog"
    build_catalog(report, texts, splits, catalog)
    output = tmp_path / "pass"
    info = publish_umr_text_pass(report, output, "motionmillion", text_catalog=catalog)
    assert info["pass_frames"] == 903
    assert info["text_counts"]["pass_with_text"] == 2
    assert info["text_counts"]["pass_without_text"] == 1
    result = [
        json.loads(line) for line in (output / "manifests/pass.jsonl").read_text().splitlines()
    ]
    assert [row["split"] for row in result] == ["train", "val", "unassigned"]
    assert len({row["motion_path"] for row in result}) == 3
    assert all((output / row["motion_path"]).is_file() for row in result)


def test_conflicting_official_split_rejected(tmp_path):
    report, texts, splits, rows = fixture(tmp_path)
    archive(
        splits,
        [
            (f"split/version1/t2m_60_300/{s}.txt", rows[0]["source_motion_id"].encode())
            for s in ("train", "val", "test")
        ],
    )
    with pytest.raises(ValueError, match="冲突"):
        build_catalog(report, texts, splits, tmp_path / "catalog")
