"""验证 BUMI V2 音乐舞蹈输入包的真实数据边界及发布流程。

构造带具名关节的可信 UMR NPZ、带表头的 50 Hz CSV、PCM 音频和已有特征，
覆盖关节重排、四元数插值、时长剔除、音频端点裁齐、UMR 特殊文件名前缀及
发布后的安全读取。集成测试仅替换耗时的音乐分析器，其他读写和哈希校验均
执行实际路径；不能以该合成测试代替正式数据的全量回读或音乐内容验证。
所有生成文件位于 pytest 的 tmp_path，调用方指定系统临时根并负责清理。
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.csv_to_npz_bumi_v2 import RETARGET_JOINT_NAMES, ROOT_COLUMNS
from tools.data.bumi import build_bumiv2_genmo_dataset as builder


def write_wav(path, samples, rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.arange(samples, dtype=np.int16).tobytes())


def write_umr(path, frames=60, object_names=False):
    names = tuple(reversed(builder.JOINT_NAMES))
    qpos = np.zeros((frames, 28), np.float32)
    qpos[:, 3] = 1
    qpos[:, 7:] = np.arange(21)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        qpos=qpos,
        fps=np.array([30]),
        frame_ids=np.arange(frames),
        robot_name=np.asarray("bumi_v2"),
        robot_joint_names=np.array(names, dtype=object if object_names else str),
        robot_xml="/assets/bumi_v2.xml",
        source_data="test",
        source_sequence_key="test",
        source_format="smplx_npz",
    )
    return qpos


def test_trusted_object_metadata_and_named_reorder(tmp_path):
    path = tmp_path / "motion.npz"
    source = write_umr(path, object_names=True)
    with pytest.raises(ValueError, match="trust-umr"):
        builder.load_umr(path, False)
    qpos, _ = builder.load_umr(path, True)
    np.testing.assert_array_equal(qpos[:, :7], source[:, :7])
    np.testing.assert_array_equal(qpos[:, 7:], source[:, 7:][:, ::-1])


def test_duration_boundary_and_audio_common_timeline(tmp_path):
    info = {"sample_rate": 48000, "samples": 2 * 48000 - 1000}
    assert builder.duration_matches(100, 50, info)
    assert builder.aligned_frames(60, info) == 59
    assert not builder.duration_matches(100, 50, {**info, "samples": 2 * 48000 - 1602})
    source, target = tmp_path / "a.wav", tmp_path / "b.wav"
    write_wav(source, info["samples"])
    before = source.read_bytes()
    builder.crop_audio(source, target, 59)
    assert builder.audio_info(target)["samples"] == 59 * 1600
    assert source.read_bytes() == before
    with wave.open(str(target), "rb") as wav:
        np.testing.assert_array_equal(
            np.frombuffer(wav.readframes(59 * 1600), np.int16), np.arange(59 * 1600, dtype=np.int16)
        )


def test_prefix_mapping_keeps_negative_video_id():
    assert (
        builder.umr_sample_id(Path("aioz_gdance__-abc_dancer_00_bumi_v2.npz")) == "-abc_dancer_00"
    )
    assert builder.umr_sample_id(Path("ordinary_bumi_v2.npz")) == "ordinary"


def test_full_release_preserves_values_and_excludes_mismatched_song(tmp_path, monkeypatch):
    args = argparse.Namespace(
        output=tmp_path / "release",
        umr_root=tmp_path / "raw",
        source_index=tmp_path / "index.jsonl",
        mine_csv_root=tmp_path / "csv",
        mine_audio_root=tmp_path / "mine_audio",
        aist_audio_root=tmp_path / "aist_audio",
        aioz_audio_root=tmp_path / "aioz_audio",
        finedance_audio_root=tmp_path / "fine_audio",
        trust_umr_metadata=True,
    )
    source_rows = []
    for ds, sid, group, audio_root in (
        ("aistpp", "aist_sequence", "mWA4", args.aist_audio_root),
        ("aioz_gdance", "-clip_dancer_00", "-clip", args.aioz_audio_root),
        ("finedance", "001", "song", args.finedance_audio_root),
    ):
        filename = ("aioz_gdance__" if ds == "aioz_gdance" else "") + sid + "_bumi_v2.npz"
        write_umr(args.umr_root / filename)
        feature_path = tmp_path / (ds + ".pt")
        torch.save(torch.arange(60 * 35, dtype=torch.float32).reshape(60, 35), feature_path)
        write_wav(audio_root / ((group if ds == "aioz_gdance" else sid) + ".wav"), 96000 - 1000)
        source_rows.append(
            {
                "sample_id": sid,
                "dataset": ds,
                "group_id": group,
                "music_id": group,
                "source_root": str(tmp_path),
                "music_feature_path": feature_path.name,
                "num_frames": 60,
                "split": "train",
            }
        )
    builder.write_jsonl(args.source_index, source_rows)
    args.mine_csv_root.mkdir()
    motion = np.zeros((100, 28))
    motion[:, 0] = np.arange(100) / 50
    angle = np.arange(100) / 100
    motion[:, 5], motion[:, 6] = np.sin(angle / 2), np.cos(angle / 2)
    motion[::2, 3:7] *= -1
    motion[:, 7:] = np.arange(21)
    for song, samples in (("accepted", 96000), ("rejected", 72000)):
        path = args.mine_csv_root / f"bumi2_01_{song}_50fps.csv"
        np.savetxt(
            path,
            motion,
            delimiter=",",
            header=",".join(ROOT_COLUMNS + RETARGET_JOINT_NAMES),
            comments="",
        )
        write_wav(args.mine_audio_root / "dance_3_音频" / (song + ".wav"), samples)
    before = {str(p): builder.sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(
        builder, "extract_edge_baseline35", lambda *a, **kw: (torch.ones(61, 35), {"test": True})
    )
    builder.build(args)
    rows = [json.loads(line) for line in (args.output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert {r["num_frames"] for r in rows if r["dataset"] != "mine"} == {59}
    mine = next(r for r in rows if r["dataset"] == "mine")
    with np.load(args.output / mine["motion_path"], allow_pickle=False) as data:
        q = data["qpos"]
        assert len(q) == 60
        np.testing.assert_allclose(q[:, 0], np.arange(60) / 30, atol=1e-7)
        np.testing.assert_allclose(abs(q[:, 3]), np.cos(np.arange(60) / 60 / 2), atol=1e-7)
        np.testing.assert_array_equal(
            q[0, 7:], [RETARGET_JOINT_NAMES.index(j) for j in builder.JOINT_NAMES]
        )
        assert np.all(np.sum(q[1:, 3:7] * q[:-1, 3:7], axis=1) > 0)
    excluded = [
        json.loads(line) for line in (args.output / "excluded.jsonl").read_text().splitlines()
    ]
    assert len(excluded) == 1 and excluded[0]["song_name"] == "rejected"
    assert not args.output.with_name("release.building").exists()
    assert all(builder.sha256_file(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        builder.build(args)
