"""physics_v3 四库完整音乐 SMPL/GMR 验证流水线的轻量合同测试。

本文件不加载 checkpoint、不运行 CUDA diffusion、GMR 或视频渲染，只验证容易在批量任务
开始前静态确认的关键边界：曲目必须同时满足人工 ``keep`` 与 ``train`` split，AIST++
按音乐 ID、其余数据集按训练特征文件名映射完整 WAV，同一真实音频只能选择一次；固定
selection seed 必须得到稳定的四库数量和逐条 seed。网页测试则确认结果不再使用多列网格，
每个条目独占一行并明确标注完整音乐和动态帧数。正式完整 EDGE35 提取、60 首生成、GMR
qpos 与 H.264/AAC 媒体验收仍由生产流水线执行并记录在结果 sidecar/summary 中。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from scripts import validate_smpl_hq4_gmr as target


def _row(dataset: str, number: int, *, split: str = "train", decision: str = "keep") -> dict:
    if dataset == "aistpp":
        music_id = f"mTR{number:03d}"
        feature_stem = f"gTR_sequence_{number:03d}"
    else:
        music_id = f"music_{number:03d}"
        feature_stem = f"{dataset}_track_{number:03d}"
    return {
        "dataset": dataset,
        "decision": decision,
        "split": split,
        "music_id": music_id,
        "sample_id": f"{dataset}_sample_{number:03d}",
        "review_id": f"{dataset}__review_{number:03d}",
        "music_feature_path": f"musicfeat_v2/{feature_stem}_musicfeat_fps30.pt",
        "music_num_frames": 900 + number,
    }


def test_train_quality_selection_is_exact_stable_and_deduplicated() -> None:
    rows: list[dict] = []
    for dataset, count in target.EXPECTED_COUNTS.items():
        rows.extend(_row(dataset, number) for number in range(count + 5))
    rows.extend(
        [
            _row("aistpp", 100, split="test"),
            _row("aioz_gdance", 101, split="val"),
            _row("finedance", 102, decision="reject"),
        ]
    )
    duplicate = _row("compas3d", 0)
    duplicate["review_id"] = "compas3d__review_duplicate"
    duplicate["sample_id"] = "duplicate_motion_same_music"
    rows.append(duplicate)

    first = target.select_train_quality_rows(rows, seed=20260826)
    second = target.select_train_quality_rows(list(reversed(rows)), seed=20260826)

    assert first == second
    assert len(first) == 60
    assert Counter(item["dataset"] for item in first) == target.EXPECTED_COUNTS
    assert all(item["split"] == "train" for item in first)
    assert all(item["quality_decision"] == "keep" for item in first)
    assert len({(item["dataset"], item["music_key"]) for item in first}) == 60
    assert [item["seed"] for item in first] == list(range(20260826, 20260886))


def test_source_audio_mapping_uses_real_full_wav_identity(tmp_path: Path) -> None:
    aist = _row("aistpp", 7)
    fine = _row("finedance", 7)

    assert target.source_audio_key(aist) == "mTR007"
    assert target.source_audio_key(fine) == "finedance_track_007"
    assert target.resolve_source_audio(
        tmp_path, {"dataset": "aistpp", "music_key": "mTR007"}
    ) == tmp_path / "aistpp" / "wav" / "mTR007.wav"
    assert target.resolve_source_audio(
        tmp_path, {"dataset": "finedance", "music_key": "finedance_track_007"}
    ) == tmp_path / "finedance" / "finedance_track_007.wav"


def test_full_music_index_uses_one_song_per_row(tmp_path: Path) -> None:
    items = [
        {
            "dataset": dataset,
            "id": f"{dataset}_01",
            "music_key": f"track_{dataset}",
            "split": "train",
            "num_frames": 3_000,
        }
        for dataset in target.EXPECTED_COUNTS
    ]
    target.write_index(SimpleNamespace(output_root=tmp_path), items)

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert page.count("<article class='song'>") == 4
    assert "grid-template-columns" not in page
    assert ".dataset-group{display:block}" in page
    assert "aspect-ratio:32/9" in page
    assert page.count("preload='none' playsinline") == 4
    assert "preload='metadata'" not in page
    assert "完整源音乐从头到尾生成，无 20 秒截断" in page
    assert "3,000 帧 · 100.00 秒" in page


def test_dynamic_frame_contract_keeps_legacy_read_compatibility() -> None:
    assert target.item_frames({"num_frames": 4_321}) == 4_321
    assert target.item_frames({"validation_frames": 600}) == 600


def test_contact_render_defaults_do_not_lift_the_source_floor() -> None:
    args = target.build_parser().parse_args(["--stage", "render"])
    assert args.ground_clearance == 0.0
    assert args.source_ground_clearance == 0.0
