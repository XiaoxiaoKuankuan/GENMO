"""四库完整音乐原始/生成网页构建器的轻量契约测试。

这些测试只在 pytest 临时目录构造最小清单，不加载 checkpoint、不执行 CUDA diffusion、
不渲染视频。覆盖的重点是服务器 1 模型使用的四库必须各取 10 条且来自高质量冻结清单，
以及完整 SMPL 网格的归一化变换保持有限值和预期几何关系。
正式 ONNX 数值对齐、长序列生成、H.264/AAC 媒体流验收由生产命令单独执行并写入报告。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import build_smpl_hq4_full_comparison as target


def test_build_selection_is_exactly_ten_per_dataset(tmp_path: Path, monkeypatch) -> None:
    human = tmp_path / "human"
    frozen_items = []
    for dataset in ("aistpp", "aioz_gdance", "finedance", "compas3d"):
        for number in range(10):
            motion = human / dataset / f"motion_{number}.npz"
            audio = tmp_path / "audio" / dataset / f"music_{number}.wav"
            motion.parent.mkdir(parents=True, exist_ok=True)
            audio.parent.mkdir(parents=True, exist_ok=True)
            motion.touch()
            audio.touch()
            frozen_items.append(
                {
                    "dataset": dataset,
                    "audio_key": f"music_{number}",
                    "audio": str(audio),
                    "representative_motion": motion.name,
                    "high_quality_motion_count": 1,
                }
            )
    frozen = tmp_path / "selection.json"
    frozen.write_text(json.dumps({"items": frozen_items}), encoding="utf-8")

    monkeypatch.setattr(target, "audio_duration", lambda _path: 10.0)

    items = target.build_selection(frozen, human, seed=100)

    assert len(items) == 40
    assert {dataset: sum(item["dataset"] == dataset for item in items) for dataset in target.DATASET_LABELS} == {
        dataset: 10 for dataset in target.DATASET_LABELS
    }
    assert [item["seed"] for item in items] == list(range(100, 140))

    normalized = tmp_path / "normalized_selection.json"
    normalized.write_text(
        json.dumps(
            {
                "contract_version": "genmo.smpl_hq4_selection.v1",
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    reconciled = target.build_selection(normalized, human, seed=200)
    assert len(reconciled) == 40
    assert [item["seed"] for item in reconciled] == list(range(200, 240))


def test_transform_vertices_applies_offset_rotation_and_translation() -> None:
    vertices = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]], dtype=np.float32)
    offset = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rotation = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    translation = np.array([0.5, 0.0, -0.5], dtype=np.float32)

    transformed = target.transform_vertices(vertices, offset, rotation, translation)

    expected = np.array([[[2.5, 1.0, -0.5], [3.5, 2.0, -1.5]]], dtype=np.float32)
    np.testing.assert_allclose(transformed, expected)
    assert np.isfinite(transformed).all()
