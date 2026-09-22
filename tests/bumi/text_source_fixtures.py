"""BUMI 历史 T5 特征读取的人工源数据夹具。

只生成微型 motion/embedding 分片以验证来源指纹、文本索引与旧特征读取兼容。
不导入旧 SMPL 模型、数据构建器或其测试模块；所有文件写入 pytest tmp_path。
"""

import json

import numpy as np
import pytest
import torch

from gem.datasets.text_source_contract import sha256_file

LENGTHS = (60, 97, 120, 183, 299, 300)
SAMPLE_INDEX_DTYPE = np.dtype(
    [("shard_id", "<i4"), ("record_index", "<i4"), ("frames", "<i2"), ("window_index", "<i2")]
)


@pytest.fixture()
def release(tmp_path):
    motions, embeddings = [], []
    for index, frames in enumerate(LENGTHS):
        pose = torch.arange(frames).float()[:, None].expand(-1, 66).clone() * 0.0001
        trans = torch.zeros(frames, 3)
        trans[:, 0] = torch.arange(frames) * 0.01
        trans[:, 1] = 1.0
        motions.append(
            {
                "motion_id": f"MotionGV/{index}",
                "pose": pose,
                "trans": trans,
                "beta": torch.zeros(10),
                "captions": ["walk forward", "a person walks"],
                "fps": 30,
                "source_up_axis": "y",
                "source_archive": "MotionGV/source.tar.gz",
                "source_archive_sha256": "a" * 64,
                "source_subset": "MotionGV",
                "source_member": f"{index}.npy",
                "source_sha256": "b" * 64,
                "source_text_member": f"{index}.txt",
                "source_text_sha256": "c" * 64,
                "split": "train",
            }
        )
        embeddings.append(
            {
                "motion_id": f"MotionGV/{index}",
                "offsets": torch.tensor([0, 3, 8]),
                "embeddings": torch.cat([torch.ones(3, 1024), torch.full((5, 1024), 2)]).half(),
            }
        )
    motion_root, embedding_root = tmp_path / "motion", tmp_path / "text"
    for root in (motion_root, embedding_root):
        (root / "manifests").mkdir(parents=True)
    torch.save(motions, motion_root / "shard.pth")
    torch.save(embeddings, embedding_root / "shard.pth")
    np.save(
        motion_root / "index.npy",
        np.array([(0, i, f, 0) for i, f in enumerate(LENGTHS)], dtype=SAMPLE_INDEX_DTYPE),
    )
    motion = {
        "schema_version": 1,
        "split": "train",
        "build_fingerprint": "f" * 64,
        "motion_frames": 120,
        "fps": 30,
        "source_up_axis": "y",
        "record_count": len(LENGTHS),
        "sample_count": len(LENGTHS),
        "sample_index_path": "index.npy",
        "shards": [
            {
                "shard_id": 0,
                "record_count": len(LENGTHS),
                "path": "shard.pth",
                "sha256": sha256_file(motion_root / "shard.pth"),
            }
        ],
    }
    embedding = {
        "schema_version": 1,
        "split": "train",
        "source_build_fingerprint": "f" * 64,
        "max_text_tokens": 150,
        "hidden_dim": 1024,
        "shards": [
            {
                "shard_id": 0,
                "record_count": len(LENGTHS),
                "path": "shard.pth",
                "sha256": sha256_file(embedding_root / "shard.pth"),
                "source_motion_path": "shard.pth",
                "source_motion_sha256": motion["shards"][0]["sha256"],
            }
        ],
    }
    for root, manifest in ((motion_root, motion), (embedding_root, embedding)):
        (root / "manifests/train.json").write_text(json.dumps(manifest))
    return motion_root, embedding_root, motions, embeddings
