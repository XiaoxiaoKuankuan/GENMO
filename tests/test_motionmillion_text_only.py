"""MotionMillion→GENMO SMPL 纯文本训练链路的 CPU 契约测试。

测试不下载 gated 数据、不加载 T5-3B 或 SMPL 资产，使用可审计的小型数组和 tar
fixture 验证官方 272D 恢复数学、长度过滤、紧凑 embedding、空逐帧条件、文本
padding mask、分片 DDP sampler、学习率计划和 Hydra 专用配置。真实 10,000 条
pilot、渲染、吞吐及 GPU forward/backward 属于服务器数据就绪后的独立验收门。
"""

from __future__ import annotations

import io
import tarfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gem.datamodule.motionmillion_sampler import ShardAwareDistributedSampler
from gem.gem import GEM, prepare_text_attention_mask
from gem.network.base_arch.transformer.encoder_rope import DecoderRoPEBlock
from gem.network.gem_denoiser import NetworkEncoderRoPE
from gem.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from scripts.demo.demo_smpl_text import (
    build_text_only_data,
    validate_text_generation_checkpoint,
)
from tools.data.motionmillion.build_motionmillion_genmo import (
    build_dataset,
    prepare_metadata_database,
)
from tools.data.motionmillion.common import (
    MAX_TEXT_TOKENS,
    MotionMillionError,
    MotionMillionFilteredError,
    accumulate_heading_rotations,
    build_sample_index,
    mirror_base_id,
    recover_smpl_from_272,
    smpl_to_272,
)
from tools.data.motionmillion.download_motionmillion import (
    _matches_stage,
    _remote_file_row,
    _reuse_verified_progress,
)
from tools.data.motionmillion.extract_t5_embeddings import (
    compact_caption_embeddings,
    extract_embeddings,
)
from tools.data.motionmillion.preflight_motionmillion import run_preflight
from tools.eval.build_motionmillion_review import build_review
from tools.eval.generate_motionmillion_val_predictions import select_caption_and_seed
from tools.eval.motionmillion_smpl_to_272 import export_motion
from tools.eval.prepare_motionmillion_official_eval import prepare_official_eval
from tools.eval.run_motionmillion_official_metrics import (
    calculate_fid,
    calculate_r_precision,
)
from tools.eval.summarize_motionmillion_metrics import choose_candidate, summarize

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_download_inventory_ignores_huggingface_repo_folders() -> None:
    """新版 huggingface_hub 的 type=None 目录不能进入文件校验清单。"""
    folder = SimpleNamespace(
        path="data_process/AIST",
        type=None,
        tree_id="fixture-tree-id",
        blob_id=None,
        size=None,
        lfs=None,
    )
    file_entry = SimpleNamespace(
        path="data_process/AIST/README.md",
        type=None,
        tree_id=None,
        blob_id="fixture-blob-id",
        size=3264,
        lfs=None,
    )
    assert _remote_file_row(folder) is None
    assert _remote_file_row(file_entry) == {
        "path": "data_process/AIST/README.md",
        "remote_size_bytes": 3264,
        "remote_blob_id": "fixture-blob-id",
        "lfs_sha256": None,
    }
    assert _matches_stage("assets/motionmillion_teaser.png", "metadata") is False
    assert _matches_stage("assets/motionmillion_teaser.png", "full") is True


def test_download_resume_reuses_only_identical_verified_prefix(tmp_path: Path) -> None:
    """已验证前缀可跳过重复哈希，远端身份漂移必须阻断。"""
    relative = "motion_272rpr/MotionGV/folder0.tar.gz"
    local_file = tmp_path / relative
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"abc")
    current_rows = [
        {
            "path": relative,
            "remote_size_bytes": 3,
            "remote_blob_id": "blob-id",
            "lfs_sha256": "a" * 64,
        }
    ]
    progress_row = {
        **current_rows[0],
        "local_size_bytes": 3,
        "sha256": "a" * 64,
        "tar_checked": True,
        "tar_members": 7,
    }
    (tmp_path / "download_progress_full.json").write_text(
        __import__("json").dumps(
            {
                "repo_id": "InternRobotics/MotionMillion",
                "resolved_revision": "fixture-revision",
                "motion_pattern": None,
                "completed_motion_file_count": 1,
                "total_motion_file_count": 1,
                "completed_motion_bytes": 3,
                "files": [progress_row],
            }
        ),
        encoding="utf-8",
    )
    count, size = _reuse_verified_progress(
        tmp_path,
        repo_id="InternRobotics/MotionMillion",
        resolved_revision="fixture-revision",
        motion_pattern=None,
        motion_rows=current_rows,
    )
    assert (count, size) == (1, 3)
    assert current_rows[0]["tar_members"] == 7

    changed_rows = [{**current_rows[0], "remote_blob_id": "changed"}]
    with pytest.raises(MotionMillionError, match="身份/顺序"):
        _reuse_verified_progress(
            tmp_path,
            repo_id="InternRobotics/MotionMillion",
            resolved_revision="fixture-revision",
            motion_pattern=None,
            motion_rows=changed_rows,
        )


def _identity_motion(frames: int = 60) -> torch.Tensor:
    """构造旋转有效、静止且根高为 1 m 的官方 272D fixture。"""
    motion = torch.zeros(frames, 272)
    identity = matrix_to_rotation_6d(torch.eye(3)).reshape(6)
    motion[:, 2:8] = identity
    motion[:, 140:272] = identity.repeat(22)
    motion[:, 8 + 1] = 1.0
    return motion


def _official_reference_recover(motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按官方 NumPy 实现逐句复写，作为被测函数之外的 parity oracle。"""
    value = motion.numpy()
    frames = len(value)
    local_rot = rotation_6d_to_matrix(torch.from_numpy(value[:, 140:272]).reshape(frames, 22, 6))
    relative = rotation_6d_to_matrix(torch.from_numpy(value[:, 2:8])).numpy()
    totals = [relative[0]]
    for item in relative[1:]:
        totals.append(np.matmul(item, totals[-1]))
    inverse = np.transpose(np.asarray(totals), (0, 2, 1))
    rotations = local_rot.numpy()
    rotations[:, 0] = np.matmul(inverse, rotations[:, 0])
    velocity = np.zeros((frames, 3), dtype=np.float32)
    velocity[:, 0] = value[:, 0]
    velocity[:, 2] = value[:, 1]
    velocity[1:] = np.matmul(inverse[:-1], velocity[1:, :, None]).squeeze(-1)
    translation = np.cumsum(velocity, axis=0)
    translation[:, 1] = value[:, 9]
    return torch.from_numpy(rotations.copy()), torch.from_numpy(translation)


def _geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    relative = left.transpose(-1, -2) @ right
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    return torch.acos(cosine)


def test_official_272_recovery_translation_rotation_and_fk_parity() -> None:
    motion = _identity_motion(80)
    angles = torch.linspace(0.0, 0.35, len(motion))
    delta = torch.zeros_like(angles)
    delta[0] = angles[0]
    delta[1:] = angles[1:] - angles[:-1]
    relative = torch.stack(
        [
            torch.cos(delta), torch.zeros_like(delta), torch.sin(delta),
            torch.zeros_like(delta), torch.ones_like(delta), torch.zeros_like(delta),
            -torch.sin(delta), torch.zeros_like(delta), torch.cos(delta),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    motion[:, 2:8] = matrix_to_rotation_6d(relative)
    motion[1:, 0] = 0.02

    actual = recover_smpl_from_272(motion)
    expected_rot, expected_trans = _official_reference_recover(motion)
    actual_rot = axis_angle_to_matrix(actual["pose"].reshape(len(motion), 22, 3))
    assert (actual["trans"] - expected_trans).abs().max().item() <= 1.0e-6
    assert _geodesic(actual_rot, expected_rot).max().item() <= 1.0e-5

    # 使用固定骨架做独立 FK；相同根平移与局部旋转必须维持 1e-5 m parity。
    offsets = torch.zeros(22, 3)
    offsets[1:, 1] = 0.05
    parents = torch.arange(22) - 1

    def fk(rotations: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        world_rot = torch.empty_like(rotations)
        joints = torch.empty(rotations.shape[0], 22, 3)
        world_rot[:, 0] = rotations[:, 0]
        joints[:, 0] = translation
        for joint in range(1, 22):
            parent = int(parents[joint])
            world_rot[:, joint] = world_rot[:, parent] @ rotations[:, joint]
            joints[:, joint] = joints[:, parent] + torch.einsum(
                "fij,j->fi", world_rot[:, parent], offsets[joint]
            )
        return joints

    assert torch.linalg.vector_norm(
        fk(actual_rot, actual["trans"]) - fk(expected_rot, expected_trans), dim=-1
    ).max().item() <= 1.0e-5


@pytest.mark.parametrize("frames", [59, 301])
def test_272_recovery_rejects_out_of_contract_lengths(frames: int) -> None:
    with pytest.raises(MotionMillionFilteredError):
        recover_smpl_from_272(_identity_motion(frames))


def test_heading_accumulation_order_is_relative_left_multiply() -> None:
    rotations = rotation_6d_to_matrix(_identity_motion(60)[:3, 2:8])
    result = accumulate_heading_rotations(rotations)
    assert torch.allclose(result[2], rotations[2] @ rotations[1] @ rotations[0])


def test_smpl_272_smpl_roundtrip_rotation_and_translation() -> None:
    frames = 60
    pose = torch.zeros(frames, 66)
    pose[:, 1] = torch.linspace(0.0, 0.4, frames)
    trans = torch.zeros(frames, 3)
    trans[:, 0] = torch.linspace(0.0, 1.0, frames)
    trans[:, 1] = 1.0
    joints = trans[:, None].repeat(1, 22, 1)
    encoded = smpl_to_272(pose, trans, joints)
    recovered = recover_smpl_from_272(encoded)
    pose_error = _geodesic(
        axis_angle_to_matrix(pose.reshape(frames, 22, 3)),
        axis_angle_to_matrix(recovered["pose"].reshape(frames, 22, 3)),
    )
    assert pose_error.max().item() <= 1.0e-5
    assert (recovered["trans"] - trans).abs().max().item() <= 1.0e-6


def test_mirror_base_id_covers_official_style_variants() -> None:
    base = mirror_base_id("MotionGV/000123")
    assert mirror_base_id("MotionGV/M000123") == base
    assert mirror_base_id("MotionGV/mirror_000123") == base
    assert mirror_base_id("MotionGV/000123_mirrored") == base
    assert mirror_base_id("MotionUnion/idea/Thumbs_Up") != mirror_base_id(
        "MotionUnion/idea/Thumbs_up"
    )


def test_compact_embedding_keeps_only_valid_fp16_tokens() -> None:
    def encode(captions):
        embeddings = torch.ones(len(captions), 150, 1024)
        mask = torch.zeros(len(captions), 150, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :5] = True
        return embeddings, mask, [3, 200]

    compact, stats = compact_caption_embeddings(
        ["walk", "a long instruction"], encode_batch=encode, batch_size=2
    )
    assert compact["embeddings"].shape == (8, 1024)
    assert compact["embeddings"].dtype == torch.float16
    assert compact["offsets"].tolist() == [0, 3, 8]
    assert stats["truncated_captions"] == 1


def test_persisted_fp16_embedding_meets_cosine_and_absolute_error_gate() -> None:
    torch.manual_seed(11)
    online = torch.randn(1, 150, 1024)
    mask = torch.zeros(1, 150, dtype=torch.bool)
    mask[:, :17] = True

    def encode(_captions):
        return online.clone(), mask.clone(), [17]

    compact, _ = compact_caption_embeddings(
        ["a person turns and raises both arms"], encode_batch=encode, batch_size=1
    )
    restored = compact["embeddings"].float()
    expected = online[0, :17]
    cosine = torch.nn.functional.cosine_similarity(restored, expected, dim=-1)
    assert cosine.min().item() >= 0.9999
    assert (restored - expected).abs().max().item() <= 5.0e-3


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def test_metadata_quarantines_cross_split_mirror_and_keeps_original_split(
    tmp_path: Path,
) -> None:
    """跨 split 镜像写入排除表，canonical 原动作仍保留官方 split。"""
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    _write_tar(
        raw / "split.tar.gz",
        {
            "split/version1/t2m_60_300/train.txt": b"MotionGV/folder0/000005\n",
            "split/version1/t2m_60_300/val.txt": b"MotionGV/folder0/000017\n",
            "split/version1/t2m_60_300/test.txt": b"Mirror_MotionGV/folder0/000005\n",
        },
    )
    _write_tar(
        raw / "texts.tar.gz",
        {
            "texts/MotionGV/folder0/000005.txt": b"walk forward\n",
            "texts/MotionGV/folder0/000017.txt": b"turn left\n",
            "texts/Mirror_MotionGV/folder0/000005.txt": b"walk forward mirrored\n",
        },
    )
    connection, report = prepare_metadata_database(raw, output, resume=False)
    try:
        assert list(
            connection.execute(
                "SELECT motion_id, split FROM split_entries ORDER BY motion_id"
            )
        ) == [
            ("MotionGV/folder0/000005", "train"),
            ("MotionGV/folder0/000017", "val"),
        ]
        assert list(
            connection.execute(
                "SELECT motion_id, split, canonical_split FROM split_exclusions"
            )
        ) == [("Mirror_MotionGV/folder0/000005", "test", "train")]
    finally:
        connection.close()
    exclusion = report["counts"]["mirror_cross_split_exclusions"]
    assert exclusion["leaking_base_group_count"] == 1
    assert exclusion["excluded_entry_count"] == 1
    assert exclusion["eligible_by_split"] == {"train": 1, "val": 1, "test": 0}
    report_path = output / "reports" / "mirror_cross_split_exclusions.jsonl"
    assert "Mirror_MotionGV/folder0/000005" in report_path.read_text(encoding="utf-8")


def test_tar_to_motion_embedding_shards_and_resume_closed_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "motion"
    embedding_output = tmp_path / "embedding"
    split_members = {}
    text_members = {}
    motion_members = {}
    ids = {
        "train": "MotionGV/000001",
        "val": "MotionGV/000002",
        "test": "MotionGV/000003",
    }
    for split, motion_id in ids.items():
        split_members[f"split/version1/t2m_60_300/{split}.txt"] = (
            motion_id + "\n"
        ).encode()
        text_members[f"texts/{motion_id}.txt"] = f"caption for {split}\n".encode()
        motion_members[f"motion_data/vector_272/{motion_id}.npy"] = _npy_bytes(
            _identity_motion(60).numpy()
        )
    _write_tar(raw / "split.tar.gz", split_members)
    _write_tar(raw / "texts.tar.gz", text_members)
    _write_tar(raw / "motion_272rpr" / "MotionGV" / "part.tar.gz", motion_members)
    mean_std = raw / "mean_std" / "vector_272"
    mean_std.mkdir(parents=True)
    np.save(mean_std / "mean.npy", np.zeros(272, dtype=np.float32))
    np.save(mean_std / "std.npy", np.ones(272, dtype=np.float32))

    pilot_release = build_dataset(
        Namespace(
            raw_root=raw,
            output_root=tmp_path / "pilot",
            records_per_shard=2,
            motion_frames=120,
            limit=1,
            only_split="train",
            archive_pattern="motion_272rpr/MotionGV/*.tar.gz",
            resume=False,
            strict=True,
            progress_every=1000,
        )
    )
    assert pilot_release["unavailable_by_release_count"] == 0
    assert pilot_release["unresolved_by_scope_count"] == 2

    build_args = Namespace(
        raw_root=raw,
        output_root=output,
        records_per_shard=2,
        motion_frames=120,
        limit=None,
        resume=False,
        strict=True,
        progress_every=1000,
    )
    release = build_dataset(build_args)
    assert release["manifests"]["train"]["record_count"] == 1
    assert release["manifests"]["train"]["total_frames"] == 60
    assert release["manifests"]["train"]["duration_seconds"] == pytest.approx(2.0)
    assert release["total_frames"] == 180
    assert release["duration_seconds"] == pytest.approx(6.0)
    assert release["duration_hours"] == pytest.approx(6.0 / 3600.0)
    assert release["unavailable_by_release_count"] == 0
    build_args.resume = True
    resumed = build_dataset(build_args)
    assert resumed["resumed_record_count"] == 3
    assert resumed["accepted_this_run"] == 0

    evaluator_data = prepare_official_eval(
        Namespace(
            raw_root=raw,
            motion_root=output,
            output_root=tmp_path / "official_evaluator",
            resume=False,
        )
    )
    assert evaluator_data["eligibility"]["record_count"] == 1
    evaluator_motion = (
        Path(evaluator_data["dataset_root"])
        / evaluator_data["records"][0]["motion_path"]
    )
    assert np.array_equal(
        np.load(evaluator_motion, allow_pickle=False), _identity_motion(60).numpy()
    )

    def fake_encoder(captions):
        values = torch.zeros(len(captions), 150, 1024)
        mask = torch.zeros(len(captions), 150, dtype=torch.bool)
        for index, caption in enumerate(captions):
            length = len(caption.split()) + 1
            values[index, :length] = index + 1
            mask[index, :length] = True
        return values, mask, [int(row.sum()) for row in mask]

    embed_args = Namespace(
        motion_root=output,
        output_root=embedding_output,
        model_name_or_path="test-t5",
        model_revision="fixture-revision",
        cache_dir=None,
        device="cpu",
        batch_size=2,
        local_files_only=True,
        limit_shards=None,
        resume=False,
    )
    embedded = extract_embeddings(
        embed_args,
        injected_encoder=fake_encoder,
        injected_revision="fixture-revision",
    )
    assert embedded["manifests"]["train"]["record_count"] == 1
    preflight = run_preflight(
        Namespace(
            motion_root=output,
            embedding_root=embedding_output,
            report=tmp_path / "preflight.json",
            verify_sha256=True,
            max_shards=None,
            normalized_stats_samples=0,
            z_threshold=8.0,
            max_outlier_fraction=0.25,
        )
    )
    assert preflight["status"] == "PASS"
    assert preflight["splits"]["train"]["records_checked"] == 1

    # 绕过 BaseDataset 的 SMPL/camera 构造，只验证 shard、裁剪/补齐和文本契约。
    from gem.datasets.pure_motion.base_dataset import BaseDataset
    from gem.datasets.pure_motion.motionmillion import MotionMillionDataset

    def metadata_only_init(self, _cam_augmentation, limit_size=None):
        self.limit_size = limit_size
        self._load_dataset()
        self._get_idx2meta()

    monkeypatch.setattr(BaseDataset, "__init__", metadata_only_init)
    dataset = MotionMillionDataset(
        root=tmp_path,
        split="train",
        motion_manifest_path=output / "manifests" / "train.json",
        embedding_manifest_path=embedding_output / "manifests" / "train.json",
        random_crop=False,
    )
    sample = dataset._load_data(0)
    assert sample["body_pose"].shape == (120, 63)
    assert sample["transl"].shape == (120, 3)
    assert sample["valid_length"] == 60
    assert sample["text_embed"].shape == (150, 1024)
    assert sample["text_attention_mask"].sum().item() == 4
    assert sample["source_subset"] == "MotionGV"
    assert sample["source_archive"].endswith("MotionGV/part.tar.gz")


def test_v1_index_has_one_unique_row_per_motion() -> None:
    records = [
        [{"pose": torch.zeros(frames, 66)} for frames in (60, 120, 240, 300)]
    ]
    index = build_sample_index(records, motion_frames=120)
    assert len(index) == 4
    assert index["record_index"].tolist() == [0, 1, 2, 3]
    assert index["window_index"].tolist() == [0, 0, 0, 0]


def _empty_condition_model() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            args=OmegaConf.create(
                {"in_attr": [], "disable_random_null_condition": True}
            )
        ),
        condition_source={
            "image": ["f_imgseq"],
            "2d": ["obs", "f_cliffcam"],
            "camera": ["f_cam_angvel", "f_cam_t_vel"],
            "audio": ["encoded_audio"],
            "music": ["encoded_music"],
        },
        music_mask_prob=0.0,
        audio_mask_prob=0.0,
        latent_dim=1024,
        model_cfg=OmegaConf.create({"use_cond_exists_as_input": False}),
    )


def test_empty_in_attr_builds_three_tensor_conditions() -> None:
    batch_size, length = 2, 7
    frame_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    batch = {
        "B": batch_size,
        "L": length,
        "device": torch.device("cpu"),
        "has_text": torch.ones(batch_size, dtype=torch.bool),
        "condition_mask": {
            "has_img_mask": frame_mask,
            "has_2d_mask": frame_mask,
            "has_cam_mask": frame_mask,
            "has_audio_mask": frame_mask,
            "has_music_mask": frame_mask,
            "j2d_visible_mask": torch.zeros(batch_size, length, 17, dtype=torch.bool),
        },
        "length": torch.tensor([7, 5]),
        "target_x": torch.randn(batch_size, length, 151),
    }
    output = GEM.create_condition_mask(
        _empty_condition_model(), batch, OmegaConf.create({}), "diffusion", train=True
    )
    for key in ("f_cond", "f_uncond", "f_empty"):
        assert isinstance(output[key], torch.Tensor)
        assert output[key].shape == (batch_size, length, 1024)
        assert torch.count_nonzero(output[key]) == 0


def test_text_mask_validation_and_all_padding_guard() -> None:
    embedding = torch.zeros(2, 150, 1024)
    embedding[:, :3] = 1
    mask = prepare_text_attention_mask(
        embedding, torch.arange(150)[None] < torch.tensor([[3], [2]]), has_text=torch.ones(2)
    )
    assert mask.sum(dim=1).tolist() == [3, 2]
    with pytest.raises(ValueError, match="no valid token"):
        prepare_text_attention_mask(
            torch.zeros(1, 150, 1024),
            torch.zeros(1, 150, dtype=torch.bool),
            has_text=torch.ones(1, dtype=torch.bool),
        )


class _MaskRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, x, _context, **kwargs):
        self.seen.append(kwargs["memory_key_padding_mask"].clone())
        return x


def test_denoiser_forwards_padding_mask_to_every_text_layer() -> None:
    network = NetworkEncoderRoPE(
        output_dim=8,
        xt_dim=8,
        latent_dim=8,
        num_layers=2,
        num_heads=2,
        encode_text=True,
        encoded_text_dim=8,
        avgbeta=False,
        pred_cam_dim=0,
        static_conf_dim=0,
        text_encoder_cfg={"mode": "all", "cross_attn_type": "mha"},
        args={"pred_fullcam": False},
    ).eval()
    recorders = []
    for index in network.text_encode_layer_idx:
        recorder = _MaskRecorder()
        network.text_encoder_layers[str(index)] = recorder
        recorders.append(recorder)
    valid = torch.tensor([[True, True, False, False]])
    network(
        torch.zeros(1, 5, 8),
        torch.zeros(1, dtype=torch.long),
        y={
            "f_cond": torch.zeros(1, 5, 8),
            "length": torch.tensor([5]),
            "encoded_text": torch.randn(1, 4, 8),
            "text_attention_mask": valid,
        },
        inputs={},
    )
    assert len(recorders) == 2
    assert all(torch.equal(recorder.seen[0], ~valid) for recorder in recorders)


def test_denoiser_exposes_measured_text_cfg_dropout_mask() -> None:
    network = NetworkEncoderRoPE(
        output_dim=8,
        xt_dim=8,
        latent_dim=8,
        num_layers=1,
        num_heads=2,
        encode_text=True,
        encoded_text_dim=8,
        avgbeta=False,
        pred_cam_dim=0,
        static_conf_dim=0,
        text_mask_prob=1.0,
        text_encoder_cfg={"mode": "all", "cross_attn_type": "mha"},
        args={"pred_fullcam": False},
    ).train()
    inputs: dict[str, torch.Tensor] = {}
    network(
        torch.zeros(2, 5, 8),
        torch.zeros(2, dtype=torch.long),
        y={
            "f_cond": torch.zeros(2, 5, 8),
            "length": torch.tensor([5, 5]),
            "encoded_text": torch.randn(2, 4, 8),
            "text_attention_mask": torch.ones(2, 4, dtype=torch.bool),
        },
        inputs=inputs,
    )
    assert inputs["text_cfg_dropout_mask"].tolist() == [True, True]


def test_cross_attention_output_ignores_changed_padding_embedding() -> None:
    torch.manual_seed(13)
    block = DecoderRoPEBlock(
        hidden_size=8,
        num_heads=2,
        dropout=0.0,
        use_self_attn=False,
        cross_attn_type="mha",
    ).eval()
    block.gate_cross_attn.data.fill_(1.0)
    block.gate_mlp.data.zero_()
    motion = torch.randn(1, 5, 8)
    context = torch.randn(1, 4, 8)
    changed = context.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 1000
    padding = torch.tensor([[False, False, True, True]])
    first = block(motion, context, memory_key_padding_mask=padding)
    second = block(motion, changed, memory_key_padding_mask=padding)
    assert torch.allclose(first, second, atol=1.0e-6, rtol=1.0e-6)


class _ShardDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.shards = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])

    def __len__(self):
        return len(self.shards)

    def __getitem__(self, index):
        return index

    def sample_shard_ids(self):
        return self.shards


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_shard_sampler_is_reproducible_and_rank_unique(world_size: int) -> None:
    dataset = _ShardDataset()
    per_rank = []
    for rank in range(world_size):
        sampler = ShardAwareDistributedSampler(
            dataset, seed=9, rank=rank, num_replicas=world_size, drop_last=True
        )
        first = list(sampler)
        assert first == list(sampler)
        per_rank.append(first)
    flattened = [index for values in per_rank for index in values]
    assert len(flattened) == len(set(flattened))
    assert len(flattened) == (len(dataset) // world_size) * world_size


def test_scheduler_warmup_cosine_and_floor() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=2.0e-4)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_steps=5, total_steps=20, min_lr=2.0e-6
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == pytest.approx(4.0e-5)
    assert rates[4] == pytest.approx(2.0e-4)
    assert rates[-1] == pytest.approx(2.0e-6)
    assert all(left >= right for left, right in zip(rates[4:], rates[5:]))


def test_motionmillion_text_only_hydra_contract() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="train", overrides=["exp=gem_smpl_motionmillion_text_only"]
        )
    assert list(cfg.train_datasets) == ["motionmillion_text_train"]
    assert list(cfg.test_datasets) == ["motionmillion_text_val"]
    assert list(cfg.pipeline.args.train_modes) == ["diffusion"]
    assert list(cfg.pipeline.args.in_attr) == []
    assert cfg.pipeline.args.disable_random_null_condition is True
    assert cfg.network.model_cfg.denoiser.encode_text is True
    assert cfg.network.model_cfg.denoiser.encoded_text_dim == 1024
    assert cfg.network.model_cfg.denoiser.text_mask_prob == pytest.approx(0.1)
    assert cfg.model.model_cfg.text_encoder.load_llm is False
    assert cfg.model.model_cfg.text_encoder.max_text_len == MAX_TEXT_TOKENS
    assert cfg.model.model_cfg.condition_mask.mask_text_prob.diffusion == 0
    assert cfg.pretrain_ckpt is None and cfg.ckpt_path is None
    assert cfg.data.loader_opts.train.batch_size * cfg.pl_trainer.devices == 512
    assert cfg.pl_trainer.precision == "bf16-mixed"
    assert cfg.pl_trainer.max_steps == 300000
    assert cfg.data.shard_aware_sampling.enabled is True


def test_evaluator_adapter_exports_272_with_identity_report(tmp_path: Path) -> None:
    frames = 60
    source = tmp_path / "motion.npz"
    output = tmp_path / "official" / "motion.npy"
    joints_path = tmp_path / "joints.npy"
    trans = np.zeros((frames, 3), dtype=np.float32)
    trans[:, 1] = 1.0
    np.savez(
        source,
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        transl=trans,
        betas=np.zeros((frames, 10), dtype=np.float32),
        fps=np.asarray(30.0, dtype=np.float32),
    )
    np.save(joints_path, np.repeat(trans[:, None], 22, axis=1))
    report = export_motion(
        Namespace(input=source, output=output, joint_positions=joints_path)
    )
    assert np.load(output, allow_pickle=False).shape == (frames, 272)
    assert report["frames"] == frames and report["fps"] == 30.0
    assert output.with_suffix(".npy.json").is_file()


def test_motionmillion_checkpoint_contract_selects_150_token_demo(tmp_path: Path) -> None:
    checkpoint = tmp_path / "text.ckpt"
    torch.save(
        {
            "state_dict": {
                "pipeline.embed_text.weight": torch.zeros(1),
                "pipeline.text_encoder_layers.0.weight": torch.zeros(1),
                "pipeline.gate_cross_attn": torch.zeros(1),
            },
            "genmo_text_contract": {
                "schema_version": 1,
                "max_text_len": 150,
                "encoded_text_dim": 1024,
                "text_only": True,
                "pipeline_in_attr": [],
            },
        },
        checkpoint,
    )
    contract = validate_text_generation_checkpoint(checkpoint)
    assert contract["max_text_len"] == 150
    assert contract["exp_name"] == "gem_smpl_motionmillion_text_only"
    data = build_text_only_data(
        "walk", torch.randn(150, 1024), 120, 1280, 720, 0.75
    )
    assert data["text_embed"].shape == (150, 1024)
    assert data["text_attention_mask"].shape == (150,)


def test_review_page_binds_protocol_and_three_scores(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("walk forward\nturn left\n", encoding="utf-8")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "000.mp4").write_bytes(b"fixture-video-0")
    (video_root / "001.mp4").write_bytes(b"fixture-video-1")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    output = tmp_path / "review" / "index.html"
    report = build_review(
        Namespace(
            prompt_file=prompt_file,
            video_root=video_root,
            checkpoint=checkpoint,
            output=output,
            seed=20260909,
            num_frames=120,
            fps=30.0,
            ddim_steps=50,
            cfg_scale=2.5,
            postprocess="shared-default",
            allow_nonofficial_count=True,
        )
    )
    content = output.read_text(encoding="utf-8")
    assert report["prompt_count"] == 2
    assert "Text Alignment" in content
    assert "Motion Smoothness" in content
    assert "Physical Plausibility" in content


def test_metric_summary_rejects_identity_drift_and_selects_tie_break(tmp_path: Path) -> None:
    identity = {
        "checkpoint_sha256": "a" * 64,
        "experiment_config_sha256": "b" * 64,
        "dataset_release_fingerprint": "c" * 64,
        "evaluator_fingerprint": "d" * 64,
        "ddim_steps": 50,
        "cfg_scale": 2.5,
    }
    paths = []
    for seed in (1, 2):
        path = tmp_path / f"{seed}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    **identity,
                    "seed": seed,
                    "fid": 1.0 + seed * 0.1,
                    "diversity": 8.0,
                    "r_precision_1": 0.5,
                    "r_precision_2": 0.6,
                    "r_precision_3": 0.7,
                    "matching_score": 2.0,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    summary = summarize(paths, required_runs=2)
    assert summary["metrics"]["fid"]["mean"] == pytest.approx(1.15)
    better_r1 = {
        **summary,
        "checkpoint_sha256": "e" * 64,
        "metrics": {**summary["metrics"], "r_precision_1": {"mean": 0.6, "ci95": 0.0}},
    }
    selected = choose_candidate([summary, better_r1], fid_tie=0.01)
    assert selected["checkpoint_sha256"] == "e" * 64
    payload = __import__("json").loads(paths[1].read_text())
    payload["evaluator_fingerprint"] = "changed"
    paths[1].write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError, match="身份链不一致"):
        summarize(paths, required_runs=2)


def test_official_metric_math_matches_identity_pairing() -> None:
    embedding = np.eye(4, dtype=np.float64)
    r_precision, matching = calculate_r_precision(embedding, embedding)
    assert r_precision.tolist() == [4, 4, 4]
    assert matching == pytest.approx(0.0)
    assert calculate_fid(embedding, embedding) == pytest.approx(0.0, abs=1.0e-10)


def test_val_caption_and_sample_seed_are_stable_per_motion() -> None:
    first = select_caption_and_seed("MotionGV/000001", ["walk", "turn"], 17)
    second = select_caption_and_seed("MotionGV/000001", ["walk", "turn"], 17)
    changed = select_caption_and_seed("MotionGV/000002", ["walk", "turn"], 17)
    assert first == second
    assert first[1] in {"walk", "turn"}
    assert first != changed
