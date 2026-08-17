"""Unit contracts for SMPL 151D physics-v1 fine-tuning."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from gem.datamodule.balanced_music_sampler import (
    HierarchicalMusicDistributedSampler,
    project_bounded_probabilities,
)
from gem.pipeline.smpl_physics_losses import (
    compute_smpl_physics_losses,
    consecutive_valid_mask,
    derivative_valid_mask,
    finite_difference,
    rollout_canonical_root,
    so3_angular_velocity,
    sole_penetration_loss,
)
from gem.utils.ground_sidecar import (
    SOLE_SMPLX_VERTEX_IDS,
    SOLE_V437_INDICES,
    estimate_ground_height,
    load_ground_sidecar,
    make_ground_record,
    sha256_file,
)
from gem.utils.rotation_conversions import axis_angle_to_matrix
from gem.utils.smpl_physics_metrics import (
    relative_regression,
    sole_penetration_metrics,
    temporal_quality_metrics,
)
from scripts.build_ground_sidecars import _aist_sources

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aist_ground_sources_accept_official_72d_pose_contract(
    tmp_path: Path,
) -> None:
    sequence_id = "gWA_sFM_cAll_d25_mWA4_ch05"
    num_frames = 4
    pose = torch.arange(num_frames * 72, dtype=torch.float32).reshape(
        num_frames, 72
    )
    annotation = {
        sequence_id: {
            "smpl_pose_global": pose,
            "smpl_trans_global": torch.zeros(num_frames, 3),
            "bbox_xyxy": torch.zeros(num_frames, 4),
        }
    }
    torch.save(annotation, tmp_path / "annot_aist_30fps.pt")
    torch.save([sequence_id], tmp_path / "train.pt")

    sources = list(
        _aist_sources(tmp_path, "annot_aist_30fps.pt", "train.pt")
    )

    assert len(sources) == 1
    assert sources[0]["num_frames"] == num_frames
    assert sources[0]["motion"]["global_orient"].shape == (num_frames, 3)
    assert sources[0]["motion"]["body_pose"].shape == (num_frames, 63)
    assert torch.equal(sources[0]["motion"]["global_orient"], pose[:, :3])
    assert torch.equal(sources[0]["motion"]["body_pose"], pose[:, 3:66])


def test_physics_experiment_is_derived_without_mutating_baseline() -> None:
    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        physics = compose(
            config_name="train",
            overrides=["exp=gem_smpl_music_only_4set_physics_v1"],
        )
        baseline = compose(
            config_name="train", overrides=["exp=gem_smpl_music_only_4set"]
        )
    assert physics.optimizer.lr == pytest.approx(2e-5)
    assert list(physics.scheduler.scheduler.milestones) == [30000, 45000]
    assert physics.pl_trainer.max_steps == 50000
    assert physics.pl_trainer.use_distributed_sampler is False
    assert physics.data.balanced_sampling.samples_per_epoch == 52224
    assert physics.pipeline.args.physics_losses.sole_penetration.weight == 0.005
    assert all(
        value.duration_aware_sampling is False
        for value in physics.train_datasets.values()
    )
    assert all(
        value.duration_aware_sampling is True
        for value in baseline.train_datasets.values()
    )
    assert "physics_losses" not in baseline.pipeline.args
    physics_network = OmegaConf.to_container(physics.network, resolve=True)
    baseline_network = OmegaConf.to_container(baseline.network, resolve=True)
    physics_network["args"].pop("physics_losses")
    physics_network["model_cfg"]["denoiser"]["args"].pop("physics_losses")
    assert physics_network == baseline_network
    assert OmegaConf.to_container(
        physics.endecoder, resolve=True
    ) == OmegaConf.to_container(baseline.endecoder, resolve=True)


def test_first_to_third_derivative_masks_exclude_padding_and_bad_intervals() -> None:
    frames = torch.tensor([[True, True, True, True, False, False]])
    assert consecutive_valid_mask(frames, 1).tolist() == [
        [True, True, True, False, False]
    ]
    assert consecutive_valid_mask(frames, 2).tolist() == [
        [True, True, False, False]
    ]
    assert consecutive_valid_mask(frames, 3).tolist() == [
        [True, False, False]
    ]
    accepted_velocity = torch.tensor([[True, False, True, True, True]])
    assert derivative_valid_mask(frames, 1, accepted_velocity).tolist() == [
        [True, False, True, False, False]
    ]
    assert derivative_valid_mask(frames, 2, accepted_velocity).tolist() == [
        [False, False, False, False]
    ]
    assert derivative_valid_mask(frames, 3, accepted_velocity).tolist() == [
        [False, False, False]
    ]


def test_so3_relative_rotation_is_continuous_across_axis_angle_pi_wrap() -> None:
    angles = torch.deg2rad(torch.tensor([178.0, 179.0, -179.0, -178.0]))
    axis_angle = torch.zeros(1, 4, 1, 3)
    axis_angle[0, :, 0, 1] = angles
    velocity = so3_angular_velocity(axis_angle_to_matrix(axis_angle), fps=30.0)
    expected = torch.deg2rad(torch.tensor([30.0, 60.0, 30.0]))
    assert torch.allclose(velocity[0, :, 0].norm(dim=-1), expected, atol=1e-4)


def test_root_rollout_and_fk_derivatives_use_physical_seconds() -> None:
    local_velocity_per_frame = torch.tensor(
        [[[1.0 / 30.0, 0.0, 0.0]] * 5]
    )
    root_orientation = torch.zeros(1, 5, 3)
    trajectory = rollout_canonical_root(local_velocity_per_frame, root_orientation)
    assert torch.allclose(trajectory[0, :, 0], torch.arange(5) / 30.0)
    assert torch.allclose(
        finite_difference(trajectory, 1, 30.0),
        torch.tensor([[[1.0, 0.0, 0.0]] * 4]),
    )

    fk = trajectory[:, :, None, :].expand(-1, -1, 22, -1)
    fk_velocity = finite_difference(fk, 1, 30.0)
    fk_acceleration = finite_difference(fk, 2, 30.0)
    assert torch.allclose(fk_velocity[..., 0], torch.ones_like(fk_velocity[..., 0]))
    assert torch.allclose(
        fk_acceleration, torch.zeros_like(fk_acceleration), atol=1e-5
    )


def test_ground_estimator_hash_contract_and_invalid_path(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.pt"
    motion_path.write_bytes(b"canonical motion")
    digest = sha256_file(motion_path)
    assert len(digest) == 64

    positions = torch.zeros(40, 8, 3)
    positions[..., 1] = -1.25
    estimate = estimate_ground_height(positions)
    assert estimate["ground_valid"] is True
    assert estimate["ground_y"] == pytest.approx(-1.25)
    record = make_ground_record(
        sample_id="dance",
        source_motion_sha256=digest,
        num_frames=40,
        fps=30.0,
        estimate=estimate,
    )
    sidecar = tmp_path / "ground.jsonl"
    sidecar.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_ground_sidecar(sidecar)["dance"]["ground_valid"] is True

    moving = positions.clone()
    moving[:, :, 0] = torch.arange(40).reshape(-1, 1)
    invalid = estimate_ground_height(moving)
    assert invalid["ground_valid"] is False
    assert invalid["ground_y"] is None

    record["source_motion_sha256"] = "not-a-hash"
    sidecar.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_ground_sidecar(sidecar)


def test_sole_proxy_indices_match_the_versioned_smplx_vertex_mapping() -> None:
    mapping_path = REPO_ROOT / "gem/utils/body_model/smplx_verts437.pt"
    mapping = torch.load(mapping_path, map_location="cpu", weights_only=False)
    actual = torch.as_tensor(mapping)[list(SOLE_V437_INDICES)].tolist()
    assert actual == list(SOLE_SMPLX_VERTEX_IDS)


def test_sole_penetration_gradient_pushes_only_low_vertices_up() -> None:
    sole_y = torch.tensor(
        [[[-0.03, 0.02], [-0.02, 0.04]]], requires_grad=True
    )
    normalized, raw = sole_penetration_loss(
        sole_y,
        ground_y_local=torch.tensor([0.0]),
        frame_valid=torch.tensor([[True, True]]),
        ground_valid=torch.tensor([True]),
    )
    normalized.backward()
    assert raw.item() > 0
    assert sole_y.grad[0, 0, 0] < 0
    assert sole_y.grad[0, 1, 0] < 0
    assert sole_y.grad[0, 0, 1] == 0
    assert sole_y.grad[0, 1, 1] == 0


def test_held_out_metrics_report_temporal_errors_and_five_percent_guard() -> None:
    gt_root = torch.zeros(1, 5, 3)
    pred_root = gt_root.clone()
    pred_root[0, :, 0] = torch.arange(5) / 30.0
    gt_fk = gt_root[:, :, None].expand(-1, -1, 22, -1)
    pred_fk = pred_root[:, :, None].expand(-1, -1, 22, -1)
    identity = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(1, 5, 21, 3, 3)
    metrics = temporal_quality_metrics(
        pred_root_position=pred_root,
        gt_root_position=gt_root,
        pred_fk_position=pred_fk,
        gt_fk_position=gt_fk,
        pred_body_rotation=identity,
        gt_body_rotation=identity,
        frame_valid=torch.ones(1, 5, dtype=torch.bool),
    )
    assert metrics["root_velocity_error"] == pytest.approx(1.0)
    assert metrics["fk_velocity_error"] == pytest.approx(1.0)
    assert metrics["pose_geodesic_error_rad"] == pytest.approx(0.0)
    penetration = sole_penetration_metrics(
        torch.tensor([[[-0.02] * 8, [0.03] * 8]]),
        ground_y_local=torch.tensor([0.0]),
        frame_valid=torch.tensor([[True, True]]),
        ground_valid=torch.tensor([True]),
    )
    assert penetration["sole_max_penetration_depth_m"] == pytest.approx(0.01)
    assert penetration["sole_penetration_frame_ratio"] == pytest.approx(0.5)
    assert relative_regression(1.04, 1.0) < 0.05
    assert relative_regression(1.06, 1.0) > 0.05


class _FakeEndecoder:
    def fk_v2(self, *, body_pose, betas, global_orient, transl):
        del betas, global_orient
        pose_offset = body_pose[..., :3].unsqueeze(-2) * 0.001
        return transl.unsqueeze(-2).expand(-1, -1, 22, -1) + pose_offset


def test_complete_physics_loss_ramps_and_backpropagates_in_fp32() -> None:
    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        config = compose(
            config_name="train",
            overrides=["exp=gem_smpl_music_only_4set_physics_v1"],
        ).pipeline.args.physics_losses
    batch, frames = 1, 5
    pred_pose = torch.zeros(batch, frames, 63, requires_grad=True)
    pred_betas = torch.zeros(batch, frames, 10, requires_grad=True)
    pred_root = torch.zeros(batch, frames, 3, requires_grad=True)
    pred_velocity = torch.zeros(batch, frames, 3, requires_grad=True)
    pred_verts = torch.zeros(batch, frames, 437, 3, requires_grad=True)
    pred_verts.data[..., 1] = -0.03
    zeros3 = torch.zeros(batch, frames, 3)
    inputs = {
        "mask": {
            "valid": torch.ones(batch, frames, dtype=torch.bool),
            "spv_incam_only": torch.zeros(batch, dtype=torch.bool),
            "2d_only": torch.zeros(batch, dtype=torch.bool),
        },
        "smpl_params_w": {
            "body_pose": torch.zeros(batch, frames, 63),
            "betas": torch.zeros(batch, frames, 10),
            "global_orient": zeros3,
            "transl": zeros3,
        },
        "smpl_params_c": {"global_orient": zeros3},
        "R_c2gv": torch.eye(3).reshape(1, 1, 3, 3).expand(batch, frames, -1, -1),
        "physics": {
            "ground_y_local": torch.zeros(batch),
            "ground_valid": torch.ones(batch, dtype=torch.bool),
        },
        "meta": [{"dataset_id": "aist++"}],
    }
    outputs = {
        "decode_dict": {
            "body_pose": pred_pose,
            "betas": pred_betas,
            "global_orient": pred_root,
            "global_orient_gv": pred_root,
            "local_transl_vel": pred_velocity,
        },
        "pred_body_params_incam": {"transl": zeros3},
        "_pred_c_verts437": pred_verts,
    }
    pipeline = SimpleNamespace(
        args={"physics_losses": config}, endecoder=_FakeEndecoder()
    )
    zero_loss, _ = compute_smpl_physics_losses(
        inputs, outputs, pipeline, global_step=0
    )
    assert zero_loss == 0
    full_loss, logs = compute_smpl_physics_losses(
        inputs, outputs, pipeline, global_step=10000
    )
    assert full_loss.dtype == torch.float32 and full_loss > 0
    assert logs["physics_weight_ramp_metric"] == 1
    assert logs["physics_sole_penetration_weighted_loss"] == pytest.approx(
        float(logs["physics_sole_penetration_normalized_loss"]) * 0.005
    )
    full_loss.backward()
    assert torch.isfinite(pred_verts.grad).all()
    assert pred_verts.grad[..., 1].min() < 0


class _SamplingDataset(Dataset):
    def __init__(self, hours: float, variants: int = 1) -> None:
        self.frames = max(round(hours * 30 * 3600), 1)
        self.variants = variants

    def __len__(self) -> int:
        return self.variants

    def __getitem__(self, index: int) -> int:
        return index

    def get_music_sampling_records(self) -> list[dict]:
        return [
            {
                "dataset_index": index,
                "sample_id": f"variant-{index}",
                "group_id": "song",
                "num_frames": self.frames,
            }
            for index in range(self.variants)
        ]


def _interleave(rank0: list[int], rank1: list[int]) -> list[int]:
    return [value for pair in zip(rank0, rank1) for value in pair]


def test_bounded_sqrt_probabilities_and_ddp_reproducibility() -> None:
    hours = [0.5118, 13.505, 5.986, 0.08384]
    expected = [0.1018, 0.50, 0.3482, 0.05]
    probabilities = project_bounded_probabilities(
        [math.sqrt(value) for value in hours], minimum=0.05, maximum=0.50
    )
    assert probabilities == pytest.approx(expected, abs=2e-4)

    datasets = [_SamplingDataset(value, variants=3) for value in hours]
    kwargs = dict(
        datasets=datasets,
        dataset_names=["aist++", "aioz", "finedance", "compas3d"],
        samples_per_epoch=1000,
        seed=123,
    )
    global_sampler = HierarchicalMusicDistributedSampler(
        **kwargs, rank=0, num_replicas=1
    )
    rank0 = HierarchicalMusicDistributedSampler(**kwargs, rank=0, num_replicas=2)
    rank1 = HierarchicalMusicDistributedSampler(**kwargs, rank=1, num_replicas=2)
    global_indices = list(global_sampler)
    assert _interleave(list(rank0), list(rank1)) == global_indices
    assert list(global_sampler) == global_indices

    offsets = [0, 3, 6, 9, 12]
    counts = [
        sum(offsets[index] <= value < offsets[index + 1] for value in global_indices)
        for index in range(4)
    ]
    assert [count / 1000 for count in counts] == pytest.approx(expected, abs=0.002)
    global_sampler.set_epoch(1)
    assert list(global_sampler) != global_indices
    restored = HierarchicalMusicDistributedSampler(
        **kwargs, rank=0, num_replicas=1
    )
    restored.load_state_dict(global_sampler.state_dict())
    assert list(restored) == list(global_sampler)
