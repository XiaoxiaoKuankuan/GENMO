"""SMPL physics_v4 支撑与异常尖峰损失的永久回归测试。

只使用合成张量、已有 Hydra 配置及替身 FK/网格函数，不加载训练数据和大模型，不创建
优化器、训练进程、分布式进程组或部署连接。覆盖 GT 接触开关、正常踮脚/滑步的豁免、
平脚与离地梯度方向、异常导数、有效 top-k、空/无效片段、FP32 反传以及旧配置兼容性。
通过这些测试只能确认损失代码和配置契约，不代表权重已在真实动作上标定或微调已完成。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gem.pipeline.smpl_physics_losses import compute_smpl_physics_losses, finite_difference
from gem.pipeline.smpl_support_losses import (
    canonical_sole_vertices,
    compute_smpl_support_losses,
    derivative_excess,
    nonnegative_penalty,
    support_masks,
    validate_support_config,
)
from gem.utils.ground_sidecar import SOLE_V437_INDICES

REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = "gem_smpl_music_only_4set_manual_q1_physics_v3_100k"
V4 = "gem_smpl_music_only_4set_manual_q1_physics_v4_contact_30k"


@pytest.fixture
def config():
    return OmegaConf.load(
        REPO_ROOT / "configs/exp" / (V4 + ".yaml")
    ).pipeline.args.physics_losses.support_losses


def _example(config, frames=8, batch=1):
    # 每脚四个矩形角点；Y 为竖直方向，XZ 为地面，左右脚不重叠。
    corners = torch.tensor(
        [
            [-0.13, 0.0, -0.10],
            [-0.07, 0.0, -0.10],
            [-0.13, 0.0, 0.10],
            [-0.07, 0.0, 0.10],
            [0.07, 0.0, -0.10],
            [0.13, 0.0, -0.10],
            [0.07, 0.0, 0.10],
            [0.13, 0.0, 0.10],
        ]
    )
    sole = corners[None, None].expand(batch, frames, -1, -1).clone()
    root = torch.zeros(batch, frames, 3)
    fk = torch.zeros(batch, frames, 22, 3)
    joint = torch.zeros(batch, frames - 1, 21, 3)
    return dict(
        pred_sole=sole.clone().requires_grad_(),
        gt_sole=sole,
        ground_y_local=torch.zeros(batch),
        ground_valid=torch.ones(batch, dtype=torch.bool),
        frame_valid=torch.ones(batch, frames, dtype=torch.bool),
        pred_root=root.clone().requires_grad_(),
        gt_root=root,
        pred_fk=fk.clone().requires_grad_(),
        gt_fk=fk,
        pred_joint_velocity=joint.clone().requires_grad_(),
        gt_joint_velocity=joint,
        root_accepted=torch.ones(batch, frames - 1, dtype=torch.bool),
        fk_accepted=torch.ones(batch, frames - 1, 22, dtype=torch.bool),
        joint_accepted=torch.ones(batch, frames - 1, 21, dtype=torch.bool),
        config=config,
        global_step=5000,
        fps=30.0,
    )


def _masks(case):
    return support_masks(
        **{
            key: case[key]
            for key in ("gt_sole", "ground_y_local", "ground_valid", "frame_valid", "fps", "config")
        }
    )


def _weighted(logs, name):
    return logs[f"physics_v4_{name}_weighted_loss"].item()


def test_matching_motion_has_zero_loss_and_finite_zero_gradients(config):
    case = _example(config)
    assert all(mask.all() for mask in _masks(case))
    loss, logs = compute_smpl_support_losses(**case)
    assert loss.dtype == torch.float32 and loss.item() == 0
    assert logs["physics_v4_foot_flat_support_candidate_count_metric"] == 16
    loss.backward()
    for key in ("pred_sole", "pred_root", "pred_fk", "pred_joint_velocity"):
        assert torch.isfinite(case[key].grad).all()
        assert torch.count_nonzero(case[key].grad) == 0


def test_planted_horizontal_sliding_is_penalized(config):
    case = _example(config)
    offset = torch.arange(8).reshape(1, 8, 1) / 30.0
    with torch.no_grad():
        case["pred_sole"][..., 0] += offset
    loss, logs = compute_smpl_support_losses(**case)
    assert _weighted(logs, "foot_slide") > 0
    assert _weighted(logs, "foot_contact_height") == 0
    loss.backward()
    assert torch.isfinite(case["pred_sole"].grad).all()
    assert case["pred_sole"].grad[..., 0].abs().sum() > 0


def test_intentional_fast_gt_slide_is_not_locked(config):
    case = _example(config)
    case["gt_sole"][..., 0] += torch.arange(8).reshape(1, 8, 1) / 30.0
    case["pred_sole"] = case["gt_sole"].clone().requires_grad_()
    contact, planted, flat = _masks(case)
    assert contact.all() and not planted.any() and not flat.any()
    loss, logs = compute_smpl_support_losses(**case)
    assert loss == 0 and _weighted(logs, "foot_slide") == 0


def test_planted_foot_lift_has_downward_gradient(config):
    case = _example(config)
    with torch.no_grad():
        case["pred_sole"][..., 1] += 0.1
    loss, logs = compute_smpl_support_losses(**case)
    assert _weighted(logs, "foot_contact_height") > 0
    # 整脚抬高不改变平脚角度，需要离地项负责拉回地面。
    assert _weighted(logs, "foot_flat_support") == 0
    loss.backward()
    assert (case["pred_sole"].grad[..., 1] > 0).all()


def test_toe_support_does_not_force_raised_heel_down(config):
    case = _example(config)
    case["gt_sole"][:, :, [0, 1, 4, 5], 1] = 0.10
    case["pred_sole"] = case["gt_sole"].clone().requires_grad_()
    with torch.no_grad():
        case["pred_sole"][:, :, [0, 1, 4, 5], 1] = 0.20
    contact, _, flat = _masks(case)
    assert contact[:, :, [2, 3, 6, 7]].all()
    assert not contact[:, :, [0, 1, 4, 5]].any()
    assert not flat.any()
    loss, _ = compute_smpl_support_losses(**case)
    loss.backward()
    assert loss == 0 and torch.count_nonzero(case["pred_sole"].grad) == 0


def test_flat_support_discourages_heel_lift_but_does_not_fix_yaw(config):
    case = _example(config)
    with torch.no_grad():
        case["pred_sole"][:, :, [0, 4], 1] = 0.1
    loss, logs = compute_smpl_support_losses(**case)
    assert _weighted(logs, "foot_flat_support") > 0
    loss.backward()
    assert (case["pred_sole"].grad[:, :, [0, 4], 1] > 0).all()
    # 静态绕 Y 轴转向，足底仍平放，不能由平脚项惩罚朝向。
    case = _example(config)
    rotation = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    case["pred_sole"] = (case["pred_sole"] @ rotation.T).detach().requires_grad_()
    _, logs = compute_smpl_support_losses(**case)
    assert _weighted(logs, "foot_flat_support") == 0


@pytest.mark.parametrize("reason", ["airborne", "invalid_ground", "padding", "single_contact"])
def test_non_support_frames_are_not_pulled_to_ground(config, reason):
    case = _example(config)
    if reason == "airborne":
        case["gt_sole"][..., 1] = 0.5
    elif reason == "invalid_ground":
        case["ground_valid"][:] = False
    elif reason == "padding":
        case["frame_valid"][:] = False
    else:
        case["gt_sole"][..., 1] = 0.5
        case["gt_sole"][:, 4, :, 1] = 0
    with torch.no_grad():
        case["pred_sole"][..., 1] = 0.2
    assert not any(mask.any() for mask in _masks(case))
    loss, _ = compute_smpl_support_losses(**case)
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(case["pred_sole"].grad) == 0


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_short_clips_have_empty_higher_derivatives(config, frames):
    case = _example(config, frames=frames)
    loss, _ = compute_smpl_support_losses(**case)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(case["pred_sole"].grad).all()


def test_masked_nan_ground_and_padding_do_not_poison_gradients(config):
    case = _example(config, batch=2)
    case["ground_valid"][0] = False
    case["ground_y_local"][0] = float("nan")
    case["frame_valid"][1, 5:] = False
    with torch.no_grad():
        for key in ("pred_sole", "gt_sole", "pred_root", "gt_root", "pred_fk", "gt_fk"):
            case[key][1, 5:] = float("nan")
        for key in ("pred_joint_velocity", "gt_joint_velocity"):
            case[key][1, 4:] = float("nan")
    loss, _ = compute_smpl_support_losses(**case)
    assert loss == 0
    loss.backward()
    for key in ("pred_sole", "pred_root", "pred_fk", "pred_joint_velocity"):
        assert torch.isfinite(case[key].grad).all()


def test_effective_topk_and_mean_are_not_diluted_by_padding():
    error = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    mean, tail, raw = nonnegative_penalty(error, torch.ones(4, dtype=torch.bool), 1.0, 0.5)
    assert mean.item() == pytest.approx(2.0)
    assert tail.item() == pytest.approx(3.0)
    assert raw.item() == pytest.approx(2.5)
    padded = torch.cat((error, torch.full((96,), float("nan"))))
    padded_mask = torch.arange(100) < 4
    actual = nonnegative_penalty(padded, padded_mask, 1.0, 0.5)
    for expected, value in zip((mean, tail, raw), actual):
        assert torch.equal(expected, value)
    (actual[0] + actual[1]).backward()
    assert torch.isfinite(error.grad).all()
    empty = nonnegative_penalty(error, torch.zeros(4, dtype=torch.bool), 1.0, 0.5)
    assert all(value.item() == 0 for value in empty)


@pytest.mark.parametrize(
    "name",
    [
        "root_acceleration_excess",
        "root_jerk_excess",
        "joint_angular_acceleration_excess",
        "joint_angular_jerk_excess",
        "fk_acceleration_excess",
        "fk_jerk_excess",
    ],
)
def test_excess_threshold_preserves_target_and_penalizes_extra_spike(config, name):
    term = config[name]
    target = torch.tensor([[float(term.allowance) * 5, 0.0, 0.0]])
    assert derivative_excess(target, target, term) == 0
    pred = (target * 3).requires_grad_()
    excess = derivative_excess(pred, target, term)
    assert excess.item() > 0
    excess.sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[0, 0] > 0


@pytest.mark.parametrize("kind", ["root", "fk", "joint_angular"])
def test_full_temporal_excess_uses_masks_fps_and_finite_gradients(config, kind):
    case = _example(config)
    pred_name, mask_name = {
        "root": ("pred_root", "root_accepted"),
        "fk": ("pred_fk", "fk_accepted"),
        "joint_angular": ("pred_joint_velocity", "joint_accepted"),
    }[kind]
    with torch.no_grad():
        case[pred_name][:, 3, ..., 0] = 100.0
    loss, logs = compute_smpl_support_losses(**case)
    for suffix in ("acceleration", "jerk"):
        assert _weighted(logs, f"{kind}_{suffix}_excess") > 0
    loss.backward()
    assert torch.isfinite(case[pred_name].grad).all()
    assert case[pred_name].grad.abs().sum() > 0
    case[mask_name][:] = False
    _, logs = compute_smpl_support_losses(**case)
    for suffix in ("acceleration", "jerk"):
        assert _weighted(logs, f"{kind}_{suffix}_excess") == 0
    # 同一位置序列采用两倍 FPS，加速度/jerk 分别扩大 4/8 倍。
    position = torch.arange(8, dtype=torch.float64).pow(3).reshape(1, 8, 1)
    for order in (2, 3):
        assert torch.allclose(
            finite_difference(position, order, 60),
            finite_difference(position, order, 30) * 2**order,
        )


def test_independent_warmup_and_autocast_stay_fp32(config):
    case = _example(config)
    with torch.no_grad():
        case["pred_sole"][..., 1] = 0.1
    full, _ = compute_smpl_support_losses(**case)
    case["global_step"] = 2500
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        half, logs = compute_smpl_support_losses(**case)
    assert half.dtype == torch.float32 and half.item() == pytest.approx(full.item() * 0.5)
    assert logs["physics_v4_weight_ramp_metric"] == 0.5
    case["global_step"] = 0
    zero, _ = compute_smpl_support_losses(**case)
    assert zero == 0
    half.backward()
    assert torch.isfinite(case["pred_sole"].grad).all()


def test_resume_warmup_starts_at_explicit_old_global_step(config):
    case = _example(config)
    with torch.no_grad():
        case["pred_sole"][..., 1] = 0.1
    full, _ = compute_smpl_support_losses(**case)
    config.warmup_start_step = 100000
    for step, fraction in ((99999, 0), (100000, 0), (102500, 0.5), (105000, 1), (200000, 1)):
        case["global_step"] = step
        value, logs = compute_smpl_support_losses(**case)
        assert value.item() == pytest.approx(full.item() * fraction)
        assert logs["physics_v4_weight_ramp_metric"] == fraction
    config.warmup_start_step = -1
    with pytest.raises(ValueError, match="warmup_start_step"):
        validate_support_config(config)


def test_resume_config_restores_full_state_and_adds_100k_steps():
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        value = compose(
            config_name="train",
            overrides=["exp=gem_smpl_music_only_4set_manual_q1_physics_v4_resume_100k"],
        )
    assert value.resume_mode == f"outputs/{V3}/version_0/checkpoints/s100000.ckpt"
    assert value.pretrain_ckpt is None and value.ckpt_path is None
    assert value.pl_trainer.max_steps == 200000
    assert value.pl_trainer.devices == 8 and value.pl_trainer.num_nodes == 1
    assert value.pl_trainer.accumulate_grad_batches == 1
    assert value.data.loader_opts.train.batch_size == 256
    assert value.optimizer.lr == pytest.approx(2.5e-6)
    assert list(value.scheduler.scheduler.milestones) == [60000, 85000]
    assert value.pipeline.args.physics_losses.support_losses.warmup_start_step == 100000
    assert value.pipeline.args.physics_losses.support_losses.warmup_steps == 5000
    assert value.pipeline.args.physics_losses.warmup_steps == 15000
    assert value.output_dir == "outputs/gem_smpl_music_only_4set_manual_q1_physics_v4_resume_100k"


@pytest.mark.parametrize(
    "key,value",
    [
        ("topk_fraction", 0.0),
        ("topk_fraction", 1.1),
        ("warmup_steps", 0),
        ("height_margin_m", -1),
        ("planted_speed_mps", float("nan")),
        ("foot_slide.scale", 0),
        ("foot_slide.weight", -1),
        ("root_jerk_excess.target_multiplier", 0.5),
        ("root_jerk_excess.allowance", float("inf")),
        ("contract_version", "wrong"),
    ],
)
def test_invalid_config_is_rejected(config, key, value):
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError):
        validate_support_config(config)


def test_canonical_sole_respects_camera_rotation_and_origin():
    relative = torch.arange(24, dtype=torch.float32).reshape(1, 1, 8, 3)
    translation = torch.tensor([[[3.0, 4.0, 5.0]]])
    root = torch.tensor([[[7.0, 8.0, 9.0]]])
    rotation = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]]])
    vertices = torch.zeros(1, 1, 437, 3)
    vertices[:, :, list(SOLE_V437_INDICES)] = relative + translation.unsqueeze(-2)
    result = canonical_sole_vertices(vertices, translation, rotation, root)
    expected = relative @ rotation[0, 0].T + root.unsqueeze(-2)
    assert torch.equal(result, expected)


@pytest.fixture(scope="module")
def experiments():
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        return tuple(compose(config_name="train", overrides=[f"exp={name}"]) for name in (V3, V4))


def test_v4_config_keeps_network_data_and_old_losses_but_isolates_finetuning(experiments):
    v3, v4 = experiments
    assert v4.exp_name == V4 and v4.output_dir == f"outputs/{V4}"
    assert v4.pretrain_ckpt == f"outputs/{V3}/version_0/checkpoints/s100000.ckpt"
    assert v4.resume_mode is None and v4.ckpt_path is None
    assert v4.optimizer.lr == pytest.approx(5e-6)
    assert v4.pl_trainer.max_steps == 30000
    assert list(v4.scheduler.scheduler.milestones) == [20000, 27000]
    for field in ("train_datasets", "data", "endecoder"):
        assert OmegaConf.to_container(v3[field], resolve=True) == OmegaConf.to_container(
            v4[field], resolve=True
        )
    base = OmegaConf.to_container(v3.pipeline.args.physics_losses, resolve=True)
    extended = OmegaConf.to_container(v4.pipeline.args.physics_losses, resolve=True)
    validate_support_config(extended.pop("support_losses"))
    assert base == extended
    networks = [OmegaConf.to_container(value.network, resolve=True) for value in (v3, v4)]
    for network in networks:
        network["args"].pop("physics_losses")
        network["model_cfg"]["denoiser"]["args"].pop("physics_losses")
    assert networks[0] == networks[1]


class _FakeEndecoder:
    """只提供测试所需 FK，避免加载真实人体模型或执行网络推理。"""

    def fk_v2(self, *, body_pose, betas, global_orient, transl):
        del betas, global_orient
        return (
            transl.unsqueeze(-2).expand(-1, -1, 22, -1) + body_pose[..., :3].unsqueeze(-2) * 0.001
        )


def _physics_case(config):
    case = _example(config)
    zeros3 = torch.zeros(1, 8, 3)
    vertices = torch.zeros(1, 8, 437, 3)
    vertices[:, :, list(SOLE_V437_INDICES)] = case["gt_sole"]
    pred_vertices = vertices.clone()
    pred_vertices[..., 1] += 0.10
    pred_vertices.requires_grad_()
    inputs = {
        "mask": {
            "valid": case["frame_valid"],
            "spv_incam_only": torch.tensor([False]),
            "2d_only": torch.tensor([False]),
        },
        "smpl_params_w": {
            "body_pose": torch.zeros(1, 8, 63),
            "betas": torch.zeros(1, 8, 10),
            "global_orient": zeros3,
            "transl": zeros3,
        },
        "smpl_params_c": {"global_orient": zeros3, "transl": zeros3},
        "R_c2gv": torch.eye(3).expand(1, 8, -1, -1),
        "physics": {"ground_y_local": case["ground_y_local"], "ground_valid": case["ground_valid"]},
        "gt_c_verts437": vertices.requires_grad_(),
        "meta": [{"dataset_id": "aist++"}],
    }
    outputs = {
        "decode_dict": {
            "body_pose": torch.zeros(1, 8, 63, requires_grad=True),
            "betas": torch.zeros(1, 8, 10),
            "global_orient": zeros3,
            "global_orient_gv": zeros3,
            "local_transl_vel": zeros3,
        },
        "pred_body_params_incam": {"transl": zeros3},
        "_pred_c_verts437": pred_vertices,
    }
    return inputs, outputs


def test_full_physics_integration_and_disabled_exact_parity(config, experiments):
    v3, v4 = experiments
    inputs, outputs = _physics_case(config)
    pipeline = SimpleNamespace(
        args={"physics_losses": v3.pipeline.args.physics_losses}, endecoder=_FakeEndecoder()
    )
    baseline, old_logs = compute_smpl_physics_losses(inputs, outputs, pipeline, global_step=15000)
    pipeline.args = {"physics_losses": deepcopy(v4.pipeline.args.physics_losses)}
    total, logs = compute_smpl_physics_losses(inputs, outputs, pipeline, global_step=15000)
    assert torch.equal(total, baseline + logs["physics_v4_total_loss"])
    assert total > baseline
    for key, value in old_logs.items():
        if key != "physics_total_loss":
            assert torch.equal(logs[key], value), key
    total.backward()
    assert torch.isfinite(outputs["_pred_c_verts437"].grad).all()
    assert torch.isfinite(outputs["decode_dict"]["body_pose"].grad).all()
    assert inputs["gt_c_verts437"].grad is None
    assert (outputs["_pred_c_verts437"].grad[:, :, list(SOLE_V437_INDICES), 1] > 0).all()
    # 关闭新项后不要求新增输入，结果及日志键集合和 v3 严格相同。
    pipeline.args["physics_losses"].support_losses.enabled = False
    inputs.pop("gt_c_verts437")
    inputs["smpl_params_c"].pop("transl")
    disabled, disabled_logs = compute_smpl_physics_losses(
        inputs, outputs, pipeline, global_step=15000
    )
    assert torch.equal(disabled, baseline)
    assert disabled_logs.keys() == old_logs.keys()
    assert all(torch.equal(value, old_logs[key]) for key, value in disabled_logs.items())
    pipeline.args["physics_losses"].support_losses.enabled = True
    with pytest.raises(ValueError, match="cached GT"):
        compute_smpl_physics_losses(inputs, outputs, pipeline, global_step=15000)


def test_projection_does_not_mutate_support_ground_truth(config, monkeypatch):
    from gem.pipeline import gem_pipeline

    inputs, outputs = _physics_case(config)
    inputs["gt_c_verts437"] = inputs["gt_c_verts437"].detach()
    original = inputs["gt_c_verts437"].clone()
    inputs["smpl_params_c"] = inputs["smpl_params_w"]
    inputs["bbx_xys"] = torch.ones(1, 8, 3)
    inputs["K_fullimg"] = torch.eye(3).expand(1, 8, -1, -1)
    outputs["pred_body_params_incam"] = inputs["smpl_params_w"]
    outputs["model_output"] = {}
    predicted = outputs["_pred_c_verts437"]
    endecoder = _FakeEndecoder()
    endecoder.smplx_model = lambda **kwargs: (predicted, torch.zeros(1, 8, 17, 3))
    pipeline = SimpleNamespace(
        endecoder=endecoder,
        args={},
        weights=OmegaConf.create(dict(cr_j3d=0, transl_c=0, j2d=0, verts2d=1)),
    )
    monkeypatch.setattr(
        gem_pipeline, "project_to_bi01", lambda verts, *args: verts[..., :2] * 0 + 0.5
    )
    loss, _ = gem_pipeline.compute_extra_incam_loss(inputs, outputs, pipeline, mode="generation")
    assert torch.isfinite(loss)
    assert torch.equal(original, inputs["gt_c_verts437"])
    assert outputs["_pred_c_verts437"] is predicted
