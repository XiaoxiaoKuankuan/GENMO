from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative: str):
    return yaml.safe_load((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_network_contract() -> None:
    config = load_yaml("configs/network/diffusion_lg_bumi93.yaml")
    denoiser = config["model_cfg"]["denoiser"]
    assert denoiser["output_dim"] == 93
    assert denoiser["xt_dim"] == 93
    assert denoiser["njoints"] == 93
    assert denoiser["pred_cam_dim"] == 0
    assert denoiser["static_conf_dim"] == 2
    assert denoiser["avgbeta"] is False
    assert denoiser["encode_text"] is False
    assert denoiser["input_remove_global"] is False


def test_pipeline_and_model_contract() -> None:
    pipeline = load_yaml("configs/pipeline/music_only_bumi.yaml")["args"]
    model = load_yaml("configs/model/bumi_music_gem.yaml")["model_cfg"]
    assert pipeline["train_modes"] == ["diffusion"]
    assert pipeline["in_attr"] == ["encoded_music"]
    assert set(pipeline["out_attr"]) == {"static_conf_logits"}
    assert "pred_cam" not in pipeline["out_attr"]
    smpl_losses = {
        "cr_j3d",
        "cr_verts",
        "j2d",
        "j2d_17",
        "verts2d",
        "transl_c",
        "transl_w",
        "shape_loss",
    }
    assert not (smpl_losses & set(pipeline["weights"]))
    assert model["motion_backend"] == "bumi"
    assert model["train_modes"] == ["diffusion"]
    assert model["text_encoder"] is None


def test_experiment_selects_only_new_bumi_layers() -> None:
    text = (REPO_ROOT / "configs/exp/gem_bumi_music_only_4set.yaml").read_text(
        encoding="utf-8"
    )
    for expected in (
        "music_robot/trainX_testY",
        "bumi_music_gem",
        "diffusion_lg_bumi93",
        "music_only_bumi",
        "bumi_93d",
    ):
        assert expected in text


def test_random_v1_disables_ground_contact_and_uses_balanced_sampling() -> None:
    pipeline = load_yaml("configs/pipeline/music_only_bumi_physical_v1.yaml")["args"]
    network = load_yaml("configs/network/diffusion_lg_bumi93_no_contact.yaml")
    experiment = load_yaml("configs/exp/gem_bumi_music_only_4set_random_v1.yaml")
    assert pipeline["loss_contract"] == "physical_v1"
    assert pipeline["auxiliary_warmup_steps"] == 10000
    assert pipeline["out_attr"] == []
    for name in ("contact_bce", "foot_slide", "penetration", "joint_jerk"):
        assert pipeline["weights"][name] == 0.0
    assert network["model_cfg"]["denoiser"]["static_conf_dim"] == 0
    assert experiment["pretrain_ckpt"] is None
    assert experiment["data"]["sampling_strategy"] == "deduplicated_hierarchical"
    assert experiment["data"]["samples_per_epoch"] == 52224
    assert experiment["pl_trainer"]["max_steps"] == 500000
    assert experiment["pl_trainer"]["devices"] == 8
