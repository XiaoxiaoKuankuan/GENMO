from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
    text = (REPO_ROOT / "configs/exp/gem_bumi_music_only_4set.yaml").read_text(encoding="utf-8")
    for expected in (
        "music_robot/trainX_testY",
        "bumi_music_gem",
        "diffusion_lg_bumi93",
        "music_only_bumi",
        "bumi_93d",
    ):
        assert expected in text


def test_random_v1_disables_ground_contact_and_uses_balanced_sampling() -> None:
    assert (REPO_ROOT / "configs/data/music_robot/trainX_testY.yaml").is_file()
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
    assert experiment["data"]["require_stats_fingerprint_match"] is True
    assert experiment["data"]["expected_train_sequences"] == 5537
    assert experiment["pl_trainer"]["max_steps"] == 500000
    assert experiment["pl_trainer"]["devices"] == 8


def test_mine_five_dataset_config_composes_with_exact_training_contract() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(config_name="train", overrides=["exp=gem_bumi_music_only_5set_mine_v1"])

    assert list(config.train_datasets) == [
        "aistpp_bumi_train",
        "aioz_gdance_bumi_train",
        "finedance_bumi_train",
        "compas3d_bumi_train",
        "mine_bumi_train",
    ]
    mine_config = OmegaConf.to_container(config.train_datasets.mine_bumi_train, resolve=False)
    assert mine_config["root"] == "${oc.env:MINE_BUMI_ROOT}"
    assert config.train_datasets.mine_bumi_train.dataset_name == "mine_bumi"
    assert all(value.joint_limit_tolerance == 0.25 for value in config.train_datasets.values())
    assert config.data.expected_train_sequences == 2623 + 99
    assert dict(config.data.dataset_sampling_weights) == {
        "aistpp_bumi": 0.29,
        "aioz_gdance_bumi": 0.47,
        "finedance_bumi": 0.16,
        "compas3d_bumi": 0.03,
        "mine_bumi": 0.05,
    }
    assert sum(config.data.dataset_sampling_weights.values()) == 1.0


def test_manual_q1_v3_uses_large_batch_and_mixed_ground_contract() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train", overrides=["exp=gem_bumi_music_only_5set_manual_q1_v3"]
        )

    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.data.expected_train_sequences == 2792 + 99
    assert config.data.loader_opts.train.batch_size == 192
    assert config.data.samples_per_epoch == 52224
    assert config.data.samples_per_epoch % (8 * config.data.loader_opts.train.batch_size) == 0
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.max_steps == 350000
    assert list(config.scheduler.scheduler.milestones) == [210000, 315000]
    assert config.pipeline.args.ground_semantics == "mixed_floor_zero_no_contact_v1"
    assert config.pipeline.args.weights.contact_bce == 0.0
    assert config.pipeline.args.weights.foot_slide == 0.0
    assert config.pipeline.args.weights.penetration == 0.0
    for dataset_name in (
        "aistpp_bumi_train",
        "aioz_gdance_bumi_train",
        "finedance_bumi_train",
        "compas3d_bumi_train",
    ):
        assert config.train_datasets[dataset_name].joint_limit_tolerance == 0.001
    assert config.train_datasets.mine_bumi_train.joint_limit_tolerance == 0.25
    for dataset_name in (
        "aistpp_bumi_music_eval",
        "aioz_gdance_bumi_music_eval",
        "finedance_bumi_music_eval",
        "compas3d_bumi_music_eval",
    ):
        assert config.test_datasets[dataset_name].joint_limit_tolerance == 0.001
    assert config.use_wandb is False


def test_manual_q1_v3_finetune_50k_preserves_data_and_tunes_physics() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=["exp=gem_bumi_music_only_5set_manual_q1_v3_finetune_50k"],
        )

    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.data.expected_train_sequences == 2792 + 99
    assert dict(config.data.dataset_sampling_weights) == {
        "aistpp_bumi": 0.29,
        "aioz_gdance_bumi": 0.47,
        "finedance_bumi": 0.16,
        "compas3d_bumi": 0.03,
        "mine_bumi": 0.05,
    }
    assert config.data.loader_opts.train.batch_size == 192
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.max_steps == 50000
    assert config.pl_trainer.val_check_interval == 5000
    assert config.optimizer.lr == 2e-5
    assert list(config.scheduler.scheduler.milestones) == [30000, 45000]
    assert config.pipeline.args.ground_semantics == "mixed_floor_zero_no_contact_v1"
    assert config.pipeline.args.weights.repr_body_pos == 1.0
    assert config.pipeline.args.weights.joint_velocity == 0.1
    assert config.pipeline.args.weights.joint_acceleration == 0.01
    assert config.pipeline.args.weights.joint_jerk == 0.005
    assert config.pipeline.args.weights.joint_limit == 0.5
    assert config.pipeline.args.weights.contact_bce == 0.0
    assert config.pipeline.args.weights.foot_slide == 0.0
    assert config.pipeline.args.weights.penetration == 0.0
    assert config.use_wandb is False
