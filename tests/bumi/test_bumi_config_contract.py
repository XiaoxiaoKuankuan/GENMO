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


def test_qpos30_contact_50k_contract_restores_main_style_feet_and_root_losses() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=["exp=gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_50k"],
        )

    denoiser = config.network.model_cfg.denoiser
    weights = config.pipeline.args.weights
    assert denoiser.output_dim == denoiser.xt_dim == denoiser.njoints == 30
    assert denoiser.static_conf_dim == 2
    assert config.endecoder.feat_dim == 30
    assert config.endecoder.clip_std_min == 0.01
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v2"
    assert config.pipeline.args.ground_semantics == "mixed_floor_zero_fk_contact_v2"
    assert "repr_body_pos" not in weights
    assert "fk_consistency" not in weights
    assert weights.repr_root_rot == 2.0
    assert weights.root_rot == 1.0
    assert weights.root_tilt == 1.0
    assert 0.0 < weights.foot_slide <= 0.05
    assert weights.contact_bce > 0.0
    assert config.model.model_cfg.checkpoint_adapter == "smpl_music_to_bumi"
    assert config.pretrain_ckpt is None
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"
    assert config.pl_trainer.max_steps == 50000


def test_qpos30_contact_scratch_350k_is_eight_gpu_random_initialization() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=["exp=gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_scratch_350k"],
        )

    denoiser = config.network.model_cfg.denoiser
    assert denoiser.output_dim == denoiser.xt_dim == denoiser.njoints == 30
    assert denoiser.static_conf_dim == 2
    assert config.endecoder.feat_dim == 30
    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.model.model_cfg.checkpoint_adapter is None
    assert config.data.loader_opts.train.batch_size == 192
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"
    assert config.pl_trainer.max_steps == 350000
    assert config.pl_trainer.val_check_interval == 5000
    assert config.optimizer.lr == 1.0e-4
    assert list(config.scheduler.scheduler.milestones) == [210000, 315000]
    assert config.pipeline.args.ground_semantics == "mixed_floor_zero_fk_contact_v2"
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v2"


def test_robot_retargeter_pass_v1_scratch_uses_only_new_four_set_contract() -> None:
    """新资产四库必须 PASS-only、严格限位且不继承旧模型。"""

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=[
                "exp=gem_bumi_music_only_4set_robot_retargeter_pass_v1_qpos30_contact_scratch_350k"
            ],
        )

    assert list(config.train_datasets) == [
        "aistpp_bumi_train",
        "aioz_gdance_bumi_train",
        "finedance_bumi_train",
        "compas3d_bumi_train",
    ]
    assert "mine_bumi_train" not in config.train_datasets
    assert config.data.expected_train_sequences == 2455
    assert dict(config.data.dataset_sampling_weights) == {
        "aistpp_bumi": 0.29,
        "aioz_gdance_bumi": 0.60,
        "finedance_bumi": 0.07,
        "compas3d_bumi": 0.04,
    }
    assert sum(config.data.dataset_sampling_weights.values()) == 1.0
    assert all(value.joint_limit_tolerance == 0.0001 for value in config.train_datasets.values())
    assert config.test_datasets.finedance_bumi_music_eval.split == "test"
    data_config = OmegaConf.to_container(config.data, resolve=False)
    assert data_config["stats_path"] == "${oc.env:BUMI_MUSIC_QPOS30_STATS_PATH}"
    assert config.network.model_cfg.denoiser.output_dim == 30
    assert config.network.model_cfg.denoiser.static_conf_dim == 2
    assert config.endecoder.feat_dim == 30
    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.model.model_cfg.checkpoint_adapter is None
    assert config.data.loader_opts.train.batch_size == 192
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"
    assert config.pl_trainer.max_steps == 350000
    assert list(config.scheduler.scheduler.milestones) == [210000, 315000]
    assert config.pipeline.args.ground_semantics == "mixed_floor_zero_fk_contact_v2"
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v2"
    assert config.use_wandb is False


def test_qpos30_contact_v3_uses_stronger_temporal_and_excess_losses() -> None:
    """v3 必须显式启用更强导数匹配及只惩罚预测超额的两个新项。"""

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=[
                "exp=gem_bumi_music_only_4set_robot_retargeter_pass_v1_qpos30_contact_scratch_350k",
                "pipeline=music_only_bumi_qpos30_contact_v3",
            ],
        )

    weights = config.pipeline.args.weights
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v3"
    assert weights.joint_velocity == 0.10
    assert weights.joint_acceleration == 0.01
    assert weights.joint_jerk == 0.003
    assert weights.joint_acceleration_excess == 0.05
    assert weights.joint_jerk_excess == 0.003
    assert config.pipeline.args.auxiliary_warmup_steps == 5000


def test_robot_retargeter_pass_v2_continues_weights_for_new_200k_steps() -> None:
    """QP PASS v2 必须使用 weights-only、batch256 和独立 200k 调度。"""

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=[
                "exp=gem_bumi_music_only_4set_robot_retargeter_pass_v2_"
                "qpos30_contact_v3_continue_200k"
            ],
        )

    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.model.model_cfg.checkpoint_adapter is None
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v3"
    assert config.data.expected_train_sequences == 2479
    assert config.data.loader_opts.train.batch_size == 256
    assert config.data.samples_per_epoch == 53248
    assert config.data.samples_per_epoch % (8 * config.data.loader_opts.train.batch_size) == 0
    assert config.optimizer.lr == 2.0e-5
    assert config.pl_trainer.max_steps == 200000
    assert config.pl_trainer.val_check_interval == 5000
    assert list(config.scheduler.scheduler.milestones) == [120000, 180000]
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"


def test_qpos30_contact_v4_adds_margin_topk_max_and_full_resume_entry() -> None:
    """v4必须显式强化稀疏越限，并保留从完整checkpoint恢复到20万步的入口。"""

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=[
                "exp=gem_bumi_music_only_4set_robot_retargeter_pass_v2_"
                "qpos30_contact_v4_resume_200k"
            ],
        )

    args = config.pipeline.args
    assert args.loss_contract == "physical_qpos30_contact_v4"
    assert args.joint_limit_margin_rad == 0.05
    assert args.joint_limit_topk_fraction == 0.01
    assert args.robust_joint_limit_warmup_steps == 5000
    assert args.weights.joint_limit == 0.1
    assert args.weights.joint_limit_margin == 0.2
    assert args.weights.joint_limit_topk == 0.5
    assert args.weights.joint_limit_max == 0.05
    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.pl_trainer.max_steps == 200000
    assert list(config.scheduler.scheduler.milestones) == [120000, 180000]
