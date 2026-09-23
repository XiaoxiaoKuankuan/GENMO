"""验证当前 BUMI 音乐实验的完整 Hydra 组合与训练边界。

本文件覆盖默认 UMR70+Mine 的四来源划分、120 帧 EDGE35 条件、
qpos30/contact2 网络、v5 损失和从头训练参数；同时保留五库 v5 的
显式权重续训回归。临时路径只用于配置解析，不写入生产数据或伪造统计量，
也不实例化网络、启动训练或访问训练服务器。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_EXP = "gem_bumi_music_only_umr70_mine_scratch_350k"


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/demo/demo_music_bumi.py",
        "tools/export/export_bumi_music_onnx.py",
        "tools/eval/select_bumi_checkpoints.py",
        "tools/eval/validate_bumi_music_onnx.py",
        "tools/eval/validate_bumi_music_tensorrt.py",
    ],
)
def test_cli_experiment_default_matches_current_training(relative_path) -> None:
    """静态检查 CLI 默认值，避免仅为测试加载真实模型或 GPU 后端。"""

    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    defaults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not any(isinstance(arg, ast.Constant) and arg.value == "--exp" for arg in node.args):
            continue
        defaults.extend(ast.literal_eval(kw.value) for kw in node.keywords if kw.arg == "default")
    assert defaults == [CURRENT_EXP]


@pytest.fixture
def current_config(monkeypatch, tmp_path):
    for dataset in ("AISTPP", "AIOZ_GDANCE", "FINEDANCE", "MINE"):
        monkeypatch.setenv(f"{dataset}_BUMI_ROOT", str(tmp_path / dataset))
    monkeypatch.setenv(
        "BUMI_KINEMATICS_PATH",
        str(REPO_ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"),
    )
    monkeypatch.setenv("BUMI_MUSIC_QPOS30_STATS_PATH", str(tmp_path / "train_stats.json"))
    monkeypatch.setenv("BUMI_EXPECTED_TRAIN_SEQUENCES", "4149")
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        return compose(config_name="train")


def test_default_matches_explicit_current_experiment(current_config) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        explicit = compose(config_name="train", overrides=[f"exp={CURRENT_EXP}"])
    assert OmegaConf.to_container(current_config, resolve=True) == OmegaConf.to_container(
        explicit, resolve=True
    )


def test_current_network_and_condition_contract(current_config) -> None:
    config = current_config
    denoiser = config.network.model_cfg.denoiser
    assert denoiser.output_dim == denoiser.xt_dim == denoiser.njoints == 30
    assert denoiser.pred_cam_dim == 0
    assert denoiser.static_conf_dim == 2
    assert denoiser.encode_text is False
    assert denoiser.input_remove_global is False
    assert config.endecoder.feat_dim == 30
    assert config.model.model_cfg.motion_backend == "bumi"
    assert config.model.model_cfg.text_encoder is None
    assert list(config.pipeline.args.train_modes) == ["diffusion"]
    assert list(config.pipeline.args.in_attr) == ["encoded_music"]
    assert config.pipeline.args.encoded_music_dim == 35
    assert dict(config.pipeline.args.out_attr) == {"static_conf_logits": 2}
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v5"
    assert config.pipeline.args.weights.contact_bce > 0
    assert config.pipeline.args.weights.foot_slide > 0


def test_umr70_four_sources_keep_train_validation_boundaries(current_config) -> None:
    config = current_config
    assert list(config.train_datasets) == [
        "aistpp_bumi_train",
        "aioz_gdance_bumi_train",
        "finedance_bumi_train",
        "mine_bumi_train",
    ]
    assert list(config.test_datasets) == [
        "aistpp_bumi_music_eval",
        "aioz_gdance_bumi_music_eval",
        "finedance_bumi_music_eval",
    ]
    assert all(value.split == "train" for value in config.train_datasets.values())
    assert all(value.split == "val" for value in config.test_datasets.values())
    for value in [*config.train_datasets.values(), *config.test_datasets.values()]:
        assert value.motion_frames == 120
        assert value.strict_contract and value.strict_alignment and value.require_quality_filter
        assert value.joint_limit_tolerance == 0.0001
    assert config.data.sampling_strategy == "deduplicated_hierarchical"
    assert config.data.require_stats_fingerprint_match
    assert int(config.data.expected_train_sequences) == 4149
    assert config.data.stats_path == config.endecoder.stats_path
    assert dict(config.data.dataset_sampling_weights) == {
        "aistpp_bumi": 0.28,
        "aioz_gdance_bumi": 0.57,
        "finedance_bumi": 0.065,
        "mine_bumi": 0.05,
    }


def test_umr70_scratch_training_parameters(current_config) -> None:
    config = current_config
    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.model.model_cfg.checkpoint_adapter is None
    assert config.data.loader_opts.train.batch_size == 256
    assert config.data.samples_per_epoch == 53248
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"
    assert config.pl_trainer.max_steps == 350000
    assert config.pl_trainer.val_check_interval == 5000
    assert config.optimizer.lr == 1e-4
    assert list(config.scheduler.scheduler.milestones) == [210000, 315000]


def test_compatible_five_set_config_runs_v5_weights_only_200k(
    monkeypatch,
) -> None:
    """保留五库 v5 兼容入口，显式 checkpoint 的续训契约不变。"""

    monkeypatch.setenv("BUMI_PRETRAIN_CKPT", "/tmp/verified-latest.ckpt")

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(
            config_name="train",
            overrides=[
                "exp=gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_latest"
            ],
        )

    args = config.pipeline.args
    assert args.loss_contract == "physical_qpos30_contact_v5"
    assert args.joint_limit_margin_rad == 0.05
    assert args.joint_limit_topk_fraction == 0.01
    assert args.robust_joint_limit_warmup_steps == 10000
    assert args.robust_joint_limit_start_step == 0
    assert args.advanced_physics_warmup_steps == 10000
    assert args.advanced_physics_start_step == 0
    assert args.advanced_physics_topk_fraction == 0.05
    assert args.root_tilt_target_margin_rad == 0.05
    assert args.weights.joint_velocity == 0.15
    assert args.weights.joint_acceleration == 0.04
    assert args.weights.joint_jerk == 0.006
    assert args.weights.joint_acceleration_excess == 0.20
    assert args.weights.joint_jerk_excess == 0.006
    assert args.weights.joint_limit == 0.40
    assert args.weights.joint_limit_margin == 0.80
    assert args.weights.joint_limit_topk == 2.00
    assert args.weights.joint_limit_max == 0.20
    assert args.weights.joint_limit_margin_topk == 0.50
    assert args.weights.foot_slide == 0.10
    assert args.weights.foot_slide_topk == 0.10
    assert args.weights.foot_contact_height == 0.10
    assert args.weights.penetration == 0.10
    assert args.weights.root_velocity == 0.05
    assert args.weights.root_angular_acceleration == 0.02
    assert args.weights.fk_acceleration == 0.02
    assert list(config.train_datasets) == [
        "aistpp_bumi_train",
        "aioz_gdance_bumi_train",
        "finedance_bumi_train",
        "compas3d_bumi_train",
        "mine_bumi_train",
    ]
    assert all(value.joint_limit_tolerance == 0.0001 for value in config.train_datasets.values())
    assert config.data.expected_train_sequences == 2479 + 99
    assert sum(config.data.dataset_sampling_weights.values()) == 1.0
    assert config.data.dataset_sampling_weights.mine_bumi == 0.05
    assert config.data.loader_opts.train.batch_size == 256
    assert config.data.samples_per_epoch == 53248
    assert config.data.samples_per_epoch % (8 * config.data.loader_opts.train.batch_size) == 0
    assert config.pretrain_ckpt == "/tmp/verified-latest.ckpt"
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.model.model_cfg.checkpoint_adapter is None
    assert config.pl_trainer.max_steps == 200000
    assert list(config.scheduler.scheduler.milestones) == [120000, 180000]
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"
