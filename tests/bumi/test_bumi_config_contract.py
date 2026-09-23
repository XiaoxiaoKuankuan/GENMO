from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative: str):
    return yaml.safe_load((REPO_ROOT / relative).read_text(encoding="utf-8"))


def test_train_config_requires_explicit_experiment_selection() -> None:
    """裸训练入口必须在 Hydra 合成阶段要求 exp，不能落入不存在的隐式默认值。"""

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        with pytest.raises(ConfigCompositionException, match="must specify 'exp'"):
            compose(config_name="train")


def test_qpos30_network_and_model_contract() -> None:
    """现行 BUMI 网络必须只暴露 30D qpos 和独立 2D 接触头。"""

    config = load_yaml("configs/network/diffusion_lg_bumi30_contact.yaml")
    denoiser = config["model_cfg"]["denoiser"]
    assert denoiser["output_dim"] == 30
    assert denoiser["xt_dim"] == 30
    assert denoiser["njoints"] == 30
    assert denoiser["pred_cam_dim"] == 0
    assert denoiser["static_conf_dim"] == 2
    assert denoiser["avgbeta"] is False
    assert denoiser["encode_text"] is False
    assert denoiser["input_remove_global"] is False
    model = load_yaml("configs/model/bumi_music_gem.yaml")["model_cfg"]
    assert model["motion_backend"] == "bumi"
    assert model["train_modes"] == ["diffusion"]
    assert model["text_encoder"] is None


def test_qpos30_v5_pipeline_excludes_smpl_losses() -> None:
    """v5 训练管线必须维持 qpos30/FK 契约且不能回引 SMPL 专用损失。"""

    pipeline = load_yaml("configs/pipeline/music_only_bumi_qpos30_contact_v5.yaml")["args"]
    assert pipeline["train_modes"] == ["diffusion"]
    assert pipeline["in_attr"] == ["encoded_music"]
    assert set(pipeline["out_attr"]) == {"static_conf_logits"}
    assert pipeline["out_attr"]["static_conf_logits"] == 2
    assert pipeline["loss_contract"] == "physical_qpos30_contact_v5"
    assert pipeline["ground_semantics"] == "mixed_floor_zero_fk_contact_v2"
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
    assert pipeline["weights"]["joint_limit_topk"] > 0.0
    assert pipeline["weights"]["foot_slide_topk"] > 0.0


def test_formal_v5_scratch_s350000_config_matches_archived_training_contract() -> None:
    """正式配置必须冻结 2026-09-09 s350000 训练快照的关键超参数。"""

    experiment = (
        "gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000"
    )
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        config = compose(config_name="train", overrides=[f"exp={experiment}"])

    assert config.exp_name_base == (
        "gem_bumi_music_only_5set_robot_retargeter_pass_v2_"
        "qpos30_contact_v5_scratch_b256_s350k_20260909"
    )
    assert config.data_name == config.exp_name_base
    assert config.pretrain_ckpt is None
    assert config.ckpt_path is None
    assert config.resume_mode is None
    assert config.pipeline.args.loss_contract == "physical_qpos30_contact_v5"
    assert config.network.model_cfg.denoiser.output_dim == 30
    assert config.network.model_cfg.denoiser.static_conf_dim == 2
    assert config.endecoder.feat_dim == 30
    assert list(config.train_datasets) == [
        "aistpp_bumi_train",
        "aioz_gdance_bumi_train",
        "finedance_bumi_train",
        "compas3d_bumi_train",
        "mine_bumi_train",
    ]
    assert config.data.expected_train_sequences == 2578
    assert config.data.loader_opts.train.batch_size == 256
    assert config.data.samples_per_epoch == 53248
    assert dict(config.data.dataset_sampling_weights) == {
        "aistpp_bumi": 0.28,
        "aioz_gdance_bumi": 0.57,
        "finedance_bumi": 0.065,
        "compas3d_bumi": 0.035,
        "mine_bumi": 0.05,
    }
    assert all(value.joint_limit_tolerance == 0.0001 for value in config.train_datasets.values())
    assert config.optimizer.lr == 1.0e-4
    assert list(config.scheduler.scheduler.milestones) == [210000, 315000]
    assert config.pl_trainer.max_steps == 350000
    assert config.pl_trainer.devices == 8
    assert config.pl_trainer.strategy == "ddp"
    assert config.pl_trainer.accumulate_grad_batches == 1
    assert config.use_wandb is False


def test_bumi_deployment_clis_default_to_frozen_v5_s350000_config() -> None:
    """所有现行生成、导出与 parity CLI 都必须默认使用可复现的正式 v5 配置。"""

    expected = "gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000"
    consumers = (
        "scripts/demo/demo_music_bumi.py",
        "tools/export/export_bumi_music_onnx.py",
        "tools/eval/select_bumi_checkpoints.py",
        "tools/eval/validate_bumi_music_onnx.py",
        "tools/eval/validate_bumi_music_tensorrt.py",
    )
    defaults: dict[str, str] = {}
    for relative in consumers:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument" or not node.args:
                continue
            if not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--exp":
                continue
            default = next(keyword.value for keyword in node.keywords if keyword.arg == "default")
            defaults[relative] = ast.literal_eval(default)
    assert defaults == {relative: expected for relative in consumers}
