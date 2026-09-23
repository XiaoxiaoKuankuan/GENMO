"""Stage 1 独立入口、严格权重迁移和 checkpoint 的小规模回归验证。

测试复用第3步真实Dataset夹具和真实BUMI Endecoder/Transformer的小型Actor，临时stats
明确标记placeholder且全部产物写入pytest临时目录。覆盖旧qpos30/contact2权重严格映射、
新增分支单独初始化、缺失/多余/形状错误拒绝、旧step不恢复、接口及资产身份检查、四库
相对采样权重、已有Dataset接通、有限反向优化和纯条件采样验证。不会读取正式训练数据，
不会生成正式统计量或启动第5步训练，不能据此声称收敛、动力学或实机性能。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import (
    CheckpointCompatibilityError,
    actor_asset_identity,
    load_stage1_checkpoint,
    save_stage1_checkpoint,
    warm_start_weights,
)
from gem.closedloop.training import (
    build_stage1_actor,
    build_stage1_loader,
    build_stage1_losses,
    load_stage1_config,
    train_stage1_steps,
    validate_stage1_batches,
)
from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory
from tests.closedloop.test_stage1_dataset import KINEMATICS_PATH, _write_dataset
from tools.train_closedloop_stage1 import main as stage1_main


def _legacy(actor):
    prefixes = {
        "denoiser.": "pipeline.denoiser3d.denoiser.",
        "music_embedder.": "music_embedder.",
        "cond_exists_embedder.": "cond_exists_embedder.encoded_music.",
    }
    state = {}
    for key, value in actor.state_dict().items():
        for new, old in prefixes.items():
            if key.startswith(new):
                state[old + key[len(new) :]] = value.clone()
    return {
        "state_dict": state,
        "bumi_representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "global_step": 350000,
        "optimizer_states": [{"must_not_be_restored": True}],
    }


def _data_config(root: Path, stats_path: str, *, history_steps=4):
    return OmegaConf.create(
        {
            "qpos30_stats": {"path": stats_path},
            "sample_contract": {"history_steps": history_steps},
            "dataset_defaults": {"kinematics_path": str(KINEMATICS_PATH)},
            "datasets": {
                "train": {
                    "synthetic": {
                        "_target_": "gem.closedloop.stage1_dataset.BumiClosedLoopStage1Dataset",
                        "root": str(root),
                        "dataset_name": "test_bumi",
                        "split": "train",
                        "kinematics_path": str(KINEMATICS_PATH),
                        "history_steps": history_steps,
                        "prefix_min_frames": 2,
                        "prefix_max_frames": 5,
                        "random_decision": False,
                        "duration_aware_sampling": False,
                        "joint_limit_tolerance": 0.0001,
                    }
                }
            },
            "train_sampling_reference": {"test_bumi": 0.7},
        }
    )


def test_warm_start_loads_every_old_weight_and_keeps_new_branches(actor_factory):
    source, _ = actor_factory(history_steps=4, starts=(45,))
    destination = copy.deepcopy(source)
    for parameter in destination.parameters():
        with torch.no_grad():
            parameter.add_(0.13)
    old = _legacy(source)
    before = {key: value.clone() for key, value in destination.state_dict().items()}
    report = warm_start_weights(destination, old, source_assets=actor_asset_identity(source))
    assert report["optimizer_restored"] is False and report["global_step_restored"] is False
    assert report["source_global_step_for_provenance_only"] == 350000
    assert report["source_asset_identity"] == "verified_exact_content"
    assert not report["missing"] and not report["unexpected"] and not report["shape_conflicts"]
    assert len(report["loaded"]) == len(old["state_dict"])
    assert report["new"]
    for entry in report["loaded"]:
        torch.testing.assert_close(
            destination.state_dict()[entry["destination"]],
            old["state_dict"][entry["source"]],
            rtol=0,
            atol=0,
        )
    for key in report["new"]:
        torch.testing.assert_close(destination.state_dict()[key], before[key], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "version", "heads"])
def test_warm_start_rejects_unexpected_errors_before_mutating(actor_factory, failure):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    checkpoint = _legacy(actor)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    extra = {}
    if failure == "missing":
        checkpoint["state_dict"].pop("music_embedder.fc1.bias")
    elif failure == "unexpected":
        checkpoint["state_dict"]["unrelated_actor.weight"] = torch.zeros(1)
    elif failure == "shape":
        checkpoint["state_dict"]["music_embedder.fc1.bias"] = torch.zeros(1)
    elif failure == "version":
        checkpoint["bumi_representation_contract_version"] = "unsupported"
    else:
        backbone = dict(actor.interface_config["backbone"])
        backbone["num_heads"] = 8
        extra["source_model_config"] = {"backbone": backbone}
    with pytest.raises(RuntimeError):
        warm_start_weights(actor, checkpoint, **extra)
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_warm_start_rejects_different_stats_identity(actor_factory):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    identity = actor_asset_identity(actor)
    identity["stats_sha256"] = "0" * 64
    with pytest.raises(CheckpointCompatibilityError, match="assets differ"):
        warm_start_weights(actor, _legacy(actor), source_assets=identity)


def test_stage1_checkpoint_roundtrip_binds_interface_assets_and_shapes(actor_factory, tmp_path):
    actor, _ = actor_factory(history_steps=4, starts=(45,))
    path = tmp_path / "stage1_actor.pt"
    payload = save_stage1_checkpoint(actor, path, config={"test_only": True}, global_step=2)
    other = copy.deepcopy(actor)
    with torch.no_grad():
        other.history_encoder.out_proj.bias.fill_(0.7)
    report = load_stage1_checkpoint(other, path)
    assert report["global_step_restored"] is False
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
    assert payload["actor_interface_config"]["history_steps"] == 4
    assert payload["global_step"] == 2
    with pytest.raises(FileExistsError):
        save_stage1_checkpoint(actor, path, config={}, global_step=3)
    corrupted = copy.deepcopy(payload)
    corrupted["actor_interface_config"]["history_steps"] = 50
    with pytest.raises(RuntimeError, match="actor_interface_config"):
        load_stage1_checkpoint(actor, corrupted)
    corrupted = copy.deepcopy(payload)
    corrupted["asset_identity"]["stats_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="asset identity"):
        load_stage1_checkpoint(actor, corrupted)


def test_dataset_builder_small_optimizer_and_condition_only_validation(actor_factory, tmp_path):
    reference, _ = actor_factory(history_steps=4, starts=(45,))
    root = _write_dataset(tmp_path / "entrypoint_dataset", length=130)
    data = _data_config(root, reference.endecoder.stats_path)
    config = load_stage1_config(
        "configs/closedloop/stage1_train.yaml",
        [
            "model.latent_dim=32",
            "model.num_layers=1",
            "model.num_heads=4",
            "model.history_hidden_dim=16",
            "model.dropout=0.0",
            "model.music_mask_prob=0.0",
            "endecoder.allow_placeholder_stats=true",
        ],
    )
    actor = build_stage1_actor(config, data)
    losses = build_stage1_losses(actor, config)
    loader = build_stage1_loader(data, split="train", batch_size=1, samples_per_epoch=2, seed=8)
    assert len(loader.dataset) == 1
    assert loader.sampler.weights.sum().item() == pytest.approx(0.7)
    before = actor.history_encoder.out_proj.weight.detach().clone()
    report = train_stage1_steps(actor, losses, loader, max_steps=2, learning_rate=1.0e-5)
    assert report["optimizer_start_step"] == 0 and report["completed_steps"] == 2
    assert all(value["gradient_norm"] > 0 for value in report["steps"])
    assert not torch.equal(before, actor.history_encoder.out_proj.weight)
    sampled = validate_stage1_batches(actor, losses, loader, max_batches=1, sample_steps=2)
    assert sampled["completed_batches"] == 1
    assert sampled["sampling_received_target_fields"] is False
    assert sampled["batches"][0]["known_max_abs_error"] == 0.0


def test_builder_rejects_different_dataset_asset(actor_factory, tmp_path):
    reference, _ = actor_factory(history_steps=4, starts=(45,))
    data = _data_config(tmp_path / "unused", reference.endecoder.stats_path)
    changed = tmp_path / "different_kinematics.json"
    changed.write_text(KINEMATICS_PATH.read_text() + "\n", encoding="utf-8")
    data.datasets.train.synthetic.kinematics_path = str(changed)
    config = load_stage1_config(
        "configs/closedloop/stage1_train.yaml", ["endecoder.allow_placeholder_stats=true"]
    )
    with pytest.raises(ValueError, match="actor/dataset kinematics mismatch"):
        build_stage1_actor(config, data)


def test_loader_supports_legacy_full_sequence_ground(actor_factory, tmp_path):
    reference, _ = actor_factory(history_steps=4, starts=(45,))
    root = _write_dataset(tmp_path / "legacy_ground_dataset", length=130)
    path = root / "meta/dataset_info.json"
    info = json.loads(path.read_text())
    info["ground_semantics"] = "legacy_body_origin_min_zero"
    path.write_text(json.dumps(info), encoding="utf-8")
    motion_path = root / "motions/sample.pt"
    payload = torch.load(motion_path, weights_only=False)
    payload["ground_semantics"] = "legacy_body_origin_min_zero"
    torch.save(payload, motion_path)
    data = _data_config(root, reference.endecoder.stats_path)
    loader = build_stage1_loader(data, split="train", batch_size=1)
    batch = next(iter(loader))
    assert (
        batch["meta"][0]["ground_supervision"]["ground_semantics"] == "legacy_body_origin_min_zero"
    )


def test_cli_train_checkpoint_then_strict_condition_validation(actor_factory, tmp_path, capsys):
    reference, _ = actor_factory(history_steps=4, starts=(45,))
    train_root = _write_dataset(tmp_path / "cli_train_dataset", length=130)
    val_root = _write_dataset(tmp_path / "cli_val_dataset", length=130, split="val")
    data = _data_config(train_root, reference.endecoder.stats_path)
    data.datasets.val = {"synthetic": OmegaConf.to_container(data.datasets.train.synthetic)}
    data.datasets.val.synthetic.root = str(val_root)
    data.datasets.val.synthetic.split = "val"
    data_path = tmp_path / "dataset_config.yaml"
    OmegaConf.save(data, data_path)
    config = load_stage1_config(
        "configs/closedloop/stage1_train.yaml",
        [
            "model.latent_dim=32",
            "model.num_layers=1",
            "model.num_heads=4",
            "model.history_hidden_dim=16",
            "model.dropout=0.0",
            "model.music_mask_prob=0.0",
            "endecoder.allow_placeholder_stats=true",
            "runtime.device=cpu",
            "sample_contract.history_steps=4",
            "train.max_steps=1",
            "data_loader.batch_size=1",
            "validation.max_batches=1",
            "validation.sample_steps=2",
        ],
    )
    config.dataset_config = str(data_path)
    config_path = tmp_path / "stage1_cli_config.yaml"
    OmegaConf.save(config, config_path)
    output = tmp_path / "cli_result"
    trained = stage1_main(["--config", str(config_path), "--output-dir", str(output)])
    assert trained["result"]["completed_steps"] == 1
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "weight_loading_report.json").is_file()
    assert trained["artifacts_retained"] is True
    validated = stage1_main(
        [
            "--config",
            str(config_path),
            "--set",
            "mode=validate",
            "--set",
            f"stage1_checkpoint={output / 'stage1_actor.pt'}",
        ]
    )
    assert validated["weight_initialization"]["mode"] == "stage1_weights_only"
    assert validated["result"]["batches"][0]["known_max_abs_error"] == 0.0
    assert validated["artifacts_retained"] is False
    capsys.readouterr()
