"""联合Stage1的真实数据工厂、持久训练器和validation-best集成回归。

使用临时30Hz音乐/文本release、预计算文本夹具、小网络与CPU，运行四步有界优化。
对比连续执行与两步后完整恢复的模型、优化器、采样游标、RNG及best元数据；同时验证
两模态固定验证与语义字段未进入监督目标。这些是工程回归，不加载真实T5、不启动
正式长训练，不证明英文理解、音乐质量或机器人执行效果；全部产物位于pytest临时根。
"""

import copy
import json

import torch
import pytest
from omegaconf import OmegaConf

from gem.closedloop import runner
from gem.closedloop.training import load_stage1_config, build_stage1_actor
from gem.closedloop.checkpoint import load_training_checkpoint
from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.test_stage1_multimodal_dataset import _make_data
from tests.closedloop.test_stage1_runner import _assert_tree_equal


def test_joint_real_loader_runner_resume_and_best(tmp_path, actor_factory):
    reference, _ = actor_factory()
    data = _make_data(tmp_path / "releases", text_mode="precomputed")
    data_path = tmp_path / "joint_data.yaml"
    OmegaConf.save(OmegaConf.create(data), data_path)
    config = load_stage1_config("configs/closedloop/stage1_train.yaml", [
        "model.latent_dim=32", "model.num_layers=1", "model.num_heads=4",
        "model.history_hidden_dim=16", "model.dropout=0.1", "model.text_conditioning=true",
        "model.semantic_mask_prob=0.1", "endecoder.allow_placeholder_stats=true",
        "runtime.device=cpu", "runtime.cpu_threads=1", "data_loader.batch_size=2",
        "data_loader.samples_per_epoch=4", "data_loader.num_workers=0", "train.max_steps=4",
        "trainer.gradient_accumulation=1", "trainer.precision=fp32", "trainer.require_warm_start=false",
        "trainer.log_every_steps=1", "trainer.validate_every_steps=1", "trainer.save_every_steps=2",
        "scheduler.total_steps=20", "scheduler.warmup_steps=2", "scheduler.min_lr_ratio=0.1",
        "validation.samples_per_source=1", "validation.seed=341", "validation.sample_steps=2",
    ])
    config.dataset_config = str(data_path)
    config.endecoder.kinematics_path = str(reference.endecoder.kinematics.kinematics_path)
    config.endecoder.stats_path = reference.endecoder.stats_path
    full = runner.run_persistent(config, tmp_path / "full")
    partial = copy.deepcopy(config)
    partial.train.max_steps = 2
    first = runner.run_persistent(partial, tmp_path / "resumed")
    partial.train.max_steps = 4
    partial.resume_checkpoint = first["last_checkpoint"]
    resumed = runner.run_persistent(partial, tmp_path / "resumed")
    left = torch.load(full["last_checkpoint"], weights_only=False)
    right = torch.load(resumed["last_checkpoint"], weights_only=False)
    for key in ("state_dict", "optimizer_state_dict", "scheduler", "runtime_state"):
        _assert_tree_equal(left[key], right[key])
    best = json.loads((tmp_path / "resumed/best.json").read_text())
    assert set(best["modalities"]) == {"music", "text"}
    assert best["score"] == sum(best["modalities"].values()) / 2
    assert best["selection_metric"] == "raw_reconstruction_loss"
    assert best["checkpoint"].endswith(f"s{best['global_step']:06d}.pt")
    assert left["stage1_contract_version"] == "genmo.bumi_closedloop.music_text.v1"
    assert resumed["resume"]["rng_restored"]
    fresh = build_stage1_actor(config, OmegaConf.create(data))
    optimizer = torch.optim.AdamW(fresh.parameters(), lr=config.train.learning_rate,
                                 weight_decay=config.train.weight_decay)
    scheduler = runner.make_scheduler(optimizer, config.scheduler)
    changed = copy.deepcopy(left["runtime_state"]["training_contract"])
    changed["text_encoder_assets"] = {"model.safetensors": {"sha256": "different-content"}}
    with pytest.raises(RuntimeError, match="runtime contract mismatch: training_contract"):
        load_training_checkpoint(fresh, left, optimizer=optimizer, scheduler=scheduler,
                                 expected_runtime={"training_contract": changed})


def test_encoder_asset_identity_tracks_bytes_and_allows_relocation(tmp_path):
    from gem.closedloop.runner import runtime_contract
    source = tmp_path / "encoder"
    source.mkdir()
    for name in ("model.safetensors", "config.json", "spiece.model"):
        (source / name).write_bytes(b"fixture-encoder-asset")
    config = load_stage1_config("configs/closedloop/stage1_train.yaml", [
        "model.text_conditioning=true", "data_loader.samples_per_epoch=4",
        "trainer.gradient_accumulation=1", "trainer.precision=fp32",
        "scheduler.total_steps=20", "scheduler.warmup_steps=2", "scheduler.min_lr_ratio=0.1",
    ])
    config.model.text_encoder_path = str(source)
    data = OmegaConf.create(_make_data(tmp_path / "data", text_mode="precomputed"))
    first = runtime_contract(config, data, 1)["training_contract"]["text_encoder_assets"]
    import shutil
    relocated = tmp_path / "encoder_relocated"
    shutil.copytree(source, relocated)
    config.model.text_encoder_path = str(relocated)
    assert runtime_contract(config, data, 1)["training_contract"]["text_encoder_assets"] == first
    (relocated / "model.safetensors").write_bytes(b"changed-weights")
    assert runtime_contract(config, data, 1)["training_contract"]["text_encoder_assets"] != first
