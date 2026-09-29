"""验证冻结 Stage1 包装器的严格加载、条件隔离、确定性和资产不变性。

在 pytest 临时目录保存小尺寸真实 Stage1Actor checkpoint，使用仓库真实训练 stats 和
fe934 kinematics。它仍使用原 Transformer、GRU、prefix MLP、codec 和 strict loader；
层数缩小仅用于 CPU 单元测试，不声称覆盖正式 s350000 的吞吐、Isaac 或完整模型验收。
测试检查 NumPy RPC 输入、已知物理前缀、原 anchor、独立噪声、额外 target 拒绝，以及
非持久化归一化 buffer 被更改时能够在关闭审计中发现。不会创建 optimizer 或写正式资产。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import save_stage1_checkpoint
from gem.closedloop.frozen_actor import FrozenStage1Actor, stable_noise_seed
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.closedloop.training import build_stage1_actor
from tests.closedloop.test_online_conditions import make_inputs

REPO = Path(__file__).resolve().parents[2]
KINEMATICS = REPO / "inputs/checkpoints/stage1_s350000_20260928/assets/bumi_kinematics_robot_retargeter_fe934_v1.json"
STATS = REPO / "inputs/checkpoints/stage1_s350000_20260928/qpos30_train_stats.json"


@pytest.fixture
def frozen(tmp_path):
    torch.set_num_threads(1)
    train = {
        "model": {"latent_dim": 32, "num_layers": 1, "num_heads": 4, "mlp_ratio": 2,
                  "dropout": 0, "history_hidden_dim": 16, "use_cond_exists_as_input": True},
        "endecoder": {"stats_path": str(STATS), "kinematics_path": str(KINEMATICS)},
    }
    data = OmegaConf.create({"sample_contract": {"history_steps": 50}, "datasets": {}})
    actor = build_stage1_actor(train, data)
    # 开启真实 attention 路径，避免零初始化 gate 让输出与条件/噪声无关。
    with torch.no_grad():
        actor.denoiser.blocks[0].gate_msa.fill_(0.3)
        actor.denoiser.blocks[0].gate_mlp.fill_(0.3)
    path = tmp_path / "small_actor.pt"
    save_stage1_checkpoint(actor, path, config=train, global_step=17)
    config = {
        "paths": {"checkpoint": str(path), "stats": str(STATS), "kinematics": str(KINEMATICS)},
        "runtime": {"genmo_device": "cpu", "torch_threads": 1},
        "model": {"history_steps": 50, "ddim_steps": 2, "guidance_scale": 2.5},
    }
    return FrozenStage1Actor(config), config


def conditions_for(worker, prefix=12):
    builder = OnlineConditionBuilder(worker.actor.endecoder.codec)
    snapshot, reservation, music = make_inputs(builder, prefix)
    conditions, meta = builder.build(snapshot, reservation, music)
    return {key: value.numpy().copy() for key, value in conditions.items()}, meta, reservation


def test_frozen_actor_numpy_sampling_preserves_prefix_anchor_and_rng(frozen):
    worker, _ = frozen
    conditions, meta, reservation = conditions_for(worker)
    state = torch.random.get_rng_state().clone()
    first = worker.generate(conditions, meta)
    second = worker.generate(conditions, meta)
    assert torch.equal(torch.random.get_rng_state(), state)
    for key in ("qpos30", "qpos_world", "contact"):
        np.testing.assert_array_equal(first[key], second[key])
        assert np.isfinite(first[key]).all()
    mask = conditions["known_qpos30_mask"][0]
    np.testing.assert_array_equal(first["qpos30"][mask], conditions["known_qpos30"][0][mask])
    np.testing.assert_allclose(first["qpos_world"][:12, :3], reservation["source_qpos"][:, :3], atol=1e-6)
    assert first["qpos_world"].shape == (120, 28)
    assert first["contact"].shape == (120, 2)
    assert first["request_id"] == meta["request_id"]
    assert first["latency_s"] > 0
    report = worker.hello()
    assert report["frozen"]
    assert report["loading"]["mode"] == "stage1_weights_only"
    assert not report["global_step_restored"] and not report["optimizer_restored"]
    assert report["checkpoint_global_step_provenance"] == 17
    assert all(worker.close()["frozen_checks"].values())


def test_sampling_accepts_p0_and_empty_history(frozen):
    worker, _ = frozen
    conditions, meta, _ = conditions_for(worker, prefix=0)
    conditions["proprio_history_valid"][:] = False
    conditions["proprio_history"][:] = 0
    result = worker.generate(conditions, meta)
    assert np.isfinite(result["qpos_world"]).all()
    assert not conditions["known_qpos30_mask"].any()
    worker.close()


@pytest.mark.parametrize("extra", ["target_qpos30", "target_contact", "unexpected"])
def test_worker_rejects_labels_and_unknown_condition_keys(frozen, extra):
    worker, _ = frozen
    conditions, meta, _ = conditions_for(worker)
    conditions[extra] = np.zeros((1, 120, 30), dtype=np.float32)
    with pytest.raises(ValueError, match="only Stage1 condition"):
        worker.generate(conditions, meta)
    worker.close()


def test_nonpersistent_normalizer_change_is_detected(frozen):
    worker, _ = frozen
    with torch.no_grad():
        worker.actor.endecoder.mean[0] += 0.5
    with pytest.raises(RuntimeError, match="integrity failed"):
        worker.close()


def test_worker_rejects_training_mode(frozen):
    worker, _ = frozen
    conditions, meta, _ = conditions_for(worker)
    worker.actor.train()
    with pytest.raises(RuntimeError, match="eval/frozen"):
        worker.generate(conditions, meta)


def test_loader_does_not_ignore_extra_weights_or_interface_mismatch(frozen, tmp_path):
    worker, config = frozen
    payload = torch.load(config["paths"]["checkpoint"], weights_only=False)
    payload["state_dict"]["unexpected_weight"] = torch.ones(2)
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    config["paths"]["checkpoint"] = str(bad)
    with pytest.raises(RuntimeError, match="unexpected"):
        FrozenStage1Actor(config)
    worker.close()


def test_noise_seed_uses_episode_and_decision_without_process_hash():
    assert stable_noise_seed(42, 3) == stable_noise_seed(np.int64(42), np.int64(3))
    assert stable_noise_seed(42, 3) != stable_noise_seed(42, 4)
    assert stable_noise_seed(42, 3) != stable_noise_seed(43, 3)
    with pytest.raises(TypeError):
        stable_noise_seed(True, 3)
