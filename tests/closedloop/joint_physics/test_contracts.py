"""新流程数据/物理课程及完整断点隔离回归。

覆盖600Hz显式换算、未知未来标签不改变已有条件、split/内容身份拒绝、实际SNR筛选及旧
checkpoint拒绝。临时文件只写pytest目录；模拟断点仅测试格式，不冒充真实优化恢复证明。
"""

import json

import pytest
import torch

from gem.closedloop.joint_physics.checkpoint import atomic_save, read_checkpoint
from gem.closedloop.joint_physics.contracts import SIDECAR_SCHEMA, frame_tick
from gem.closedloop.joint_physics.curriculum import physics_selection, physics_weights
from gem.closedloop.joint_physics.data import SidecarStore


def test_time_and_split_are_explicit(tmp_path):
    assert frame_tick(3) == 60 and frame_tick(30) == 600
    with pytest.raises(ValueError):
        frame_tick(.5)
    (tmp_path/"index.json").write_text(json.dumps({"schema": SIDECAR_SCHEMA, "split": "val"}))
    with pytest.raises(ValueError, match="split"):
        SidecarStore(tmp_path, "train")


def test_no_old_checkpoint_or_overwrite(tmp_path):
    path = tmp_path/"old.pt"
    atomic_save({"schema": "genmo.supermimic.bptt.training.v3", "actor": {}}, path)
    with pytest.raises(ValueError, match="旧Stage2"):
        read_checkpoint(path)
    with pytest.raises(FileExistsError):
        atomic_save({"other": "do_not_overwrite"}, path)
    assert torch.load(path, weights_only=False)["schema"] == "genmo.supermimic.bptt.training.v3"


def test_full_schedule_snr_and_dropout(actor_factory):
    base, batch = actor_factory(motion_frames=60, starts=(0,))
    cfg = {"enabled": True, "start_step": 10, "warmup_steps": 10, "lambda_exec": .1,
           "lambda_safety": .01, "snr_min": .1, "snr_max": 100.}
    assert physics_weights(0, cfg) == (0., 0.)
    assert physics_weights(19, cfg) == (.1, .01)
    out = {"timesteps": torch.arange(1000), "music_dropout_mask": torch.zeros(1000, dtype=torch.bool)}
    eligible, snr = physics_selection(out, base.train_diffusion, cfg)
    assert 0 < eligible.sum() < 1000
    out["music_dropout_mask"].fill_(True)
    assert not physics_selection(out, base.train_diffusion, cfg)[0].any()
    assert snr[0] > snr[-1] and torch.isfinite(snr).all()


def test_offline_requests_random_lookup_and_prefix_identity(tmp_path):
    (tmp_path/"index.json").write_text(json.dumps({"schema": SIDECAR_SCHEMA, "split": "train"}))
    store = SidecarStore(tmp_path, "train")
    episode = {"offline_requests": {"frames": torch.tensor([0, 3, 6]), "prefix_frames": torch.tensor([12, 13, 14]),
        "valid": torch.tensor([True, False, True]), "reasons": ["available", "offline_failed", "available"],
        "features": torch.arange(3*58).reshape(3,58).float(), "forecast_ticks": torch.tensor([120, 180, 240]),
        "state_offsets": torch.tensor([0, 1, 1, 2]), "predicted_states": torch.zeros(2,55)}}
    from gem.closedloop.joint_physics.contracts import PhysicsRejected
    for frame, prefix in [(6,14),(0,12),(6,14)]:
        context = store.request(episode, frame, prefix, "cpu")
        assert context.request_tick == frame*20
        assert context.features[0,0] == (frame//3)*58
    with pytest.raises(ValueError, match="前缀"):
        store.request(episode, 6, 12, "cpu")
    with pytest.raises(PhysicsRejected, match="offline_failed"):
        store.request(episode, 3, 13, "cpu")
