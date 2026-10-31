"""统一音乐/文本生成入口的时序、纯条件与初态契约测试。

使用真实小型联合Transformer、BUMI codec及临时统计夹具，验证滚动前缀与超出音频
结束的模型尾部支持，不需要下载T5或启动物理。单独覆盖state55具名关节重排、请求
互斥及文本显式时长；这些是代码和数值回归，不代表已训练模型的语义/整曲质量。
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from gem.runtime.multimodal_stage1 import generate_sequence, initial_qpos, validate_request
from tests.closedloop.test_stage1_actor import actor_factory as actor_factory
from tests.closedloop.test_stage1_multimodal_actor import joint_actor as joint_actor


@pytest.mark.parametrize("changed", [False, True])
def test_text_reference_checks_trained_encoder_before_encoding(joint_actor, tmp_path, monkeypatch, changed):
    from gem.runtime import frozen_text_encoder, multimodal_stage1
    actor, batch = joint_actor
    asset_names = ("t5_model_safetensors", "t5_config_json", "t5_spiece_model")
    expected = {key: {"sha256": key} for key in asset_names}
    actor.__dict__["_text_encoder_assets"] = expected
    calls = []

    class Encoder:
        def __init__(self, path, **kwargs):
            self.assets = {key: {"path": "new/location", "sha256": key} for key in asset_names}
            if changed:
                self.assets[asset_names[0]]["sha256"] = "changed"

        def encode(self, captions, device):
            calls.append(captions)
            return batch["text_features"][1:2], batch["text_attention_mask"][1:2]

    monkeypatch.setattr(frozen_text_encoder, "FrozenTextEncoder", Encoder)
    monkeypatch.setattr(multimodal_stage1, "load_actor", lambda *args: (actor, OmegaConf.create({"model": {}})))
    cfg = {"text_encoder_path": "new/location", "sample_steps": 2}
    for name in ("checkpoint", "stats", "kinematics"):
        path = tmp_path / name
        path.write_text(name)
        cfg[name] = str(path)
    task = {"text": "wave your hand", "duration_s": .2}
    if changed:
        with pytest.raises(ValueError, match="文本编码器内容"):
            multimodal_stage1.generate_stage1_reference(cfg, task, device="cpu", required_end_tick=120)
        assert not calls
    else:
        result = multimodal_stage1.generate_stage1_reference(cfg, task, device="cpu", required_end_tick=120)
        assert calls == [[task["text"]]]
        assert result["world_qpos"].shape == (1, 7, 28)


def test_music_file_rejects_transposed_features(joint_actor, tmp_path, monkeypatch):
    from gem.runtime import multimodal_stage1
    actor, _ = joint_actor
    monkeypatch.setattr(multimodal_stage1, "load_actor", lambda *args: (actor, {}))
    path = tmp_path / "wrong_shape.pt"
    torch.save(torch.zeros(35, 120), path)
    with pytest.raises(ValueError, match="不能自动改排"):
        multimodal_stage1.generate_stage1_reference({}, {"music_features": str(path)}, device="cpu", required_end_tick=200)


def test_request_exclusive_and_duration():
    for task in ({}, {"text": ""}, {"text": "walk"},
                 {"text": "walk", "duration_s": 2, "music_features": "x.pt"},
                 {"text": "walk", "duration_s": float("nan")}):
        with pytest.raises(ValueError):
            validate_request(task)
    assert validate_request({"text": "walk", "duration_s": 4}) == ("text", 0, 2400)
    assert validate_request({"music_features": "song.pt"}) == ("music", 0, None)


def test_named_state55_numpy_and_single_initial(joint_actor, tmp_path):
    actor, _ = joint_actor
    names = list(actor.endecoder.kinematics.joint_order)[::-1]
    state = np.zeros(55, dtype=np.float32)
    state[2:4] = [.8, 1]
    state[13:34] = np.linspace(-.1, .1, 21)
    path = tmp_path / "initial.npy"
    np.save(path, state)
    actual = initial_qpos(actor, {"initial_state": str(path)}, names)
    torch.testing.assert_close(actual[7:], torch.from_numpy(state[13:34].copy()).flip(0))
    assert actual[2] == state[2]
    with pytest.raises(ValueError, match="robot_spec_joint_names"):
        initial_qpos(actor, {"initial_state": str(path)})


def test_music_rollout_generates_tail_and_causal_conditions(joint_actor):
    actor, _ = joint_actor
    observed = []
    original = actor.sample

    def record(conditions, **kwargs):
        observed.append({k: v.clone() for k, v in conditions.items()})
        return original(conditions, **kwargs)

    actor.sample = record
    result = generate_sequence(actor, total_frames=138, music=torch.zeros(119, 35), steps=2)
    assert result["world_qpos"].shape == (1, 138, 28)
    assert len(result["windows"]) == 2
    assert result["contact_valid"][12:].all() and not result["contact_valid"][:12].any()
    assert not observed[-1]["music_valid"][0, 11:].any()
    assert observed[-1]["future_valid"][0, 11:30].all()
    assert all(not c["has_text"].any() for c in observed)
    assert all((c["proprio_history_times"] <= c["decision_time"][:, None]).all() for c in observed)
    assert all(w["prefix_max_abs_error"] == 0 for w in result["windows"])


def test_short_text_still_generates_unknown_frame(joint_actor):
    actor, batch = joint_actor
    result = generate_sequence(actor, total_frames=4, text_features=batch["text_features"][1:2],
                               text_attention_mask=batch["text_attention_mask"][1:2], steps=2)
    assert result["world_qpos"].shape == (1, 4, 28)
    assert result["contact_valid"].tolist() == [False, False, False, True]
    assert len(result["windows"]) == 1
