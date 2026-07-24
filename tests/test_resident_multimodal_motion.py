"""CPU-only tests for the single-stack resident multimodal generator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import gem.runtime.resident_multimodal_motion as runtime


class DummyTokenizer:
    def __call__(self, prompts, **_kwargs):
        token = sum(map(ord, prompts[0])) % 97
        return SimpleNamespace(
            input_ids=torch.full((1, 50), token, dtype=torch.long),
            attention_mask=torch.ones(1, 50, dtype=torch.long),
        )


class DummyT5:
    def __init__(self) -> None:
        self.calls = 0

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask):
        self.calls += 1
        hidden = input_ids[:, :1].float().unsqueeze(-1).expand(1, 50, 1024).clone()
        hidden *= attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class DummyDenoiser:
    def __init__(self) -> None:
        self.regression_only = False
        self.init_calls = 0
        self.model_cfg = SimpleNamespace(
            diffusion=SimpleNamespace(
                guidance_param=None,
                test_timestep_respacing=None,
                gen_only_test_timestep_respacing=None,
            )
        )

    def init_diffusion(self):
        self.init_calls += 1


class DummyGEM:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace(
            args=SimpleNamespace(in_attr=["encoded_text", "encoded_music"]),
            denoiser3d=DummyDenoiser(),
        )
        self.music_embedder = nn.Sequential(nn.Linear(35, 8))
        self.predict_calls = 0
        self.inputs: list[dict] = []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def predict(self, data, *, static_cam, postproc):
        assert static_cam and isinstance(postproc, bool)
        self.predict_calls += 1
        self.inputs.append(data)
        length = int(data["length"])
        group = {
            "body_pose": torch.zeros(length, 63),
            "global_orient": torch.zeros(length, 3),
            "transl": torch.zeros(length, 3),
            "betas": torch.ones(length, 10),
        }
        return {
            "body_params_global": {key: value.clone() for key, value in group.items()},
            "body_params_incam": {key: value.clone() for key, value in group.items()},
        }


@pytest.fixture()
def engine_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "song.wav"
    audio.write_bytes(b"dummy")
    calls = {"checkpoint": 0, "classes": 0, "t5": 0, "gem": 0, "features": 0}
    tokenizer = DummyTokenizer()
    t5 = DummyT5()
    gem = DummyGEM()
    feature_length = {"value": 5}

    def validate(_path):
        calls["checkpoint"] += 1
        return 35

    def classes():
        calls["classes"] += 1
        return object, object

    def load_t5(*_args):
        calls["t5"] += 1
        return tokenizer, t5, "dummy"

    def load_gem(_path):
        calls["gem"] += 1
        return gem

    def extract(path, *, start_sec, duration_sec, target_fps):
        calls["features"] += 1
        length = feature_length["value"]
        return torch.zeros(length, 35), {
            "source_path": str(path),
            "original_duration_sec": 100.0,
            "selected_start_sec": start_sec,
            "selected_duration_sec": duration_sec,
            "sample_rate": 15360,
            "hop_length": 512,
            "target_fps": target_fps,
            "feature_dim": 35,
            "feature_names": [f"f{i}" for i in range(35)],
            "estimated_or_prior_bpm": 120.0,
            "bpm_source": "test",
        }

    monkeypatch.setattr(runtime, "_validate_multimodal_checkpoint", validate)
    monkeypatch.setattr(runtime, "_load_transformers_classes", classes)
    monkeypatch.setattr(runtime, "_load_t5_components", load_t5)
    monkeypatch.setattr(runtime, "_load_gem_model", load_gem)
    monkeypatch.setattr(runtime, "extract_edge_baseline35", extract)
    engine = runtime.ResidentMultimodalMotionEngine(
        ckpt_path=tmp_path / "dummy.ckpt",
        device="cpu",
        text_dtype="float32",
        clip_frames=4,
        output_root=tmp_path / "out",
        allowed_audio_roots=[audio_root],
        warmup_enabled=False,
        _allow_cpu_for_tests=True,
    )
    return engine, gem, t5, calls, audio, feature_length


def test_initialize_loads_t5_gem_and_ddim_once(engine_bundle) -> None:
    engine, gem, _, calls, _, _ = engine_bundle
    engine.initialize()
    engine.initialize()
    assert calls == {
        "checkpoint": 1,
        "classes": 1,
        "t5": 1,
        "gem": 1,
        "features": 0,
    }
    assert gem.pipeline.denoiser3d.init_calls == 1
    assert engine.status()["gem_instances"] == 1
    assert engine.status()["t5_instances"] == 1
    assert engine.status()["ddim_init_count"] == 1


@pytest.mark.parametrize(
    ("mode", "has_text", "has_music"),
    [
        ("text", True, False),
        ("music", False, True),
        ("text_music", True, True),
    ],
)
def test_all_modes_use_fixed_contract_and_one_predict(
    engine_bundle, mode, has_text, has_music
) -> None:
    engine, gem, _, _, audio, _ = engine_bundle
    engine.initialize()
    request = {"mode": mode, "seed": 7}
    if has_text:
        request["prompt"] = "walk and wave"
    if has_music:
        request["audio_path"] = str(audio)
    before = gem.predict_calls
    result = engine.generate(request)
    assert result["ok"], result
    assert gem.predict_calls == before + 1
    data = gem.inputs[-1]
    assert int(data["length"]) == 4
    assert ("text_embed" in data) is has_text
    assert bool(data["mask"]["has_music_mask"].all()) is has_music
    assert not data["mask"]["has_img_mask"].any()
    assert not data["mask"]["has_2d_mask"].any()
    assert not data["mask"]["has_cam_mask"].any()
    assert not data["mask"]["has_audio_mask"].any()


def test_text_music_is_joint_input_not_two_generated_motions(engine_bundle) -> None:
    engine, gem, _, _, audio, _ = engine_bundle
    engine.initialize()
    result = engine.generate_text_music("dance", str(audio), request_id="joint")
    assert result["ok"] and gem.predict_calls == 1
    data = gem.inputs[0]
    assert data["text_embed"].shape == (50, 1024)
    assert data["music_embed"].shape == (4, 35)
    metadata = json.loads(
        (Path(result["output_dir"]) / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["source"] == "text_music"
    assert metadata["fusion_mode"] == "joint_gem_condition"
    assert metadata["fusion_training_status"] == "zero_shot_cross_dataset"


def test_music_boundary_alignment_allows_one_frame_but_rejects_large_difference(
    engine_bundle,
) -> None:
    engine, _, _, _, audio, feature_length = engine_bundle
    engine.initialize()
    assert engine.generate_music(str(audio))["ok"]
    engine.clear_music_cache()
    feature_length["value"] = 7
    failed = engine.generate_music(str(audio), start_sec=1)
    assert failed["ok"] is False
    assert "difference=3" in failed["error"]


def test_cpu_text_and_music_lru_caches_hit(engine_bundle) -> None:
    engine, _, t5, calls, audio, _ = engine_bundle
    engine.initialize()
    assert engine.generate_text("walk")["ok"]
    assert engine.generate_text("walk")["text_cache_hit"]
    assert t5.calls == 1
    assert engine.generate_music(str(audio))["ok"]
    assert engine.generate_music(str(audio))["music_cache_hit"]
    assert calls["features"] == 1
    assert next(iter(engine.embedding_cache.values())).device.type == "cpu"
    assert next(iter(engine.feature_cache.values()))[0].device.type == "cpu"


def test_ready_artifacts_are_direct_children_and_zero_shape(engine_bundle) -> None:
    engine, _, _, _, audio, _ = engine_bundle
    engine.initialize()
    result = engine.generate_text_music("turn", str(audio))
    output = Path(result["output_dir"])
    assert output.parent == engine.output_root
    assert (output / "READY").is_file()
    assert (output / "prompt.txt").is_file()
    assert (output / "music_features.pt").is_file()
    assert (output / "source_audio.txt").is_file()
    payload = torch.load(output / "smpl_params.pt", weights_only=False)
    for name in ("body_params_global", "body_params_incam"):
        assert torch.count_nonzero(payload[name]["betas"]) == 0
    assert not list(engine.output_root.glob(".tmp_*"))


def test_failed_request_does_not_stop_service(engine_bundle) -> None:
    engine, _, _, _, _, _ = engine_bundle
    engine.initialize()
    bad = engine.generate({"mode": "video_text", "prompt": "walk"})
    good = engine.generate_text("walk")
    assert bad["error_type"] == "UnsupportedModeError"
    assert good["ok"] and engine.initialized


def test_build_text_music_data_contract() -> None:
    data = runtime.build_text_music_data(
        "wave",
        torch.zeros(50, 1024),
        torch.zeros(120, 35),
    )
    assert data["has_text"].item()
    assert data["mask"]["has_music_mask"].all()
    assert data["text_embed"].shape == (50, 1024)
    assert data["music_embed"].shape == (120, 35)
