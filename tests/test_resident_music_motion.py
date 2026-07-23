"""CPU-only tests for the resident GEM music-to-motion engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import gem.runtime.resident_music_motion as resident


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

    def init_diffusion(self) -> None:
        self.init_calls += 1


class DummyGEM:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace(
            args=SimpleNamespace(in_attr=["encoded_music"]),
            denoiser3d=DummyDenoiser(),
        )
        self.music_embedder = nn.Sequential(nn.Linear(35, 8))
        self.predict_calls = 0
        self.failures: list[Exception] = []
        self.device = None

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def predict(self, data, *, static_cam, postproc):
        assert static_cam is True
        assert isinstance(postproc, bool)
        assert data["mask"]["has_music_mask"].all()
        self.predict_calls += 1
        if self.failures:
            raise self.failures.pop(0)
        length = int(data["length"])
        body = {
            "body_pose": torch.zeros(length, 63),
            "global_orient": torch.zeros(length, 3),
            "transl": torch.zeros(length, 3),
            "betas": torch.ones(length, 10),
        }
        return {
            "body_params_global": {key: value.clone() for key, value in body.items()},
            "body_params_incam": {key: value.clone() for key, value in body.items()},
            "net_outputs": {"model_output": {"pred_x": torch.zeros(1, length, 151)}},
        }


@pytest.fixture()
def engine_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio = audio_root / "song.wav"
    audio.write_bytes(b"dummy-wave")
    calls = {"checkpoint": 0, "gem": 0, "features": 0}
    gem = DummyGEM()

    def validate(_path):
        calls["checkpoint"] += 1
        return 35

    def load_gem(_path):
        calls["gem"] += 1
        return gem

    def extract(path, start_sec=0.0, duration_sec=None, target_fps=30):
        del target_fps
        calls["features"] += 1
        length = 8 if duration_sec == 99 else 4
        features = torch.zeros(length, 35)
        features[:, 0] = float(start_sec)
        return features, {
            "source_path": str(Path(path).resolve()),
            "original_duration_sec": 120.0,
            "selected_start_sec": float(start_sec),
            "selected_duration_sec": length / 30,
            "sample_rate": 15360,
            "hop_length": 512,
            "target_fps": 30,
            "feature_dim": 35,
            "feature_names": [f"feature_{index}" for index in range(35)],
            "estimated_or_prior_bpm": 120.0,
            "bpm_source": "test",
            "feature_frames": length,
            "audio_decode_mode": "range",
        }

    monkeypatch.setattr(resident, "_validate_checkpoint", validate)
    monkeypatch.setattr(resident, "_load_gem_model", load_gem)
    monkeypatch.setattr(resident, "extract_edge_baseline35", extract)
    engine = resident.ResidentMusicMotionEngine(
        ckpt_path=tmp_path / "dummy.ckpt",
        device="cpu",
        output_root=tmp_path / "motions",
        latest_file=tmp_path / "motions" / "latest_ready.json",
        feature_cache_size=2,
        warmup_enabled=False,
        max_frames=5,
        allowed_audio_roots=[audio_root],
        _allow_cpu_for_tests=True,
    )
    return engine, gem, calls, audio, audio_root


def test_initialize_loads_gem_and_ddim_only_once(engine_bundle) -> None:
    engine, gem, calls, _, _ = engine_bundle
    engine.initialize()
    engine.initialize()
    assert calls["checkpoint"] == 1
    assert calls["gem"] == 1
    assert gem.pipeline.denoiser3d.init_calls == 1
    diffusion = gem.pipeline.denoiser3d.model_cfg.diffusion
    assert diffusion.guidance_param == 2.5
    assert diffusion.test_timestep_respacing == "20"
    assert not hasattr(engine, "text_encoder")
    assert not hasattr(engine, "tokenizer")


def test_two_requests_do_not_reload_model_or_ddim(engine_bundle) -> None:
    engine, gem, calls, audio, _ = engine_bundle
    engine.initialize()
    assert engine.generate({"audio_path": str(audio)})["ok"]
    assert engine.generate({"audio_path": str(audio), "start_sec": 1})["ok"]
    assert calls["gem"] == 1
    assert gem.pipeline.denoiser3d.init_calls == 1
    assert gem.predict_calls == 2


def test_same_file_and_range_hits_feature_cache(engine_bundle) -> None:
    engine, _, calls, audio, _ = engine_bundle
    engine.initialize()
    first = engine.generate({"audio_path": str(audio)})
    second = engine.generate({"audio_path": str(audio)})
    assert first["feature_cache_hit"] is False
    assert second["feature_cache_hit"] is True
    assert calls["features"] == 1
    status = engine.status()
    assert status["feature_cache_hits"] == 1
    assert status["feature_cache_misses"] == 1
    assert status["feature_cache_bytes"] == 4 * 35 * 4


def test_different_range_and_changed_file_miss_cache(engine_bundle) -> None:
    engine, _, calls, audio, _ = engine_bundle
    engine.initialize()
    engine.generate({"audio_path": str(audio)})
    engine.generate({"audio_path": str(audio), "start_sec": 1})
    engine.generate({"audio_path": str(audio), "duration_sec": 2})
    audio.write_bytes(b"changed-wave-larger")
    os.utime(audio, None)
    engine.generate({"audio_path": str(audio)})
    assert calls["features"] == 4


def test_lru_capacity_and_clear_cache_preserve_gem(engine_bundle) -> None:
    engine, gem, _, audio, _ = engine_bundle
    engine.initialize()
    for start in (0, 1, 2):
        assert engine.generate({"audio_path": str(audio), "start_sec": start})["ok"]
    assert engine.status()["feature_cache_size"] == 2
    assert engine.clear_cache() == 2
    assert engine.status()["feature_cache_size"] == 0
    assert engine.gem_model is gem and engine.initialized


def test_required_artifacts_shapes_metadata_and_zero_betas(engine_bundle) -> None:
    engine, _, _, audio, _ = engine_bundle
    engine.initialize()
    result = engine.generate(
        {
            "request_id": "music-1",
            "audio_path": str(audio),
            "metadata": {"operator": "test"},
        }
    )
    assert result["ok"]
    output = Path(result["output_dir"])
    required = {
        "smpl_params.pt",
        "motion.npz",
        "raw_motion_151d.pt",
        "music_features.pt",
        "metadata.json",
        "source_audio.txt",
        "READY",
    }
    assert required <= {path.name for path in output.iterdir()}
    payload = torch.load(output / "smpl_params.pt", map_location="cpu", weights_only=False)
    for group_name in ("body_params_global", "body_params_incam"):
        group = payload[group_name]
        assert group["body_pose"].shape == (4, 63)
        assert group["betas"].shape == (4, 10)
        assert torch.count_nonzero(group["betas"]) == 0
    motion = np.load(output / "motion.npz")
    assert motion["betas"].shape == (4, 10)
    assert np.count_nonzero(motion["betas"]) == 0
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["request_id"] == "music-1"
    assert metadata["request_metadata"] == {"operator": "test"}
    assert metadata["service"] == "resident_music_motion"
    assert metadata["source"] == "music_only"
    assert not list(output.parent.glob(".tmp_*"))


def test_latest_ready_is_atomic_and_contains_music_fields(engine_bundle) -> None:
    engine, _, _, audio, _ = engine_bundle
    engine.initialize()
    result = engine.generate({"request_id": "latest-1", "audio_path": str(audio)})
    latest = json.loads(engine.latest_file.read_text(encoding="utf-8"))
    assert latest["request_id"] == "latest-1"
    assert latest["music_features"] == result["music_features"]
    assert latest["sequence_number"] == 1
    assert Path(latest["ready"]).is_file()
    assert not list(engine.latest_file.parent.glob(".*latest_ready*.tmp-*"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"audio_path": "/does/not/exist.wav"}, "No such file"),
        ({"audio_path": ""}, "non-empty"),
        ({"audio_path": "unused", "start_sec": -1}, "No such file"),
        ({"audio_path": "unused", "duration_sec": 0}, "No such file"),
        ({"audio_path": "unused", "render": True}, "unsupported request"),
    ],
)
def test_invalid_requests_return_structured_errors(engine_bundle, payload, message) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    result = engine.generate(payload)
    assert result["ok"] is False
    assert message in result["error"]
    assert engine.initialized


def test_wrong_suffix_and_invalid_times_are_rejected(engine_bundle) -> None:
    engine, _, _, audio, audio_root = engine_bundle
    engine.initialize()
    text = audio_root / "song.txt"
    text.write_text("not audio", encoding="utf-8")
    assert engine.generate({"audio_path": str(text)})["ok"] is False
    assert engine.generate({"audio_path": str(audio), "start_sec": -1})["ok"] is False
    assert engine.generate({"audio_path": str(audio), "duration_sec": 0})["ok"] is False


def test_max_frames_failure_does_not_break_next_request(engine_bundle) -> None:
    engine, _, _, audio, _ = engine_bundle
    engine.initialize()
    failed = engine.generate({"audio_path": str(audio), "duration_sec": 99})
    good = engine.generate({"audio_path": str(audio), "duration_sec": 1})
    assert failed["ok"] is False and "max_frames=5" in failed["error"]
    assert good["ok"] is True


def test_predict_failure_does_not_break_next_request(engine_bundle) -> None:
    engine, gem, _, audio, _ = engine_bundle
    engine.initialize()
    gem.failures.append(RuntimeError("injected failure"))
    failed = engine.generate({"audio_path": str(audio)})
    good = engine.generate({"audio_path": str(audio), "start_sec": 1})
    assert failed["ok"] is False and "injected failure" in failed["error"]
    assert good["ok"] is True


def test_oom_clears_cache_once_and_recovers(engine_bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, gem, _, audio, _ = engine_bundle
    engine.initialize()
    empty_cache_calls = []
    monkeypatch.setattr(resident.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resident.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))
    gem.failures.append(torch.cuda.OutOfMemoryError("dummy OOM"))
    failed = engine.generate({"audio_path": str(audio)})
    good = engine.generate({"audio_path": str(audio), "start_sec": 1})
    assert failed["error_type"] == "OutOfMemoryError"
    assert empty_cache_calls == [True]
    assert good["ok"] is True


def test_normal_requests_never_empty_cuda_cache(
    engine_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _, _, audio, _ = engine_bundle
    engine.initialize()
    calls = []
    monkeypatch.setattr(resident.torch.cuda, "empty_cache", lambda: calls.append(True))
    assert engine.generate({"audio_path": str(audio)})["ok"]
    assert engine.generate({"audio_path": str(audio)})["ok"]
    assert calls == []


def test_warmup_is_unpublished_and_does_not_increment_count(engine_bundle) -> None:
    engine, gem, _, _, _ = engine_bundle
    engine.initialize()
    result = engine.warmup()
    assert result["frames"] == 30
    assert engine.request_count == 0
    assert gem.predict_calls == 1
    assert not list(engine.output_root.glob("*READY*"))


def test_allowlist_and_symlink_escape_are_rejected(engine_bundle, tmp_path: Path) -> None:
    engine, _, _, _, audio_root = engine_bundle
    engine.initialize()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    link = audio_root / "escape.wav"
    link.symlink_to(outside)
    direct = engine.generate({"audio_path": str(outside)})
    escaped = engine.generate({"audio_path": str(link)})
    assert direct["ok"] is False and "outside" in direct["error"]
    assert escaped["ok"] is False and "outside" in escaped["error"]


def test_failed_write_removes_own_temporary_directory(
    engine_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _, _, audio, _ = engine_bundle
    engine.initialize()
    demo = resident._music_demo_helpers()

    def fail_write(path, **_kwargs):
        path.mkdir()
        (path / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("write failed")

    monkeypatch.setattr(demo, "write_music_artifacts", fail_write)
    result = engine.generate({"audio_path": str(audio)})
    assert result["ok"] is False
    assert not list(engine.output_root.glob(".tmp_*"))
    assert not list(engine.output_root.glob("*/READY"))


def test_close_releases_model_reference(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    engine.close()
    assert engine.initialized is False
    assert engine.gem_model is None
    assert engine.denoiser3d is None
