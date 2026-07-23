"""CPU-only tests for the resident T5 + GEM text-motion engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gem.runtime.resident_text_motion as resident


class DummyTokenizer:
    """Return a deterministic fully valid token batch."""

    def __call__(self, prompts, **kwargs):
        del kwargs
        value = sum(ord(character) for character in prompts[0]) % 97
        return SimpleNamespace(
            input_ids=torch.full((1, 50), value, dtype=torch.long),
            attention_mask=torch.ones(1, 50, dtype=torch.long),
        )


class DummyT5:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.device = None
        self.training = True

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        self.forward_calls += 1
        value = input_ids[:, :1].float().unsqueeze(-1)
        hidden = value.expand(1, 50, 1024).clone()
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

    def init_diffusion(self) -> None:
        self.init_calls += 1


class DummyGEM:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace(denoiser3d=DummyDenoiser())
        self.predict_calls = 0
        self.failures: list[Exception] = []
        self.training = True
        self.device = None

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.training = False
        return self

    def predict(self, data, *, static_cam, postproc):
        assert static_cam is True
        assert isinstance(postproc, bool)
        self.predict_calls += 1
        if self.failures:
            raise self.failures.pop(0)
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
    calls = {"classes": 0, "t5": 0, "gem": 0, "checkpoint": 0}
    tokenizer = DummyTokenizer()
    text_encoder = DummyT5()
    gem_model = DummyGEM()

    def load_classes():
        calls["classes"] += 1
        return object, object

    def load_t5(*_args, **_kwargs):
        calls["t5"] += 1
        return tokenizer, text_encoder, "dummy"

    def load_gem(_path):
        calls["gem"] += 1
        return gem_model

    def validate(_path):
        calls["checkpoint"] += 1

    monkeypatch.setattr(resident, "_load_transformers_classes", load_classes)
    monkeypatch.setattr(resident, "_load_t5_components", load_t5)
    monkeypatch.setattr(resident, "_load_gem_model", load_gem)
    monkeypatch.setattr(resident, "_validate_checkpoint", validate)
    engine = resident.ResidentTextMotionEngine(
        ckpt_path=tmp_path / "dummy.ckpt",
        device="cpu",
        text_dtype="float32",
        output_root=tmp_path / "motions",
        latest_file=tmp_path / "motions" / "latest_ready.json",
        warmup_enabled=False,
        embedding_cache_size=2,
        _allow_cpu_for_tests=True,
    )
    return engine, tokenizer, text_encoder, gem_model, calls


def test_initialize_loads_tokenizer_t5_gem_once(engine_bundle) -> None:
    engine, _, _, _, calls = engine_bundle
    engine.initialize()
    engine.initialize()
    assert calls == {"classes": 1, "t5": 1, "gem": 1, "checkpoint": 1}


def test_initialize_configures_and_initializes_ddim_once(engine_bundle) -> None:
    engine, _, _, gem, _ = engine_bundle
    engine.initialize()
    diffusion = gem.pipeline.denoiser3d
    assert diffusion.init_calls == 1
    assert diffusion.model_cfg.diffusion.test_timestep_respacing == "20"
    assert diffusion.model_cfg.diffusion.gen_only_test_timestep_respacing == "20"
    assert diffusion.model_cfg.diffusion.guidance_param == 2.5


def test_two_requests_do_not_reload_models_or_ddim(engine_bundle) -> None:
    engine, _, _, gem, calls = engine_bundle
    engine.initialize()
    assert engine.generate({"prompt": "walk", "num_frames": 2})["ok"]
    assert engine.generate({"prompt": "wave", "num_frames": 2})["ok"]
    assert calls["t5"] == 1 and calls["gem"] == 1
    assert gem.pipeline.denoiser3d.init_calls == 1


def test_same_normalized_prompt_hits_cpu_embedding_cache(engine_bundle) -> None:
    engine, _, t5, _, _ = engine_bundle
    engine.initialize()
    first = engine.generate({"prompt": "  Walk forward  ", "num_frames": 2})
    second = engine.generate({"prompt": "Walk forward", "num_frames": 2})
    assert first["text_cache_hit"] is False
    assert second["text_cache_hit"] is True
    assert t5.forward_calls == 1
    assert next(iter(engine.embedding_cache.values())).device.type == "cpu"


def test_different_prompt_executes_new_t5_forward(engine_bundle) -> None:
    engine, _, t5, _, _ = engine_bundle
    engine.initialize()
    engine.generate({"prompt": "Walk", "num_frames": 2})
    engine.generate({"prompt": "Wave", "num_frames": 2})
    assert t5.forward_calls == 2
    assert engine.status()["cache_misses"] == 2


def test_generated_files_have_expected_shapes_and_zero_betas(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    result = engine.generate({"prompt": "Squat", "num_frames": 4, "fps": 20})
    payload = torch.load(result["smpl_params"], map_location="cpu", weights_only=False)
    for group_name in ("body_params_global", "body_params_incam"):
        group = payload[group_name]
        assert group["body_pose"].shape == (4, 63)
        assert group["global_orient"].shape == (4, 3)
        assert group["transl"].shape == (4, 3)
        assert group["betas"].shape == (4, 10)
        assert torch.count_nonzero(group["betas"]) == 0


def test_ready_is_last_and_latest_notification_is_updated(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    result = engine.generate({"request_id": "ready-1", "prompt": "Stand", "num_frames": 2})
    output = Path(result["output_dir"])
    assert (output / "READY").is_file()
    assert (output / "smpl_params.pt").is_file()
    assert (output / "motion.npz").is_file()
    latest = json.loads(engine.latest_file.read_text(encoding="utf-8"))
    assert latest["request_id"] == "ready-1"
    assert latest["sequence_number"] == 1
    assert latest["output_dir"] == str(output.resolve())
    assert not list(engine.latest_file.parent.glob(f".{engine.latest_file.name}.tmp-*"))


def test_invalid_request_returns_error_without_stopping_engine(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    bad = engine.generate({"prompt": "", "num_frames": 2})
    good = engine.generate({"prompt": "Stand", "num_frames": 2})
    assert bad["ok"] is False
    assert good["ok"] is True
    assert engine.initialized is True
    assert engine.failed_count == 1 and engine.successful_count == 1


def test_generation_failure_does_not_break_next_request(engine_bundle) -> None:
    engine, _, _, gem, _ = engine_bundle
    engine.initialize()
    gem.failures.append(RuntimeError("injected generation failure"))
    failed = engine.generate({"prompt": "First", "num_frames": 2})
    succeeded = engine.generate({"prompt": "Second", "num_frames": 2})
    assert failed["ok"] is False and "injected" in failed["error"]
    assert succeeded["ok"] is True


def test_oom_is_structured_and_recoverable(
    engine_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _, _, gem, _ = engine_bundle
    engine.initialize()
    empty_cache_calls = []
    monkeypatch.setattr(resident.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        resident.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True)
    )
    gem.failures.append(torch.cuda.OutOfMemoryError("dummy OOM"))
    failed = engine.generate({"request_id": "oom", "prompt": "Jump", "num_frames": 2})
    succeeded = engine.generate({"prompt": "Stand", "num_frames": 2})
    assert failed["ok"] is False and failed["error_type"] == "OutOfMemoryError"
    assert empty_cache_calls == [True]
    assert succeeded["ok"] is True


def test_normal_requests_never_empty_cuda_cache(
    engine_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    empty_cache_calls = []
    monkeypatch.setattr(
        resident.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True)
    )
    assert engine.generate({"prompt": "Walk", "num_frames": 2})["ok"]
    assert engine.generate({"prompt": "Wave", "num_frames": 2})["ok"]
    assert empty_cache_calls == []


def test_clear_cache_preserves_resident_models(engine_bundle) -> None:
    engine, _, _, gem, _ = engine_bundle
    engine.initialize()
    engine.encode_prompt("one")
    assert engine.clear_cache() == 1
    assert engine.embedding_cache == {}
    assert engine.gem_model is gem and engine.text_encoder is not None


def test_lru_cache_capacity_is_enforced(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    for prompt in ("one", "two", "three"):
        engine.encode_prompt(prompt)
    assert len(engine.embedding_cache) == 2
    assert all(key[0] != "one" for key in engine.embedding_cache)


def test_warmup_does_not_increment_request_count(engine_bundle) -> None:
    engine, _, _, gem, _ = engine_bundle
    engine.initialize()
    result = engine.warmup()
    assert result["frames"] == 30
    assert engine.request_count == 0
    assert gem.predict_calls == 1


def test_status_contains_required_resident_fields(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    status = engine.status()
    for field in (
        "initialized",
        "pid",
        "device",
        "checkpoint",
        "t5_model",
        "ddim_steps",
        "request_count",
        "embedding_cache_size",
        "startup_timings",
        "gpu_memory",
        "uptime_seconds",
    ):
        assert field in status


def test_close_releases_model_references(engine_bundle) -> None:
    engine, _, _, _, _ = engine_bundle
    engine.initialize()
    engine.close()
    assert engine.initialized is False
    assert engine.tokenizer is None
    assert engine.text_encoder is None
    assert engine.gem_model is None
    assert engine.denoiser3d is None


def test_loaded_t5_encoding_masks_padding_and_returns_contiguous_cpu() -> None:
    class HalfMaskTokenizer(DummyTokenizer):
        def __call__(self, prompts, **kwargs):
            result = super().__call__(prompts, **kwargs)
            result.attention_mask[:, 25:] = 0
            return result

    encoded = resident.encode_prompt_with_loaded_t5(
        "stand",
        HalfMaskTokenizer(),
        DummyT5(),
        "cpu",
    )
    assert encoded.shape == (50, 1024)
    assert encoded.dtype == torch.float32
    assert encoded.device.type == "cpu" and encoded.is_contiguous()
    assert torch.count_nonzero(encoded[25:]) == 0


def test_gem_diffusion_default_initialization_behavior_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gem.network.gem_diffusion as diffusion_module

    init_calls = []
    monkeypatch.setattr(diffusion_module, "instantiate", lambda _config: object())
    monkeypatch.setattr(
        diffusion_module.GEMDiffusion,
        "init_diffusion",
        lambda self: init_calls.append(self),
    )
    class Config(SimpleNamespace):
        def get(self, name, default=None):
            return getattr(self, name, default)

    config = Config(denoiser=object(), regression_only=False)
    normal = diffusion_module.GEMDiffusion(config, args={})
    deferred = diffusion_module.GEMDiffusion(
        config,
        args={},
        defer_diffusion_init=True,
    )
    assert init_calls == [normal]
    assert deferred not in init_calls
