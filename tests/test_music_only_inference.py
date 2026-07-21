"""CPU contract tests and conditional CUDA smoke test for music-only GEM inference."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.gem import prepare_predict_text_condition
from gem.utils.music_features import EDGE_SAMPLE_RATE
from scripts.demo.demo_music import build_music_only_data, main


def _write_sine_wav(path: Path, seconds: float) -> None:
    count = int(round(seconds * EDGE_SAMPLE_RATE))
    time = np.arange(count, dtype=np.float32) / EDGE_SAMPLE_RATE
    signal = (0.2 * np.sin(2 * np.pi * 220.0 * time) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(EDGE_SAMPLE_RATE)
        output.writeframes(signal.tobytes())


def test_music_only_data_enables_only_music() -> None:
    data = build_music_only_data(torch.randn(7, 35), width=640, height=480)
    assert data["music_embed"].shape == (7, 35)
    assert data["has_text"].tolist() == [False]
    assert data["mask"]["has_music_mask"].all()
    for name in (
        "has_img_mask",
        "has_2d_mask",
        "has_cam_mask",
        "has_audio_mask",
    ):
        assert not data["mask"][name].any()
    assert torch.all(data["bbx_xys"][:, 2] > 0)
    assert torch.count_nonzero(data["kp2d"]) == 0
    assert torch.count_nonzero(data["f_imgseq"]) == 0
    assert torch.isfinite(data["cam_angvel"]).all()


def test_predict_false_has_text_stays_false_and_skips_encoder() -> None:
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("text encoder must not run")

    caption, has_text, encoded = prepare_predict_text_condition(
        {"has_text": torch.tensor([False]), "caption": "ignored"},
        max_text_len=50,
        encoded_text_dim=1024,
        device="cpu",
        encode_text_fn=fail_if_called,
    )
    assert caption == [""]
    assert has_text.tolist() == [False]
    assert encoded.shape == (1, 50, 1024)
    assert torch.count_nonzero(encoded) == 0
    assert not called


def test_predict_true_has_text_preserves_text_demo_behavior() -> None:
    def encoder(captions, has_text):
        assert captions == ["walk forward"]
        assert has_text.tolist() == [True]
        return torch.ones(1, 50, 1024)

    caption, has_text, encoded = prepare_predict_text_condition(
        {"has_text": torch.tensor([True]), "caption": "walk forward"},
        max_text_len=50,
        encoded_text_dim=1024,
        device="cpu",
        encode_text_fn=encoder,
    )
    assert caption == ["walk forward"]
    assert has_text.tolist() == [True]
    assert torch.count_nonzero(encoded) == encoded.numel()


_CHECKPOINT = Path("inputs/pretrained/gem_smpl.ckpt")
_BODY_MODEL = Path("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
_SMOKE_AVAILABLE = (
    torch.cuda.is_available()
    and _CHECKPOINT.is_file()
    and _BODY_MODEL.is_file()
    and importlib.util.find_spec("librosa") is not None
)


@pytest.mark.skipif(
    not _SMOKE_AVAILABLE,
    reason="CUDA, full gem_smpl checkpoint, SMPL-X model, or librosa unavailable",
)
def test_cuda_checkpoint_music_smoke(tmp_path: Path) -> None:
    audio = tmp_path / "two_seconds.wav"
    output = tmp_path / "output"
    _write_sine_wav(audio, 2.0)
    assert (
        main(
            [
                "--audio",
                str(audio),
                "--ckpt_path",
                str(_CHECKPOINT),
                "--duration_sec",
                "2",
                "--output_root",
                str(output),
                "--max_frames",
                "100",
                "--no_render",
                "--save_features",
            ]
        )
        == 0
    )
    generations = [path.parent for path in output.glob("*/READY")]
    assert len(generations) == 1
    generation = generations[0]
    features = torch.load(
        generation / "music_features.pt", map_location="cpu", weights_only=False
    )
    saved = torch.load(
        generation / "smpl_params.pt",
        map_location="cpu",
        weights_only=False,
    )
    length = int(features.shape[0])
    assert saved["body_params_global"]["body_pose"].shape == (length, 63)
    for group_name in ("body_params_global", "body_params_incam"):
        for value in saved[group_name].values():
            assert value.shape[0] == length
            assert torch.isfinite(value).all()
        assert torch.count_nonzero(saved[group_name]["betas"]) == 0
    assert saved["source"] == "music_only"
