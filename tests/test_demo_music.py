"""CPU-only CLI, checkpoint, shape, and dry-run tests for demo_music."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from gem.runtime.artifact_publish import enforce_zero_shape
from gem.utils.music_features import EDGE_SAMPLE_RATE
from scripts.demo.demo_music import (
    build_parser,
    inspect_model_music_input_dim,
    main,
    render_global_motion,
    validate_arguments,
    validate_music_checkpoint,
)


def _write_wav(path: Path, seconds: float = 0.5) -> None:
    count = int(seconds * EDGE_SAMPLE_RATE)
    timeline = np.arange(count, dtype=np.float32) / EDGE_SAMPLE_RATE
    signal = (np.sin(2 * np.pi * 220 * timeline) * 4000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(EDGE_SAMPLE_RATE)
        output.writeframes(signal.tobytes())


def test_cli_defaults_and_dry_run_does_not_publish(tmp_path: Path, capsys) -> None:
    audio = tmp_path / "beat.wav"
    _write_wav(audio)
    args = build_parser().parse_args(["--audio", str(audio)])
    assert args.output_root == Path("outputs/music_motion")
    assert args.shape_mode == "zero"
    assert args.guidance_scale == 2.5
    assert args.ddim_steps == 50
    output = tmp_path / "output"
    assert main(["--audio", str(audio), "--output_root", str(output), "--dry_run"]) == 0
    printed = capsys.readouterr().out
    assert "music_embed:" in printed and "has_music:" in printed
    assert "no checkpoint/GEM/CUDA/render" in printed
    assert not output.exists()


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        (["--audio", "missing.wav", "--dry_run"], FileNotFoundError, "does not exist"),
        (["--start_sec", "-1"], ValueError, "start_sec"),
        (["--duration_sec", "0"], ValueError, "duration_sec"),
        (["--max_frames", "0"], ValueError, "max_frames"),
        (["--guidance_scale", "-1"], ValueError, "guidance_scale"),
        (["--ddim_steps", "0"], ValueError, "ddim_steps"),
    ],
)
def test_invalid_arguments_are_clear(
    tmp_path: Path, arguments: list[str], exception: type[Exception], message: str
) -> None:
    audio = tmp_path / "valid.wav"
    _write_wav(audio)
    values = ["--audio", str(audio), "--dry_run", *arguments]
    if arguments[:2] == ["--audio", "missing.wav"]:
        values = arguments
    args = build_parser().parse_args(values)
    with pytest.raises(exception, match=message):
        validate_arguments(args)


def test_checkpoint_music_markers_and_dimension(tmp_path: Path) -> None:
    good = tmp_path / "good.ckpt"
    torch.save(
        {"state_dict": {"model.music_embedder.fc1.weight": torch.zeros(8, 35)}}, good
    )
    assert validate_music_checkpoint(good) == 35

    missing = tmp_path / "missing.ckpt"
    torch.save({"state_dict": {"other.weight": torch.zeros(1)}}, missing)
    with pytest.raises(RuntimeError, match="music-conditioned diffusion"):
        validate_music_checkpoint(missing)

    wrong = tmp_path / "wrong.ckpt"
    torch.save(
        {"state_dict": {"model.music_embedder.fc1.weight": torch.zeros(8, 34)}}, wrong
    )
    with pytest.raises(RuntimeError, match="dimension is 34"):
        validate_music_checkpoint(wrong)


def _mock_model(*, dimension: int = 35, regression_only: bool = False):
    pipeline = SimpleNamespace(
        args=SimpleNamespace(in_attr=["encoded_music"]),
        denoiser3d=SimpleNamespace(regression_only=regression_only),
    )
    return SimpleNamespace(pipeline=pipeline, music_embedder=nn.Sequential(nn.Linear(dimension, 8)))


def test_loaded_model_contract_rejects_wrong_or_regression_only() -> None:
    assert inspect_model_music_input_dim(_mock_model()) == 35
    with pytest.raises(RuntimeError, match="expects 34"):
        inspect_model_music_input_dim(_mock_model(dimension=34))
    with pytest.raises(RuntimeError, match="music-conditioned diffusion"):
        inspect_model_music_input_dim(_mock_model(regression_only=True))


def test_zero_shape_applies_to_both_groups_and_render_boundary(tmp_path: Path) -> None:
    groups = enforce_zero_shape(
        {
            "body_params_global": {"betas": torch.randn(3, 10)},
            "body_params_incam": {"betas": torch.randn(3, 10)},
        }
    )
    assert torch.count_nonzero(groups["body_params_global"]["betas"]) == 0
    assert torch.count_nonzero(groups["body_params_incam"]["betas"]) == 0
    with pytest.raises(RuntimeError, match="non-zero betas"):
        render_global_motion(tmp_path, {"betas": torch.ones(1, 10)}, 64, 64)
