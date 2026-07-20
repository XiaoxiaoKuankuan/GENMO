"""CPU-only tests for the pure-text GEM-SMPL demo input contract."""

from argparse import Namespace

import pytest
import torch

from gem.gem import prepare_precomputed_text_embed
from scripts.demo.demo_smpl_text import (
    MAX_TEXT_LEN,
    TEXT_EMBED_DIM,
    _load_t5_components_cached_first,
    build_arg_parser,
    build_text_only_data,
    enforce_zero_shape,
    prompt_slug,
    publish_ready_directory,
    resolve_prompt,
    validate_arguments,
)


@pytest.fixture()
def text_data() -> dict:
    """Create a short synthetic input without loading T5 or GEM."""
    embedding = torch.randn(MAX_TEXT_LEN, TEXT_EMBED_DIM)
    return build_text_only_data("a person walks", embedding, 12, 1280, 720, 0.75)


def test_text_only_data_shapes(text_data: dict) -> None:
    assert text_data["kp2d"].shape == (12, 17, 3)
    assert text_data["bbx_xys"].shape == (12, 3)
    assert text_data["K_fullimg"].shape == (12, 3, 3)
    assert text_data["R_w2c"].shape == (12, 3, 3)
    assert text_data["cam_angvel"].shape == (12, 6)
    assert text_data["cam_tvel"].shape == (12, 3)
    assert text_data["f_imgseq"].shape == (12, 1024)


def test_virtual_bbox_is_nonzero_and_centered(text_data: dict) -> None:
    assert torch.all(text_data["bbx_xys"][:, 2] > 0)
    assert torch.allclose(text_data["bbx_xys"][0], torch.tensor([640.0, 360.0, 540.0]))


def test_synthetic_intrinsics_have_positive_focal_length(text_data: dict) -> None:
    assert torch.all(text_data["K_fullimg"][:, 0, 0] > 0)
    assert torch.all(text_data["K_fullimg"][:, 1, 1] > 0)


def test_all_non_text_condition_masks_are_false(text_data: dict) -> None:
    for condition_mask in text_data["mask"].values():
        assert condition_mask.dtype == torch.bool
        assert not condition_mask.any()
    assert text_data["has_text"].tolist() == [True]


def test_cam_angvel_is_finite_including_single_frame() -> None:
    embedding = torch.zeros(MAX_TEXT_LEN, TEXT_EMBED_DIM)
    one_frame = build_text_only_data("stand", embedding, 1, 640, 480, 0.5)
    assert one_frame["cam_angvel"].shape == (1, 6)
    assert torch.isfinite(one_frame["cam_angvel"]).all()


def test_video_conditions_are_zero(text_data: dict) -> None:
    assert torch.count_nonzero(text_data["kp2d"]) == 0
    assert torch.count_nonzero(text_data["f_imgseq"]) == 0
    assert torch.count_nonzero(text_data["cam_tvel"]) == 0


def test_text_embedding_is_preserved() -> None:
    embedding = torch.randn(MAX_TEXT_LEN, TEXT_EMBED_DIM)
    data = build_text_only_data("wave", embedding, 3, 1280, 720, 0.75)
    assert data["text_embed"].shape == (50, 1024)
    assert torch.equal(data["text_embed"], embedding)
    assert "multi_text_data" not in data
    assert "multi_text_data" not in data["meta"][0]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("A Person Walks Forward!", "a_person_walks_forward"),
        ("keep-this_and_that", "keep-this_and_that"),
        ("中文动作", "text_motion"),
        ("word " * 30, ("word_" * 10)[:48].rstrip("_-")),
    ],
)
def test_prompt_slug_rules(prompt: str, expected: str) -> None:
    result = prompt_slug(prompt)
    assert result == expected
    assert len(result) <= 48
    assert all(
        character.isascii() and (character.isalnum() or character in "_-") for character in result
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_frames": 0}, "num_frames"),
        ({"fps": 0}, "fps"),
        ({"bbox_scale": 0}, "bbox_scale"),
        ({"bbox_scale": 1.6}, "bbox_scale"),
        ({"ddim_steps": 0}, "ddim_steps"),
        ({"guidance_scale": -0.1}, "guidance_scale"),
    ],
)
def test_invalid_numeric_arguments_raise_clear_errors(overrides: dict, message: str) -> None:
    values = {
        "num_frames": 60,
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "bbox_scale": 0.75,
        "ddim_steps": 50,
        "guidance_scale": 2.5,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_arguments(Namespace(**values))


def test_prompt_inputs_are_mutually_exclusive_and_nonempty(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("turn left", encoding="utf-8")
    assert resolve_prompt(None, prompt_file) == "turn left"
    with pytest.raises(ValueError, match="exactly one"):
        resolve_prompt("walk", prompt_file)
    with pytest.raises(ValueError, match="must not be empty"):
        resolve_prompt("   ", None)


def test_predict_text_embed_helper_batches_valid_input() -> None:
    embedding = torch.randn(50, 1024)
    batched = prepare_precomputed_text_embed(
        embedding, expected_dim=1024, expected_length=50, device="cpu"
    )
    assert batched.shape == (1, 50, 1024)
    assert torch.equal(batched[0], embedding)


@pytest.mark.parametrize(
    "embedding, message",
    [
        (torch.zeros(2, 50, 1024), "exactly one"),
        (torch.zeros(1, 49, 1024), "length"),
        (torch.zeros(1, 50, 768), "feature dimension"),
        (torch.zeros(1024), "2 or 3 dimensions"),
    ],
)
def test_predict_text_embed_helper_rejects_invalid_shapes(
    embedding: torch.Tensor, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_precomputed_text_embed(
            embedding, expected_dim=1024, expected_length=50, device="cpu"
        )


def test_shape_mode_defaults_to_zero_and_overrides_both_parameter_groups() -> None:
    args = build_arg_parser().parse_args(["--prompt", "stand"])
    assert args.shape_mode == "zero"
    params = {
        "body_params_global": {"betas": torch.randn(4, 10)},
        "body_params_incam": {"betas": torch.randn(4, 10)},
    }
    enforce_zero_shape(params)
    assert torch.count_nonzero(params["body_params_global"]["betas"]) == 0
    assert torch.count_nonzero(params["body_params_incam"]["betas"]) == 0


def test_ready_is_created_only_after_atomic_directory_publish(tmp_path) -> None:
    temporary = tmp_path / ".tmp_test"
    final = tmp_path / "motion"
    temporary.mkdir()
    (temporary / "smpl_params.pt").write_bytes(b"complete")
    assert not (temporary / "READY").exists()
    publish_ready_directory(temporary, final, "2026-01-01T00:00:00Z")
    assert not temporary.exists()
    assert (final / "smpl_params.pt").read_bytes() == b"complete"
    assert (final / "READY").is_file()


def test_t5_loader_prefers_complete_local_cache_without_network() -> None:
    class Tokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            cls.calls.append(kwargs["local_files_only"])
            return "tokenizer"

    class Encoder:
        calls = []

        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            cls.calls.append(kwargs["local_files_only"])
            return "encoder"

    tokenizer, encoder, source = _load_t5_components_cached_first(
        "t5-3b", torch.float16, False, Tokenizer, Encoder
    )
    assert (tokenizer, encoder, source) == ("tokenizer", "encoder", "local cache/path")
    assert Tokenizer.calls == [True]
    assert Encoder.calls == [True]


def test_t5_loader_uses_network_only_after_incomplete_local_cache() -> None:
    class Tokenizer:
        calls = []

        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            cls.calls.append(kwargs["local_files_only"])
            return object()

    class Encoder:
        calls = []

        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            local = kwargs["local_files_only"]
            cls.calls.append(local)
            if local:
                raise OSError("cached weights incomplete")
            return object()

    _, _, source = _load_t5_components_cached_first(
        "t5-3b", torch.float16, False, Tokenizer, Encoder
    )
    assert source == "Hugging Face Hub"
    assert Tokenizer.calls == [True, False]
    assert Encoder.calls == [True, False]


def test_t5_loader_reports_actionable_socks_proxy_error() -> None:
    class AlwaysFails:
        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            if kwargs["local_files_only"]:
                raise OSError("cache missing")
            raise ValueError("Unknown scheme for proxy URL socks://127.0.0.1:7897")

    with pytest.raises(RuntimeError, match="unset ALL_PROXY/all_proxy"):
        _load_t5_components_cached_first("t5-3b", torch.float16, False, AlwaysFails, AlwaysFails)
