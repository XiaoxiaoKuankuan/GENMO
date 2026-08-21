from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from gem.utils.bumi_checkpoint_adapter import adapt_smpl_music_checkpoint_to_bumi


class FakeBumiModel(nn.Module):
    def __init__(self, latent: int = 4) -> None:
        super().__init__()
        self.music_embedder = nn.Linear(35, latent)
        self.blocks = nn.Sequential(nn.Linear(latent, latent), nn.LayerNorm(latent))
        self.add_cond_linear = nn.Linear(latent + 93, latent)
        self.final_layer = nn.Linear(latent, 93)


def source_state(model: FakeBumiModel) -> dict[str, torch.Tensor]:
    state = {key: torch.full_like(value, 3.0) for key, value in model.state_dict().items()}
    state["add_cond_linear.weight"] = torch.full((4, 4 + 151), 7.0)
    state["final_layer.weight"] = torch.full((151, 4), 9.0)
    state["final_layer.bias"] = torch.full((151,), 9.0)
    return state


def test_exact_partial_and_expected_skip_are_reported() -> None:
    model = FakeBumiModel()
    motion_columns_before = model.add_cond_linear.weight[:, 4:].detach().clone()
    _, report = adapt_smpl_music_checkpoint_to_bumi(
        model, {"state_dict": source_state(model)}
    )
    assert torch.all(model.music_embedder.weight == 3.0)
    assert torch.all(model.blocks[0].weight == 3.0)
    assert torch.all(model.add_cond_linear.weight[:, :4] == 7.0)
    torch.testing.assert_close(model.add_cond_linear.weight[:, 4:], motion_columns_before)
    assert any(item["key"] == "add_cond_linear.weight" for item in report["loaded_partial"])
    assert "final_layer.weight" in report["skipped_expected"]
    assert report["unclassified_shape_mismatch"] == []


def test_unclassified_allowlisted_shape_mismatch_raises() -> None:
    model = FakeBumiModel()
    state = source_state(model)
    state["music_embedder.weight"] = torch.zeros(5, 35)
    with pytest.raises(RuntimeError, match="unclassified shape mismatches"):
        adapt_smpl_music_checkpoint_to_bumi(model, {"state_dict": state})
