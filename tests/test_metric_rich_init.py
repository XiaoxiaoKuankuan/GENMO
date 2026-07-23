"""Regression tests for the RICH metric callback's body-model dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import gem.callbacks.metric.metric_rich as metric_rich_module


class _DummySMPLXModel:
    """Minimal identity-bearing stand-in for an SMPL-X model."""

    def __init__(self, index: int) -> None:
        self.index = index


def test_metric_rich_init_does_not_load_unused_smpl_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MetricRICH must initialize without constructing the unused SMPL model."""

    body_model_calls: list[str] = []
    loaded_paths: list[Path] = []
    j_regressor = torch.zeros(2, 3)
    smplx2smpl = torch.zeros(3, 4)

    def fake_make_smplx(model_type: str):
        body_model_calls.append(model_type)
        if model_type == "smpl":
            pytest.fail("MetricRICH must not request the unused SMPL_NEUTRAL.pkl model")
        assert model_type == "supermotion"
        return _DummySMPLXModel(len(body_model_calls))

    def fake_torch_load(path: str | Path):
        artifact = Path(path)
        loaded_paths.append(artifact)
        if artifact.name == "smpl_neutral_J_regressor.pt":
            return j_regressor
        if artifact.name == "smplx2smpl_sparse.pt":
            return smplx2smpl
        pytest.fail(f"MetricRICH requested an unexpected artifact: {artifact}")

    monkeypatch.setattr(metric_rich_module, "make_smplx", fake_make_smplx)
    monkeypatch.setattr(metric_rich_module.torch, "load", fake_torch_load)

    metric = metric_rich_module.MetricRICH()

    assert body_model_calls == ["supermotion", "supermotion", "supermotion"]
    assert list(metric.smplx_model) == ["male", "female", "neutral"]
    assert len({id(model) for model in metric.smplx_model.values()}) == 3
    assert metric.J_regressor is j_regressor
    assert metric.smplx2smpl is smplx2smpl
    assert [path.name for path in loaded_paths] == [
        "smpl_neutral_J_regressor.pt",
        "smplx2smpl_sparse.pt",
    ]
    assert not hasattr(metric, "faces_smpl")
