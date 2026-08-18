"""Tests for the non-blocking GMR MuJoCo visualization sidecar client."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from gem.runtime.gmr_mujoco_viewer import (
    VIEWER_FRAME_BYTES,
    GMRMujocoViewerClient,
    encode_qpos_viewer_frame,
)
from gem.runtime.gmt_trajectory import BUMI_QPOS_DIM


def test_qpos_viewer_frame_contract_and_validation() -> None:
    qpos = np.arange(BUMI_QPOS_DIM, dtype=np.float32)
    payload = encode_qpos_viewer_frame(qpos)
    assert len(payload) == VIEWER_FRAME_BYTES
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4"), qpos)

    with pytest.raises(ValueError, match="shape"):
        encode_qpos_viewer_frame(qpos[:-1])
    qpos[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        encode_qpos_viewer_frame(qpos)


def test_viewer_client_readiness_and_binary_delivery(tmp_path) -> None:
    output = tmp_path / "qpos.bin"
    child = (
        "import pathlib,sys; "
        "sys.stdout.buffer.write(b'V'); sys.stdout.buffer.flush(); "
        f"pathlib.Path({str(output)!r}).write_bytes(sys.stdin.buffer.read({VIEWER_FRAME_BYTES}))"
    )
    qpos = np.linspace(-1.0, 1.0, BUMI_QPOS_DIM, dtype=np.float32)
    client = GMRMujocoViewerClient([sys.executable, "-c", child])
    assert client.alive
    assert client.publish(qpos)
    client.close()
    assert client.published_frames == 1
    np.testing.assert_array_equal(
        np.frombuffer(output.read_bytes(), dtype="<f4"), qpos
    )
