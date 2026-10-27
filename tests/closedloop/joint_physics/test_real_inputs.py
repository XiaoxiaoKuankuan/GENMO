"""可选的原配对数据与真实Newton sidecar无泄漏验收。

仅在显式JOINT_REAL_TEST_CONFIG和JOINT_SIDECAR存在时运行，不生成替代数据。修改当前
未知标签、contact和地面监督后，请求历史/旧计划/ForecastContext必须逐元素不变；真实
state55必须区别于GT参考。只用真实codec和冻结SM，不加载任何GENMO权重。
"""

import copy
import os

import pytest
import torch

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.closedloop.joint_physics.config import read_config, build_adapter, ROOT
from gem.closedloop.joint_physics.data import SourceCatalog, SidecarStore
from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
from gem.closedloop.joint_physics.reference import attach_sidecars
from gem.robots.bumi.endecoder import BumiEndecoder


def test_actual_sidecar_conditions_do_not_read_unknown_labels(monkeypatch):
    path = os.getenv("JOINT_REAL_TEST_CONFIG")
    if not path:
        pytest.skip("需要真实原配对数据和Newton sidecar，不能用模拟数据替代")
    cfg = read_config(path)
    adapter = build_adapter(cfg, "cpu")
    def forbidden(*args, **kwargs):
        raise AssertionError("随机读取离线缓存时不得重新运行WM预处理")
    monkeypatch.setattr(adapter.bundle, "step_model", forbidden)
    options = dict(cfg["stage1"]["endecoder"])
    options["kinematics_path"] = str(ROOT/options["kinematics_path"])
    codec = BumiEndecoder(**options).codec
    store = SidecarStore(cfg["data"]["sidecar"], "train", asset_contract(cfg, adapter))
    dataset, row = next(SourceCatalog(cfg, "train").sources())
    for frame in (30, 60, 90, 120):
        batch = collate_stage1_training_samples([dataset.get_window(row, frame)])
        before, context, windows, _ = attach_sidecars(batch, store, adapter, codec, 8)
        if windows[0] is not None:
            break
    assert windows[0] is not None
    modified = copy.deepcopy(batch)
    modified["target_qpos30"][~batch["known_qpos30_mask"]] += 3
    modified["target_contact"] = 1-modified["target_contact"]
    modified["meta"][0]["ground_supervision"]["ground_height_world_m"] += 2
    after, other_context, other_windows, _ = attach_sidecars(modified, store, adapter, codec, 8)
    for key in STAGE1_CONDITION_KEYS:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    torch.testing.assert_close(context[0].features, other_context[0].features, atol=0, rtol=0)
    torch.testing.assert_close(windows[0].request.reference.world_qpos, other_windows[0].request.reference.world_qpos, atol=0, rtol=0)
    target, _, _ = windows[0].request.reference.read_current_future(windows[0].request.tick, 0)
    assert (windows[0].request.state[:, 13:34]-target.q).abs().max() > 1e-5
    assert before["meta"][0]["proprio_source"] == "real_continuous_newton_sidecar"
    assert context[0].valid.all()
