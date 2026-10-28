"""批量参考分页与原单歌曲ReferenceBuffer的数值一致性和可见域测试。

读取只读SM机器人资产运行真实解析FK，不启动Newton或训练。不同槽位使用不同运动与
控制年龄，对照按原提交版本裁剪后的current/future/goal，并核对页切换、槽位复用和
右侧halo不足时严格拒绝，防止完整预计算无意泄漏尚未提交的标签。
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.kinematics import Kinematics
from supermimic.robots import get_robot_spec
from gem.closedloop.joint_physics.batched_reference import ReferencePageCache
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer


def fixture():
    spec = get_robot_spec("bumi3_4340")
    # 只核对相同FK下分页数学一致性，使用仓库源MJCF，避免要求生成USD/仿真资产。
    spec = replace(spec, mjcf_path=spec.source_mjcf_path)
    kin = Kinematics("cpu", spec)
    adapter = SimpleNamespace(runtime=SimpleNamespace(kin=kin), to_sm=tuple(range(21)))
    episodes = []
    for sign in (1, -1):
        frames = 240
        t = torch.arange(frames)/30
        qpos = torch.zeros(frames, 28)
        qpos[:, 0] = sign*.07*t
        qpos[:, 1] = sign*.03*torch.sin(t)
        qpos[:, 2] = .65
        qpos[:, 3] = torch.cos(.1*t)
        qpos[:, 6] = sign*torch.sin(.1*t)
        qpos[:, 7:] = torch.tensor(spec.default_joint_pos) + .015*torch.sin(t[:, None])
        versions = torch.div((torch.arange(frames)-120).clamp_min(0)*20, 300, rounding_mode="floor")+1
        versions[:120] = 0
        episodes.append({"reference_qpos": qpos, "source_versions": versions,
            "old_plan_frames": 120, "old_plan_period_ticks": 300})
    return adapter, episodes


def old_read(adapter, episode, age):
    tick = age*12
    upper = min(len(episode["reference_qpos"])-1, 119+(tick//300)*15)
    ticks = torch.arange(upper+1)*20
    ref = ReferenceBuffer(ticks, episode["reference_qpos"][:upper+1][None],
        torch.zeros(1, upper+1, 2), episode["source_versions"][:upper+1], adapter.runtime.kin, adapter.to_sm)
    return ref.read_current_future(tick)


def test_independent_age_pages_match_original_version_reference():
    adapter, episodes = fixture()
    pages = ReferencePageCache(3, 8, adapter, "cpu")
    pages.assign(0, episodes[0]); pages.assign(1, episodes[1])
    assert pages.initial(0).shape == (1, 55)
    for a, b in ((0, 1), (1, 25), (25, 50), (76, 7)):
        actual = pages.read(np.array([a, b, 0]), np.array([True, True, False]), [*episodes, None])
        for slot, age in enumerate((a, b)):
            expected = old_read(adapter, episodes[slot], age)
            for got_ref, want_ref in zip(actual[:2], expected[:2]):
                for got, want in zip(got_ref, want_ref):
                    torch.testing.assert_close(got[slot:slot+1], want, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(actual[2][slot:slot+1], expected[2], rtol=1e-6, atol=1e-6)
        for reference in actual[:2]:
            for field in reference:
                assert torch.isfinite(field).all()
                torch.testing.assert_close(field[2], field[0])
    pages.release(0)
    pages.assign(0, episodes[1])
    actual = pages.read([0, 50, 0], [True, True, False], [episodes[1], episodes[1], None])
    torch.testing.assert_close(actual[2][:1], old_read(adapter, episodes[1], 0)[2])
    assert pages.statistics()["page_uploads"] > 2


def test_precomputed_future_does_not_extend_visible_halo():
    adapter, episodes = fixture()
    episode = {**episodes[0], "old_plan_frames": 60, "old_plan_period_ticks": 1200}
    pages = ReferencePageCache(1, 8, adapter, "cpu")
    pages.assign(0, episode)
    with pytest.raises(ReferenceExhausted, match="已提交参考"):
        pages.read([90], [True], [episode])  # tick1080+144超出首个已提交计划1180，整曲仍有支持。
    with pytest.raises(ValueError, match="来源身份"):
        pages.read([0], [True], [dict(episode)])
    with pytest.raises(ReferenceExhausted, match="没有合法活动"):
        pages.read([0], [False], [episode])
