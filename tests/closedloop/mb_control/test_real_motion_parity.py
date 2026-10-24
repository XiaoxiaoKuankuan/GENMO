"""原MotionLib与动态缓存的真实动作资产parity测试源，本轮未执行。

仅接受显式SM_BRIDGE_MOTION_FILE中的30Hz真实UMR动作、同源checkpoint与RobotSpec。
将源关节故意倒序后按名称恢复，在起点、非零朝向和尾部安全范围对照goal/obs/reward/action。
这属于接口验收，不能代替Newton真实闭环或物理梯度方向实验。
"""

import os

import pytest
import torch


def test_motionlib_dynamic_buffer_goal_obs_reward_action():
    motion_file, checkpoint = os.getenv("SM_BRIDGE_MOTION_FILE"), os.getenv("SM_BRIDGE_CHECKPOINT")
    if not motion_file or not checkpoint:
        pytest.skip("需要显式真实UMR动作与SM编号checkpoint")
    from supermimic.bridge.bundle import load_bundle
    from supermimic.bridge.runtime import SuperMimicRuntime
    from supermimic.motion_cache import read_umr_npz
    from supermimic.motion import MotionLib
    from supermimic.mdp import MDP, flatten_state
    from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
    bundle = load_bundle(checkpoint, robot_asset_dir=os.getenv("SM_ROBOT_ASSET_DIR"))
    runtime = SuperMimicRuntime(bundle)
    p, quat, q, fps = read_umr_npz(motion_file, bundle.spec)
    assert fps == 30
    motion = MotionLib([motion_file], runtime.kin, "cpu", spec=bundle.spec)
    original = MDP(runtime.kin, motion, runtime.mdp.default_q)
    native = tuple(reversed(bundle.spec.joint_names))
    to_native = [bundle.spec.joint_names.index(n) for n in native]
    to_sm = tuple(native.index(n) for n in bundle.spec.joint_names)
    world = torch.cat((p, quat, q[:, to_native]), -1)[None]
    ticks = torch.arange(len(p)) * 20
    buffer = ReferenceBuffer(ticks, world, torch.zeros(1, len(p), 2), torch.zeros_like(ticks), runtime.kin, to_sm)
    xi = torch.zeros(1, 25)
    xi[:, 0] = 1.
    previous = torch.zeros(1, 21)
    for frame in (0, min(9, motion.num_frames - 13), motion.num_frames - 13):
        if frame < 0:
            pytest.skip("动作过短，无法覆盖未来10帧与速度halo")
        f = torch.tensor([frame])
        ref, future, goal = buffer.read_current_future(frame * 12)
        state = motion.ref_state(f)
        old_goal = motion.goal(f, 10)
        torch.testing.assert_close(goal, old_goal, atol=2e-4, rtol=2e-4)
        obs = runtime.observation(flatten_state(state), previous, xi, ref)
        old_obs = original.obs(state, previous, xi, f)
        torch.testing.assert_close(obs, old_obs, atol=2e-4, rtol=2e-4)
        a = bundle.act(obs[:, None].expand(-1, 10, -1), goal)
        old_a = bundle.act(old_obs[:, None].expand(-1, 10, -1), old_goal)
        torch.testing.assert_close(a, old_a, atol=5e-4, rtol=5e-4)
        reward, _ = runtime.reward(flatten_state(state), a, previous, ref)
        old_reward = original.reward(state, f, old_a, previous)
        torch.testing.assert_close(reward, old_reward, atol=2e-4, rtol=2e-4)
