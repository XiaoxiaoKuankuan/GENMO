"""音乐 MuJoCo 入口的真实资产合同回归，不启动训练或修改已有模型。

使用固定 SM 权重及 RobotSpec/MJCF 检查世界/局部角速度、21关节名称映射、
真实200Hz物理推进、PD执行动作和力矩上限；影子测试证明可视参考不回写实体。
同时以原 VectorControl 检查零/非零十帧 Goal 与冻结 Actor 的数值一致性。
合成短参考只用于接口回归，不代表真实整曲跟踪改善。
"""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
from gem.closedloop.reference_adapter.mujoco_env import MujocoPhysicalEnv, NominalGhost


def test_music_reference_cache_tracks_sampling_parameters(system, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gem.closedloop.reference_adapter.mujoco_demo import prepare_reference
    from gem.runtime import multimodal_stage1
    engine, original = system
    calls = []

    def generate(config, task, **kwargs):
        calls.append((config["sample_steps"], config["guidance_scale"]))
        return dict(world_qpos=original.nominal_reference.world_qpos,
                    joint_names=engine.step.bundle.spec.joint_names,
                    contact=torch.zeros(1, len(original.nominal_reference.world_qpos[0]), 2))

    monkeypatch.setattr(multimodal_stage1, "generate_stage1_reference", generate)
    genmo = {"sample_steps": 2, "guidance_scale": 1.5}
    for name in ("checkpoint", "stats", "kinematics"):
        path = tmp_path / name
        path.write_text(name)
        genmo[name] = str(path)
    config = dict(genmo=genmo, cache_root=str(tmp_path / "cache"))
    prepared = SimpleNamespace(identity=dict(audio_sha256="audio", feature_values_sha256="feature"),
                               features=np.zeros((31, 35)), duration=1.)
    _, first = prepare_reference(config, prepared, engine.step.bundle, 42, "cpu")
    _, repeat = prepare_reference(config, prepared, engine.step.bundle, 42, "cpu")
    genmo["sample_steps"] = 3
    _, changed_steps = prepare_reference(config, prepared, engine.step.bundle, 42, "cpu")
    genmo["guidance_scale"] = 2.
    _, changed_cfg = prepare_reference(config, prepared, engine.step.bundle, 42, "cpu")
    assert first["path"] == repeat["path"]
    assert len({first["path"], changed_steps["path"], changed_cfg["path"]}) == 3
    assert calls == [(2, 1.5), (3, 1.5), (3, 2.)]


def initial_for(spec):
    from supermimic.rotations import mat_to_quat, so3_exp
    state = torch.zeros(1, 55)
    state[:, :3] = torch.tensor([.1, -.2, .9])
    state[:, 3:7] = mat_to_quat(so3_exp(torch.tensor([[.15, -.2, .7]])))
    state[:, 7:13] = torch.tensor([.11, -.08, .07, .4, -.3, .2])
    state[:, 13:34] = torch.tensor(spec.default_joint_pos)
    state[:, 34:] = torch.linspace(-.1, .1, 21)
    return state


def test_state_roundtrip_and_named_pd(assets):
    bundle, _ = assets
    env = MujocoPhysicalEnv(bundle.spec)
    state = initial_for(bundle.spec)
    observed = env.reset(state)
    torch.testing.assert_close(observed["state"], state, atol=2e-7, rtol=0)
    np.testing.assert_allclose(env.model.actuator_gainprm[:, 0], bundle.spec.kp)
    np.testing.assert_allclose(env.model.actuator_biasprm[:, 2], -np.asarray(bundle.spec.kd))
    np.testing.assert_allclose(env.model.dof_armature[env.vadr], bundle.spec.armature)
    np.testing.assert_allclose(env.model.actuator_forcerange[:, 1], bundle.spec.effort_limits)
    action = torch.linspace(-.3, .3, 21)[None]
    actual = env.step_physics(action)
    assert env.data.time == pytest.approx(.02)
    assert env.control_steps == 1
    assert not torch.equal(actual, state)
    np.testing.assert_allclose(env.data.ctrl, np.asarray(bundle.spec.default_joint_pos)+
                              np.asarray(bundle.spec.action_scales)*action[0].numpy())
    assert (np.abs(env.data.actuator_force) <= np.asarray(bundle.spec.effort_limits)+1e-9).all()
    torch.testing.assert_close(env.observe()["previous_action"], action)


def test_actual_action_changes_physics_and_nonfinite_rejected(assets):
    env = MujocoPhysicalEnv(assets[0].spec)
    state = initial_for(env.spec)
    env.reset(state)
    plus = env.step_physics(torch.ones(1, 21)*.2)
    env.reset(state)
    minus = env.step_physics(torch.ones(1, 21)*-.2)
    assert (plus[:, 34:]-minus[:, 34:]).abs().max() > .001
    before = env.data.time
    with pytest.raises(FloatingPointError):
        env.step_physics(torch.full((1, 21), float("nan")))
    assert env.data.time == before


def test_blue_ghost_does_not_change_actual_state(assets):
    from types import SimpleNamespace
    import mujoco
    env = MujocoPhysicalEnv(assets[0].spec)
    env.reset(initial_for(env.spec))
    before = env.data.qpos.copy(), env.data.qvel.copy(), env.data.time, env.data.ncon
    ghost = NominalGhost(env)
    ref = SimpleNamespace(root_p=torch.tensor([[2., 1., .6]]), root_R=torch.eye(3)[None],
                          q=torch.tensor(env.spec.default_joint_pos)[None])
    ghost.update(ref)
    scene = mujoco.MjvScene(env.model, maxgeom=1500)
    camera = mujoco.MjvCamera()
    ghost.append(scene, camera)
    assert scene.ngeom > 0
    assert scene.geoms[0].rgba[2] > scene.geoms[0].rgba[0]
    assert scene.geoms[0].rgba[3] == pytest.approx(.25)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.qvel, before[1])
    assert (env.data.time, env.data.ncon) == before[2:]
    np.testing.assert_allclose(ghost.data.qpos[:3], ref.root_p[0])


@torch.no_grad()
def test_same_training_goal_actor_and_history_with_real_physics(system):
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.music_reward import AudioTrack
    from supermimic.motion import goal_from_reference, select_reference
    engine, original = system
    audio = AudioTrack.from_features(original.audio.features[:, :70])
    bank = VectorBank.from_entries([(dict(id="test", split="train"), original.executed_state,
                                    original.nominal_reference, audio)], engine.step.runtime.kin, engine.step.rewards.music)
    control = VectorControl(engine, bank)
    env = MujocoPhysicalEnv(engine.step.bundle.spec)
    ids = torch.zeros(1, dtype=torch.long)
    z = control.initial(env.reset(original.executed_state), ids, ids, ids)
    zero = control.propose(z, zero=True)
    assert zero["goal"].shape == (1, 570)
    _, expected_goal = control.nominal_goal(z)
    torch.testing.assert_close(zero["goal"], expected_goal, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(zero["action"], engine.step.bundle.act(z["history"], expected_goal), atol=2e-5, rtol=2e-5)
    engine.step.adapter.net[-1].bias.fill_(.05)
    changed = control.propose(z)
    assert changed["residual"].shape == (1, 10, 27)
    assert (changed["action"]-zero["action"]).abs().max() > 1e-6
    rebuilt = goal_from_reference(changed["current"], changed["future"])
    torch.testing.assert_close(changed["goal"], rebuilt)
    actual = env.step_physics(changed["action"])
    end, _, raw, _, _, _, _ = control.advance(z, changed, actual)
    expected, _ = control.runtime.reward(actual, changed["action"], z["action"], bank.ref(ids, ids+1), diagnostics=True)
    torch.testing.assert_close(raw["sm"], expected)
    obs = control.runtime.observation(actual, changed["action"], z["xi"],
                                      select_reference(changed["future"], (slice(None), 0)))
    torch.testing.assert_close(end["history"][:, -1], obs)
    torch.testing.assert_close(end["anchor"], changed["residual"][:, 0])
