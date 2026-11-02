"""验证restart_h32的真实权重、物理条件和本地MuJoCo执行合同。

复用明确指定的真实checkpoint与MJCF，不用合成模型代替新WM；检查119维条件的
质量参数能影响预测并反传、超界动作在想象和执行中一致裁剪，实际XML自碰撞和
被动摩擦仍有效。仅为接口与有限接触回归，不能证明完整舞蹈或硬件效果。
"""
import numpy as np
import torch
import mujoco
from supermimic.physics_context import context_slices, normalize_context
from test_vector import vector


def test_actual_world_model_context_gradient_and_clipping(system):
    engine,z=system
    xi=z.xi.clone().requires_grad_()
    assert xi.shape[-1]==119 and engine.dynamics.model.in_dim==195
    predicted=engine.dynamics.step(z.executed_state,torch.zeros(1,21),xi)
    gradient,=torch.autograd.grad(predicted.square().sum(),xi)
    assert torch.isfinite(gradient).all() and gradient[:,context_slices(21)['torso_mass']].abs().sum()>0
    a=torch.full((1,21),100.)
    torch.testing.assert_close(engine.dynamics.step(z.executed_state,a,z.xi),engine.dynamics.step(z.executed_state,a.clamp(-20,20),z.xi))
    from dataclasses import replace
    from gem.closedloop.reference_adapter.music_reward import AudioTrack
    probe=replace(z,audio=AudioTrack.from_features(z.audio.features[:,:-10]))
    _,_,control,state=vector((engine,probe))
    assert control.propose(state,action_offset=a)['action'].abs().max()<=20
    normalized=normalize_context(z.xi)
    torch.testing.assert_close(normalized[:,25:26],z.xi[:,25:26]/10.)


def test_native_mujoco_keeps_latest_friction_and_self_contacts(assets):
    from gem.closedloop.reference_adapter.mujoco_env import MujocoPhysicalEnv
    bundle,_=assets
    env=MujocoPhysicalEnv(bundle.spec)
    np.testing.assert_allclose(env.model.dof_damping[env.vadr],bundle.spec.viscous_friction)
    np.testing.assert_allclose(env.model.dof_frictionloss[env.vadr],bundle.spec.coulomb_friction)
    assert env.xi.shape==(1,119) and env.xi[0,25]>0
    limits=np.array([env.model.joint(n).range for n in bundle.spec.joint_names])
    rng=np.random.default_rng(4340)
    observed=False
    env.data.qpos[2:4]=[2.,1.]
    for _ in range(40):
        env.data.qpos[env.qadr]=rng.uniform(limits[:,0],limits[:,1])
        mujoco.mj_forward(env.model,env.data)
        for contact in env.data.contact:
            bodies={env.model.body(int(env.model.geom_bodyid[g])).name for g in contact.geom}
            if 'waist_yaw_link' in bodies and any('elbow_pitch_link' in n for n in bodies):observed=True
    assert observed
