"""显式启用的本机GPU＋真实MuJoCo反馈验收。

SIM2SIM_REAL_TESTS=1才运行：同一初态/音乐/前缀/随机种子，仅在一个分支施加
短时真实外力，比较实际50帧历史及后续GENMO输出。外力通过xfrc_applied作用
于动力学，非reset绝不写qpos。另检查20次denoiser的实际长度60、真正no_grad
与参数冻结。它是受控因果检验，不是完整歌曲质量或实时性统计。
按当前YAML的基础模型和测试模型验证外力反馈；另用同条件/初始噪声对照原生DDIM，
测试模型有冻结前16步时，独立原生参考也使用基础模型16步加测试模型4步，
不能拿全20步均使用测试权重的另一种算法作为参考。
核对只消耗一次初始随机数、20步全程不消耗探索噪声，且原Stage2训练身份保留。
可用SIM2SIM_EVIDENCE_DIR保存本次用户要求的证据；默认pytest临时目录自动清理。
"""
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from gem.closedloop.sim2sim.config import load_config

pytestmark=pytest.mark.skipif(os.environ.get('SIM2SIM_REAL_TESTS')!='1',reason='需显式SIM2SIM_REAL_TESTS=1和本机GPU/模型资产')
CONFIG=Path(__file__).resolve().parents[3]/'configs/closedloop/sim2sim_mujoco.yaml'
MODEL_INDICES=range(len(load_config(CONFIG)['models']))


@pytest.mark.parametrize('model_index',MODEL_INDICES)
def test_real_perturbation_reaches_genmo_and_changes_generation(tmp_path,model_index):
    from gem.closedloop.sim2sim.runtime import InferenceActor
    from gem.closedloop.sim2sim.music import load_playlist,prepare_music
    from gem.closedloop.online_conditions import FrozenMusicFeatures
    from gem.closedloop.sim2sim.diagnostics import write_json
    root=Path(__file__).resolve().parents[3]
    config=load_config(root/'configs/closedloop/sim2sim_mujoco.yaml')
    sys.path.insert(0,str(Path(config['paths']['gmt_repo'])/'scripts'))
    from bumi4340_mujoco_backend import BumiMujocoBackend
    backend=BumiMujocoBackend(config);actor=InferenceActor(config,config['models'][model_index])
    track=next(t for t in load_playlist(config['playlist'],config['paths']['dataset_root']) if t.source=='AIOZ-GDANCE')
    music=prepare_music(track,tmp_path/'music_cache',decode_audio=False)
    frozen=FrozenMusicFeatures(music.features);observations=[];plans=[];shapes=[];grad_modes=[]
    def hook(module,args):
        shapes.append(list(args[0].shape));grad_modes.append(torch.is_grad_enabled())
    denoisers=[actor.actor.denoiser]
    if getattr(actor.policy,'frozen_prefix_steps',0):
        denoisers.append(actor.policy.frozen_policy.actor.denoiser)
    handles=[denoiser.register_forward_pre_hook(hook) for denoiser in denoisers]
    try:
        for disturbed in (False,True):
            backend.reset_episode('perturbed' if disturbed else 'normal')
            for step in range(50):
                backend.step_control(check_termination=False,
                    external_force=[50.,0.,0.] if disturbed and 40<=step<45 else None)
            snap=backend.snapshot();observations.append(snap)
            reservation=backend.reserve_prefix(dict(episode_id=backend.episode_id,request_id='request',decision_tick=600,deadline_tick=840))
            plans.append(actor.generate(snap,reservation,frozen,42,0,600))
        assert shapes and len(shapes)==40 and all(shape[1]==60 for shape in shapes)
        assert not any(grad_modes) and all(p.grad is None for p in actor.actor.parameters())
        assert all(o['physics_steps']==200 and o['control_steps']==50 and o['history_valid'].sum()==50 for o in observations)
        history_difference=float(np.max(np.abs(observations[0]['history_values']-observations[1]['history_values'])))
        actual_difference=float(np.max(np.abs(observations[0]['robot_qpos']-observations[1]['robot_qpos'])))
        p=plans[0]['prefix_frames']
        # 行为噪声、音乐和既有参考保持相同；只有真实状态历史不同。
        assert plans[0]['noise_seed']==plans[1]['noise_seed']
        np.testing.assert_allclose(plans[0]['qpos_world'][:p],plans[1]['qpos_world'][:p],atol=1e-5,rtol=0)
        generation_difference=float(np.max(np.abs(plans[0]['qpos_world'][p:]-plans[1]['qpos_world'][p:])))
        assert history_difference>1e-3 and actual_difference>1e-4 and generation_difference>1e-5
        assert actor.close()['frozen'] and backend.close()['frozen']
        evidence=Path(os.environ.get('SIM2SIM_EVIDENCE_DIR',str(tmp_path)))
        evidence.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(evidence/f'model{model_index+1}_physical_feedback.npz',normal_history=observations[0]['history_values'],
            disturbed_history=observations[1]['history_values'],normal_output=plans[0]['qpos_world'],disturbed_output=plans[1]['qpos_world'])
        write_json(evidence/f'model{model_index+1}_physical_feedback.json',dict(passed=True,force_n=[50.,0.,0.],force_control_steps=5,
            control_hz=50,physics_hz=200,history_max_abs_difference=history_difference,
            actual_qpos_max_abs_difference=actual_difference,generation_max_abs_difference=generation_difference,
            prefix_frames=p,denoiser_input_shapes=shapes,grad_enabled_any=any(grad_modes),
            model=actor.identity,scope='controlled physical perturbation; not a real-time benchmark'))
    finally:
        for handle in handles:handle.remove()


def native_ddim_reference(owner,batch,noise):
    """使用原生扩散实现独立核对16+4分派，不调用部署的transition_parameters。"""
    if not owner.source_kernel or not owner.source_kernel.get('frozen_prefix'):
        return owner.actor.sample(batch,steps=20,guidance_scale=2.5,noise=noise)['qpos']
    frozen=owner.policy.frozen_policy.actor
    actors=(frozen,owner.actor);conditions=[]
    for actor in actors:
        adapted=actor.adapt_conditions(batch)
        drop=torch.ones(len(noise),dtype=torch.bool,device=noise.device)
        conditions.append((adapted,actor.encode_conditions(adapted),actor.encode_conditions(adapted,drop)))
    state=owner.actor._constrain(noise,conditions[1][0]);diffusion=owner.actor._diffusion(20)
    for step in range(20):
        index=0 if step<owner.source_kernel['frozen_prefix']['steps'] else 1
        actor=actors[index];adapted,conditional,unconditional=conditions[index]
        def denoise(value,timestep,**kwargs):
            a=actor._denoise(value,timestep,adapted,conditional)
            b=actor._denoise(value,timestep,adapted,unconditional)
            return dict(pred_x_start=actor._constrain(b['pred_x_start']+2.5*(a['pred_x_start']-b['pred_x_start']),adapted),
                static_conf_logits=b['static_conf_logits']+2.5*(a['static_conf_logits']-b['static_conf_logits']))
        t=torch.full((len(noise),),19-step,dtype=torch.long,device=noise.device)
        output=diffusion.ddim_sample(denoise,state,t,clip_denoised=False,model_kwargs={'y':{}},eta=0.)
        state=actor._constrain(output['sample'],adapted)
    adapted=conditions[1][0];physical=owner.actor.endecoder.denormalize(state)
    physical=torch.where(adapted['known_mask'],adapted['known_physical'],physical)
    physical=torch.where(adapted['future_valid'][...,None],physical,0.)
    qpos=owner.actor.endecoder.codec.decode_to_canonical_qpos(physical)
    return torch.where(adapted['future_valid'][...,None],qpos,0.)


@pytest.mark.parametrize('model_index',MODEL_INDICES)
def test_deployment_matches_native_ddim_without_extra_action_noise(tmp_path,model_index):
    from gem.closedloop.sim2sim.runtime import InferenceActor
    from gem.closedloop.sim2sim.music import load_playlist,prepare_music
    from gem.closedloop.online_conditions import FrozenMusicFeatures
    from gem.closedloop.sim2sim.diagnostics import write_json
    root=Path(__file__).resolve().parents[3]
    config=load_config(root/'configs/closedloop/sim2sim_mujoco.yaml')
    sys.path.insert(0,str(Path(config['paths']['gmt_repo'])/'scripts'))
    from bumi4340_mujoco_backend import BumiMujocoBackend
    backend=BumiMujocoBackend(config);owner=InferenceActor(config,config['models'][model_index])
    track=load_playlist(config['playlist'],config['paths']['dataset_root'])[0]
    music=prepare_music(track,tmp_path/'music_cache',decode_audio=False)
    for _ in range(50):backend.step_control(check_termination=False)
    reservation=backend.reserve_prefix(dict(episode_id=backend.episode_id,request_id='parity',decision_tick=600,deadline_tick=888))
    conditions,_=owner.builder.build(backend.snapshot(),reservation,FrozenMusicFeatures(music.features),music_start_tick=600)
    batch={k:v.to(owner.device) for k,v in conditions.items()};errors=[]
    with torch.no_grad():
        for seed in (42,43,44):
            generator=torch.Generator(device=owner.device).manual_seed(seed)
            noise=torch.randn((1,60,30),device=owner.device,generator=generator)
            rng_after_initial=generator.get_state().clone()
            demo=owner._sample(batch,generator,initial_noise=noise)
            assert torch.equal(generator.get_state(),rng_after_initial)
            native=native_ddim_reference(owner,batch,noise)
            error=float((demo['qpos']-native).abs().max());errors.append(error)
            torch.testing.assert_close(demo['qpos'],native,atol=1e-5,rtol=1e-5)
            implicit_generator=torch.Generator(device=owner.device).manual_seed(seed)
            implicit=owner._sample(batch,implicit_generator)
            assert torch.equal(implicit_generator.get_state(),rng_after_initial)
            torch.testing.assert_close(implicit['qpos'],demo['qpos'],atol=0,rtol=0)
    assert owner.policy.steps==20 and owner.policy.eta==0.
    assert owner.identity['sampler']=='ddim20_eta0_initial_noise_only'
    assert owner.identity['kernel']['noise']=='initial_only'
    assert owner.identity['kernel']['effective_std_floors']==[0.]*20
    assert owner.identity['kernel']['std_floor']==0. and not owner.identity['kernel']['final_step_noise']
    if model_index:
        assert owner.identity['source_training_kernel']==owner.source_identity['kernel']
        assert owner.identity['source_training_kernel']['eta']==.1
        assert owner.identity['source_training_kernel']['effective_std_floors'][-1]==.12
        assert owner.identity['source_training_kernel']['frozen_prefix']['steps']==16
        from gem.closedloop.dppo.evaluation import _model_fingerprint
        assert _model_fingerprint(owner.actor)==owner.source_checkpoint['actor_fingerprint']
    assert owner.close()['frozen'] and backend.close()['frozen']
    evidence=Path(os.environ.get('SIM2SIM_EVIDENCE_DIR',str(tmp_path)));evidence.mkdir(parents=True,exist_ok=True)
    write_json(evidence/f'model{model_index+1}_native_ddim_parity.json',dict(seeds=[42,43,44],qpos_max_errors=errors,
        sampler=owner.identity['sampler'],model=owner.identity,only_initial_noise_consumed=True,
        scope='same weights/conditions/initial noise; native eta=0 DDIM semantics'))
