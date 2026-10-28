#!/usr/bin/env python3
"""导出最新联合GENMO的三类同音乐轨迹，供原参考/自由生成/真实执行同步观看。

显式选择当前划分中的样本，原参考只提供单个初态；随后复用已有因果滚动生成器，
以60帧、12帧旧生成前缀、DDIM20/CFG2.5/eta0从独立纯噪声生成全曲。历史来自自身
已生成姿态代理，没有实际状态缓存，因此FutureContext显式无效，不伪造在线物理反馈。
额外生成9个源帧支持真实SM未来目标及速度halo，歌曲尾部不复制姿态，音乐缺失明确mask。
生成完成后，冻结SM在Newton/MuJoCo-Warp中逐控制步跟踪同一参考；每曲只重置一次，
保留原终止和实际硬限位检查，失败后不拼接、不补物理帧。这里不是墙钟异步闭环评价。
输出只写全新目录，保留模型/源文件/资产身份、30Hz参考、50Hz实测、接触和结束原因。
此入口不训练、不修改模型、原配对库、缓存或共享SuperMimic源码。
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-id', action='append', required=True)
    parser.add_argument('--split', default='val', choices=['train','val','test'])
    parser.add_argument('--seed', type=int, default=42)
    prelim, _ = parser.parse_known_args()
    from gem.closedloop.joint_physics.config import read_config, build_adapter, seed_all
    cfg = read_config(prelim.config)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.mdp import flatten_state
    from gem.closedloop.mb_control.contracts import ForecastContext, ReferenceExhausted
    from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
    from gem.closedloop.joint_physics.checkpoint import atomic_save, read_checkpoint
    from gem.closedloop.joint_physics.contracts import digest
    from gem.closedloop.joint_physics.data import SourceCatalog
    from gem.closedloop.joint_physics.evaluation import load_actor
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    from tools.eval.generate_stage1_music_dances import generate
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = 'cuda:0'
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg['data']['source_ids'] = args.source_id
    catalog = SourceCatalog(cfg, args.split)
    selected = {dataset.rows[row]['sample_id']:(dataset,row) for dataset,row in catalog.sources()}
    if set(selected) != set(args.source_id):
        raise ValueError('所选来源没有全部位于指定划分')
    actor,_ = load_actor(cfg,args.checkpoint,device)
    actor.requires_grad_(False)
    adapter = build_adapter(cfg,device)
    assets = asset_contract(cfg,adapter)
    value = read_checkpoint(args.checkpoint)
    if value['contract']['assets'] != assets:
        raise ValueError('评价与训练冻结资产不同')
    step = value['global_step']
    del value
    frozen = adapter.bundle.frozen_snapshot()
    checkpoint_sha = digest(args.checkpoint)
    names = list(actor.stage1.endecoder.kinematics.joint_order)
    from_sm = [adapter.bundle.spec.joint_names.index(name) for name in names]

    class Runtime:
        def sample(self, conditions, steps, guidance_scale, noise):
            if steps != 20 or guidance_scale != 2.5:
                raise ValueError('必须使用完整原采样合同')
            context = ForecastContext.invalid(noise, round(float(conditions['decision_time'][0])*600))
            result = actor.sample_differentiable(conditions,context,noise)
            return dict(result,qpos=result['qpos28'])

    manifest = dict(schema='genmo.joint_physics.three_view.v1',checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=checkpoint_sha,step=step,assets=assets,sampling=cfg['sampling'],split=args.split,
        history_source='own_generated_causal_kinematic_proxy',forecast_valid=False,
        generation='offline_autoregressive_pure_noise_60frames_prefix12',
        physics='real_newton_mujoco_warp_frozen_sm_tracking_fixed_generated_reference',
        control_fps=50,reference_fps=30,tail_generated_halo_frames=9,sources=[])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    sim_cfg = make_sim(device,args)
    sim_cfg = replace(sim_cfg,physics=replace(sim_cfg.physics,class_type=None,
        solver_cfg=replace(sim_cfg.physics.solver_cfg,disable_sensors=True),deterministic_mode=cfg['physics']['determinism']))
    with launch_simulation(sim_cfg,args):
        env = StreamingPhysicalEnv(sim_cfg,1,adapter.bundle)
        for number,source in enumerate(args.source_id,1):
            dataset,row = selected[source]
            original_row = dataset.rows[row]
            sequence = dataset.reader.load_aligned_sequence(original_row)
            ground = dataset._ground_supervision(sequence)
            original = sequence['qpos'].clone().cpu()
            original[:,2] -= ground['ground_height_world_m']
            n = len(original)
            directory = output/f'{number:02d}'
            directory.mkdir()
            meta = dataset.get_window(row,0)['meta']
            audio = Path(meta['motion_path']).parent.parent/original_row['audio_path']
            if digest(audio) != original_row['source_audio_sha256']:
                raise ValueError('原音乐内容身份改变')
            shutil.copy2(audio,directory/'audio.wav')
            started = time.perf_counter()
            generated,logits,contact_valid,windows = generate(actor.stage1,Runtime(),sequence['music'].cpu(),n+9,
                args.seed+number*1000,directory,window_frames=60,prefix_frames=12,initial_qpos=original[0])
            common = dict(robot_name='bumi',quaternion_convention='wxyz',qpos_order='mujoco_native',joint_names=names)
            atomic_save(dict(common,qpos=original,fps=30,evidence='original_dataset_reference'),directory/'original.pt')
            atomic_save(dict(common,qpos=generated[:n],fps=30,evidence='pure_noise_joint_genmo_reference',
                generated_halo=generated[n:],contact_logits=logits[:n],windows=windows),directory/'generated.pt')
            ref = ReferenceBuffer(torch.arange(len(generated),device=device)*20,generated.to(device)[None],
                logits.sigmoid().to(device)[None],torch.zeros(len(generated),dtype=torch.long,device=device),
                adapter.runtime.kin,adapter.to_sm)
            records,actions,contacts,tracking,errors=[],[],[],[],[]
            end_reason = 'music_end'
            tick=0
            with torch.no_grad():
                first,_,_ = ref.read_current_future(0)
                initial = flatten_state((first.root_p,first.root_R,first.root_v,first.root_w,first.q,first.qd))
                env.reset(torch.arange(1,device=device),initial)
                obs = env.observe()
                records.append(obs['state'][0].cpu())
                history = None
                while tick < n*20:
                    try:
                        current,_,goal = ref.read_current_future(tick)
                    except ReferenceExhausted:
                        end_reason='reference_support_exhausted';break
                    before=env.observe()
                    observation=adapter.runtime.observation(before['state'],before['previous_action'],before['xi'],current)
                    history=observation[:,None].expand(-1,10,-1).clone() if history is None else torch.cat((history[:,1:],observation[:,None]),1)
                    action=adapter.bundle.act(history,goal)
                    if not torch.isfinite(action).all():end_reason='nonfinite_action';break
                    state=env.step_physics(action)
                    if not torch.isfinite(state).all():end_reason='nonfinite_measured_state';break
                    tick+=12
                    target,_,_=ref.read_current_future(tick,0)
                    reward,_=adapter.runtime.reward(state,action,before['previous_action'],target)
                    records.append(state[0].cpu());actions.append(action[0].cpu());contacts.append(env.contact_events())
                    tracking.append(float(reward));errors.append(float((state[:,13:34]-target.q).square().mean()))
                    measured=state[:,13:34]
                    if bool(((measured<adapter.runtime.kin.q_lo-1e-4)|(measured>adapter.runtime.kin.q_hi+1e-4)).any()):
                        end_reason='measured_hard_joint_limit';break
                    if bool(adapter.runtime.termination(state,target).any()):
                        end_reason='sm_physical_termination';break
            states=torch.stack(records)
            qpos=torch.cat((states[:,:7],states[:,13:34][:,from_sm]),-1)
            atomic_save(dict(common,qpos=qpos,fps=50,evidence='measured_newton_state',state55=states,
                ticks=torch.arange(len(states))*12,actions=torch.stack(actions) if actions else torch.empty(0,21),
                real_contacts=contacts,end_reason=end_reason,physical_end_seconds=tick/600),directory/'physics.pt')
            result=dict(number=number,directory=directory.name,source=source,dataset=dataset.dataset_name,split=args.split,
                title=original_row.get('song_name',source),frames=n,music_seconds=n/30,physical_seconds=tick/600,
                physical_control_steps=len(actions),complete_song=end_reason=='music_end',end_reason=end_reason,
                tracking_reward_mean=sum(tracking)/max(1,len(tracking)),joint_tracking_rmse_rad=math.sqrt(sum(errors)/max(1,len(errors))),
                source_motion_sha256=digest(meta['motion_path']),source_music_feature_sha256=digest(meta['music_feature_path']),
                audio_sha256=digest(audio),ground_metadata=ground,seed=args.seed+number*1000,
                startup_hold_seconds=.4,demonstration_frames_used=1,generated_windows=len(windows),wall_seconds=time.perf_counter()-started)
            (directory/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            manifest['sources'].append(result)
            (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            print(json.dumps(result,ensure_ascii=False),flush=True)
    adapter.bundle.assert_unchanged(frozen)
    manifest['frozen_unchanged']=True
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
