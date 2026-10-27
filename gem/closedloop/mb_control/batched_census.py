"""每卡多个真实Newton环境的原Stage1全量评价，完全关闭WM前瞻计算。

数据流：原s595000和原统计→真实机器人历史/原前缀/音乐→拼批DDIM20→各环境独立
参考缓存→冻结SM Actor→同款Newton四子步。不同长度音乐有独立有效掩码，噪声按
音乐种子和请求仿真时刻确定，不因批大小改变。每批同一时间网格，环境间不共享状态。
评价使用明确固定的仿真时钟接管延迟，计算期间不等待墙钟、不读取未来实测状态。
它验证物理跟踪能力，不宣称实际部署延迟合格；原墙钟失败证据保留在旧目录。
逐步状态/参考/指标在GPU暂存，整批结束后才搬回CPU和保存，避免逐帧同步。
每张卡一个进程管理多个环境，持久队列每条音乐恰好领取一次，工程错误独立留证。
"""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .stage1_census import connect, finish, report, utc, write_json


def claim_batch(root,count):
    with connect(root) as db:
        db.execute('BEGIN IMMEDIATE')
        rows=db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY ordinal LIMIT ?",(count,)).fetchall()
        for row in rows:
            db.execute("UPDATE jobs SET status='running',owner=?,attempts=attempts+1,started=? WHERE key=?",(os.getpid(),utc(),row['key']))
    return [(json.loads(row['task']),row['attempts']+1) for row in rows]


def music_masks(ticks,lengths):
    """逐环境音乐有效域，不把短曲的补零区当成真实音乐。"""
    return (ticks[None]>=0)&(ticks[None]//20<lengths[:,None])


def generate(actor,state,ticket,lengths,noise):
    import torch
    from .online_conditions import build_conditions
    from .contracts import ForecastContext,GeneratedPlan
    conditions,anchor=build_conditions(state,ticket,actor.stage1.endecoder.codec)
    ticks=state.tick+torch.arange(60,device=state.state.device)*20
    valid=music_masks(ticks,lengths)
    conditions['music_valid']=valid
    conditions['music_features']=torch.where(valid[...,None],conditions['music_features'],0.)
    output=actor.sample_differentiable(conditions,ForecastContext.invalid(state.state,state.tick),noise)
    world=actor.stage1.endecoder.codec.apply_world_anchor(output['qpos28'],anchor)
    return GeneratedPlan(ticket,world,output['contact'])


def initial_batch(tasks,actor,adapter,env,device):
    import torch
    from .application import read_music_features
    from .contracts import ClosedLoopState
    from .reference_buffer import ReferenceBuffer
    B=env.N;actual=len(tasks);padded=tasks+[tasks[-1]]*(B-actual)
    music_list=[read_music_features(t['music_features']) for t in padded]
    lengths=torch.tensor([len(x) for x in music_list],device=device,dtype=torch.long)
    music=torch.zeros(B,int(lengths.max()),35,device=device)
    for i,value in enumerate(music_list):music[i,:len(value)]=value.to(device)
    initial=torch.zeros(B,55,device=device);initial[:,2]=adapter.bundle.spec.spawn_height;initial[:,3]=1.
    initial[:,13:34]=initial.new_tensor(adapter.bundle.spec.default_joint_pos)
    env.reset(torch.arange(B,device=device),initial)
    observed=env.observe();s=observed['state']
    pose=torch.cat((s[:,:7],s[:,13:34][:,adapter.to_genmo]),-1)
    ref=ReferenceBuffer(torch.arange(60,device=device)*20,pose[:,None].expand(-1,60,-1).clone(),
        torch.zeros(B,60,2,device=device),torch.zeros(60,device=device,dtype=torch.long),adapter.runtime.kin,adapter.to_sm)
    current,_,_=ref.read_current_future(0)
    obs=adapter.runtime.observation(s,observed['previous_action'],observed['xi'],current)
    history=torch.zeros(B,50,48,device=device);history[:,-1]=adapter.proprio(s,observed['xi'])
    valid=torch.zeros(B,50,device=device,dtype=torch.bool);valid[:,-1]=True
    z=ClosedLoopState(0,s,observed['xi'],observed['previous_action'],obs[:,None].expand(-1,10,-1).clone(),
        history,valid,ref,music,0,int(lengths.max())*20)
    return z,lengths,actual


def contact_reader(env):
    """GPU读取真实接触；按nacon掩码过滤有效槽，不把旧槽或足高当成接触。"""
    import torch
    import warp as wp
    import mujoco
    solver=env.sim.backend_manager._solver;data=solver.mjw_data;model=solver.mj_model
    left=[];right=[]
    for g in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or ''
        left.append('l_ankle_roll' in name);right.append('r_ankle_roll' in name)
    lookup=torch.tensor([left,right],device=env.device,dtype=torch.bool)
    def read():
        contact=data.contact;geom=wp.to_torch(contact.geom);world=wp.to_torch(contact.worldid)
        dist=wp.to_torch(contact.dist);margin=wp.to_torch(contact.includemargin)
        count=wp.to_torch(data.nacon).reshape(-1)[0]
        active=(torch.arange(len(world),device=env.device)<count)&(dist<margin)&(geom>=0).all(-1)&(world>=0)&(world<env.N)
        active=active&(geom<model.ngeom).all(-1)
        sides=lookup[:,geom.clamp(0,model.ngeom-1)].any(-1).T&active[:,None]
        result=torch.zeros(env.N,2,device=env.device,dtype=torch.long)
        result.scatter_add_(0,world.clamp(0,env.N-1).long()[:,None].expand(-1,2),sides.long())
        return result>0
    return read


def run_batch(actor,adapter,scheduler,rewards,env,tasks,cfg,device,*,maximum_seconds=None,on_progress=None):
    import numpy as np
    import torch
    from .contracts import PendingGeneration
    from .census_metrics import episode_statistics
    from .evaluation import motion_metrics
    z,lengths,actual=initial_batch(tasks,actor,adapter,env,device)
    B=env.N;end_steps=(lengths*20+11)//12
    limit=int(end_steps[:actual].max())
    if maximum_seconds is not None:limit=min(limit,math.ceil(maximum_seconds*50))
    done=torch.arange(B,device=device)>=actual
    stop=torch.full((B,),limit,device=device,dtype=torch.long)
    reasons=torch.zeros(B,device=device,dtype=torch.long)
    states=[];references=[];metrics=[];contacts=[];actions=[];events=[];metric_names=None;pending_plan=None
    read_contact=contact_reader(env)
    interval=cfg['clock']['request_interval_ticks'];lag=cfg['clock']['bootstrap_latency_ticks']
    started=time.perf_counter();generation_seconds=0.;requests=0
    for step in range(limit):
        if z.tick%interval==0:
            z=replace(z,reference=z.reference.trim(z.tick))
            ticket=scheduler.reserve(z)
            g=torch.Generator(device=device).manual_seed(42+(z.tick//60)*1009)
            noise=torch.randn(1,60,30,device=device,generator=g).expand(B,-1,-1).clone()
            begin=time.perf_counter();pending_plan=generate(actor,z,ticket,lengths,noise)
            torch.cuda.synchronize();generation_seconds+=time.perf_counter()-begin;requests+=1
            z=replace(z,pending=True,pending_request=PendingGeneration(z,ticket,noise,0))
        if pending_plan is not None and z.tick==pending_plan.ticket.request_tick+lag:
            z,event=scheduler.commit(z,pending_plan,actual_ready_tick=z.tick)
            event['timing_mode']='sim_clock_fixed_latency';events.append(event);pending_plan=None
            z=replace(z,decision_id=z.decision_id+1)
        z=adapter.refresh_current_observation(z)
        _,_,goal=z.reference.read_current_future(z.tick)
        action=adapter.bundle.act(z.sm_history,goal)
        action=torch.where(done[:,None],0.,action)
        ref,_,_=z.reference.read_current_future(z.tick+12)
        state=env.step_physics(action)
        _,values=rewards(state,action,z.previous_action,ref)
        failed=adapter.runtime.termination(state,ref)
        ended=(step+1)>=end_steps
        new=(~done)&(failed|ended)
        stop=torch.where(new,step+1,stop)
        reasons=torch.where(new,torch.where(failed,1,2),reasons)
        done=done|new
        if metric_names is None:metric_names=list(values)
        metrics.append(torch.stack([values[k] for k in metric_names],-1))
        states.append(state)
        references.append(torch.cat((ref.root_p,ref.root_R.flatten(-2),ref.q,ref.root_v,ref.root_w,ref.qd),-1))
        actions.append(action);contacts.append(read_contact())
        obs=adapter.runtime.observation(state,action,z.xi,ref)
        z=replace(z,tick=z.tick+12,state=state,previous_action=action,
            sm_history=torch.cat((z.sm_history[:,1:],obs[:,None]),1),
            genmo_history=torch.cat((z.genmo_history[:,1:],adapter.proprio(state,z.xi)[:,None]),1),
            genmo_valid=torch.cat((z.genmo_valid[:,1:],torch.ones_like(z.genmo_valid[:,:1])),1))
        if on_progress is not None and (step+1)%50==0:
            on_progress(dict(simulation_seconds=z.tick/600,maximum_seconds=limit/50,
                active_environments=int((~done).sum()),wall_seconds=time.perf_counter()-started,
                generation_requests=requests,wm_forward_calls=0))
        if (step+1)%25==0 and bool(done.all()):break
    torch.cuda.synchronize();wall=time.perf_counter()-started
    state_np=torch.stack(states).cpu().numpy();ref_np=torch.stack(references).cpu().numpy()
    action_np=torch.stack(actions).cpu().numpy();contact_np=torch.stack(contacts).cpu().numpy()
    metric_np=torch.stack(metrics).cpu().numpy();stop_np=stop.cpu().tolist();reason_np=reasons.cpu().tolist()
    results=[];arrays=[]
    for i,task in enumerate(tasks):
        n=stop_np[i];s=state_np[:n,i];r=ref_np[:n,i]
        data=dict(ticks=np.arange(1,n+1)*12,state=s,root_p=s[:,None,:3],root_quat=s[:,None,3:7],
            q=s[:,None,13:34],ref_root_p=r[:,:3],ref_root_R=r[:,3:12].reshape(-1,3,3),ref_q=r[:,12:33],
            ref_root_v=r[:,33:36],ref_root_w=r[:,36:39],ref_qd=r[:,39:60],action=action_np[:n,i],
            contacts=contact_np[:n,i],alive=np.ones((n,1),dtype=bool),f=np.arange(n)[:,None],
            joint_names=np.asarray(adapter.bundle.spec.joint_names),robot_id=np.asarray(adapter.bundle.spec.robot_id),
            names=np.asarray([task['id']]),clip_start=np.asarray([0]),physics_backend=np.asarray('isaaclab3_newton_mujoco_warp'))
        reason={0:'administrative_truncation',1:'physical_failure',2:'music_end'}[reason_np[i]]
        if reason=='physical_failure':data['alive'][-1]=False
        means=metric_np[:n,i].mean(0,dtype=np.float64)
        row=dict(task=task['id'],source=task['source'],split=task['split'],seed=42,mode='original_stage1_no_wm_batched',
            status='physical_result',census_key=task['census_key'],alias_count=len(task['aliases']),complete=reason=='music_end',
            reason=reason,physical_seconds=n*.02,music_seconds=task['music_seconds'],wall_seconds=wall/actual,
            batch_wall_seconds=wall,batch_environments=B,batch_actual_tasks=actual,
            wall_seconds_definition='shared_batch_compute_wall_divided_by_actual_tasks',timing_mode='sim_clock_fixed_latency',
            fixed_latency_seconds=lag/600,request_interval_seconds=interval/600,
            physical_metrics_mean=dict(zip(metric_names,means.tolist())),events=[e for e in events if e['request_tick']<n*12],
            commits=sum(e['request_tick']<n*12 for e in events),timeouts=0,world_model_forward_calls=0,
            generation_seconds_per_batch=generation_seconds,generation_requests_per_batch=requests,
            contacts={'source':'mujoco_warp_solver_active_contacts','left_steps':int(data['contacts'][:,0].sum()),
                'right_steps':int(data['contacts'][:,1].sum()),'airborne_steps':int((~data['contacts'].any(-1)).sum()),
                'contact_switches':int(np.any(data['contacts'][1:]!=data['contacts'][:-1],axis=-1).sum())},
            motion={},result_utc=utc(),evidence='real_isaaclab3_newton_mujoco_warp')
        row['statistics']=episode_statistics(row,data,list(adapter.bundle.spec.joint_names))
        row['motion']=dict(joint_range_mean_rad=float(np.ptp(s[:,13:34],axis=0).mean()),joint_speed_rms_rad_s=float(np.sqrt(np.mean(s[:,34:55]**2))))
        data.update(termination_reason=np.asarray(reason),music_seconds=np.asarray(task['music_seconds']))
        results.append(row);arrays.append(data)
    perf=dict(batch_wall_seconds=wall,physical_seconds=sum(r['physical_seconds'] for r in results),
        environment_seconds_per_wall_second=sum(r['physical_seconds'] for r in results)/wall,
        generation_seconds=generation_seconds,generation_requests=requests,wm_forward_calls=0,environments=B)
    return results,arrays,perf


def prepare(args):
    from omegaconf import OmegaConf
    from .stage1_census import prepare as base_prepare
    base_prepare(args.config,args.output,args.gpus)
    root=Path(args.output);cfg=OmegaConf.to_container(OmegaConf.load(root/'config.yaml'),resolve=True)
    cfg['model']['forecast_enabled']=False
    cfg['clock'].update(bootstrap_latency_ticks=args.latency_ticks,latency_margin_ticks=0,request_mode='fixed_interval',request_interval_ticks=300)
    cfg['execution']['physical_batch']=args.num_envs
    cfg['evaluation']=dict(timing_mode='sim_clock_fixed_latency',fixed_latency_ticks=args.latency_ticks)
    cfg['census'].update(num_envs=args.num_envs,maximum_seconds=args.max_seconds,compile_kinematics=not args.no_compile_kinematics)
    tasks=json.loads((root/'all_tasks.json').read_text())
    if args.task_ids:tasks=[t for t in tasks if t['id'] in set(args.task_ids)]
    if not tasks:raise ValueError('实际任务清单为空')
    # 时长相近的音乐同批，减少短曲结束后占用空闲环境。
    tasks.sort(key=lambda t:t['music_seconds'],reverse=True)
    write_json(root/'all_tasks.json',tasks)
    with connect(root) as db:
        db.execute('DELETE FROM jobs')
        for i,t in enumerate(tasks):db.execute('INSERT INTO jobs(key,ordinal,task,status) VALUES(?,?,?,?)',(t['census_key'],i,json.dumps(t,ensure_ascii=False),'pending'))
    OmegaConf.save(OmegaConf.create(cfg),root/'config.yaml')
    manifest=json.loads((root/'manifest.json').read_text())
    from collections import Counter
    manifest.update(schema='genmo.stage1.batched_music_census.v1',unique_music_inputs=len(tasks),music_hours=sum(t['music_seconds'] for t in tasks)/3600,
        original_dataset_entries=sum(len(t['aliases']) for t in tasks),
        unique_by_source=dict(Counter(t['source'] for t in tasks)),unique_by_split=dict(Counter(t['split'] for t in tasks)),
        model=cfg['model'],clock=cfg['clock'],timing_mode='sim_clock_fixed_latency',fixed_latency_seconds=args.latency_ticks/600,
        num_envs_per_gpu=args.num_envs,wm_forward_calls_required=0,
        future_encoder='invalid_context_exact_zero; no WM forecast',maximum_seconds=args.max_seconds,
        wall_clock_deployment_claim=False,batch_wall_cost_attribution='shared_compute_wall_divided_by_actual_tasks')
    write_json(root/'manifest.json',manifest)


def worker(args):
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    sys.path.insert(0,cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args,launch_simulation
    p=argparse.ArgumentParser();add_launcher_args(p);simulation_args=p.parse_args([])
    import numpy as np
    import torch
    from .application import build_components,seed_all
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from isaaclab.utils import replace as config_replace
    seed_all(42);device='cuda:0';root=Path(cfg['census']['root']);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    actor,adapter,scheduler,rewards,critic,engine,_=build_components(cfg,device)
    actor.eval().requires_grad_(False)
    if cfg['model']['forecast_enabled']:raise ValueError('原Stage1批量评价禁止WM前瞻')
    if any(bool(v.detach().count_nonzero()) for v in actor.future_encoder.output.parameters()):raise ValueError('禁止Stage2未来编码参数混入Stage1')
    def forbidden(*a,**kw):raise RuntimeError('原Stage1评价意外调用WM：立即保留工程错误')
    adapter.bundle.step_model=forbidden
    before=adapter.bundle.frozen_snapshot()
    sim_cfg=make_sim(device,simulation_args)
    solver=config_replace(sim_cfg.physics.solver_cfg,disable_sensors=True)
    sim_cfg=config_replace(sim_cfg,physics=config_replace(sim_cfg.physics,class_type=None,solver_cfg=solver,deterministic_mode=cfg['execution']['physics_determinism']))
    with launch_simulation(sim_cfg,simulation_args),torch.no_grad():
        env=StreamingPhysicalEnv(sim_cfg,cfg['census']['num_envs'],adapter.bundle)
        for _ in range(3):env.step_physics(torch.zeros(env.N,21,device=device))
        if cfg['census'].get('compile_kinematics',False):
            from supermimic.mdp import unflatten_state
            # 融合原FK公式，减少每步几十个小算子；参数、几何、FP32及奖励定义不变。
            original_fk=adapter.runtime.kin.forward
            adapter.runtime.kin.forward=torch.compile(original_fk,fullgraph=True,dynamic=False)
            observed=unflatten_state(env.observe()['state'])
            differences={}
            for all_bodies in (False,True):
                raw=original_fk(*observed,all_bodies=all_bodies)
                fused=adapter.runtime.kin.forward(*observed,all_bodies=all_bodies)
                differences[str(all_bodies)]=max(float((a-b).abs().max()) for a,b in zip(raw,fused))
            write_json(out/'fk_compilation_parity.json',dict(actual_physical_state=True,max_abs=differences,fp32=True))
        # 所有编译/推理预热都显式使用invalid未来条件，不经过原来会调用WM的预热器。
        warm_tasks=json.loads((root/'all_tasks.json').read_text())[:env.N]
        warm_state,warm_lengths,_=initial_batch(warm_tasks,actor,adapter,env,device)
        warm_ticket=scheduler.reserve(warm_state)
        warm_noise=torch.randn(1,60,30,device=device).expand(env.N,-1,-1).clone()
        warm_started=time.perf_counter()
        for _ in range(2):
            generate(actor,warm_state,warm_ticket,warm_lengths,warm_noise)
            _,_,goal=warm_state.reference.read_current_future(0)
            adapter.bundle.act(warm_state.sm_history,goal)
        torch.cuda.synchronize()
        write_json(out/'inference_warmup.json',dict(seconds=time.perf_counter()-warm_started,calls=2,wm_forward_calls=0))
        write_json(out/'worker_ready.json',dict(pid=os.getpid(),utc=utc(),environments=env.N,wm_forward_calls=0,forecast_enabled=False))
        batch_id=0
        while True:
            claimed=claim_batch(root,env.N)
            if not claimed:break
            tasks=[v[0] for v in claimed]
            write_json(out/'current_batch.json',dict(batch=batch_id,tasks=[t['id'] for t in tasks],started=utc()))
            try:
                rows,data,perf=run_batch(actor,adapter,scheduler,rewards,env,tasks,cfg,device,maximum_seconds=cfg['census']['maximum_seconds'],
                    on_progress=lambda x:write_json(out/'current_progress.json',dict(batch=batch_id,utc=utc(),**x)))
                for (task,attempt),row,arrays in zip(claimed,rows,data):
                    directory=root/'episodes'/task['census_key']/f'attempt_{attempt}';directory.mkdir(parents=True,exist_ok=False)
                    path=directory/'physical_with_reference_velocity.npz'
                    np.savez(path,**arrays)
                    row.update(trajectory_artifact=str(path),attempt=attempt)
                    write_json(directory/'result.json',row);finish(root,task,row)
                adapter.bundle.assert_unchanged(before)
                write_json(out/f'batch_{batch_id:04d}.json',perf)
                print(json.dumps(dict(event='batch_complete',batch=batch_id,**perf),ensure_ascii=False),flush=True)
                batch_id+=1
            except Exception as error:
                write_json(out/f'batch_{batch_id:04d}_error.json',dict(error=repr(error),traceback=traceback.format_exc()))
                for task,attempt in claimed:
                    with connect(root) as db:status=db.execute('SELECT status FROM jobs WHERE key=?',(task['census_key'],)).fetchone()[0]
                    if status=='done':continue
                    finish(root,task,dict(status='engineering_error',task=task['id'],source=task['source'],split=task['split'],census_key=task['census_key'],reason='batch_exception',error=repr(error)),retry=attempt<2)
                raise
        write_json(out/'worker_complete.json',dict(utc=utc(),batches=batch_id,wm_forward_calls=0))


def supervise(args):
    root=Path(args.output);manifest=json.loads((root/'manifest.json').read_text());children={};attempts={g:0 for g in manifest['gpus']}
    write_json(root/'supervisor.json',dict(pid=os.getpid(),utc=utc(),gpus=manifest['gpus'],environments_per_gpu=manifest['num_envs_per_gpu']))
    def launch(gpu):
        attempts[gpu]+=1;out=root/'workers'/f'gpu{gpu}_attempt{attempts[gpu]}';out.parent.mkdir(exist_ok=True)
        log=out.with_suffix('.log').open('w')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-u','-m','gem.closedloop.mb_control.batched_census','worker','--config',str(root/'config.yaml'),'--output',str(out)]
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[gpu]=(child,log,out);write_json(out.with_name(out.name+'_launch.json'),dict(pid=child.pid,utc=utc(),command=cmd))
    for gpu in manifest['gpus']:launch(gpu)
    last=0
    while children:
        for gpu,(p,log,out) in list(children.items()):
            if p.poll() is None:continue
            log.close();del children[gpu]
            with connect(root) as db:
                db.execute("UPDATE jobs SET status='pending',owner=NULL WHERE status='running' AND owner=?",(p.pid,))
                pending=db.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
            if p.returncode:write_json(out.with_name(out.name+'_exit.json'),dict(returncode=p.returncode,utc=utc()))
            if pending and attempts[gpu]<3:launch(gpu)
        if time.time()-last>15:
            s=report(root);last=time.time();write_json(root/'progress.json',dict(utc=utc(),states=s['states'],active_workers={str(g):p.pid for g,(p,_,_) in children.items()}))
        time.sleep(1)
    s=report(root);write_json(root/'supervisor_complete.json',dict(utc=utc(),complete=s['complete'],states=s['states']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','worker','supervise'])
    p.add_argument('--output',required=True);p.add_argument('--config');p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)));p.add_argument('--latency-ticks',type=int,default=120)
    p.add_argument('--max-seconds',type=float);p.add_argument('--task-ids',nargs='+')
    p.add_argument('--no-compile-kinematics',action='store_true')
    args=p.parse_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='worker':worker(args)
    else:supervise(args)


if __name__=='__main__':main()
