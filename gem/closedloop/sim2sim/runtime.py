"""GENMO＋GMT实时sim2sim的单进程调度与冻结Actor加载。

物理线程独占MjData和reference提交；唯一推理线程只消费复制的音乐/历史/前缀。
行为固定完整20步，60帧输入直接进入Transformer，模型从明确checkpoint加载。
Stage1和Stage2统一采用20步eta=0 DDIM，仅初始噪声随机；不采样DPPO探索噪声。
缓存条件/合批CFG只改执行组织，须与Stage1Actor.sample同条件/初始噪声核验。
Stage2先校验checkpoint声明的训练随机核，再显式构造独立确定性推理身份。
保留原训练身份、CFG、Horizon、数值算子和冻结前段；不构造Critic/optimizer。

600Hz整数tick组织30Hz源动作、50HzGMT和200Hz物理。生成期间物理继续执行；
默认连续仿真：物理误差、迟到和音频漂移记录告警后继续；迟到/旧episode结果
不提交，参考不足时显式追加标记的保持参考，不改真实qpos。不可恢复的数值
异常记录后重新开始episode。严格验收可关闭continue_on_failure，所有降级
仍在报告中判为未通过。启动先在本机同一线程预热/计时，不沿用服务器预算。
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import copy
import hashlib
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time
import uuid
import numpy as np
import torch
from omegaconf import OmegaConf
from gem.closedloop.frozen_actor import FrozenStage1Actor, _fingerprint, stable_noise_seed
from gem.closedloop.online_conditions import OnlineConditionBuilder, FrozenMusicFeatures
from gem.closedloop.dppo.run_control import build_policy
from .music import prepare_music, AudioPlayer, file_sha
from .diagnostics import EpisodeDiagnostics, write_json


def settings_from_kernel(kernel):
    """只从checkpoint声明的真实随机核重建，不读当前正在修改的训练YAML。"""
    execution=kernel['numerical_execution']
    settings=dict(generation_frames=kernel['generation_horizon_frames'],denoising_steps=kernel['steps'],
        eta=kernel['eta'],guidance_scale=kernel['guidance_scale'],std_schedule=kernel['effective_std_floors'])
    if kernel.get('frozen_prefix'):settings['frozen_prefix_steps']=kernel['frozen_prefix']['steps']
    precision=dict(mode=execution['precision'],linear_execution=execution['linear'],
        attention=execution['attention'],compile=execution.get('compiler') is not None,
        precise_output=execution['final_projection']=='float64_local',
        weight_gradient=execution['weight_gradient'])
    if precision['compile']:
        raise ValueError('首版sim2sim不接受未经本机验证的compile checkpoint；请显式提供未编译身份')
    return settings,precision


class InferenceActor:
    def __init__(self,config,entry):
        paths=config['paths'];self.path=Path(entry['checkpoint']).resolve()
        self.name=entry['name'];self.sha=file_sha(self.path)
        self.final_report=None
        payload=torch.load(self.path,map_location='cpu',weights_only=False,mmap=True)
        is_stage1='checkpoint_version' in payload and 'actor_interface_config' in payload
        self.is_stage1=is_stage1
        base=dict(paths=dict(checkpoint=str(self.path) if is_stage1 else paths['stage1_checkpoint'],
                            stats=paths['stats'],kinematics=paths['kinematics']),
            runtime=dict(torch_threads=config['runtime']['torch_threads'],genmo_device=config['runtime']['device']),
            model=dict(history_steps=50,ddim_steps=20,guidance_scale=config['sampling']['guidance_scale']))
        self.owner=FrozenStage1Actor(base);self.actor=self.owner.actor;self.device=self.owner.device
        self.source_identity=None;self.source_kernel=None
        sampling,precision=copy.deepcopy(config['sampling']),copy.deepcopy(config['precision'])
        self.source_checkpoint=payload.get('source_checkpoint',dict(path=str(self.path),sha256=self.sha,
            schema=payload.get('schema',payload.get('checkpoint_version')),
            iteration=payload.get('state',{}).get('iteration'),stage1_step=payload.get('global_step')))
        frozen_actor=None
        if not is_stage1:
            identity=payload.get('identity')
            if not isinstance(identity,dict) or 'kernel' not in identity or 'actor' not in payload:
                raise ValueError('Stage2权重必须携带actor、identity.kernel和资产身份')
            if payload.get('schema') not in ('genmo.closedloop.stage2.asymmetric_tracking.checkpoint.v1',
                                           'genmo.closedloop.sim2sim.actor.v1'):
                raise ValueError('不支持的Stage2 checkpoint schema')
            for key,local in (('checkpoint','stage1_checkpoint'),('stats','stats'),('kinematics','kinematics'),('gmt_policy','gmt_policy')):
                expected=identity['assets'].get(key)
                if expected is None or file_sha(paths[local])!=expected:
                    raise ValueError('Stage2绑定资产不一致: '+key)
            if identity['kernel'].get('frozen_prefix'):frozen_actor=copy.deepcopy(self.actor)
            self.actor.load_state_dict(payload['actor'],strict=True)
            self.source_identity=identity;self.source_kernel=identity['kernel']
            sampling,precision=settings_from_kernel(self.source_kernel)
        del payload
        if any(not bool(torch.isfinite(p).all()) for p in self.actor.state_dict().values()):
            raise ValueError('Actor checkpoint包含非有限权重或buffer')
        if sampling['generation_frames']!=config['generation_frames']:
            raise ValueError('Actor行为Horizon与MuJoCo参考Horizon不同，不能静默覆盖')
        self.actor.eval().requires_grad_(False)
        if self.source_kernel:
            source_policy=build_policy(self.actor,sampling,precision,frozen_actor=frozen_actor)
            old=copy.deepcopy(self.source_kernel);new=copy.deepcopy(source_policy.kernel_config)
            # 跨机器Torch版本可能不同：记录新行为身份并重新标定，不声称旧核逐位等价。
            old['numerical_execution'].pop('torch_version',None);new['numerical_execution'].pop('torch_version',None)
            if old!=new:raise ValueError('本机重建随机核与checkpoint声明不同')
            del source_policy
        # 概率参数构造器要求正floor；推理仅取mean，绝不使用其std采样。
        # eta必须在构造前归零，确保DDIM均值中的方向系数也属于确定性路径。
        sampling.update(denoising_steps=20,eta=0.,std_schedule=[.001]*20)
        self.policy=build_policy(self.actor,sampling,precision,frozen_actor=frozen_actor)
        self.initial_fingerprint=_fingerprint(self.actor)
        self.builder=OnlineConditionBuilder(self.actor.endecoder.codec,history_steps=50,
                                            motion_frames=sampling['generation_frames'])
        kernel=dict(version='genmo.sim2sim.deterministic_ddim.v1',steps=20,eta=0.,
            guidance_scale=sampling['guidance_scale'],generation_horizon_frames=sampling['generation_frames'],
            noise='initial_only',std_floor=0.,effective_std_floors=[0.]*20,final_step_noise=False,
            timestep_map=list(self.policy.timestep_map),
            numerical_execution=copy.deepcopy(self.policy.kernel_config['numerical_execution']))
        if self.policy.kernel_config.get('frozen_prefix'):
            kernel['frozen_prefix']=copy.deepcopy(self.policy.kernel_config['frozen_prefix'])
        self.identity=dict(name=self.name,checkpoint=str(self.path),checkpoint_sha256=self.sha,
            actor_fingerprint=self.initial_fingerprint,source_training_identity=self.source_identity,
            source_checkpoint=self.source_checkpoint,source_training_kernel=copy.deepcopy(self.source_kernel),
            kernel=kernel,device=str(self.device),torch_version=torch.__version__,
            physics_backend='mujoco',purpose='inference_only_no_training_resume')
        self.identity['sampler']='ddim20_eta0_initial_noise_only'
        self.identity['behavior_id']=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()


    @torch.no_grad()
    def _sample(self,batch,generator,*,initial_noise=None):
        """Stage1和Stage2共用eta=0 DDIM，仅抽取初始噪声，后续每步只取均值。

        两阶段均执行20个真实去噪步，合批CFG和条件缓存复用既有执行算子；每一步
        都恢复已知前缀，物理解码后再次精确回写。这里不做动作平滑/限速。
        """
        started=time.perf_counter()
        prepared=self.policy.prepare_conditions(batch);adapted=prepared['adapted']
        noise=(torch.randn(adapted['known_x'].shape,device=self.device,dtype=torch.float32,generator=generator)
               if initial_noise is None else initial_noise.to(adapted['known_x']))
        state=self.actor._constrain(noise,adapted)
        for step in range(self.policy.steps):
            parameters=self.policy.transition_parameters(batch,state,step,prepared=prepared)
            state=parameters['mean']
        physical=self.actor.endecoder.denormalize(state)
        physical=torch.where(adapted['known_mask'],adapted['known_physical'],physical)
        physical=torch.where(adapted['future_valid'][...,None],physical,0.)
        qpos=self.actor.endecoder.codec.decode_to_canonical_qpos(physical)
        qpos=torch.where(adapted['future_valid'][...,None],qpos,0.)
        contact=torch.where(adapted['future_valid'][...,None],parameters['contact_logits'].sigmoid(),0.)
        if torch.device(self.device).type=='cuda':torch.cuda.current_stream(self.device).synchronize()
        self.sampling_timing=dict(total_wall_seconds=time.perf_counter()-started,
            interval_scope='deterministic DDIM20; initial noise only; cached conditions and batched CFG',steps=self.policy.steps)
        return dict(qpos=qpos,qpos30=physical,contact=contact,normalized=state)

    @torch.no_grad()
    def generate(self,snapshot,reservation,music,seed,decision_id,music_start_tick):
        if self.final_report is not None:raise RuntimeError('Actor已经关闭')
        started=time.perf_counter()
        conditions,meta=self.builder.build(snapshot,reservation,music,music_start_tick=music_start_tick)
        generator=torch.Generator(device=self.device).manual_seed(stable_noise_seed(seed,decision_id))
        batch={k:v.to(self.device) for k,v in conditions.items()}
        sample=self._sample(batch,generator)
        anchor=torch.as_tensor(meta['world_anchor'],dtype=torch.float32,device=self.device)
        world=self.actor.endecoder.codec.apply_world_anchor(sample['qpos'],anchor)
        plan=dict(meta,plan_id=f"{reservation['request_id']}:plan",behavior_id=self.identity['behavior_id'])
        for key,value in (('qpos_world',world),('qpos30',sample['qpos30']),('contact',sample['contact'])):
            if not torch.isfinite(value).all():raise FloatingPointError('GENMO产生非有限动作')
            plan[key]=value[0].cpu().numpy().copy()
        plan.update(latency_seconds=time.perf_counter()-started,
            history_sha256=hashlib.sha256(conditions['proprio_history'].numpy().tobytes()).hexdigest(),
            history_valid_count=int(conditions['proprio_history_valid'].sum()),
            timing=dict(self.sampling_timing),noise_seed=stable_noise_seed(seed,decision_id))
        return plan

    def close(self):
        if self.final_report is not None:return dict(self.final_report)
        frozen=_fingerprint(self.actor)==self.initial_fingerprint and file_sha(self.path)==self.sha
        frozen=frozen and all(not p.requires_grad for p in self.actor.parameters())
        frozen=frozen and {k:file_sha(p) for k,p in self.owner.paths.items()}==self.owner.asset_hashes
        if not frozen:raise RuntimeError('运行过程中GENMO权重发生变化')
        self.final_report=dict(frozen=True,checkpoint_sha256=self.sha,actor_fingerprint=self.initial_fingerprint)
        self.policy=None;self.builder=None;self.actor=None;self.owner=None
        return dict(self.final_report)


class LogicalClock:
    """显式暂停冻结逻辑墙钟；暂停不是模型运行慢时的自动补偿。"""
    def __init__(self,now=time.monotonic):
        self.now=now;self.origin=now();self.paused_at=None;self.paused_total=0.;self.delayed_total=0.
    def elapsed(self):
        return (self.paused_at if self.paused_at is not None else self.now())-self.origin-self.paused_total
    def pause(self):
        if self.paused_at is None:self.paused_at=self.now()
    def resume(self):
        if self.paused_at is not None:self.paused_total+=self.now()-self.paused_at;self.paused_at=None
    def rebase(self, delay):
        """连续仿真允许节拍降速，但保留全部延迟供真实RTF统计。"""
        self.origin+=delay;self.delayed_total+=delay
    def measured_elapsed(self):return self.elapsed()+self.delayed_total


class Sim2SimRuntime:
    def __init__(self,config,tracks,output,*,seed=42,headless=False,record=False,max_seconds=None,scripted_events=()):
        self.config=config;self.tracks=tracks;self.output=Path(output);self.seed=seed
        self.headless=headless;self.record=record;self.max_seconds=max_seconds
        self.output.mkdir(parents=True,exist_ok=False)
        scripts=Path(config['paths']['gmt_repo'])/'scripts'
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        from bumi4340_mujoco_backend import BumiMujocoBackend
        self.backend=BumiMujocoBackend(config)
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='genmo_inference')
        self.commands=queue.Queue();self.view_lock=threading.Lock();self.view={'status':'loading'}
        self.stop=threading.Event();self.finished=threading.Event();self.failure=None
        self.viewer_heartbeat=None
        self.models=config['models'];self.model_index=0;self.track_index=0;self.actor=None
        self._announced_music=None
        self.episodes=[];self.pending=None;self.episode_number=0;self.audio=None
        self.scripted_events=deque(sorted(scripted_events,key=lambda e:e['at_seconds']))
        self.total_controls=0;self.frozen_models=[];self.calibrations={}
        self.continuous=bool(config['runtime'].get('continue_on_failure',True))
        genmo=Path(config['paths']['genmo_repo']);gmt=Path(config['paths']['gmt_repo'])
        sources=[*sorted(Path(__file__).parent.glob('*.py')),
            genmo/'scripts/demo/run_genmo_gmt_sim2sim.py',genmo/'gem/closedloop/actor.py',
            genmo/'gem/closedloop/online_conditions.py',genmo/'gem/closedloop/dppo/dense_policy.py',
            genmo/'gem/closedloop/dppo/policy.py',genmo/'gem/closedloop/dppo/run_control.py',
            genmo/'gem/robots/bumi/feature_codec.py',gmt/'scripts/bumi4340_mujoco_backend.py',
            gmt/'scripts/motion/closedloop_reference.py',gmt/'scripts/motion/bumi4340_reference_kinematics.py']
        self.source_hashes={str(p):file_sha(p) for p in sources}

    def command(self,action):self.commands.put(action)
    def snapshot(self):
        with self.view_lock:return copy.deepcopy(self.view)
    def _publish(self,**value):
        with self.view_lock:self.view.update(value)
    def start(self):
        self.thread=threading.Thread(target=self._run,name='mujoco_physics',daemon=False);self.thread.start()
    def close(self):
        self.command('stop')
        if hasattr(self,'thread'):self.thread.join()
    def _drain_generation(self):
        if self.pending is not None:
            try:self.pending.result()
            finally:self.pending=None

    def _load_actor(self):
        self._publish(status='loading_model',model=self.models[self.model_index]['name'])
        self._drain_generation()
        if self.actor is not None:
            self.frozen_models.append(self.worker.submit(self.actor.close).result())
            self.actor=None
            if torch.cuda.is_available():torch.cuda.empty_cache()
        self.actor=self.worker.submit(InferenceActor,self.config,self.models[self.model_index]).result()
        write_json(self.output/f'model_{self.model_index+1}_identity.json',self.actor.identity)

    def _warn(self, diagnostics, code, **details):
        if diagnostics.warn(code,self.backend.tick,**details):
            print(f'[CONTINUE] tick={self.backend.tick} {code}',flush=True)
        self._publish(last_warning=code)

    def _ensure_reference(self, through_tick, diagnostics=None):
        if not self.continuous:return
        if self.backend.timeline.reference_valid_end_tick>=through_tick:return
        event=self.backend.timeline.extend_hold(through_tick+600)
        if event is not None and diagnostics is not None:
            diagnostics.events.append(dict(event,tick=self.backend.tick))
            self._warn(diagnostics,'reference_hold_extended')

    def _calibrate(self,music):
        self._publish(status='calibrating_local_latency')
        samples=[];warmup=self.config['runtime']['calibration_warmup']
        calibration_controls=0
        for i in range(warmup+self.config['runtime']['calibration_samples']):
            self.backend.reset_episode('calibration-'+uuid.uuid4().hex)
            # 标定也包含并行的GMT/物理和窗口负载，结束后显式重置为正式episode。
            for _ in range(self.config['runtime']['warmup_control_steps']):
                self.backend.step_control(check_termination=False);calibration_controls+=1
            self._publish(qpos=self.backend.data.qpos.copy(),reference_qpos=self.backend.data.qpos.copy())
            tick=self.backend.tick
            reservation=self.backend.reserve_prefix(dict(episode_id=self.backend.episode_id,request_id='calibration:'+str(i),
                decision_tick=tick,deadline_tick=tick+300,min_prefix=12))
            # 准备耗时在请求时冻结的参考副本上测量；并发物理的保护水位继续推进，
            # 不把已过时的标定轨迹提交到真实时间轴，也不会先被protection_advanced打断。
            preparation_timeline=copy.deepcopy(self.backend.timeline)
            start=time.perf_counter()
            future=self.worker.submit(self.actor.generate,self.backend.snapshot(),reservation,music,
                self.seed,i,self.config['runtime']['warmup_control_steps']*12)
            count=0
            # 首次Triton编译只在正式episode之前做离线预热；实测样本同时推进物理。
            if i<warmup:future.result()
            while not future.done():
                if time.perf_counter()-start>=count*.02:
                    self._ensure_reference(self.backend.tick+180)
                    self.backend.step_control(check_termination=False);count+=1;calibration_controls+=1
                else:time.sleep(.001)
            result=future.result()
            ticket=preparation_timeline.prepare_plan(result)
            preparation_timeline.discard_plan(ticket['ticket_id'],'calibration_only')
            duration=time.perf_counter()-start
            if i>=warmup:samples.append(duration)
        guard=float(self.config['runtime']['latency_guard_seconds'])
        budget=math.ceil((max(samples)*1.2+guard)/.02)*.02
        # 0.5秒重规划减去一个完整20ms控制边界；不额外虚构0.46秒硬上限。
        result=dict(machine='local',samples_seconds=samples,warmup=warmup,
            maximum_seconds=max(samples),median_seconds=float(np.median(samples)),guard_seconds=guard,
            deadline_budget_seconds=budget,deadline_ticks=int(round(budget*600)),
            behavior_id=self.actor.identity['behavior_id'],scope='condition_build+20_step_sampling+world_decode+reference_prepare',
            reference_future_control_steps=10)
        result['calibration_control_steps']=calibration_controls
        result['concurrent_load']='GMT CPU ONNX + four MuJoCo substeps; active viewer when enabled'
        result['accepted']=budget<=(300-12)/600
        result['measured_budget_seconds']=budget
        result['preparation_reference']='frozen_request_timeline; never committed to live calibration'
        if self.continuous and not result['accepted']:
            result['deadline_budget_seconds']=.48;result['deadline_ticks']=288
            result['continuation']='best_effort; late results discarded; explicit hold if needed'
        write_json(self.output/f'calibration_model{self.model_index+1}_{len(self.calibrations):03d}.json',result)
        if not result['accepted'] and not self.continuous:
            raise RuntimeError(f'本机生成＋准备预算{budget:.3f}s超过0.5s重规划可用时间，不能实时执行')
        self.calibrations[self.actor.identity['behavior_id']]=result
        return result

    def _episode(self):
        track=self.tracks[self.track_index]
        if self._announced_music!=track.audio:
            print(f'[MUSIC] 当前音乐 [{self.track_index+1}/{len(self.tracks)}]: {track.name}',flush=True)
            self._announced_music=track.audio
        self._publish(status='preparing_music',music=track.name)
        music=prepare_music(track,Path(self.config['paths']['output_root'])/'music_cache',
                            decode_audio=self.config['runtime']['audio'])
        frozen_music=FrozenMusicFeatures(music.features)
        calibration=self.calibrations.get(self.actor.identity['behavior_id']) or self._calibrate(frozen_music)
        music_start=self.config['runtime']['warmup_control_steps']*12
        self.episode_number+=1
        episode=f'episode{self.episode_number:04d}-{uuid.uuid4().hex[:8]}'
        self.backend.reset_episode(episode)
        diagnostics=EpisodeDiagnostics(music,music_start)
        self.audio=AudioPlayer(music,enabled=self.config['runtime']['audio'])
        clock=LogicalClock();paused=False;decision_id=0;pending_meta=None;reason='user_stop';next_action=None
        initial=self.backend.data.qpos.copy()
        diagnostics.events.append(dict(event='reset',episode_id=episode,tick=0,initial_qpos=initial))
        self._publish(status='running',episode_id=episode,reason=None,model=self.actor.name,
            music=track.name,music_duration=music.duration,qpos=initial,reference_qpos=initial,
            music_seconds=0.,paused=False,last_warning=None)
        while not self.stop.is_set():
            while self.scripted_events and self.total_controls*.02>=self.scripted_events[0]['at_seconds']:
                event=self.scripted_events.popleft();self.command(event['action'])
            while not self.commands.empty():
                command=self.commands.get_nowait()
                diagnostics.events.append(dict(event='command',command=command,tick=self.backend.tick,logical_seconds=clock.elapsed()))
                if command=='toggle_pause':
                    paused=not paused
                    clock.pause() if paused else clock.resume()
                    self.audio.update((self.backend.tick-music_start)/600,not paused)
                    self._publish(paused=paused,status='paused' if paused else 'running')
                elif command=='stop':reason='user_stop';self.stop.set();break
                elif command.startswith('abort:'):
                    reason='realtime_'+command.split(':',1)[1];self.stop.set();break
                elif command in ('restart','next','previous') or command.startswith('model:'):
                    reason='user_'+command;next_action=command;break
            if self.stop.is_set() or next_action:break
            if paused:
                time.sleep(.005);continue
            if self.backend.tick>=music_start+math.ceil(music.duration*50)*12:
                reason='music_end';next_action='auto_next' if self.config['runtime']['auto_next'] else None;break
            if self.max_seconds is not None and self.total_controls*.02>=self.max_seconds:
                reason='administrative_time_limit';self.stop.set();break
            due=self.backend.tick/600
            remaining=due-clock.elapsed()
            if remaining>0:
                time.sleep(min(remaining,.003));continue
            lateness=clock.elapsed()-due
            if lateness>self.config['runtime']['lateness_limit_seconds']:
                if not self.continuous:reason='realtime_control_late';break
                self._warn(diagnostics,'realtime_control_late',lateness_seconds=lateness)
                # 重新对齐节拍，保留每个真实物理步；降速/跳音频均显式记录，不能算实时通过。
                clock.rebase(lateness)
            if (not self.headless and self.viewer_heartbeat is not None and
                    time.monotonic()-self.viewer_heartbeat>self.config['runtime']['lateness_limit_seconds']):
                if not self.continuous:reason='realtime_viewer_unresponsive';break
                self._warn(diagnostics,'realtime_viewer_unresponsive')
            try:
                if self.pending is not None:
                    if self.pending.done():
                        completed=self.pending;self.pending=None
                        plan=completed.result()
                        captured=dict(qpos_world=plan['qpos_world'].copy(),decision_tick=plan['decision_tick'],
                            prefix_frames=plan['prefix_frames'],committed=False)
                        diagnostics.plans.append(captured)
                        if plan['episode_id']!=episode or plan['behavior_id']!=self.actor.identity['behavior_id']:
                            raise ValueError('stale_generation_identity')
                        if clock.elapsed()*600>plan['deadline_tick']:
                            diagnostics.events.append(dict(event='reject_completed_late_generation',
                                request_id=plan['request_id'],decision_tick=plan['decision_tick'],
                                deadline_tick=plan['deadline_tick'],arrival_logical_seconds=clock.elapsed(),
                                generation_seconds=plan['latency_seconds'],gpu_stage_timing=plan['timing']))
                            if not self.continuous:reason='generation_deadline_missed';break
                            self._warn(diagnostics,'generation_deadline_missed',request_id=plan['request_id'])
                        else:
                            prepared_start=time.perf_counter();ticket=self.backend.prepare_plan(plan)
                            if clock.elapsed()*600>plan['deadline_tick']:
                                self.backend.timeline.discard_plan(ticket['ticket_id'],'late_wall_clock')
                                if not self.continuous:reason='generation_deadline_missed';break
                                self._warn(diagnostics,'generation_prepare_late',request_id=plan['request_id'])
                            else:
                                ack=self.backend.commit_plan(ticket);captured['committed']=True
                                diagnostics.requests.append(dict(request_id=plan['request_id'],decision_tick=plan['decision_tick'],
                                    commit_tick=self.backend.tick,deadline_tick=plan['deadline_tick'],prefix_frames=plan['prefix_frames'],
                                    generation_seconds=plan['latency_seconds'],prepare_seconds=time.perf_counter()-prepared_start,
                                    history_sha256=plan['history_sha256'],history_valid_count=plan['history_valid_count'],
                                    generated_qpos_sha256=hashlib.sha256(plan['qpos_world'].tobytes()).hexdigest(),
                                    noise_seed=plan['noise_seed'],ack=ack))
                                diagnostics.requests[-1]['gpu_stage_timing']=plan['timing']
                    elif clock.elapsed()*600>pending_meta['deadline_tick']:
                        if not self.continuous:reason='generation_deadline_missed';break
                        if not pending_meta.get('late_reported'):
                            self._warn(diagnostics,'generation_deadline_missed',request_id=pending_meta['request_id'])
                            pending_meta['late_reported']=True
                tick=self.backend.tick
                if tick>=music_start and (tick-music_start)%300==0:
                    if self.pending is not None:
                        if not self.continuous:reason='generation_overlap';break
                        self._warn(diagnostics,'generation_overlap')
                    else:
                        self._ensure_reference(tick+calibration['deadline_ticks']+180,diagnostics)
                        request=dict(episode_id=episode,request_id=f'{episode}:{decision_id}',decision_tick=tick,
                            deadline_tick=tick+calibration['deadline_ticks'],min_prefix=12)
                        reservation=self.backend.reserve_prefix(request)
                        self.pending=self.worker.submit(self.actor.generate,self.backend.snapshot(),reservation,
                            frozen_music,self.seed,decision_id,music_start)
                        pending_meta=request;decision_id+=1
                self.audio.update((tick-music_start)/600,True)
                self._ensure_reference(tick+180,diagnostics)
                record=self.backend.step_control(check_termination=not self.continuous)
                self.total_controls+=1
                physical=[k for k,v in self.config['termination'].items() if record[k]>float(v)]
                record['physical_warning_count']=len(physical)
                record['reference_fallback']=any(str(p).startswith('fallback_hold:') for p in
                    self.backend.timeline.window(self.backend.tick)['origin_plan_ids'])
                for code in physical:self._warn(diagnostics,'physical_failure:'+code,value=record[code])
                record['gmt_policy_observation']=self.backend.last_feed['policy'][0]
                record['gmt_history']=self.backend.last_feed['history_obs'][0]
                record['gmt_command_window']=self.backend.last_feed['command_window'][0]
                audio_state=self.audio.snapshot()
                row=diagnostics.append(record,lateness=lateness,wall_elapsed=clock.measured_elapsed(),audio=audio_state)
                self._publish(status='running',qpos=record['qpos'],reference_qpos=record['reference_qpos'],
                    reference_source='fallback_hold' if record['reference_fallback'] else 'GENMO',
                    music_seconds=row['music_seconds'],activity=row['activity_rad_s'],yaw_error=record['yaw_error_rad'],
                    reference_remaining=record['reference_remaining_seconds'],
                    inference_seconds=diagnostics.requests[-1]['generation_seconds'] if diagnostics.requests else 0.,
                    real_time_factor=self.backend.tick/600/max(clock.measured_elapsed(),.001),tick=self.backend.tick)
                if audio_state['enabled'] and tick>=music_start+60 and abs(audio_state['drift_seconds'])>.1:
                    if not self.continuous:reason='realtime_audio_drift';break
                    self._warn(diagnostics,'realtime_audio_drift',drift_seconds=audio_state['drift_seconds'])
                    self.audio.resync((self.backend.tick-music_start)/600)
                if record['done']:
                    reason=record['reason'];break
                # 无在途请求时裁剪历史参考，既有时钟、世界状态和承诺均不重置。
                if self.pending is None:self.backend.timeline.trim(self.backend.tick)
            except Exception as exc:
                diagnostics.events.append(dict(event='exception',type=type(exc).__name__,detail=str(exc),tick=self.backend.tick))
                if not self.continuous:
                    reason='reference_or_execution_error:'+str(exc);break
                self._warn(diagnostics,'reference_or_execution_error',detail=str(exc))
                if isinstance(exc,FloatingPointError) or self.backend.done:
                    reason='recovery_reset:'+str(exc);next_action='restart';break
                # 参考拒绝/生成异常不终止物理；下一轮仍检查命令与截止，且可重新请求。
                # 控制本身持续异常时显式重置，不在同一tick无限重试。
                if diagnostics.warnings.get('reference_or_execution_error',0)>3 and self.backend.tick==getattr(self,'_last_error_tick',None):
                    reason='recovery_reset:'+str(exc);next_action='restart';break
                self._last_error_tick=self.backend.tick
        # 最后一控制周期也计入墙钟；不因提前计算下一状态制造大于1的RTF。
        while not paused and clock.elapsed()<self.backend.tick/600:time.sleep(.001)
        active_wall=clock.measured_elapsed();self.audio.close();audio_final=self.audio.snapshot();self.audio=None
        self._publish(status='finished',reason=reason,paused=True)
        if self.pending is not None:
            try:
                discarded=self.pending.result()
                diagnostics.events.append(dict(event='discard_inflight_at_episode_end',reason=reason,
                    request_id=discarded['request_id'],generation_seconds=discarded['latency_seconds'],
                    decision_tick=discarded['decision_tick'],deadline_tick=discarded['deadline_tick']))
            except Exception as exc:
                diagnostics.events.append(dict(event='generation_exception_at_episode_end',detail=str(exc)))
                self.failure=self.failure or dict(phase='generation',message=str(exc))
            finally:self.pending=None
        directory=self.output/episode
        report=diagnostics.finish(directory,reason=reason,actor=self.actor.identity,backend=dict(
            physics_backend='mujoco',asset_sha256=self.backend.asset_hashes),seed=self.seed,
            active_wall_seconds=active_wall,calibration=calibration)
        report['audio']=audio_final;write_json(directory/'report.json',report)
        self.episodes.append(dict(directory=str(directory),reason=reason,report=report))
        print(f'[MuJoCo] {episode}: {reason}, {report["simulated_seconds"]:.2f}s, RTF={report["real_time_factor"]}',flush=True)
        if self.record and diagnostics.records:
            # 延后到整个交互执行结束后编码，期间不占控制线程/渲染上下文。
            self.video_jobs.append((directory,music.track.audio,initial,music_start))
        self._drain_generation()
        return next_action,reason

    def _run(self):
        self.video_jobs=[]
        try:
            self._publish(status='verifying_GMT140500')
            parity=self.backend.verify_native_policy(self.config['paths']['gmt_validation_python'])
            write_json(self.output/'gmt_native_onnx_parity.json',parity)
            self._load_actor()
            while not self.stop.is_set():
                action,reason=self._episode()
                if self.stop.is_set():break
                if action is None:
                    if self.headless:break
                    while not self.stop.is_set():
                        try:action=self.commands.get(timeout=.05)
                        except queue.Empty:continue
                        if action=='stop':self.stop.set();break
                        if action in ('restart','next','previous') or action.startswith('model:'):break
                    if self.stop.is_set():break
                if action in ('next','previous','auto_next'):
                    offset=-1 if action=='previous' else 1
                    if action=='auto_next' and self.track_index+1>=len(self.tracks):break
                    self.track_index=(self.track_index+offset)%len(self.tracks)
                elif action and action.startswith('model:'):
                    index=int(action.split(':')[1])
                    if not 0<=index<len(self.models):
                        self._publish(status='finished',reason='unbound_model_slot');continue
                    if index!=self.model_index:self.model_index=index;self._load_actor()
        except BaseException as exc:
            import traceback
            self.failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            self._publish(status='error',reason=str(exc));print(self.failure['traceback'],flush=True)
        finally:
            if self.audio:self.audio.close()
            try:
                self._drain_generation()
                if self.actor is not None:self.frozen_models.append(self.worker.submit(self.actor.close).result())
                backend_final=self.backend.close()
            except BaseException as exc:
                backend_final=dict(frozen=False,error=str(exc))
                self.failure=self.failure or dict(message=str(exc))
            self.worker.shutdown(wait=True)
            write_json(self.output/'run_report.json',dict(schema='genmo.gmt.mujoco_run.v1',physics_backend='mujoco',
                config=self.config,seed=self.seed,episodes=self.episodes,failure=self.failure,
                frozen_models=self.frozen_models,backend_final=backend_final,source_files=self.source_hashes,
                sources_unchanged={p:file_sha(p) for p in self.source_hashes}==self.source_hashes))
            self.finished.set()
