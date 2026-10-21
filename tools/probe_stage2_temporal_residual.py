"""真实行为条件和同一Adam增量上的独立时间残差反事实探针。

输入原始case、批量对照候选Actor及真实消费分析结果；由实测前瞻源帧范围选择
自由帧支持长度，明确添加一个30Hz帧的速度差分支持，余弦过渡长度显式给定。
同一条件、Stage1前15步状态和四步真实innovation下，对比原60帧策略与残差门控。
先验证Stage1起点两种核的旧路径均值、最终qpos及contact完全相同，再施加同一真实
候选参数增量；这种探针不等价于用门控核重新训练，更不代表新Actor已在PhysX执行。

输出每帧变化、零门控均值、最终远期跨帧耦合、世界系qpos/速度变化及实际前向耗时。
有真实world_anchor才重建世界坐标，缺失时明确未验证；不猜测根运动特征为世界XYZ，
不删任何自由概率坐标，不改变高斯标准差、Actor输入或正式训练配置。
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from gem.closedloop.dppo.actor_step_diagnostics import output_probe
from gem.closedloop.dppo.attribution_runtime import probe_difference,probe_summary
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import immutable_cpu_snapshot,file_sha256
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.temporal_residual_policy import residual_gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--consumption',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--taper-frames',type=int,default=4)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    with gzip.open(args.consumption/'plans.jsonl.gz','rt') as f:plans=[json.loads(line) for line in f]
    supports=[max(p['center_source_frames']+p['future_only_source_frames'])-p['prefix_frames']+1
        for p in plans if p['full_observed_horizon'] and p['prefix_frames'] is not None and
        p['center_source_frames']+p['future_only_source_frames']]
    if not supports:raise ValueError('缺少完整保留时域中的真实中心及前瞻消费证据')
    focus=int(np.ceil(np.quantile(supports,.95)))+1
    config=configuration(args.config,genmo_repo=ROOT);config['runtime']['genmo_device']='cuda:0'
    sys.path.insert(0,str(Path(config['paths']['gmt_repo'])/'scripts/motion'))
    from bumi4340_reference_kinematics import resample,linear_velocity
    actor,_,_=load_actor(config)
    base=build_policy(actor,config['stage9'],config['stage10']['precision'])
    settings=dict(config['stage9'],temporal_residual=dict(focus_free_frames=focus,taper_frames=args.taper_frames,
        evidence=str(args.consumption/'summary.json')))
    gated=build_policy(actor,settings,config['stage10']['precision'],frozen_actor=base.frozen_policy.actor)
    payload=load_rollout(args.case/f'rank{args.rank:02d}')
    if payload['kernel']!=base.kernel_config:raise ValueError('原始行为核与完整60帧对照不一致')
    learning=torch.load(args.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
    candidate=torch.load(args.candidate,map_location='cpu',weights_only=False,mmap=True)
    if candidate.get('source_case_sha256')!=file_sha256(args.case/'case.json'):
        raise ValueError('候选不来自同一不可变真实case')
    wanted=torch.load(args.candidate.parent/f'probe_rank{args.rank:02d}.pt',map_location='cpu',weights_only=False)
    identities=wanted['before']['identities']
    rows=[r for r in payload['transitions'] if r.identity in identities]
    cache=RolloutTensorCache(payload['transitions'],payload['targets'],'cuda:0',max_device_bytes=1024**3)
    def measure(policy):
        torch.cuda.synchronize();start=time.perf_counter()
        value=output_probe(policy,rows,cache,max_chains=len(rows))
        torch.cuda.synchronize()
        return value,time.perf_counter()-start
    try:
        actor.load_state_dict(learning['actor']);before,_=measure(base);initial_gate,_=measure(gated)
        for name in ('mean','actual_std','final','qpos','contact'):
            if not torch.equal(before[name],initial_gate[name]):raise AssertionError('Stage1初始门控不等价: '+name)
        actor.load_state_dict(candidate['actor'])
        full,full_seconds=measure(base);after,gate_seconds=measure(gated)
        result={}
        for name,value in [('full60',full),('time_gate',after)]:
            result[name]=probe_summary(probe_difference(before,value))
        known=cache.context(rows)
        with torch.no_grad():
            prepared=gated.prepare_conditions(known)
            gate=residual_gate(prepared['adapted']['known_mask'],prepared['adapted']['future_valid'],
                focus_free_frames=focus,taper_frames=args.taper_frames)
            zero=gate.expand_as(after['final'])==0
            # 这里零gate区域的最终变化可非零：末步冻结Transformer跨时间耦合。
            free=after['mask'];far=zero&free
            result['far_final_normalized_rms']=float((after['final']-before['final']).double()[far].square().mean().sqrt()) if far.any() else None
            step_max=[]
            for step in gated.trainable_steps:
                state=cache.get('chain',rows,[step]*len(rows))
                actual=gated.transition_parameters(known,state,step,prepared=prepared)
                prior=gated._frozen_transition(prepared,state,step,False)
                step_max.append(float((actual['mean']-prior['mean'])[zero].abs().max()))
            result['zero_gate_old_path_mean_max_abs_by_step']=step_max
            if any(step_max):raise AssertionError('零gate区域并非严格冻结先验均值')
            anchors=[r.metadata.get('generation_world_anchor') for r in rows]
            if all(a is not None for a in anchors):
                anchor=torch.tensor(anchors,device=after['qpos'].device,dtype=torch.float32)
                old_world=actor.endecoder.codec.apply_world_anchor(before['qpos'],anchor)
                world=actor.endecoder.codec.apply_world_anchor(after['qpos'],anchor)
                result['world_root_position_rms_m']=float((world[...,:3]-old_world[...,:3]).double().square().mean().sqrt())
                root_velocity=[]
                for old,new in zip(old_world.cpu().numpy(),world.cpu().numpy()):
                    a=resample(old);b=resample(new)
                    root_velocity.append(float(np.sqrt(np.mean((linear_velocity(b[:,:3])-linear_velocity(a[:,:3]))**2))))
                result['world_root_velocity_delta_rms_m_s']=root_velocity
            else:result['world_qpos']='unverified_missing_world_anchor'
        with (args.output_dir/'frame_changes.csv').open('w') as f:
            f.write('frame,full_joint_rms_rad,gated_joint_rms_rad,gated_normalized_rms,gate_mean\n')
            for frame in range(60):
                rms=lambda x:float(x.double().square().mean().sqrt())
                f.write(f"{frame},{rms(full['qpos'][:,frame,7:]-before['qpos'][:,frame,7:])},{rms(after['qpos'][:,frame,7:]-before['qpos'][:,frame,7:])},{rms(after['final'][:,frame]-before['final'][:,frame])},{float(gate[:,frame].mean())}\n")
        result.update(focus_free_frames=focus,taper_frames=args.taper_frames,
            evidence=str(args.consumption/'summary.json'),actual_support_p95=float(np.quantile(supports,.95)),
            original_probe_seconds=full_seconds,gated_probe_seconds=gate_seconds,
            probe_cost_ratio=gate_seconds/full_seconds,chains=len(rows),initial_exact_equivalence=True,
            scope='same_actual_parameter_increment_counterfactual_not_independently_optimized_or_PhysX_executed',
            far_frames_not_guaranteed_unchanged=True)
        # 反事实探针复用前15步，成本比例不能冒充完整20步生成的开销。
        # 另测同一batch/seed的真实完整采样；首遍预热，不改std或innovation分布。
        timings={};traces={}
        with torch.no_grad():
            for name,policy in [('full60',base),('time_gate',gated)]:
                samples=[]
                for repeat in range(4):
                    generator=torch.Generator(device='cuda:0').manual_seed(739)
                    torch.cuda.synchronize();start=time.perf_counter()
                    trace=policy.sample_rollout(known,generator=generator)
                    torch.cuda.synchronize()
                    if repeat:samples.append(time.perf_counter()-start)
                timings[name]=dict(seconds=samples,median_seconds=float(np.median(samples)))
                traces[name]=trace
            if (not torch.equal(traces['full60']['sampling_noise'],traces['time_gate']['sampling_noise']) or
                    not torch.equal(traces['full60']['chain'][:,:16],traces['time_gate']['chain'][:,:16])):
                raise AssertionError('完整采样前15步或实际innovation协议不同')
        result['actual_full20_sampling_timing']=timings
        result['actual_full20_sampling_cost_ratio']=timings['time_gate']['median_seconds']/timings['full60']['median_seconds']
        result['full20_frozen_first15_and_actual_innovations_equal']=True
        atomic_json(args.output_dir/'summary.json',result)
        torch.save(dict(before=immutable_cpu_snapshot(before),full=immutable_cpu_snapshot(full),
            gated=immutable_cpu_snapshot(after),gate=gate.cpu()),args.output_dir/'probe.pt')
        print(json.dumps({k:result[k] for k in ('focus_free_frames','probe_cost_ratio','far_final_normalized_rms')}))
    finally:cache.close()


if __name__=='__main__':main()
