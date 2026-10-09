"""服务器1八卡固定参考和提交时刻的CPU/GPU PhysX单环境迁移对照。

读取既有CPU闭环重放记录，不重采样、不改变参考或到达时刻；每卡启动一个GPU
场景、一个真实机器人，重放同样的reset/reserve/prepare/commit/advance请求。
只重映射新worker的episode和ticket身份，输入参考数组保持原字节。GMT的同输入
数值一致性由独立批量策略验收负责；这里测量相同控制框架下物理状态引起的动作、
历史、跟踪和奖励差异，不能要求不同PhysX后端的浮点轨迹逐比特一致。

容限在测试前写死：参考字段2e-6；根位置RMS2cm/最大5cm；关节RMS0.03rad/最大
0.1rad；关节速度RMS0.3rad/s/最大1rad/s；四元数几何角RMS0.03rad/最大0.1rad；
逐步积分奖励RMS0.005/最大0.02。实际执行步数/终止必须一致。失败原样输出全部
误差，不自动扩大容限。默认检查150控制步（含50步初始预热），不称作长期质量
或完整训练验收。所有输出只进入明确的新目录，原CPU证据按SHA绑定且不修改。
"""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
import yaml
from tools.train_closedloop_stage10 import configuration,_assert_frozen
from tools.train_closedloop_stage10_8gpu import _available_gpus
from tools.eval.run_closedloop_baseline import Workers
from tools.replay_stage10_runtime_v4 import reward_rows
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import root_call,local_call
from gem.closedloop.dppo.vector_collector import VectorWorldClient,VectorLaneBackend
from gem.closedloop.dppo.run_management import DiskGuard,GuardedStepJournal
from gem.closedloop.dppo.full_dataset import FullMusicCatalog,SOURCES
from gem.runtime.closedloop_protocol import RemoteError


class SingleLaneTransport:
    """N=1固定重放的同步transport，仍执行世界及环境两层持久化后ACK。"""
    def __init__(self,world):self.world=world;self.sequence=0;self.last_call_timing={}
    def call(self,method,**payload):
        self.sequence+=1;key=str(self.sequence)
        result=self.world.call('exchange',requests=[dict(env_id=0,method=method,payload=payload,request_id=key)])
        if result.get('fatal'):raise RuntimeError(str(result['fatal']))
        matches=[r for r in result['replies'] if r['request_id']==key]
        if len(matches)!=1:raise RuntimeError('Single environment replay unexpectedly deferred')
        reply=matches[0]
        if not reply['ok']:raise RemoteError(reply['error'])
        return reply['result']
    def close(self):pass


def remap(value,mapping):
    if isinstance(value,str):
        for old,new in sorted(mapping.items(),key=lambda p:-len(p[0])):value=value.replace(old,new)
        return value
    if isinstance(value,dict):return {k:remap(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [remap(v,mapping) for v in value]
    if isinstance(value,tuple):return tuple(remap(v,mapping) for v in value)
    return copy.deepcopy(value)


def error_stats(difference,rms_limit,max_limit):
    a=np.asarray(difference,dtype=np.float64)
    rms=float(np.sqrt(np.mean(a*a))) if a.size else 0.
    maximum=float(np.abs(a).max()) if a.size else 0.
    return dict(rms=rms,max_abs=maximum,rms_limit=rms_limit,max_limit=max_limit,
        passed=bool(np.isfinite(a).all() and rms<=rms_limit and maximum<=max_limit))


def compare(cpu,gpu,cpu_rewards,gpu_rewards):
    rows=lambda records:[step for r in records if r['method']=='advance' for step in r['result']['trace']]
    a,b=rows(cpu),rows(gpu)
    if len(a)!=len(b):return dict(passed=False,reason='different_executed_controls',counts=[len(a),len(b)])
    q=np.array([r['actual_qpos'] for r in a]);p=np.array([r['actual_qpos'] for r in b])
    qa=q[:,3:7]/np.linalg.norm(q[:,3:7],axis=-1,keepdims=True)
    qb=p[:,3:7]/np.linalg.norm(p[:,3:7],axis=-1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs((qa*qb).sum(-1)),0,1))
    stats=dict(root_position=error_stats(q[:,:3]-p[:,:3],.02,.05),
        root_quaternion_angle=error_stats(angle,.03,.1),
        joint_position=error_stats(q[:,7:]-p[:,7:],.03,.1),
        joint_velocity=error_stats(np.array([r['actual_joint_vel'] for r in a])-np.array([r['actual_joint_vel'] for r in b]),.3,1.))
    reference=[]
    for left,right in zip(a,b):
        for name,value in left['reference'].items():
            if isinstance(value,np.ndarray) and value.dtype.kind in 'fiu':
                reference.extend((np.asarray(value)-right['reference'][name]).reshape(-1))
    stats['reference']=error_stats(reference,2e-6,2e-6)
    stats['reward']=error_stats(np.array([r['reward'] for r in cpu_rewards])-np.array([r['reward'] for r in gpu_rewards]),.005,.02)
    history=[]
    for left,right in zip(cpu,gpu):
        if left['method'] not in {'advance','reset_episode'}:continue
        x=left['result'].get('snapshot',left['result']);y=right['result'].get('snapshot',right['result'])
        if np.shape(x['history_values'])!=np.shape(y['history_values']):
            return dict(passed=False,reason='history_shape',shapes=[np.shape(x['history_values']),np.shape(y['history_values'])])
        if not np.array_equal(x['history_ticks'],y['history_ticks']) or not np.array_equal(x['history_valid'],y['history_valid']):
            return dict(passed=False,reason='history_time_or_mask')
        history.extend((np.asarray(x['history_values'])-y['history_values']).reshape(-1))
    stats['history']=error_stats(history,.3,1.)
    actions=np.array([r['applied_action'] for r in a])-np.array([r['applied_action'] for r in b])
    termination_equal=all(x['terminated']==y['terminated'] and x['tick']==y['tick'] for x,y in zip(a,b))
    return dict(passed=termination_equal and all(v['passed'] for v in stats.values()),criteria=stats,
        action_difference=dict(rms=float(np.sqrt(np.mean(actions**2))),max_abs=float(np.abs(actions).max()),
            interpretation='different_physical_inputs_same_frozen_controller_not_same_input_policy_parity'),
        termination_and_ticks_equal=termination_equal,control_steps=len(a),physics_steps=len(a)*4,
        cpu_reward=sum(r['reward'] for r in cpu_rewards),gpu_reward=sum(r['reward'] for r in gpu_rewards))


def replay_requests(backend,records,control_limit):
    before=[];after=[];mapping={};controls=0;start=time.perf_counter()
    for old in records:
        method=old['method'];payload=remap(old['payload'],mapping)
        if method=='reset_episode' and before:break
        if method=='advance' and controls+payload['control_steps']>control_limit:break
        try:value=backend.call(method,**payload)
        except RemoteError as error:
            if not old.get('remote_error') or error.code!=old['remote_error']['code']:raise
            continue
        if old.get('remote_error'):raise AssertionError('Previously rejected reference was accepted')
        if method=='reset_episode':mapping[old['result']['episode_id']]=value['episode_id']
        if method=='prepare_plan':mapping[old['result']['prepared_plan_id']]=value['prepared_plan_id']
        before.append(old);after.append(dict(method=method,payload=payload,result=value))
        if method=='advance':
            controls+=value['executed_control_steps']
            if value['done']:break
    return before,after,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','gmt-repo','cpu-replay','data-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--controls',type=int,default=150)
    args=p.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('Eight GPUs required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=10))
    group=DistributedCollectives(rank,8,device='cpu')
    root_call(group,_available_gpus);root_call(group,lambda:args.output.mkdir(parents=True,exist_ok=False))
    output=args.output/f'rank{rank:02d}';output.mkdir()
    cfg=configuration(args.config)
    cfg['paths'].update(genmo_repo=str(Path(__file__).resolve().parents[1]),gmt_repo=str(args.gmt_repo),
        compat_profile=str(args.gmt_repo/'configs/sim2sim/model_135000_stage2.json'))
    cfg['runtime'].update(backend='gpu_vectorized.v1',physics_device='cuda:0',num_envs=1,rank=rank,
        asset_conversion_dir=str(output/'usd'),headless=True,video_path=None)
    from gem.closedloop.dppo.vector_devices import bind_vector_device
    device_environment=bind_vector_device(cfg,rank)
    path=output/'resolved_config.yaml';path.write_text(yaml.safe_dump(cfg,allow_unicode=True))
    catalog=FullMusicCatalog(cfg['paths']['data_root']);catalog.apply_audit(json.loads(args.data_audit.read_text()))
    sample=catalog.samples['val'][SOURCES[rank%4]][0];music=catalog.load_music(sample)
    source=args.cpu_replay/f'rank{rank:02d}/baseline/requests_and_full_replies.pt'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    records=torch.load(source,map_location='cpu',weights_only=False)
    worker=Workers(cfg,output);journals=[];world=None
    report=dict(status='started',cpu_source=dict(path=str(source),sha256=digest))
    try:
        socket=Path(worker.temp.name)/'gmt.sock'
        client=local_call(group,lambda:worker.start('gmt',[cfg['paths']['isaac_python'],'-B',
            str(args.gmt_repo/'scripts/rsl_rl/serve_frozen_gmt_vector.py'),'--config',str(path),'--socket',str(socket),'--headless'],
            args.gmt_repo,socket,strip_distributed=True,environment=device_environment))
        guard=DiskGuard(output,min_free_bytes=10*2**30,max_run_bytes=4*2**30)
        for name in ('world','lane'):journals.append(GuardedStepJournal(output/(name+'.sqlite'),guard,format='genmo.execution_journal.ndarray.v2'))
        world=VectorWorldClient(client,journals[0],socket_path=socket);backend=VectorLaneBackend(SingleLaneTransport(world),journals[1])
        before,after,elapsed=local_call(group,lambda:replay_requests(backend,records,args.controls))
        result=local_call(group,lambda:compare(before,after,reward_rows(before,cfg,sample,music),reward_rows(after,cfg,sample,music)))
        frozen=local_call(group,lambda:world.call('verify_frozen'));_assert_frozen(frozen)
        torch.save(after,output/'gpu_requests_and_full_replies.pt')
        report.update(status='passed' if result['passed'] else 'failed',comparison=result,seconds=elapsed,frozen=frozen,
            scope='fixed_reference_and_arrival_short_GPU_PhysX_replay_not_training_quality')
        reports=group.all_gather_object(report)
        if rank==0:(args.output/'report.json').write_text(json.dumps(dict(status='passed' if all(r['status']=='passed' for r in reports) else 'failed',ranks=reports),ensure_ascii=False,indent=2))
        local_call(group,lambda:None if result['passed'] else (_ for _ in ()).throw(AssertionError('CPU/GPU comparison failed; unchanged tolerances')))
    finally:
        if world is not None:
            try:world.call('close')
            except Exception:pass
        worker.close()
        for journal in journals:journal.close()
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
