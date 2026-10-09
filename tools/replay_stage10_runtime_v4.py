"""服务器1八卡固定参考/提交时刻的GMT、journal和状态复用有限对照。

每rank只启动一个真实GMT实例。首先用冻结Stage1模型产生最多20条上层转移，记录
全部有副作用的RPC请求及完整回复；随后依次重建同初态实例，重放完全相同的请求、
参考和提交tick。分别启用二进制journal、状态复用、列式轨迹及组合，比较全量状态、
四物理子步、参考、计数、终止与每一步奖励。只有计时和worker会话身份不参与数值
比较，内容不近似截断。每次推进之前预占预算，回复先FULL持久化再ACK。

这是物理数据通路对照，不训练Actor，不作为真实闭环延迟收益；首次生成的实际
到达时刻被固定用于所有重放。报告每卡各段wall计时、真实控制步、数组逐字段差异，
完整journal和请求作为审计证据保留。全程显式八卡运行且不使用本地测试资源。
"""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
import yaml
from tools.train_closedloop_stage10 import configuration, runtime_preflight, _assert_frozen
from tools.train_closedloop_stage10_8gpu import _available_gpus
from tools.eval.run_closedloop_baseline import Workers
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import local_call
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.full_dataset import FullMusicCatalog, SOURCES
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from gem.closedloop.dppo.run_management import DiskGuard, GuardedStepJournal
from gem.closedloop.dppo.budget_ledger import IncrementalBudget
from gem.closedloop.dppo.performance import PhaseProfiler, activate, deactivate
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.closedloop.dppo.target_activity import load_paired_activity
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics


def differences(left, right, path='', output=None):
    """列出真实差异，数值要求逐元素相等，不将物理差异隐藏到容差内。"""
    output = [] if output is None else output
    if len(output) >= 20:
        return output
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        equal_nan=left.dtype.kind in 'fc' and right.dtype.kind in 'fc'
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right, equal_nan=equal_nan):
            output.append(dict(path=path, kind='array', shapes=[list(left.shape), list(right.shape)],
                max_abs=float(np.max(np.abs(left.astype(float)-right.astype(float))))
                if left.shape==right.shape and left.dtype.kind in 'iufc' and right.dtype.kind in 'iufc' and left.size else None))
    elif isinstance(left, dict) and isinstance(right, dict):
        ignored={'backend_session_id','cpu_timing','prepare_seconds','gmt_inference_seconds',
                 'physics_seconds','video_capture_seconds','step_seconds','trace_encoding_seconds'}
        keys=(set(left)|set(right))-ignored
        for key in sorted(keys):
            if key=='schema' and {left.get(key),right.get(key)} <= {
                    'genmo.gmt_execution_feedback.v2','genmo.gmt_execution_feedback.columns.v3'}:
                continue
            if key not in left or key not in right:
                output.append(dict(path=path+'/'+key,kind='missing_field'))
            else:
                differences(left[key],right[key],path+'/'+key,output)
    elif isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        if len(left)!=len(right):output.append(dict(path=path,kind='length'))
        else:
            for i,(a,b) in enumerate(zip(left,right)):differences(a,b,path+f'/{i}',output)
    elif left != right:
        output.append(dict(path=path,kind='value',left=str(left)[:200],right=str(right)[:200]))
    return output


class Recorder:
    def __init__(self, backend):
        self.backend,self.records=backend,[]
    def __getattr__(self,name):return getattr(self.backend,name)
    def call(self,method,**payload):
        value=self.backend.call(method,**payload)
        if method in self.backend.MUTATIONS:
            self.records.append(dict(method=method,payload=copy.deepcopy(payload),result=copy.deepcopy(value),
                timing=dict(self.backend.last_call_timing)))
        return value


def reward_rows(records, config, sample, music):
    target=load_paired_activity(config['stage9']['bc_data_root'],sample,music_start_tick=600,
        window_s=config['stage9']['reward']['activity']['window_s'],dt=.02,split=sample['row']['split'],music_start_frame=0)
    reward=ExecutionReward(config['stage9']['reward'],music,music_start_tick=600,target_activity=target)
    rows=[]
    for record in records:
        if record['method']!='advance':continue
        for row in record['result']['trace']:
            if row['tick']==600:reward.seed_previous_target(row['joint_position_target'])
            elif row['tick']>600:rows.append(reward.evaluate_step(row))
    if any(not row['transition_valid'] for row in rows):raise ValueError('Replay reward invalid')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    rank,world,local_rank=(int(os.environ.get(k,'-1')) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=8 or rank!=local_rank:raise ValueError('Use single-node torchrun with exactly eight GPUs')
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    collective=DistributedCollectives(rank,world,device='cpu')
    startup=[None]
    if rank==0:
        try:
            if args.output_dir.exists():raise FileExistsError(args.output_dir)
            startup[0]=dict(devices=_available_gpus())
        except Exception as error:startup[0]=dict(error=str(error))
    dist.broadcast_object_list(startup,0)
    if 'error' in startup[0]:raise RuntimeError(startup[0]['error'])
    root=args.output_dir;output=root/f'rank{rank:02d}';output.mkdir(parents=True)
    config=configuration(args.config)
    check=collective.broadcast_object(runtime_preflight(config,check_gpu=False) if rank==0 else None)
    if not check['ready']:raise RuntimeError('Runtime assets failed preflight')
    torch.cuda.set_device(rank);torch.set_num_threads(config['runtime']['torch_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config['runtime']['genmo_device']=f'cuda:{rank}'
    actor,_,_=local_call(collective,lambda:load_actor(config))
    policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',defer_checks=True)
    catalog=FullMusicCatalog(config['paths']['data_root'])
    catalog.apply_audit(collective.broadcast_object(catalog.audit_files(require_audio=True) if rank==0 else None))
    sample=catalog.samples['val'][SOURCES[rank%4]][0]
    music=catalog.load_music(sample)
    results=[];reference=None;reference_reward=None
    variants=[('baseline',False,False,False),('binary',True,False,False),
              ('state_reuse',False,True,False),('columns',False,False,True),('combined',True,True,True)]
    for label,binary,reuse,blocks in variants:
        directory=output/label;directory.mkdir()
        child=copy.deepcopy(config)
        child['runtime']['asset_conversion_dir']=str(directory/'usd_assets')
        child['stage10']['performance'].update(gmt_state_reuse=reuse,gmt_trace_blocks=blocks)
        path=directory/'resolved_config.yaml';path.write_text(yaml.safe_dump(child,allow_unicode=True))
        worker=Workers(child,directory);socket=Path(worker.temp.name)/'gmt.sock'
        journal=budget=None;token=None
        try:
            client=local_call(collective,lambda:worker.start('gmt',[child['paths']['isaac_python'],'-B',
                str(Path(child['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt.py'),'--config',str(path),
                '--socket',str(socket),'--headless'],child['paths']['gmt_repo'],socket,strip_distributed=True,
                environment=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank])))
            guard=DiskGuard(directory,min_free_bytes=10*2**30,max_run_bytes=4*2**30)
            journal=GuardedStepJournal(directory/'execution_journal.sqlite',guard,
                format='genmo.execution_journal.ndarray.v2' if binary else 'json.v1')
            budget=IncrementalBudget(directory/'budget.json',dict(accepted_iterations=1,optimizer_attempts=1,
                generations=40,control_steps=5000,physics_steps=20000),disk_guard=guard)
            backend=AcknowledgedBackend(client,journal,socket_path=socket)
            recorder=Recorder(backend)
            profile=PhaseProfiler(f'cuda:{rank}',rank,detailed=True);token=activate(profile)
            collective.barrier();start=time.perf_counter()
            if reference is None:
                builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(child['paths']['kinematics'])))
                env=UpperEnvironment(child,recorder,builder,policy,budget,directory/'collection');env.disk_guard=guard
                def collect():
                    env.reset_task(sample,music,seed=42 if rank<4 else 1729,phase='fixed_replay',music_start_frame=0)
                    for _ in range(20):
                        row=env.step()
                        if row.terminated or row.truncated:break
                    return recorder.records
                records=local_call(collective,collect)
                reference=records
            else:
                def replay():
                    for record in reference:
                        method,payload=record['method'],record['payload']
                        if method=='advance':
                            count=payload['control_steps'];budget.reserve('fixed_replay',control_steps=count,physics_steps=count*4)
                        value=recorder.call(method,**payload)
                        if method=='advance':budget.settle_control('fixed_replay',count,value)
                    return recorder.records
                records=local_call(collective,replay)
            elapsed=time.perf_counter()-start
            rewards=reward_rows(records,child,sample,music)
            if reference_reward is None:reference_reward=rewards
            diff=differences([r['result'] for r in reference],[r['result'] for r in records])
            reward_diff=differences(reference_reward,rewards)
            trace=[row for r in records if r['method']=='advance' for row in r['result']['trace']]
            timing={}
            for row in trace:
                for key,value in row.get('cpu_timing',{}).items():
                    if isinstance(value,(int,float)):timing[key]=timing.get(key,0.)+value
            rpc_timing={key:sum(r['timing'].get(key,0.) for r in records) for key in
                ('total_seconds','transport_seconds','journal_seconds','ack_seconds')}
            frozen=backend.call('verify_frozen');_assert_frozen(frozen)
            record=dict(variant=label,seconds=elapsed,control_steps=len(trace),physics_steps=4*len(trace),
                exact_feedback=not diff,feedback_differences=diff,exact_reward=not reward_diff,reward_differences=reward_diff,
                reward_sum=sum(r['reward'] for r in rewards),backend_cpu_timing=timing,rpc_timing=rpc_timing,
                profile=profile.report(),frozen=frozen,journal_bytes=journal.path.stat().st_size)
            results.append(record)
            torch.save(records,directory/'requests_and_full_replies.pt')
            (directory/'report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
            local_call(collective,lambda:None if not diff and not reward_diff else (_ for _ in ()).throw(
                ValueError(f'Fixed physical replay differs: {diff}; reward {reward_diff}')))
            if rank==0:print(json.dumps(dict(variant=label,seconds=elapsed,exact_feedback=not diff,exact_reward=not reward_diff)),flush=True)
        finally:
            if token is not None:deactivate(token)
            worker.close()
            if journal is not None:journal.close()
            if budget is not None:budget.close()
    reports=collective.all_gather_object(dict(rank=rank,variants=results))
    if rank==0:(root/'report.json').write_text(json.dumps(dict(scope='eight_gpu_fixed_reference_rpc_replay_not_closedloop_training',ranks=reports),ensure_ascii=False,indent=2))
    dist.destroy_process_group()


if __name__=='__main__':main()
