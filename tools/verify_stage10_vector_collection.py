"""服务器1八卡GPU多环境闭环采集的有限正确性与吞吐验收。

每rank只启动一个GPU PhysX世界，默认8个独立环境，完整一轮仍是20条真实上层
转移、全局160条。真实GENMO条件按就绪队列批量扩散，执行冻结GMT和真实物理，
保留原奖励、前缀保护、终止及逐环境journal。采集后使用同一Actor对保存链执行
全部20步严格零更新概率检查，并核对身份、自由坐标和片段边界，不更新任何参数。
本工具不是正式训练入口；通过只能证明采样闭环，DPPO、回滚和恢复另行验收。
输出仅限显式新目录；八卡空闲检查在初始化CUDA之前，异常只清理本工具的进程。
"""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import yaml
from tools.train_closedloop_stage10 import configuration,_assert_frozen
from tools.train_closedloop_stage10_8gpu import _available_gpus
from tools.eval.run_closedloop_baseline import Workers
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import local_call,root_call
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.vector_collector import VectorWorldClient,VectorLaneBackend,VectorEnvironmentCollector
from gem.closedloop.dppo.full_dataset import FullMusicCatalog,FullMusicSampler
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from gem.closedloop.dppo.run_management import DiskGuard,GuardedStepJournal
from gem.closedloop.dppo.budget_ledger import IncrementalBudget
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import probability_check_local


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--gmt-repo',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--num-envs',type=int,default=8)
    p.add_argument('--rounds',type=int,default=2)
    args=p.parse_args()
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    if world!=8 or rank!=int(os.environ['LOCAL_RANK']):raise ValueError('Single-node eight GPUs required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    collective=DistributedCollectives(rank,world,device='cpu')
    devices=root_call(collective,_available_gpus)
    root_call(collective,lambda:args.output.mkdir(parents=True,exist_ok=False))
    output=args.output/f'rank{rank:02d}';output.mkdir()
    config=configuration(args.config)
    config['paths'].update(genmo_repo=str(Path(__file__).resolve().parents[1]),gmt_repo=str(args.gmt_repo.resolve()),
        compat_profile=str(args.gmt_repo/'configs/sim2sim/model_135000_stage2.json'))
    config['runtime'].update(rank=rank,genmo_device=f'cuda:{rank}',backend='gpu_vectorized.v1',num_envs=args.num_envs,
        physics_device='cuda:0',gmt_precision='float32',asset_conversion_dir=str(output/'usd'),headless=True,video_path=None)
    config['stage9']['run_id']='vector-finite-'+str(uuid.uuid4())
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    actor,_,_=local_call(collective,lambda:load_actor(config))
    policy=DPPODiffusionPolicy(actor,steps=20,eta=config['stage9']['eta'],std_floor=config['stage9']['std_floor'],
        guidance_scale=config['stage9']['guidance_scale'],cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',defer_checks=True)
    catalog=FullMusicCatalog(config['paths']['data_root'])
    catalog.apply_audit(root_call(collective,lambda:catalog.audit_files(require_audio=True)))
    config_path=output/'resolved_config.yaml';config_path.write_text(yaml.safe_dump(config,allow_unicode=True))
    workers=Workers(config,output);collector=None;journal=None
    report=dict(rank=rank,rounds=[],status='started',num_envs=args.num_envs)
    try:
        socket=Path(workers.temp.name)/'vector.sock'
        client=local_call(collective,lambda:workers.start('gmt',[config['paths']['isaac_python'],'-B',
            str(args.gmt_repo/'scripts/rsl_rl/serve_frozen_gmt_vector.py'),'--config',str(config_path),
            '--socket',str(socket),'--headless'],args.gmt_repo,socket,strip_distributed=True,
            environment=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank])))
        guard=DiskGuard(output,min_free_bytes=10*1024**3,max_run_bytes=10*1024**3)
        journal=GuardedStepJournal(output/'world_journal.sqlite',guard,format='genmo.execution_journal.ndarray.v2')
        remote=VectorWorldClient(client,journal,socket_path=socket)
        def factory(slot,proxy):
            directory=output/f'env{slot:03d}';directory.mkdir()
            lane_journal=GuardedStepJournal(directory/'execution_journal.sqlite',guard,format='genmo.execution_journal.ndarray.v2')
            budget=IncrementalBudget(directory/'budget.json',dict(accepted_iterations=args.rounds,optimizer_attempts=1,
                generations=200,control_steps=15000,physics_steps=60000),disk_guard=guard)
            backend=VectorLaneBackend(collector.lane_client(slot),lane_journal)
            child=copy.deepcopy(config);child['stage9']['seed']+=(rank*max(32,args.num_envs)+slot)*100003
            builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
            env=UpperEnvironment(child,backend,builder,proxy,budget,directory/'collection');env.disk_guard=guard
            sampler=FullMusicSampler(catalog,seed=child['stage9']['seed'],window_seconds=config['stage9']['episode_seconds'],random_start=True)
            def close():
                lane_journal.close();budget.close();backend.client.close()
            return SimpleNamespace(env=env,sampler=sampler,close=close)
        collector=VectorEnvironmentCollector(policy,factory,remote,num_envs=args.num_envs)
        for iteration in range(args.rounds):
            collective.barrier()
            fragments,timing=local_call(collective,lambda:collector.collect(count_per_rank=20,policy_version=0))
            rows=[r for fragment in fragments for r in fragment]
            if len(rows)!=20:raise AssertionError('Real local batch changed')
            torch.save(fragments,output/f'rollout_{iteration:06d}.pt')
            def audit():
                for slot,fragment in enumerate(fragments):
                    for row in fragment:
                        row.validate()
                        if row.metadata['collector_env_slot']!=slot:raise AssertionError('Cross-environment identity')
                    if not(fragment[-1].terminated or fragment[-1].truncated):raise AssertionError('Missing GAE boundary')
                cache=RolloutTensorCache(rows,{},f'cuda:{rank}')
                try:return probability_check_local(policy,rows,denoising_microbatch=32,tensor_cache=cache)
                finally:cache.close()
            check=local_call(collective,audit)
            frozen=local_call(collective,lambda:remote.call('verify_frozen'));_assert_frozen(frozen)
            timing.update(probability=check,control_steps=sum(r.executed_control_steps for r in rows),
                reward=sum(float(r.rewards.sum()) for r in rows),physical_failures=sum(bool(r.metadata.get('terminal_snapshot',{}).get('terminated')) for r in rows),
                peak_memory_allocated=torch.cuda.max_memory_allocated(),frozen=frozen)
            report['rounds'].append(timing)
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        report.update(status='passed',scope='real_vector_collection_and_probability_no_DPPO')
        results=collective.all_gather_object(report)
        if rank==0:(args.output/'report.json').write_text(json.dumps(dict(status='passed',ranks=results,devices=devices),ensure_ascii=False,indent=2))
    except BaseException as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        if collector is not None:collector.close()
        if journal is not None:journal.close()
        workers.close()
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
