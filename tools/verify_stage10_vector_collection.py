"""服务器1八卡GPU多环境闭环采集的有限正确性与吞吐验收。

每rank只启动一个GPU PhysX世界，默认8个独立环境，完整一轮仍是20条真实上层
转移、全局160条。真实GENMO条件按就绪队列批量扩散，执行冻结GMT和真实物理，
保留原奖励、前缀保护、终止及逐环境journal。采集后使用同一Actor对保存链执行
全部20步严格零更新概率检查，并核对身份、自由坐标和片段边界。显式--updates
调用原完整DPPO/BC/Critic/KL更新，--resume用于独立进程间完整正常边界恢复验收。
本工具不是正式训练入口；报告必须区分采样模式与完整有限更新模式。
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
from gem.closedloop.dppo.vector_environment import VectorUpperEnvironment,DEADLINE_CONTRACT
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
    p.add_argument('--batch-wait-ms',type=float,default=100.)
    p.add_argument('--sampling-graph',action='store_true',help='启用已独立验收的无梯度采样图；PPO反向保持原路径')
    p.add_argument('--data-audit',type=Path,help='显式复用完整数据审计报告；仍核对清单并逐次核验实际加载文件')
    p.add_argument('--updates',action='store_true',help='采集后运行原完整DPPO/BC/Critic更新与KL验收')
    p.add_argument('--resume',type=Path,help='仅恢复本工具完整GPU向量验收断点')
    p.add_argument('--stop-after-iteration',type=int,help='有限验收提前正常退出，例如第一轮保存后退出进程')
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
        physics_device='cuda:0',gmt_precision='float32',asset_conversion_dir=str(output/'usd'),headless=True,video_path=None,
        prefix_deadline_contract=DEADLINE_CONTRACT,vector_audit_contract='nested_world_journal_excluded.v1')
    if args.resume and not args.updates:raise ValueError('Resume requires the complete finite training mode')
    config['stage9']['run_id']=root_call(collective,lambda:'vector-finite-'+str(uuid.uuid4()))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    actor,train_config,_=local_call(collective,lambda:load_actor(config))
    policy=DPPODiffusionPolicy(actor,steps=20,eta=config['stage9']['eta'],std_floor=config['stage9']['std_floor'],
        guidance_scale=config['stage9']['guidance_scale'],cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',defer_checks=True)
    graph=None
    if args.sampling_graph:
        from gem.closedloop.dppo.sampling_graph import install_sampling_graph
        graph=install_sampling_graph(actor)
    catalog=FullMusicCatalog(config['paths']['data_root'])
    audit=root_call(collective,lambda:json.loads(args.data_audit.read_text()) if args.data_audit else catalog.audit_files(require_audio=True))
    catalog.apply_audit(audit)
    root_call(collective,lambda:(args.output/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False)))
    learner=None
    if args.updates:
        group=dist.new_group(backend='nccl',timeout=timedelta(minutes=10))
        collective=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{rank}')
        from gem.closedloop.dppo.vector_validation_learning import FiniteVectorLearner
        learner=FiniteVectorLearner(actor,policy,train_config,config,collective,args.output,
            max_iterations=args.rounds,num_envs=args.num_envs)
    config_path=output/'resolved_config.yaml';config_path.write_text(yaml.safe_dump(config,allow_unicode=True))
    workers=Workers(config,output);collector=None;journal=None
    report=dict(rank=rank,rounds=[],status='started',num_envs=args.num_envs,
        audit_reused=args.data_audit is not None,data_content_sha256=audit['data_content_sha256'])
    try:
        socket=Path(workers.temp.name)/'vector.sock'
        client=local_call(collective,lambda:workers.start('gmt',[config['paths']['isaac_python'],'-B',
            str(args.gmt_repo/'scripts/rsl_rl/serve_frozen_gmt_vector.py'),'--config',str(config_path),
            '--socket',str(socket),'--headless'],args.gmt_repo,socket,strip_distributed=True,
            environment=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank])))
        guard=DiskGuard(output,min_free_bytes=10*1024**3,max_run_bytes=10*1024**3)
        def world_factory():
            world_journal=GuardedStepJournal(output/'world_journal.sqlite',guard,format='genmo.execution_journal.ndarray.v2')
            return VectorWorldClient(client,world_journal,socket_path=socket)
        workers.entries[0]['client']=None  # socket转交唯一RPC线程；关闭同样由该线程完成。
        def factory(slot,proxy):
            directory=output/f'env{slot:03d}';directory.mkdir()
            lane_journal=GuardedStepJournal(directory/'execution_journal.sqlite',guard,format='genmo.execution_journal.ndarray.v2')
            if collector.restore_records is not None and collector.restore_records[slot] is not None:
                from gem.closedloop.dppo.budget import atomic_json
                atomic_json(directory/'budget.json',collector.restore_records[slot]['budget'])
            budget=IncrementalBudget(directory/'budget.json',dict(accepted_iterations=args.rounds,optimizer_attempts=1,
                generations=200,control_steps=15000,physics_steps=60000),disk_guard=guard)
            backend=VectorLaneBackend(collector.lane_client(slot),lane_journal)
            child=copy.deepcopy(config);child['stage9']['seed']+=(rank*max(32,args.num_envs)+slot)*100003
            builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
            env=VectorUpperEnvironment(child,backend,builder,proxy,budget,directory/'collection');env.disk_guard=guard
            sampler=FullMusicSampler(catalog,seed=child['stage9']['seed'],window_seconds=config['stage9']['episode_seconds'],random_start=True)
            def close():
                lane_journal.close();budget.close();backend.client.close()
            return SimpleNamespace(env=env,sampler=sampler,close=close)
        collector=VectorEnvironmentCollector(policy,factory,world_factory,num_envs=args.num_envs,batch_wait_seconds=args.batch_wait_ms/1000.)
        if args.resume:
            report['restored']=learner.restore(args.resume,collector)
        else:
            report['calibration']=local_call(collective,lambda:collector.calibrate())
            (output/'calibration.json').write_text(json.dumps(report['calibration'],indent=2))
            if rank==0:print(f'[VECTOR] calibration budget={report["calibration"]["latency_budget_s"]:.3f}s',flush=True)
        start=0 if learner is None else learner.iteration
        stop=args.rounds if args.stop_after_iteration is None else args.stop_after_iteration
        if not start<stop<=args.rounds:raise ValueError('Invalid finite iteration boundary')
        for iteration in range(start,stop):
            collective.barrier()
            outer_begin=time.perf_counter()
            version=0 if learner is None else learner.policy_version
            for state in collector.states:
                if state is not None:state.resource.env.iteration=iteration
            fragments,timing=local_call(collective,lambda:collector.collect(count_per_rank=20,policy_version=version))
            if rank==0:print(f'[VECTOR] round {iteration+1} collection={timing["seconds"]:.3f}s batches={[b["effective_rows"] for b in timing["batches"]]}',flush=True)
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
            frozen=local_call(collective,lambda:collector.world_call('verify_frozen'));_assert_frozen(frozen)
            timing.update(probability=check,control_steps=sum(r.executed_control_steps for r in rows),
                reward=sum(float(r.rewards.sum()) for r in rows),physical_failures=sum(bool(r.metadata.get('terminal_snapshot',{}).get('terminated')) for r in rows),
                peak_memory_allocated=torch.cuda.max_memory_allocated(),frozen=frozen)
            if graph is not None:timing['sampling_graph']=graph.report()
            # 更新失败也保留已完成采集的耗时、物理工作量和概率验收，不能只剩错误字符串。
            report['rounds'].append(timing)
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            if learner is not None:
                timing['update']=learner.update(fragments)
                if rank==0:print(f'[VECTOR] accepted round {iteration+1}: {timing["update"]["timings"]}',flush=True)
                timing['outer_seconds_excluding_checkpoint']=time.perf_counter()-outer_begin
                timing['checkpoint']=learner.save(collector,args.output/f'checkpoint_{learner.iteration:06d}.pt')
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        report.update(status='passed',scope='real_vector_full_DPPO_finite_validation' if learner else 'real_vector_collection_and_probability_no_DPPO')
        results=collective.all_gather_object(report)
        if rank==0:(args.output/'report.json').write_text(json.dumps(dict(status='passed',ranks=results,devices=devices),ensure_ascii=False,indent=2))
    except BaseException as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}')
        if collector is not None:
            report['failed_collection_diagnostics']=dict(batches=collector.batch_reports,world_timing=collector.world_timing)
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        raise
    finally:
        try:
            if collector is not None:collector.close()
        finally:
            try:
                if journal is not None:journal.close()
                workers.close()
            finally:
                (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
                if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
