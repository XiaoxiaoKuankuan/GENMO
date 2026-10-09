"""服务器1八卡、每卡两个独立 GMT 环境的有限真实采样验收入口。

本工具只采集两段各20条上层转移（每环境各10条），不更新Actor/Critic、不启动
正式训练、不写正式run。必须torchrun启动八个rank并显式暴露八张空闲GPU；先检查
计算进程和CPU容量，各卡启动第一实例测RSS，跨卡核对剩余内存后才启动第二实例。
SQLite/预算/连接/音乐游标均在所属环境线程中创建。结束保存完整转移、冻结证据、
双游标边界和运行资源报告，关闭自己创建的工作进程。全局每轮保持160条上层转移，
不宣称两环境完整训练入口已经接入；真实8卡恢复仍由独立验收决定。
"""
import argparse
import copy
import csv
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid
from datetime import timedelta

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import yaml
from tools.train_closedloop_stage10 import configuration,runtime_preflight,_assert_frozen
from tools.eval.run_closedloop_baseline import Workers
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.dual_collector import DualEnvironmentCollector
from gem.closedloop.dppo.full_dataset import FullMusicCatalog,FullMusicSampler
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from gem.closedloop.dppo.rpc import AcknowledgedBackend
from gem.closedloop.dppo.run_management import DiskGuard,GuardedStepJournal
from gem.closedloop.dppo.budget_ledger import IncrementalBudget
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.closedloop_protocol import RpcClient
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import local_call


def available_memory():
    fields=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0])*1024


def process_usage(pid):
    """读取自己创建进程的累计CPU时间/RSS，分别报告，不将CPU核数当百分比。"""
    raw=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(cpu_seconds=(int(raw[11])+int(raw[12]))/os.sysconf('SC_CLK_TCK'),
        rss_bytes=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--inject-worker-exit',action='store_true',help='After two timed rounds, terminate only rank3/env1 owned test worker')
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
    root=args.output_dir
    args.output_dir=root/f'rank{rank:02d}'
    config=configuration(args.config)
    if len(os.sched_getaffinity(0))<2*world*config['runtime']['torch_threads']:
        raise RuntimeError('Insufficient CPU affinity for two independent workers')
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    check=collective.broadcast_object(runtime_preflight(config,check_gpu=False) if rank==0 else None)
    if not check['ready']:raise RuntimeError('Runtime assets failed preflight')
    args.output_dir.mkdir(parents=True)
    config['runtime'].update(rank=rank,genmo_device=f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    config['stage9']['run_id']='dual-prototype-'+str(uuid.uuid4())
    torch.set_num_threads(config['runtime']['torch_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    actor,_,_=local_call(collective,lambda:load_actor(config))
    policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',defer_checks=True)
    catalog=FullMusicCatalog(config['paths']['data_root'])
    catalog.apply_audit(collective.broadcast_object(catalog.audit_files(require_audio=True) if rank==0 else None))
    workers=[];sockets=[];resources=[];collector=None
    report=dict(scope='eight_gpu_two_real_workers_per_rank_collection_only',rank=rank,
        original_gpu=startup[0]['devices'][rank],cpu_affinity=len(os.sched_getaffinity(0)),memory_before=available_memory(),rounds=[])
    try:
        for slot in range(2):
            directory=args.output_dir/f'env{slot}'
            directory.mkdir()
            child=copy.deepcopy(config)
            child['runtime']['asset_conversion_dir']=str(directory/'usd_assets')
            path=directory/'resolved_config.yaml';path.write_text(yaml.safe_dump(child,allow_unicode=True))
            worker=Workers(child,directory);workers.append(worker)
            socket=Path(worker.temp.name)/'gmt.sock';sockets.append(socket)
            client=local_call(collective,lambda:worker.start('gmt',[config['paths']['isaac_python'],'-B',str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt.py'),
                '--config',str(path),'--socket',str(socket),'--headless'],config['paths']['gmt_repo'],socket,
                strip_distributed=True,environment=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank])))
            client.close()
            worker.entries[0]['client']=None  # 连接转交所属环境线程重建，避免退出再次使用关闭的socket。
            pid=worker.entries[0]['proc'].pid
            rss=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
            report.setdefault('workers',[]).append(dict(slot=slot,pid=pid,rss_bytes=rss,memory_available=available_memory()))
            rss_by_rank=collective.all_gather_object(rss)
            if slot==0 and min(collective.all_gather_object(available_memory()))<max(2*sum(rss_by_rank),32*1024**3):
                raise MemoryError('Measured first-worker RSS leaves insufficient capacity for a second worker')
        def factory(slot,proxy):
            directory=args.output_dir/f'env{slot}'
            guard=DiskGuard(directory,min_free_bytes=10*1024**3,max_run_bytes=2*1024**3)
            journal=GuardedStepJournal(directory/'execution_journal.sqlite',guard,
                format=config['stage10']['performance'].get('journal_format','json.v1'))
            budget=IncrementalBudget(directory/'budget.json',dict(accepted_iterations=2,optimizer_attempts=1,
                generations=50,control_steps=5000,physics_steps=20000),disk_guard=guard)
            backend=AcknowledgedBackend(RpcClient(sockets[slot],timeout_s=120),journal,socket_path=sockets[slot])
            builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
            child=copy.deepcopy(config);child['stage9']['seed']+=(rank*2+slot)*100003
            env=UpperEnvironment(child,backend,builder,proxy,budget,directory/'collection')
            env.disk_guard=guard
            sampler=FullMusicSampler(catalog,seed=config['stage10']['seed']+(rank*2+slot)*100003,
                window_seconds=config['stage9']['episode_seconds'],random_start=True)
            def close():
                try:
                    frozen=backend.call('verify_frozen');_assert_frozen(frozen)
                    (directory/'frozen.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2))
                    backend.call('close')
                finally:
                    journal.close();budget.close();backend.client.close()
            resource=SimpleNamespace(env=env,sampler=sampler,close=close)
            resources.append(resource)
            return resource
        collector=DualEnvironmentCollector(policy,factory,timeout_seconds=600.)
        for iteration in range(2):
            collective.barrier()
            pids=[os.getpid(),*[w.entries[0]['proc'].pid for w in workers]]
            before={pid:process_usage(pid) for pid in pids}
            fragments,metrics=local_call(collective,lambda:collector.collect(count_per_rank=20,policy_version=0))
            after={pid:process_usage(pid) for pid in pids}
            torch.save(fragments,args.output_dir/f'rollout_{iteration}.pt')
            metrics['control_steps']=sum(r.executed_control_steps for f in fragments for r in f)
            metrics['timing']=[r.metadata.get('timing',{}) for f in fragments for r in f]
            metrics['processes']=[dict(pid=pid,**after[pid],cpu_core_equivalents=
                (after[pid]['cpu_seconds']-before[pid]['cpu_seconds'])/metrics['seconds']) for pid in pids]
            metrics['prefix_frames']=[int(r.context['known_qpos30_mask'][0].any(-1).sum()) for f in fragments for r in f]
            metrics['environment_fragments']=[dict(slot=i,transitions=len(f),
                control_steps=sum(r.executed_control_steps for r in f),terminated=sum(r.terminated for r in f),
                truncated=sum(r.truncated for r in f),bootstrap=sum(not r.terminated and r.next_context is not None for r in f))
                for i,f in enumerate(fragments)]
            metrics['sampling_network_useful_fraction']=1.
            metrics['paired_request_fraction']=sum(b['effective_rows'] for b in metrics['batches'] if b['effective_rows']==2)/20
            report['rounds'].append(metrics)
        torch.save(collector.state_dict(),args.output_dir/'collector_boundary.pt')
        report['status']='eight_gpu_collection_completed_pending_independent_audit_and_training_integration'
        if args.inject_worker_exit:
            # 先在各自SQLite/连接线程验证全部16个冻结实例，再终止明确属于本测试的一个PID。
            frozen=[]
            for executor,state in zip(collector.executors,collector.states):
                value=executor.submit(lambda r=state.resource:r.env.backend.call('verify_frozen')).result(timeout=120)
                _assert_frozen(value);frozen.append(value)
            report['frozen_before_worker_exit']=frozen
            collective.barrier()
            if rank==3:
                owned=workers[1].entries[0]['proc']
                if owned.poll() is not None:raise RuntimeError('Fault-injection worker already exited unexpectedly')
                owned.send_signal(signal.SIGTERM)
                owned.wait(timeout=30)
            collective.barrier()
            caught=False
            try:
                local_call(collective,lambda:collector.collect(count_per_rank=2,policy_version=0))
            except RuntimeError as error:
                caught=True
                report['worker_exit_diagnostic']=str(error)
            if not all(collective.all_gather_object(caught)):raise AssertionError('A rank missed actual worker exit')
            report['worker_exit_test']=dict(actual_owned_worker_terminated=True,failed_rank=3,failed_environment=1,
                all_ranks_notified=True,actor_optimizer_steps=0,spent_budget_refunded=False,
                scope='after_two_timed_rounds_not_included_in_sampling_throughput')
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if collector is not None:
                try:collector.close()
                except Exception as error:
                    if not report.get('worker_exit_test'):raise
                    report['expected_close_error_after_injected_exit']=str(error)
        finally:
            for worker in reversed(workers):worker.close()
            (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    reports=collective.all_gather_object(report)
    if rank==0:
        (root/'report.json').write_text(json.dumps(dict(ranks=reports,global_upper_per_round=160,
            wall_seconds_per_round=[max(r['rounds'][i]['seconds'] for r in reports) for i in range(2)]),ensure_ascii=False,indent=2))
    dist.destroy_process_group()


if __name__=='__main__':main()
