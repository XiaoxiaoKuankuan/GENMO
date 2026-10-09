"""服务器1八卡验证采样专用CUDA图，原保存rollout与学习反向不变。

每卡读取自己封存的真实条件，分别用B1/4/8的不同真实链和独立随机种子，执行原
20步FP32采样及图采样。逐元素比较完整随机链、均值、标准差、两个head和输出，
并用普通有梯度执行路径对图产生的链重算全部概率，门槛固定1e-4/1e-3/1e-8。
有限稳态测速不含第一次图捕获，捕获单列；不含GMT或物理，不冒充闭环轮耗时。
一次真实有梯度前向必须进入原forward，防止图专用静态梯度污染PPO。
所有证据写入显式新文件，源tar SHA前后不变；仅允许服务器1八卡执行。
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tools.train_closedloop_stage10 import configuration
from tools.train_closedloop_stage10_8gpu import _available_gpus
from tools.verify_stage10_saved_learning import read_saved_rank
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.sampling_graph import install_sampling_graph
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import probability_check_local
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.parallel_support import root_call,local_call
from gem.closedloop.dppo.dual_collector import split_trace
from gem.closedloop.dppo.run_management import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','iteration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--target', choices=('denoiser','conditions','both'),default='denoiser')
    a=p.parse_args();rank=int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('Eight Server1 GPUs required')
    dist.init_process_group('gloo',timeout=timedelta(minutes=15))
    group=DistributedCollectives(rank,8,device='cpu')
    root_call(group,_available_gpus)
    root_call(group,lambda:None if not a.output.exists() else (_ for _ in ()).throw(FileExistsError(a.output)))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    cfg=configuration(a.config);cfg['runtime']['genmo_device']=f'cuda:{rank}'
    actor,_,_=local_call(group,lambda:load_actor(cfg))
    policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',defer_checks=True)
    rows,_,source_sha=local_call(group,lambda:read_saved_rank(a.iteration,rank))
    original=actor.denoiser.forward;original_condition=policy._encode_prepared
    wrappers={}
    if a.target in ('denoiser','both'):wrappers['denoiser']=install_sampling_graph(actor)
    if a.target in ('conditions','both'):
        from gem.closedloop.dppo.condition_sampling_graph import install_condition_sampling_graph
        wrappers['conditions']=install_condition_sampling_graph(policy)
    def select_graph(enabled):
        actor.denoiser.forward=wrappers.get('denoiser',original) if enabled else original
        policy._encode_prepared=wrappers.get('conditions',original_condition) if enabled else original_condition
    def execution_report():return {name:wrapper.report() for name,wrapper in wrappers.items()}
    report=dict(rank=rank,status='running',source_sha256=source_sha,target=a.target,results=[])
    def run():
        for batch in (1,4,8):
            selected=rows[:batch]
            context={k:torch.cat([row.context[k] for row in selected]).cuda(rank) for k in selected[0].context}
            def sample():
                return policy.sample_rollout(context,generator=[torch.Generator(device=f'cuda:{rank}').manual_seed(9000+rank*100+i) for i in range(batch)])
            select_graph(False)
            eager=sample();torch.cuda.synchronize()
            begin=time.perf_counter();sample();torch.cuda.synchronize();eager_seconds=time.perf_counter()-begin
            eager_components=dict(policy.last_sample_timing)
            select_graph(True)
            graph=sample();torch.cuda.synchronize()
            difference={k:float((eager[k]-graph[k]).abs().max()) for k in
                ('chain','old_means','old_stds','old_log_probs','qpos','contact','contact_logits')}
            if any(value!=0 for value in difference.values()):
                raise AssertionError(f'Graph changed FP32 eager output: {difference}')
            check_rows=[]
            for i,row in enumerate(selected):
                item=copy.copy(row);trace=split_trace(graph,i,batch)
                item.context=trace['conditions'];item.chain=trace['chain'][0];item.old_log_prob=trace['old_log_probs'][0]
                item.free_mask=trace['free_mask'][0];item.metadata=dict(row.metadata,sampler_trace=trace)
                check_rows.append(item)
            cache=RolloutTensorCache(check_rows,{},f'cuda:{rank}')
            # 强制有梯度上下文，图模块应走原forward；校验函数内部no_grad也允许图执行。
            select_graph(False)
            check=probability_check_local(policy,check_rows,denoising_microbatch=32,tensor_cache=cache)
            select_graph(True)
            with torch.enable_grad():
                params=policy.transition_parameters(context,graph['chain'][:,0],0)
                params['mean'].sum().backward()
            if not all(w.eager_gradient_calls for w in wrappers.values()):raise AssertionError('PPO did not use original gradient path')
            actor.zero_grad(set_to_none=True);cache.close()
            times=[];components=[]
            for _ in range(3):
                torch.cuda.synchronize();begin=time.perf_counter();sample();torch.cuda.synchronize()
                times.append(time.perf_counter()-begin)
                components.append(dict(policy.last_sample_timing))
            report['results'].append(dict(batch=batch,exact_output_difference=difference,probability=check,
                eager_seconds=eager_seconds,graph_seconds=times,eager_components=eager_components,graph_components=components))
        if file_sha256(a.iteration/'execution_evidence.tar.gz')!=source_sha:raise AssertionError('Source changed')
    try:
        local_call(group,run);report.update(status='passed',execution=execution_report())
    except BaseException as error:
        report.update(status='failed',error=str(error),execution=execution_report())
        raise
    finally:
        select_graph(False)
        records=group.all_gather_object(report)
        if rank==0:a.output.write_text(json.dumps(dict(status='passed' if all(r['status']=='passed' for r in records) else 'failed',ranks=records),ensure_ascii=False,indent=2))
        dist.destroy_process_group()


if __name__=='__main__':main()
