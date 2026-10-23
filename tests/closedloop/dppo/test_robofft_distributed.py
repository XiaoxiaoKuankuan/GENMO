"""真实双进程 Gloo 的 RoboFFT 梯度、尾批与空 rank 验收。

父进程用真实小型 Stage1Actor 准备一次固定动作/MC/旧损失，子进程只按全局 manifest
分片。比较单进程拼接梯度与双进程 SUM 梯度，第二组让 rank1 完全为空；另检查动作
minibatch=1 的尾批更新跨 rank 一致。只用 CPU 临时目录，不启动 CUDA 或物理训练。
"""
from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_robofft import (cpu_threads,make_policy,make_rows,settings,gradients)
from gem.closedloop.robofft.policy import RoboFFTDiffusionPolicy
from gem.closedloop.robofft.updater import prepare_mc,objective_batch,actor_update,_cache_signature
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer


def _worker(rank,directory):
    directory=Path(directory);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(directory/'rendezvous').as_uri(),
        rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        distributed=DistributedCollectives(rank,2,device='cpu')
        for empty_rank in (False,True):
            data=torch.load(directory/'inputs.pt',weights_only=False)
            actor=data['actor'];t=settings()
            policy=RoboFFTDiffusionPolicy(actor,frozen_actor=data['frozen'],frozen_prefix_steps=9,
                generation_frames=60,std_schedule=t['std_schedule'],robofft=t['robofft'])
            owned=([0,1] if rank==0 else []) if empty_rank else [rank]
            rows=[data['rows'][i] for i in owned]
            targets=dict(data['targets'],advantages=data['targets']['advantages'][owned])
            cache=RolloutTensorCache(rows,targets,'cpu');cache.robofft_input_signature=_cache_signature(cache)
            for row in rows:
                objective_batch(policy,[row],cache,clip=.01,global_count=2,backward=True)
            distributed.sum_gradients(actor)
            result={'gradients':gradients(actor)}
            manifest=([dict(valid=True,has_free=True,owner_rank=0,local_index=i) for i in range(2)]
                if empty_rank else [dict(valid=True,has_free=True,owner_rank=i,local_index=0) for i in range(2)])
            optimizer=make_actor_optimizer(actor,t)
            result['update']=actor_update(policy,optimizer,rows,targets,cache,manifest,distributed=distributed,
                generator=torch.Generator().manual_seed(15),minibatch_actions=1,microbatch=1)
            result['state']=copy.deepcopy(actor.state_dict())
            # 同一全局 surrogate 触发全部 rank 的相同早停，包含完全空的 rank。
            policy.robofft['surrogate_kl_stop']=1e-30
            stopped=actor_update(policy,optimizer,rows,targets,cache,manifest,distributed=distributed,
                generator=torch.Generator().manual_seed(16),minibatch_actions=1,microbatch=1)
            result['early_stop']=(stopped['optimizer_steps'],stopped['stop_reason'])
            if rank==0:
                def broken(*args,**kwargs):raise FloatingPointError('injected_rank0_numeric_failure')
                policy.losses=broken
            try:
                actor_update(policy,optimizer,rows,targets,cache,manifest,distributed=distributed,
                    generator=torch.Generator().manual_seed(17),microbatch=1)
            except FloatingPointError as error:
                result['cooperative_failure']='injected_rank0_numeric_failure' in str(error)
            torch.save(result,directory/f'rank{rank}_empty{empty_rank}.pt')
            cache.close()
    finally:
        dist.destroy_process_group()


def test_gloo_sum_matches_concatenated_and_empty_rank(actor_factory,tmp_path):
    policy,context=make_policy(actor_factory)
    rows,targets,cache,_=make_rows(policy,context)
    prepare_mc(policy,rows,cache,torch.Generator().manual_seed(444),microbatch=1)
    actor, frozen=copy.deepcopy(policy.actor),copy.deepcopy(policy.frozen_policy.actor)
    # 夹具跨进程仅传网络；原 GaussianDiffusion 带局部 lambda，策略会从参数重建日程。
    actor.train_diffusion=None;frozen.train_diffusion=None
    torch.save(dict(actor=actor,frozen=frozen,rows=rows,targets=targets),tmp_path/'inputs.pt')
    objective_batch(policy,rows,cache,clip=.01,global_count=2,backward=True)
    expected=gradients(policy.actor)
    mp.spawn(_worker,args=(str(tmp_path),),nprocs=2,join=True)
    for empty in (False,True):
        results=[torch.load(tmp_path/f'rank{rank}_empty{empty}.pt',weights_only=False) for rank in range(2)]
        for result in results:
            assert result['update']['optimizer_steps']==2
            assert result['early_stop']==(1,'surrogate_kl_early_stop')
            assert result['cooperative_failure']
            for name,value in expected.items():
                torch.testing.assert_close(result['gradients'][name],value,atol=3e-5,rtol=3e-4,msg=name)
        for name,value in results[0]['state'].items():
            torch.testing.assert_close(results[1]['state'][name],value,atol=0,rtol=0,msg=name)
    cache.close()
