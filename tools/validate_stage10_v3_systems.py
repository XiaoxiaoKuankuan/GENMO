"""服务器1八卡的 Stage10 v3 梯度归约、恢复和原生GMT有限验收。

systems阶段用独立小模型验证真正的全局均值：不等长rank分片、空分片、仅BC头、
未使用参数、单次配对SUM、SGD尺度及非空Adam动量都与单卡全数据参考比较。
同一阶段验证CPU不可变快照、异步原子发布、损坏身份拒绝、有界内存ACK记录和
资源预算，不把小模型验收当作真实GENMO性能。所有文件位于指定的新测试目录。

gmt阶段严格加载model_140500和ONNX来源SHA，比较GPU批量、单样本、置换及
CPU ONNX动作；容限沿用原ONNX导出atol/rtol=1e-4，不放宽DPPO概率检查。
这里只证明策略数值和冻结边界，真实机器人观测、奖励和终止由闭环入口验收。
脚本不启动长期训练、不写入正式checkpoint、不修改已有模型或运行任务。
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from gem.closedloop.dppo.checkpoint_v3 import SCHEMA,AsyncCheckpointWriter,immutable_cpu_snapshot,file_sha256,load_v3
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.gradient_merge_v3 import BCGradientController,merge_global_gradients,take_gradients
from gem.closedloop.dppo.memory_execution_v3 import MemoryReceiveJournal,PrepaidMemoryBudget


class SmallActor(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.shared=nn.Linear(3,2,device=device)
        self.contact=nn.Linear(3,1,device=device)
        self.unused=nn.Parameter(torch.ones(3,device=device))


def compare_state(actual,expected):
    if isinstance(actual,torch.Tensor):
        torch.testing.assert_close(actual,expected.to(actual.device),atol=3e-7,rtol=2e-5)
    elif isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:compare_state(actual[key],expected[key])
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):compare_state(a,b)
    else:assert actual==expected


def systems_checks(output,distributed):
    d=distributed;torch.manual_seed(841);device=d.device
    results=[]
    for optimizer_type in (torch.optim.SGD,torch.optim.AdamW):
        actor=SmallActor(device);reference=copy.deepcopy(actor)
        opts=dict(lr=.01)
        if optimizer_type is torch.optim.SGD:opts['momentum']=.7
        else:opts['weight_decay']=0.
        optimizer=optimizer_type(actor.parameters(),**opts)
        expected_optimizer=optimizer_type(reference.parameters(),**opts)
        controller=BCGradientController(mode='fixed')
        x=torch.linspace(-1.,1.2,69,device=device).reshape(23,3)
        z=torch.linspace(.9,-.7,51,device=device).reshape(17,3)
        # rank7没有PPO样本，rank0拥有一个额外尾样本，BC八卡独立分担。
        ids=list(range(d.rank,23,7)) if d.rank<7 else []
        bids=list(range(d.rank,17,8))
        for step in range(3):
            optimizer.zero_grad(set_to_none=True);expected_optimizer.zero_grad(set_to_none=True)
            expected_loss=(reference.shared(x)-.2*step).square().mean()+.1*(
                reference.shared(z).square().mean()+reference.contact(z).square().mean())
            expected_loss.backward()
            if ids:((actor.shared(x[ids])-.2*step).square().sum()/46).backward()
            ppo=take_gradients(actor)
            if bids:(actor.shared(z[bids]).square().sum()/34+actor.contact(z[bids]).square().sum()/17).backward()
            bc=take_gradients(actor)
            report=merge_global_gradients(actor,ppo,bc,controller,distributed=d,bucket_bytes=32)
            for (name,p),(qname,q) in zip(actor.named_parameters(),reference.named_parameters()):
                assert name==qname
                if q.grad is None:assert p.grad is None
                else:torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-6)
            optimizer.step();expected_optimizer.step()
            compare_state(actor.state_dict(),reference.state_dict())
            compare_state(optimizer.state_dict(),expected_optimizer.state_dict())
        results.append(dict(optimizer=optimizer_type.__name__,steps=3,gradients_and_nonempty_state=True,
            empty_PPO_rank=7,gradient_reduction=report['communication']['gradient_reduction']))
    dynamic=BCGradientController()
    before=dynamic.value;after=dynamic.update(1000.,.01)
    assert before<=after<=before*2
    state=copy.deepcopy(dynamic.state_dict());dynamic.update(0.,0.);dynamic.load_state_dict(state)
    assert dynamic.state_dict()==state
    journal=MemoryReceiveJournal(output/'ephemeral',capacity=1)
    receipt=dict(backend_session_id='test',mutation_seq=1,payload=np.array([1.,-0.],dtype=np.float64))
    assert journal.append_result(receipt) and not journal.append_result(receipt)
    try:journal.append_result(dict(receipt,payload=np.array([2.,-0.],dtype=np.float64)))
    except ValueError:pass
    else:raise AssertionError('Changed receipt was silently accepted')
    journal.append_result(dict(receipt,mutation_seq=2));assert len(journal.entries)==1
    journal.close()
    budget=PrepaidMemoryBudget(dict(control_steps=10,physics_steps=40))
    budget.reserve('test',control_steps=10,physics_steps=40)
    budget.settle_control('test',10,dict(executed_control_steps=7,executed_physics_steps=28))
    assert budget.used==dict(control_steps=7,physics_steps=28)
    try:budget.reserve('excess',control_steps=4)
    except RuntimeError:pass
    else:raise AssertionError('Prepaid credit exceeded without rejection')
    critic=nn.Linear(3,1,device=device);critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=.001)
    critic(torch.ones(2,3,device=device)).sum().backward();critic_optimizer.step()
    identity=dict(schema='test.v3',rank=d.rank)
    payload=immutable_cpu_snapshot(dict(schema=SCHEMA,identity=identity,actor=actor.state_dict(),
        critic=critic.state_dict(),actor_optimizer=optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
        bc_controller=dynamic.state_dict(),actor_fingerprint='test',state=dict(iteration=1,rollouts_completed=2,
        actor_updates=3,critic_updates=1),rank_states=[dict(rank=d.rank)]))
    writer=AsyncCheckpointWriter(output)
    writer.submit(payload,reason='finite_test')
    with torch.no_grad():next(actor.parameters()).add_(1.)
    publication=writer.poll(wait=True);writer.close()
    path=output/publication['path'];assert file_sha256(path)==publication['sha256']
    load_v3(path,identity=identity,actor=actor,critic=critic,actor_optimizer=optimizer,
        critic_optimizer=critic_optimizer,controller=dynamic)
    compare_state(actor.state_dict(),payload['actor'])
    try:load_v3(path,identity=dict(identity,schema='wrong'),actor=actor,critic=critic,
        actor_optimizer=optimizer,critic_optimizer=critic_optimizer,controller=dynamic)
    except ValueError:pass
    else:raise AssertionError('Incompatible identity restored')
    # 保存发布后文件截断，独立SHA检查必须识别；只破坏本脚本自己的测试文件。
    with path.open('r+b') as f:f.truncate(path.stat().st_size//2)
    assert file_sha256(path)!=publication['sha256']
    path.unlink();(output/'latest.json').unlink()
    return dict(passed=True,global_gradient_tests=results,bounded_BC_controller=True,
        ephemeral_receipt_identity_checked=True,prepaid_budget_checked=True,
        immutable_checkpoint_and_identity_checked=True,corrupt_checkpoint_SHA_detected=True,
        scope='eight_GPU_system_correctness_not_real_physics_or_GENMO_performance')


def gmt_checks(output,distributed,gmt_repo,assets):
    for path in [gmt_repo/'rsl_rl',gmt_repo/'scripts/rsl_rl']:
        sys.path.insert(0,str(path))
    from bumi4340_native_frozen_policy import BumiNativeFrozenPolicy
    from bumi4340_onnx_policy import INPUT_WIDTHS
    d=distributed
    model=BumiNativeFrozenPolicy(assets/'policy.onnx',assets/'model_140500.pt',assets/'agent.yaml',device=d.device)
    assert all(not p.requires_grad for p in model.module.parameters())
    rng=np.random.default_rng(140500);cases=[]
    for count in [1,2,8,32,64]:
        for scale in [0.,.1,1.,3.]:
            cpu={k:(rng.standard_normal((count,w))*scale).astype(np.float32) for k,w in INPUT_WIDTHS.items()}
            gpu={k:torch.from_numpy(v).to(d.device) for k,v in cpu.items()}
            start=time.perf_counter();batch=model.raw_actions(gpu).cpu().numpy();seconds=time.perf_counter()-start
            expected=np.concatenate([model.session.run(['actions'],{k:v[i:i+1] for k,v in cpu.items()})[0]
                for i in range(count)])
            np.testing.assert_allclose(batch,expected,atol=1e-4,rtol=1e-4)
            scalar=torch.cat([model.raw_actions({k:v[i:i+1] for k,v in gpu.items()}) for i in range(count)]).cpu().numpy()
            np.testing.assert_allclose(batch,scalar,atol=1e-4,rtol=1e-4)
            order=torch.arange(count-1,-1,-1,device=d.device)
            perm=model.raw_actions({k:v[order] for k,v in gpu.items()}).flip(0).cpu().numpy()
            np.testing.assert_allclose(batch,perm,atol=1e-4,rtol=1e-4)
            cases.append(dict(batch=count,scale=scale,max_error=float(abs(batch-expected).max()),seconds=seconds))
    assert model.module_fingerprint()==model.initial_module_sha256
    return dict(passed=True,cases=cases,checkpoint_sha=model.checkpoint_sha256,
        raw_and_clipped_action_contract_preserved=True,weights_frozen=True,
        tolerance=dict(atol=1e-4,rtol=1e-4),scope='CPU_ONNX_GPU_native_control_numerical_not_physics')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['systems','gmt'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gmt-repo',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/data0/user/liwei/closedloop_stage10_v3_assets/20261012/gmt140500'))
    args=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    for key,value in dict(NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_DISABLE='1',
                         NCCL_SOCKET_IFNAME='lo',TORCH_NCCL_BLOCKING_WAIT='1').items():
        os.environ.setdefault(key,value)
    if int(os.environ['WORLD_SIZE'])!=8:raise ValueError('Eight Server1 GPU ranks required')
    torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('gloo',timeout=timedelta(minutes=10));group=dist.new_group(backend='nccl')
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    output=args.output/f'rank{rank:02d}';output.mkdir(parents=True,exist_ok=False)
    try:
        result=systems_checks(output,d) if args.phase=='systems' else gmt_checks(output,d,args.gmt_repo,args.assets)
    except BaseException as error:
        import traceback
        result=dict(passed=False,error=repr(error),traceback=traceback.format_exc())
    result.update(rank=rank,phase=args.phase,torch_version=torch.__version__)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    dist.destroy_process_group(group);dist.destroy_process_group()
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
