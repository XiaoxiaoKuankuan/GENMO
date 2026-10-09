"""服务器1八卡逐因素定位真实模型的候选数值差异，不修改概率检查容限。

使用同一原模型与真实环境条件，分别切换条件批量和注意力后端，比较两次重复编码、
逐条件编码、同批次概率重算以及置换后的末端输出。每项从相同参数开始且只读原档；
用于查明接口/实现错误，不能代替梯度、闭环或吞吐验收。报告写入独立路径。
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tools.verify_stage10_saved_learning import read_saved_rank
from tools.train_closedloop_stage10 import configuration
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy, masked_joint_log_prob
from gem.closedloop.dppo.execution_checks import policy_phase


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','iteration','weights','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('gloo')
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    rows,_,_=read_saved_rank(args.iteration,rank)
    contexts={k:torch.cat([r.context[k] for r in rows[:8]]).to(f'cuda:{rank}') for k in rows[0].context}
    config=configuration(args.config);config['runtime']['genmo_device']=f'cuda:{rank}'
    actor,_,_=load_actor(config);payload=torch.load(args.weights,map_location='cpu',weights_only=False,mmap=True)
    reports=[]
    for batch,attention in ((False,'manual'),(True,'manual'),(False,'sdpa_math'),(True,'sdpa_math')):
        actor.load_state_dict(payload['actor'])
        policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',
            precision_mode='fp32_fast',attention_backend=attention,defer_checks=True)
        policy.batched_conditions=batch
        with policy_phase(policy):
            first=policy.prepare_conditions(contexts);second=policy.prepare_conditions(contexts)
            singles=[policy.prepare_conditions({k:v[i:i+1] for k,v in contexts.items()}) for i in range(8)]
            components={name:dict(repeat=float((first[name]-second[name]).abs().max()),
                scalar=float((first[name]-torch.cat([entry[name] for entry in singles])).abs().max()))
                for name in ('conditional','unconditional')}
            trace=policy.sample_rollout(contexts,generator=[torch.Generator(device=f'cuda:{rank}').manual_seed(17+i) for i in range(8)])
            errors=[]
            for order in (torch.arange(8,device=f'cuda:{rank}'),torch.arange(7,-1,-1,device=f'cuda:{rank}')):
                context={k:v[order] for k,v in contexts.items()}
                prepared=policy.prepare_conditions(context)
                result=policy.transition_parameters(context,trace['chain'][order,19],19,prepared=prepared)
                logp=masked_joint_log_prob(trace['chain'][order,20],result['mean'],result['std'],result['free_mask'])
                errors.append(dict(mean=float((result['mean']-trace['old_means'][order,19]).abs().max()),
                    logp=float((logp-trace['old_log_probs'][order,19]).abs().max())))
            captures=[]
            for repeat in range(2):
                captured={}
                hooks=[module.register_forward_hook(lambda m,a,o,n=name: captured.update({n:(a[0].clone(),o.clone())}))
                    for name,module in actor.named_modules() if name.endswith('.attn')]
                try:
                    result=policy.transition_parameters(contexts,trace['chain'][:,19],19)
                finally:
                    for hook in hooks:hook.remove()
                captures.append(captured)
            repeat_layers={name:dict(input=float((a[0]-captures[1][name][0]).abs().max()),
                output=float((a[1]-captures[1][name][1]).abs().max())) for name,a in captures[0].items()}
        reports.append(dict(batch=batch,attention=attention,conditions=components,recompute=errors,
            attention_states=[dict(name=name,training=module.training,head_dim=module.head_dim,heads=module.num_heads)
                for name,module in actor.named_modules() if name.endswith('.attn')], repeat_layers=repeat_layers))
    (args.output/f'rank{rank:02d}.json').write_text(json.dumps(reports,indent=2)+'\n')
    if rank==0:print(json.dumps(reports),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
