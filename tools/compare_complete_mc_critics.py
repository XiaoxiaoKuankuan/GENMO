"""用已采完整歌曲的相同MC数据对照预训练时序Critic与原独立MLP Critic。

只读取固定Stage1 Actor下32条训练和32条独立验证物理episode，不推进Isaac，不
重采轨迹，不把TD bootstrap标签当真实MC。预训练方案读取已经通过验证的实际
价值权重；原Critic从零训练相同256 epoch/全局1024批，使用同一真实交互预算。
只输入生成动作之前的十个条件及剩余音乐时长，校验Actor和冻结副本不被更新。
输出各自独立MC的MSE/EV、音乐和前缀顺序敏感性、GPU时间及实际权重；此工具
仅做对照，不会切换主训练网络或解冻其预训练编码器。
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.mc_calibration import ValueConditionCache, independent_mc_metrics
from gem.closedloop.dppo.parallel_support import root_call
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import file_sha256, immutable_cpu_snapshot
from gem.closedloop.dppo.updater import critic_update


class Bank:
    def __init__(self,path,actor_fingerprint,kernel):
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if (data['schema']!='genmo.complete_episode.value_dataset.v1' or data['policy_version']!=0
                or data['actor_fingerprint']!=actor_fingerprint or data['kernel']!=kernel
                or data['report']['pending_episodes'] or data['report']['excluded']):
            raise ValueError('Complete fixed Stage1 MC identity or terminal boundary differs')
        self.rows=[SimpleNamespace(context=r['context'],metadata=r['metadata']) for r in data['rows']]
        self.targets=data['returns'];self.record=data['report']
    def training_data(self,**kwargs):return self.rows,self.targets
    def report(self):return self.record


@torch.no_grad()
def order_sensitivity(c,bank):
    rows=bank.rows[:32];cache=ValueConditionCache(rows,c.distributed.device)
    context=cache.context(rows);remaining=cache.get('remaining_music_seconds',rows)
    base=c.critic(context,remaining);report={}
    for name,field,mask_key in [('music','music_features','music_valid'),('prefix','known_qpos30','known_qpos30_mask')]:
        changed={k:v.clone() for k,v in context.items()};mask=context[mask_key]
        if mask.ndim==3:mask=mask.any(-1)
        eligible=[]
        for i in range(len(rows)):
            idx=mask[i].nonzero().flatten();eligible.append(len(idx)>1)
            changed[field][i,idx]=context[field][i,idx.flip(0)]
        delta=(c.critic(changed,remaining)-base).abs()[torch.tensor(eligible,device=c.distributed.device)]
        values=[v for part in c.distributed.all_gather_object(delta.cpu().tolist()) for v in part]
        report[name]=dict(states=len(values),mean_abs=sum(values)/len(values),maximum=max(values),
            changed_fraction=sum(v>1e-7 for v in values)/len(values),
            scope='reverse_valid_values_keep_original_relative_times_and_masks')
    return report


@torch.no_grad()
def validation_points(c, bank):
    predictions=[]
    for start in range(0,len(bank.rows),128):
        rows=bank.rows[start:start+128];cache=ValueConditionCache(rows,c.distributed.device)
        predictions.extend(c.critic(cache.context(rows),cache.get('remaining_music_seconds',rows)).cpu().tolist())
    parts=c.distributed.all_gather_object(dict(predictions=predictions,targets=bank.targets.tolist()))
    return dict(predictions=[x for p in parts for x in p['predictions']],targets=[x for p in parts for x in p['targets']])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mc-run',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=256)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank);torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('gloo',timeout=timedelta(minutes=20));group=dist.new_group(backend='nccl')
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    root_call(d,lambda:a.output.mkdir(parents=True,exist_ok=False))
    artifact=torch.load(a.artifact,map_location='cpu',weights_only=False,mmap=True)
    source=torch.load(artifact['source']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    config=source['config'];config['runtime']['genmo_device']=str(d.device)
    actor,train,_=load_actor(config);fingerprint=_model_fingerprint(actor)
    if fingerprint!=artifact['actor_fingerprint']:raise ValueError('Source Stage1 Actor changed')
    kernel=artifact['matched_identity']['kernel']
    banks={split:Bank(next(a.mc_run.glob(f'sessions/*/mc_calibration/preheat_{split}/rank{rank:02d}/complete_mc.pt')),
        fingerprint,kernel) for split in ('train','val')}
    rows,targets=banks['train'].training_data();manifest=[v for shard in d.all_gather_object([
        dict(owner_rank=rank,local_index=i,valid=True,has_free=True)for i in range(len(rows))])for v in shard]
    c=SimpleNamespace(distributed=d,policy=SimpleNamespace(kernel_config=kernel),state=dict(policy_version=0))
    report=dict(schema='genmo.complete_MC.critic_architecture_comparison',source=str(a.mc_run),
        artifact_sha256=file_sha256(a.artifact),fixed_actor=fingerprint,real_training_states=len(manifest),cases=[])
    for kind in ('pretrained_temporal','legacy_mlp_control'):
        torch.manual_seed(42)
        c.critic=(PretrainedConditionCritic(actor,dimension=256,layers=2,heads=4,freeze_encoders=True)
            if kind=='pretrained_temporal' else UpperCritic(qpos_mean=actor.endecoder.mean,qpos_std=actor.endecoder.std,
            proprio_scales=tuple(train.model.proprio_scales))).to(d.device)
        d.broadcast_module(c.critic);start=time.perf_counter();training=None
        if kind=='pretrained_temporal':c.critic.load_state_dict(artifact['critic'])
        else:
            optimizer=torch.optim.AdamW(c.critic.parameters(),lr=config['stage10']['critic']['learning_rate'],weight_decay=0.)
            training=critic_update(c.critic,optimizer,rows,dict(returns=targets),ValueConditionCache(rows,d.device),manifest,
                distributed=d,generator=torch.Generator().manual_seed(config['stage10']['seed']+2002),epochs=a.epochs,batch=1024)
        c.critic.eval();metrics=independent_mc_metrics(c,banks['val'])
        item=dict(architecture=kind,validation_MC=metrics,order_sensitivity=order_sensitivity(c,banks['val']),
            validation_points=validation_points(c,banks['val']),
            training=training,training_updates=(artifact['selected_updates']if training is None else training['optimizer_steps']),
            wall_seconds=max(d.all_gather_object(time.perf_counter()-start)))
        report['cases'].append(item)
        root_call(d,lambda:torch.save(immutable_cpu_snapshot(c.critic.state_dict()),a.output/(kind+'.pt')))
        if _model_fingerprint(actor)!=fingerprint or any(x.grad is not None for x in actor.parameters()):
            raise RuntimeError('Critic comparison modified Actor')
        root_call(d,lambda:atomic_json(a.output/'report.json',report))
        del c.critic
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
