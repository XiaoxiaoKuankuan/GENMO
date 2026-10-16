"""在同一套新奖励完整MC轨迹上复核非对称Critic初始化和预热权重。

仅使用新的90维真实历史MC银行及新schema预热成果，对照同一三编码器网络在初始化
和完成预热后的独立验证MSE/EV、音乐/前缀顺序敏感性。MC/TD混合候选的拟合与选择
由主预热流程完成，这里不再保留旧MLP或旧48维Critic训练入口，也不推进Isaac。
固定Actor、行为核、数据源SHA必须一致，输出实际价值权重以支持复核。
初始化对照复用Stage1加载后的真实RNG位置，避免重新seed造成不同价值头；
另按音乐来源报告MSE、EV和偏差，防止长歌曲的状态数量掩盖短歌曲来源的问题。
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
        if (data['schema']!='genmo.stage2.causal_tracking.complete_episode.value_dataset.v1' or data['policy_version']!=0
                or data['actor_fingerprint']!=actor_fingerprint or data['kernel']!=kernel
                or data['report']['pending_episodes'] or data['report']['excluded']):
            raise ValueError('Complete fixed Stage1 MC identity or terminal boundary differs')
        self.rows=[SimpleNamespace(context=r['context'],metadata=r['metadata']) for r in data['rows']]
        self.targets=data['returns'];self.record=data['report']
        self.sources=[e['task']['dataset'] for e in data['episodes'] for _ in range(e['states'])]
        if len(self.sources)!=len(self.rows):raise ValueError('MC source labels differ from recorded state counts')
        self.evidence=dict(path=str(path.resolve()),sha256=file_sha256(path))
    def training_data(self,**kwargs):return self.rows,self.targets
    def report(self):return self.record


@torch.no_grad()
def order_sensitivity(c,bank):
    rows=bank.rows[:32];cache=ValueConditionCache(rows,c.distributed.device)
    context=cache.critic_context(rows);remaining=cache.get('remaining_music_seconds',rows)
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
        predictions.extend(c.critic(cache.critic_context(rows),cache.get('remaining_music_seconds',rows)).cpu().tolist())
    parts=c.distributed.all_gather_object(dict(predictions=predictions,targets=bank.targets.tolist(),sources=bank.sources))
    return dict(predictions=[x for p in parts for x in p['predictions']],targets=[x for p in parts for x in p['targets']],
        sources=[x for p in parts for x in p['sources']])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mc-run',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank);torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('gloo',timeout=timedelta(minutes=20));group=dist.new_group(backend='nccl')
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    root_call(d,lambda:a.output.mkdir(parents=True,exist_ok=False))
    artifact=torch.load(a.artifact,map_location='cpu',weights_only=False,mmap=True)
    source=torch.load(artifact['source']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    config=source['config'];config['runtime']['genmo_device']=str(d.device)
    actor,train,_=load_actor(config);initial_critic_rng=torch.get_rng_state().clone()
    fingerprint=_model_fingerprint(actor)
    if fingerprint!=artifact['actor_fingerprint']:raise ValueError('Source Stage1 Actor changed')
    kernel=artifact['matched_identity']['kernel']
    banks={split:Bank(next(a.mc_run.glob(f'sessions/*/mc_calibration/preheat_{split}/rank{rank:02d}/complete_mc.pt')),
        fingerprint,kernel) for split in ('train','val')}
    rows,targets=banks['train'].training_data();manifest=[v for shard in d.all_gather_object([
        dict(owner_rank=rank,local_index=i,valid=True,has_free=True)for i in range(len(rows))])for v in shard]
    c=SimpleNamespace(distributed=d,policy=SimpleNamespace(kernel_config=kernel),state=dict(policy_version=0))
    report=dict(schema='genmo.complete_MC.critic_architecture_comparison',source=str(a.mc_run),
        artifact_sha256=file_sha256(a.artifact),fixed_actor=fingerprint,real_training_states=len(manifest),cases=[],
        initial_value_head_rng='actual_post_Stage1_load_state',
        bank_evidence=d.all_gather_object({k:v.evidence for k,v in banks.items()}),
        scope='same_architecture_before_after_preheat_not_architecture_ablation')
    for kind in ('stage1_initialized_asymmetric','mc_preheated_asymmetric'):
        torch.set_rng_state(initial_critic_rng)
        settings=config['stage10']['critic']
        c.critic=PretrainedConditionCritic(actor,dimension=settings['dimension'],layers=settings['layers'],
            heads=settings['heads'],history_scales=settings['history_scales']).to(d.device)
        d.broadcast_module(c.critic);start=time.perf_counter();training=None
        if kind=='mc_preheated_asymmetric':c.critic.load_state_dict(artifact['critic'])
        c.critic.eval();metrics=independent_mc_metrics(c,banks['val'])
        points=validation_points(c,banks['val'])
        from gem.closedloop.dppo.periodic_monitor import summarize_critic_predictions
        by_source={}
        for name in sorted(set(points['sources'])):
            indices=[i for i,s in enumerate(points['sources']) if s==name]
            by_source[name]=summarize_critic_predictions([points['predictions'][i] for i in indices],
                [points['targets'][i] for i in indices])
            by_source[name]['bias']=by_source[name]['mean_prediction']-by_source[name]['mean_target']
        item=dict(architecture=kind,validation_MC=metrics,order_sensitivity=order_sensitivity(c,banks['val']),
            validation_points=points,validation_by_source=by_source,
            source_balanced_mse=sum(v['mse'] for v in by_source.values())/len(by_source),
            training=training,training_updates=artifact['selected_updates'] if kind=='mc_preheated_asymmetric' else 0,
            wall_seconds=max(d.all_gather_object(time.perf_counter()-start)))
        report['cases'].append(item)
        root_call(d,lambda:torch.save(immutable_cpu_snapshot(c.critic.state_dict()),a.output/(kind+'.pt')))
        if _model_fingerprint(actor)!=fingerprint or any(x.grad is not None for x in actor.parameters()):
            raise RuntimeError('Critic comparison modified Actor')
        root_call(d,lambda:atomic_json(a.output/'report.json',report))
        del c.critic
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
