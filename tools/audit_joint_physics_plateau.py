"""只读定位联合训练执行Loss平台，不改变训练进程、权重或物理合同。

在独立CPU进程加载同一run的多个完整checkpoint，固定真实来源、请求、噪声和H32，
比较学生去噪参考、数据动作参考及真实缓存本身的跟踪误差。数据动作只用于诊断基线，
绝不替换学生训练输出。逐步误差、参考消费范围、Loss分项和x0梯度均保留原执行算法。
额外在最新checkpoint的小批样本上分解数据/执行参数梯度及夹角，明确它不是八卡全局
梯度的精确重放。只读取现有sidecar；不启动Newton，不创建优化器，不使用训练GPU。
输出固定样本清单和逐项JSON证据；所有拒绝保留，不按Loss或梯度筛选好看的窗口。
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def scalar(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else value


def gradient_stats(named, first, second, weight):
    """按实际参数名计算两路梯度范数/余弦；不以x0梯度替代参数梯度。"""
    groups = defaultdict(lambda: [0., 0., 0.])
    for (name, _), a, b in zip(named, first, second):
        group = 'future_encoder' if name.startswith('future_encoder') else '.'.join(name.split('.')[:2])
        aa = float(a.float().square().sum()) if a is not None else 0.
        bb = float(b.float().square().sum()) if b is not None else 0.
        ab = float((a.float()*b.float()).sum()) if a is not None and b is not None else 0.
        for key in ('all', group):
            groups[key] = [x+y for x, y in zip(groups[key], (aa, bb, ab))]
    return {k: {'data_norm': a**.5, 'execution_norm': b**.5,
                'weighted_execution_to_data': weight*b**.5/max(a**.5, 1e-30),
                'cosine': c/max((a*b)**.5, 1e-30)} for k, (a, b, c) in groups.items()}


@contextmanager
def capture_terms(execution, override_reference=None):
    """仅在独立诊断进程捕获原函数返回值，退出时恢复；训练模块文件不改写。"""
    original, candidate = execution.execution_terms, execution.candidate_reference
    rows = []
    def wrapped(*args, **kwargs):
        loss, safety, terms = original(*args, **kwargs)
        rows.append({'loss': scalar(loss.mean()), **{k: scalar(v.mean()) for k, v in terms.items()}})
        return loss, safety, terms
    execution.execution_terms = wrapped
    if override_reference is not None:
        execution.candidate_reference = lambda *args, **kwargs: override_reference
    try:
        yield rows
    finally:
        execution.execution_terms, execution.candidate_reference = original, candidate


def run(args):
    from gem.closedloop.joint_physics.config import read_config, build_actor, build_adapter, seed_all
    cfg = read_config(args.config)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from gem.closedloop.joint_physics import execution
    from gem.closedloop.joint_physics.checkpoint import read_checkpoint
    from gem.closedloop.joint_physics.contracts import PhysicsRejected, execution_contract
    from gem.closedloop.joint_physics.data import SourceCatalog, SidecarStore, collate_joint_samples, slice_batch, JointDataStream
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    from gem.closedloop.joint_physics.reference import attach_sidecars, candidate_reference, attach_supervised_sidecars
    from gem.closedloop.joint_physics.requests import PhysicsRequestStream
    from gem.closedloop.joint_physics.losses import execution_terms
    from gem.closedloop.joint_physics.curriculum import physics_noise_levels
    from gem.closedloop.mb_control.contracts import ReferenceExhausted
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); seed_all(args.seed)
    # 限制只读映射预算和线程，不修改正在运行的配置文件。
    cfg['data']['loading'].update(workers=0, prefetch_batches=0, pin_memory=False,
                                  sequence_cache_mb=2048, sidecar_cache_mb=2048)
    actor, data_loss = build_actor(cfg, 'cpu'); actor.eval()
    adapter = build_adapter(cfg, 'cpu'); frozen = adapter.bundle.frozen_snapshot()
    assets = asset_contract(cfg, adapter)
    catalog = SourceCatalog(cfg, 'train')
    store = SidecarStore(cfg['data']['sidecar'], 'train', assets,
                        sample_contract=cfg['data']['sample_contract'], cache_bytes=2*1024**3)
    support = execution.ValidatedSupport(cfg['physics']['validation_report'], assets, 32, 'cpu', execution_contract(cfg))
    stream = PhysicsRequestStream(catalog, store, adapter.bundle.spec.joint_names, 32, cfg['training']['seed'],
        certification=support, historical_mode='diagnostic', noise_levels=cfg['physics']['noise_levels'],
        cache_directory=cfg['data']['loading']['request_cache'])
    rng = np.random.default_rng(args.seed)
    by_dataset = defaultdict(list)
    for key in stream.keys:
        by_dataset[stream.sources[key][0].dataset_name].append(key)
    selections, samples = [], []
    for dataset, keys in sorted(by_dataset.items()):
        for key in rng.choice(keys, min(args.per_dataset, len(keys)), replace=False):
            ds, row = stream.sources[key]
            groups = stream.groups[key]; group = groups[int(rng.integers(len(groups)))]
            request = group[int(rng.integers(len(group)))]
            samples.append(ds.get_window(row, request['frame']))
            selections.append(dict(request))
    batch = collate_joint_samples(samples)
    batch, contexts, windows, reasons = attach_sidecars(batch, store, adapter, actor.stage1.endecoder.codec, 32)
    if any(w is None for w in windows):
        raise ValueError('固定静态合法请求装配失败: '+str(reasons))
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(len(samples), 60, 30, generator=generator)
    levels = physics_noise_levels(actor.stage1.train_diffusion, cfg['physics'], support.report).tolist()
    manifest = dict(config=args.config, checkpoints=args.checkpoints, seed=args.seed, device='cpu',
        precision=actor.precision, threads=args.threads, selection='equal_sources_per_dataset; random_profile_and_time_before_prediction',
        samples=selections, timesteps=levels, assets=assets, data_gradient_batch=args.data_batch,
        no_optimizer=True, no_newton=True, original_training_unchanged=True)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    records = []
    def emit(row):
        records.append(row)
        with (output/'rows.jsonl').open('a') as f: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k not in ('per_step','x0_frame_grad_norm','terms','parameter_gradients')},ensure_ascii=False),flush=True)
    def evaluate(index, prediction, logits, *, reference=None, gradients=False):
        window = windows[index]
        guard = execution.DiagnosticSupport(support, selections[index]['tags'])
        with capture_terms(execution, reference) as per_step:
            result = execution.execute_window(window, prediction, logits, actor, adapter, cfg['execution_loss'],
                support=guard, training_policy=cfg['physics']['training_safety'])
        row = {'loss': scalar(result['loss']), 'terms': {k:scalar(v) for k,v in result['terms'].items()},
               'coverage': result['coverage'], 'per_step': per_step,
               'effective_horizon':result['effective_horizon'], 'failure_steps':result['failure_steps']}
        if gradients:
            gradient, = torch.autograd.grad(result['loss'], prediction, retain_graph=True)
            row['x0_gradient_norm'] = float(gradient.norm())
            row['x0_frame_grad_norm'] = gradient[0].norm(dim=-1).tolist()
        return row, result
    # 缓存真实状态跟踪原数据的误差，以及冻结SM/WM执行完美数据参考的非零基线。
    with torch.no_grad():
        adapted = actor.stage1.adapt_conditions(batch)
        target = actor.stage1._constrain(actor.stage1.endecoder.normalize(batch['target_qpos30']), adapted)
        for i, window in enumerate(windows):
            episode = store.episode(batch['meta'][i]); start = window.start.tick//12
            real = []
            for j in range(32):
                ref, _, _ = window.target.read_current_future(window.start.tick+(j+1)*12, 0)
                loss, _, terms = execution_terms(episode['state'][start+j+1:start+j+2], ref, adapter.runtime.kin, cfg['execution_loss'])
                real.append({'loss':scalar(loss.mean()), **{k:scalar(v.mean()) for k,v in terms.items()}})
            emit({'kind':'real_tracking_baseline','sample':i,'loss':sum(x['loss'] for x in real)/32,'per_step':real})
            for name, reference in [('gt_codec_reference',None),('original_committed_reference',window.request.reference)]:
                try:
                    row, _ = evaluate(i,target[i:i+1],torch.zeros(1,60,2),reference=reference)
                    emit({'kind':name,'sample':i,**row})
                except (PhysicsRejected, ReferenceExhausted) as error:
                    emit({'kind':name,'sample':i,'rejection':getattr(error,'reason',str(error))})
    for checkpoint in args.checkpoints:
        value = read_checkpoint(checkpoint); step = value['global_step']
        actor.load_state_dict(value['actor'],strict=True); actor.eval()
        del value
        for level in levels:
            with torch.no_grad():
                predictions = actor.denoise_supervised(batch,contexts,torch.full((len(samples),),level,dtype=torch.long),
                                                       noise,torch.zeros(len(samples),dtype=torch.bool))
            for i, window in enumerate(windows):
                prediction = predictions['pred_x_start'][i:i+1].detach().requires_grad_()
                try:
                    row, result = evaluate(i,prediction,predictions['static_conf_logits'][i:i+1],gradients=True)
                    unknown = predictions['unknown_mask'][i:i+1]
                    row['unknown_x0_mse'] = float(((prediction-predictions['target_x_start'][i:i+1]).square()*unknown).sum()/unknown.sum())
                    candidate = candidate_reference(window,prediction,predictions['static_conf_logits'][i:i+1],actor.stage1.endecoder.codec,actor.stage1.endecoder)
                    ts=candidate.ticks[candidate.ticks>=window.ticket.first_mutable_source_tick]
                    # 仅比较实际H32依赖到的新参考点，不把完整60帧误差混成执行覆盖。
                    ts=ts[ts<=max(row['coverage']['supported_source_ticks'])]
                    delta=candidate.source_at(ts)[...,7:]-window.target.source_at(ts)[...,7:]
                    row['consumed_reference_joint_rmse_rad']=float(delta.square().mean().sqrt())
                    emit({'kind':'student','step':step,'sample':i,'noise_level':level,**row})
                    del result
                except (PhysicsRejected, ReferenceExhausted) as error:
                    emit({'kind':'student','step':step,'sample':i,'noise_level':level,'rejection':getattr(error,'reason',str(error))})
        del predictions
    # 小批参数梯度分解：保留同网络、全Stage1 Loss和真实物理链；不执行任何参数更新。
    if args.parameter_gradients:
        named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
        params=[p for _,p in named]
        data_stream=JointDataStream(cfg)
        try:
            data_batch=data_stream.next(args.data_batch,'cpu')
            data_batch,data_contexts,_=attach_supervised_sidecars(data_batch,store,adapter,actor.stage1.endecoder.codec,32,'cpu')
            actor.train(); seed_all(args.seed+1)
            out=actor.denoise_supervised(data_batch,data_contexts,
                torch.randint(1000,(args.data_batch,),generator=generator),
                torch.randn(args.data_batch,60,30,generator=generator),
                torch.rand(args.data_batch,generator=generator)<actor.stage1.music_mask_prob)
            bc,_=data_loss(data_batch,out['pred_x_start'],out['static_conf_logits'],step)
            data_grad=torch.autograd.grad(bc,params,allow_unused=True)
            del out
            actor.eval()
            # 每个来源取第一个预先固定样本，绝不根据Loss/梯度筛选。
            selected=list(range(0,len(samples),args.per_dataset))
            pb={k:(torch.cat([slice_batch(batch,i)[k] for i in selected]) if isinstance(v,torch.Tensor)
                   else [batch['meta'][i] for i in selected] if k=='meta' else len(selected) if k=='B' else v) for k,v in batch.items()}
            pc=[contexts[i] for i in selected]
            po=actor.denoise_supervised(pb,pc,torch.full((len(selected),),100,dtype=torch.long),noise[selected],torch.zeros(len(selected),dtype=torch.bool))
            total=po['pred_x_start'].sum()*0; accepted=0
            for j,i in enumerate(selected):
                try:
                    _,result=evaluate(i,po['pred_x_start'][j:j+1],po['static_conf_logits'][j:j+1])
                    total=total+result['loss'];accepted+=1
                except PhysicsRejected as error:emit({'kind':'parameter_gradient_rejection','sample':i,'rejection':error.reason})
            if accepted:
                total=total/accepted
                physics_grad=torch.autograd.grad(total,params,allow_unused=True)
                emit({'kind':'parameter_gradients','step':step,'data_batch':args.data_batch,'physics_batch':accepted,
                      'data_loss':scalar(bc),'execution_loss':scalar(total),
                      'parameter_gradients':gradient_stats(named,data_grad,physics_grad,cfg['physics']['lambda_exec']),
                      'evidence':'local_small_batch_snapshot_not_exact_8rank_optimizer_update'})
        finally:data_stream.close()
    adapter.bundle.assert_unchanged(frozen)
    summary={'rows':len(records),'elapsed_seconds':time.perf_counter()-started,'frozen_unchanged':True,
             'checkpoints':args.checkpoints,'samples':len(samples),'completed':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--per-dataset',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261031);p.add_argument('--data-batch',type=int,default=32)
    p.add_argument('--parameter-gradients',action='store_true')
    args=p.parse_args()
    if min(args.threads,args.per_dataset,args.data_batch)<1:p.error('线程和样本数必须为正数')
    run(args)


if __name__=='__main__':main()
