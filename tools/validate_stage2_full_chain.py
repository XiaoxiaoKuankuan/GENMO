"""服务器八卡真实旧rollout的逐去噪步骤DPPO概率、梯度与学习率标定入口。

输入为已有不可变fixed_data_validation目录及当前显式配置。先核对原始文件SHA，
重建冻结Stage1先验并加载采样时Actor，使用原状态/创新噪声/均值/实际标准差/mask，
验证零更新概率、FP64独立Normal公式、完整Autograd与流式微批梯度、多卡SUM。
随后各学习率从相同Actor/Adam/BC/控制器/RNG起点执行有限完整链更新，保存所有
接受、拒绝、裁剪、参数变化、输出变化和耗时；候选之间完整恢复并核验字节。

本入口不创建Isaac世界、不产生新物理交互、不改变旧行为数据，也不把数值通过
当作物理收益。旧rollout的Critic/GAE目标仅用于同数据数值对照，不作为新策略MC。
输出必须是新的独立目录。原rollout、模型及正式日志只读；复用同一个脚本研究不同
LR/Epoch，不复制实验脚本。生产仍通过真实闭环训练和完整音乐配对评估判断质量。

四个诊断开关默认关闭：audit-condition-chunk-boundary在每rank至少1025条真实
有效链上，一次性比较无界条件图与默认1024+1块，保持微批128与全局SUM，单独
记录完整梯度差、loss、重算误差、耗时和显存；样本不足明确拒绝，不以16链代替。
audit-full-gradient-direction复用更新器的整批真实PPO
反向，比较小批次梯度与实际Adam方向，增加反向成本但不改变候选接受规则；
save-probability-records把更新前、每次接受后及最终重新测量的[N,K,8]统计和固定
优势保存到独立子目录，不保存扩散链或改变概率。前四列是每链每步的联合KL、
坐标平均KL、联合log-ratio和自由坐标数；后四个物理变化槽保持未测量标记，不能
把占位零解释为动作不变。export-evaluation-actors只导出
完成全部原计划步骤且生产soft/margin/active-clip与hard KL均通过的Actor。
导出格式仅供load_evaluation_actor重新做物理评价，不含Adam/Critic/RNG，绝不能
用作继续训练的checkpoint。全部候选仍从相同学习状态开始，结束或异常均恢复起点。
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
import torch.distributed as dist

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.execution_checks import policy_phase
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.optimization_support import _parameters,_old_kernel,_owned,probability_check_local
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.rollback_audit import compare_recovered_state
from gem.closedloop.dppo.run_control import build_policy
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.updater import actor_update,final_kl
from gem.closedloop.dppo.actor_backtracking import check_candidate_kl


RECORD_COLUMNS=('joint_kl','mean_coordinate_kl','joint_log_ratio','free_count',
    'normalized_mean_change_rms','root_mean_change_rms','orientation_mean_change_rms','joint_mean_change_rms')


def save_new_tensor_payload(path,payload):
    """仅发布本诊断新文件；临时文件失败即清理，不覆盖原证据或半成品。"""
    path=Path(path);partial=path.with_suffix(path.suffix+'.partial')
    if path.exists() or partial.exists():raise FileExistsError(path)
    created=False
    try:
        with partial.open('xb') as stream:
            created=True
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.link(partial,path)
    finally:
        if created:partial.unlink(missing_ok=True)
    return dict(path=str(path),sha256=file_sha256(path),size_bytes=path.stat().st_size)


class ProbabilityRecordWriter:
    """rank0持久化原全局链顺序的小型概率统计，明确首步、接受步与最终边界。"""
    def __init__(self,directory,*,case_sha256,manifest,trainable_steps):
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.case_sha256=case_sha256;self.steps=list(trainable_steps)
        self.manifest=[dict(global_manifest_index=i,**m) for i,m in enumerate(manifest) if m['valid'] and m['has_free']]
        self.entries=[];self.advantages=None;self.sources=None

    def observe(self,stage,records,advantages,sources):
        if stage not in ('before_update','accepted_Adam','final'):
            raise ValueError('Unknown full-rollout probability observation stage')
        if ((not self.entries and stage!='before_update') or
                self.entries and (stage=='before_update' or self.entries[-1]['stage']=='final')):
            raise ValueError('Probability observations must begin once and finish once')
        records=records.detach().cpu().clone();advantages=advantages.detach().cpu().clone()
        if (records.shape!=(len(self.manifest),len(self.steps),8) or records.dtype!=torch.float64
                or advantages.shape!=(len(self.manifest),) or not torch.isfinite(records).all()
                or not torch.isfinite(advantages).all() or len(sources)!=len(self.manifest)):
            raise ValueError('Probability record shape, finite values or chain order differs')
        if self.advantages is None:
            self.advantages=advantages.clone();self.sources=list(sources)
        elif not torch.equal(self.advantages,advantages) or self.sources!=list(sources):
            raise ValueError('Fixed advantages or source order changed between diagnostic observations')
        accepted=sum(e['stage']=='accepted_Adam' for e in self.entries)+(stage=='accepted_Adam')
        path=self.directory/f'{len(self.entries):03d}_{stage}.pt'
        artifact=save_new_tensor_payload(path,dict(schema='genmo.stage2.full_chain_probability_records.v1',
            stage=stage,accepted_optimizer_steps_observed=accepted,source_case_sha256=self.case_sha256,
            trainable_steps=self.steps,columns=list(RECORD_COLUMNS),records=records,advantages=advantages,
            sources=self.sources,global_manifest=self.manifest,
            physical_change_columns_measured=False,
            scope='fixed_behavior_full_rollout_conditional_Gaussian_statistics_not_physical_return'))
        self.entries.append(dict(stage=stage,accepted_optimizer_steps_observed=accepted,
            **{**artifact,'path':path.name}))
        summary=dict(source_case_sha256=self.case_sha256,trainable_steps=self.steps,
            chains=len(self.manifest),entries=self.entries,complete=stage=='final')
        atomic_json(self.directory/'index.json',summary)
        return dict(directory=str(self.directory),**summary)


@torch.no_grad()
def observe_final_probability(policy,rows,cache,manifest,distributed,microbatch,observer):
    """最终模型另做一次只读整批前向，包含提前停步后的真实保留状态。"""
    selected=[i for i,m in enumerate(manifest) if m['valid'] and m['has_free']]
    owned=_owned(selected,manifest,distributed);local=[rows[i] for _,i in owned]
    def evaluate():
        with policy_phase(policy):
            records=evaluate_chain_records(policy,local,cache,microbatch=microbatch)
            advantages=cache.get('advantages',local).double() if local else torch.zeros(
                0,device=distributed.device,dtype=torch.float64)
        return records,advantages
    records,advantages=local_call(distributed,evaluate)
    width=len(policy.trainable_steps)*8
    full=distributed.gather_rows(torch.cat((records.reshape(len(local),width),advantages[:,None]),1),
        [position for position,_ in owned],len(selected))
    labels=[(position,rows[i].metadata.get('training_task',{}).get('dataset','unknown')) for position,i in owned]
    names={position:name for shard in distributed.all_gather_object(labels) for position,name in shard}
    return observer('final',full[:,:width].reshape(-1,len(policy.trainable_steps),8),full[:,width],
        [names[i] for i in range(len(selected))])


def export_evaluation_actor(path,*,actor,kernel,source_identity,case_sha256,report,hard,acceptance,
                            learning_rate,epochs,clip):
    """只发布完整且满足现行soft/hard KL的Actor；未通过时仅返回原因，不写模型。"""
    steps=report.get('optimizer_steps');planned=report.get('planned_optimizer_steps')
    if (type(steps)is not int or type(planned)is not int or steps<1 or steps!=planned
            or len(report.get('steps',[]))!=steps):
        return dict(exported=False,reason='incomplete_planned_optimizer_update')
    if hard.get('passed') is not True or acceptance.get('passed') is not True:
        return dict(exported=False,reason='production_soft_or_hard_KL_failed')
    if (source_identity.get('kernel')!=kernel or not source_identity.get('assets')
            or not isinstance(case_sha256,str) or len(case_sha256)!=64
            or any(c not in '0123456789abcdef' for c in case_sha256)):
        raise ValueError('Evaluation export requires bound behavior kernel/assets and immutable case SHA')
    from gem.closedloop.dppo.checkpoint import _validate_model_state
    from gem.closedloop.dppo.evaluation import _model_fingerprint
    state=immutable_cpu_snapshot(actor.state_dict());_validate_model_state(actor,state,'evaluation-only Actor')
    update=dict(optimizer_steps=steps,planned_optimizer_steps=planned,learning_rate=learning_rate,
        epochs=epochs,clip=clip,actor_fingerprint=_model_fingerprint(actor),
        source='fixed_real_full_chain_validation_only')
    artifact=save_new_tensor_payload(path,dict(schema='genmo.stage2.asymmetric_actor_evaluation.v1',
        evaluation_only=True,training_resume=False,actor=state,kernel=copy.deepcopy(kernel),
        source_identity=copy.deepcopy(source_identity),source_case_sha256=case_sha256,
        hard_kl=copy.deepcopy(hard),complete_update_and_soft_kl_passed=True,
        production_candidate_checks=copy.deepcopy(acceptance),update=update))
    return dict(exported=True,evaluation_only=True,training_resume=False,update=update,**artifact)


def candidate_update_checks(report,kl,config):
    """分开报告硬KL和正式候选KL；PPO目标变化仅作诊断，不作为接受条件。

    LR试探可显式启用回溯，仍可能KL通过但PPO目标下降或未覆盖完整轮次。按生产原有margin、
    soft KL与active clip重新判读；这里只解释结果，不改变本次真实更新。
    """
    hard=check_candidate_kl(kl,config['stage10']['kl'])
    settings=config['stage9'];trust=settings['trust_region']
    limits={k:v*trust['margin'] for k,v in config['stage10']['kl'].items() if k.startswith('denoising_step_')}
    limits['denoising_step_mean']=min(limits['denoising_step_mean'],settings['kl_soft_stop_denoising_step']*trust['margin'])
    limits['maximum_active_clip']=settings['max_active_clip_fraction']
    eligible=check_candidate_kl(kl,limits)
    if report['optimizer_steps']<1:eligible['failures']['no_optimizer_step']=True
    for step in report['steps']:
        after=step.get('full_rollout_after')
        if after is not None:
            step_check=check_candidate_kl(after,limits)
            if step_check['failures']:eligible['failures'][f'step_{step["optimizer_step"]}']=step_check['failures']
    eligible['passed']=not eligible['failures']
    eligible['scope']='existing_production_conditions_diagnostic_only_no_formal_step_committed'
    return hard,eligible


@torch.no_grad()
def rejected_probability_diagnostics(policy,rows,cache,manifest,distributed,microbatch):
    """高LR指数溢出时只报告真实log-ratio/KL，不截断概率、不把拒绝更新当作成功。"""
    from gem.closedloop.dppo.gaussian_probability import distribution
    selected=[rows[m['local_index']] for m in manifest
        if m['owner_rank']==distributed.rank and m['valid'] and m['has_free']]
    with policy_phase(policy):
        records=evaluate_chain_records(policy,selected,cache,microbatch=microbatch)
    local=torch.stack((records[:,:,0].reshape(-1),records[:,:,2].reshape(-1)),1).cpu()
    full=torch.cat(distributed.all_gather_object(local),0)
    return dict(denoising_step_kl=distribution(full[:,0]),denoising_step_log_ratio=distribution(full[:,1]),
        gaussian_steps=list(policy.trainable_steps),chains=len(full),
        ratio_underflow_fraction=float((full[:,1]<math.log(float.fromhex('0x0.0000000000001p-1022'))).double().mean()),
        ratio_overflow_fraction=float((full[:,1]>math.log(torch.finfo(torch.float64).max)).double().mean()),
        accepted=False,scope='unclamped_Gaussian_statistics_after_rejected_Adam_attempt')


def gradient_validation(policy,rows,cache,distributed):
    """本地计算先协作传播异常，再执行SUM；独立Normal公式和2e-5门槛不变。"""
    actor=policy.actor;device=distributed.device
    try:
        from gem.closedloop.dppo.executed_action_mask import action_mask
        selected=local_call(distributed,lambda:[row for row in rows if action_mask(row).any()][:2])
        denominator=sum(distributed.all_gather_object(len(selected)))
        actor.zero_grad(set_to_none=True)
        def independent_backward():
            with policy_phase(policy):
                terms=[]
                for step in policy.trainable_steps:
                    params,mask=_parameters(policy,selected,[step]*len(selected),device,cache,executed_only=True)
                    old_mean,old_std=_old_kernel(selected,[step]*len(selected),device,cache)
                    value=cache.get('chain',selected,[step+1]*len(selected)).double()
                    a=torch.distributions.Normal(params['mean'].double(),params['std'].double())
                    b=torch.distributions.Normal(old_mean,old_std)
                    terms.append((a.log_prob(value)-b.log_prob(value)).masked_fill(~mask,0.).sum((-2,-1)))
                log=torch.stack(terms,1);adv=cache.get('advantages',selected).double()[:,None]
                loss=-torch.minimum(log.exp()*adv,log.exp().clamp(.9,1.1)*adv).sum()/(denominator*len(policy.trainable_steps))
                loss.backward()
        local_call(distributed,independent_backward)
        distributed.sum_gradients(actor)
        reference=local_call(distributed,lambda:{n:p.grad.detach().clone()
            for n,p in actor.named_parameters() if p.grad is not None})
        variants=[]
        # 同一独立Normal参考额外核验条件图分块；每rank两条真实链可实际触发1+1尾块。
        # None保留原无界执行参考，1024对应正式默认；此小样本不替代全批量显存测量。
        for micro,condition_chunk in ((1,None),(3,1024),(8,1024),(128,None),(128,1)):
            actor.zero_grad(set_to_none=True)
            def streamed_backward():
                with policy_phase(policy):
                    records=evaluate_chain_records(policy,selected,cache,microbatch=micro)
                    return backward_chain_surrogate(policy,selected,cache,records,clip=.1,
                        global_chain_count=denominator,microbatch=micro,
                        condition_chunk_chains=condition_chunk)
            streamed=local_call(distributed,streamed_backward)
            distributed.sum_gradients(actor)
            def compare():
                current={n:p.grad for n,p in actor.named_parameters() if p.grad is not None}
                if set(current)!=set(reference):raise AssertionError('Streamed gradient presence differs')
                delta=sum((current[n].double()-reference[n].double()).square().sum() for n in reference)
                scale=sum(g.double().square().sum() for g in reference.values())
                relative=float((delta/scale).sqrt())
                maximum=max(float((current[n]-reference[n]).abs().max()) for n in reference)
                result=dict(relative_gradient_l2=relative,maximum_coordinate_error=maximum,
                    recompute_max_error=float(streamed['recompute_max_error']))
                if not all(math.isfinite(value) for value in result.values()) or relative>2e-5:
                    raise AssertionError('Real GPU streamed gradient mismatch: '+repr(result))
                return result
            comparisons=distributed.all_gather_object(local_call(distributed,compare))
            variants.append(dict(microbatch=micro,condition_chunk_chains=condition_chunk,
                **{key:max(value[key] for value in comparisons) for key in comparisons[0]}))
        return dict(passed=True,global_chains=denominator,variants=variants,
            reference=f'independent_FP64_Normal_full_{len(policy.trainable_steps)}_step_Autograd_global_SUM',
            scope='actual_stage1_network_and_real_behavior_rows_not_physics_quality')
    finally:
        actor.zero_grad(set_to_none=True)


def condition_chunk_boundary_validation(policy,rows,cache,manifest,distributed):
    """每rank真实1025链比较默认块边界；不更新参数/RNG，异常也清空诊断梯度。

    两次反向共用同一份概率统计、完整链优势和全局分母，均在结束后执行一次SUM。
    无界参考梯度移到CPU，避免额外GPU参考副本污染第二次峰值；每次峰值仅描述
    本工具的反向与SUM，包含已有模型/cache常驻，不含PhysX或候选实际Adam。
    参数/buffer指纹及Python/NumPy/Torch/CUDA RNG前后核对，失败不放宽2e-5门槛。
    """
    from gem.closedloop.dppo.evaluation import _model_fingerprint
    from gem.closedloop.dppo.rollback_audit import exact_state_digest
    actor=policy.actor
    actor.zero_grad(set_to_none=True)
    selected=[i for i,m in enumerate(manifest) if m['valid'] and m['has_free']]
    owned=_owned(selected,manifest,distributed)
    available=distributed.all_gather_object(len(owned))
    if any(count<1025 for count in available):
        raise ValueError('Condition boundary audit requires at least 1025 eligible chains per rank: '+repr(available))
    local=[rows[i] for _,i in owned[:1025]]
    counts=distributed.all_gather_object(len(local));denominator=sum(counts)
    device=torch.device(distributed.device)
    started=time.perf_counter()
    rng=local_call(distributed,capture_local_rng)
    before=local_call(distributed,lambda:dict(actor=_model_fingerprint(actor),rng=exact_state_digest(rng)))
    reference=None;reference_loss=None;variants=[]
    try:
        def collect():
            with policy_phase(policy):return evaluate_chain_records(policy,local,cache,microbatch=128)
        records=local_call(distributed,collect)
        for condition_chunk in (None,1024):
            actor.zero_grad(set_to_none=True)
            if device.type=='cuda':
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            phase_start=time.perf_counter()
            def backward():
                with policy_phase(policy):
                    return backward_chain_surrogate(policy,local,cache,records,clip=.1,
                        global_chain_count=denominator,microbatch=128,condition_chunk_chains=condition_chunk)
            streamed=local_call(distributed,backward)
            distributed.sum_gradients(actor)
            loss=float(distributed.sum_tensor(streamed['loss']))
            if device.type=='cuda':torch.cuda.synchronize(device)
            measurements=distributed.all_gather_object(dict(wall_seconds=time.perf_counter()-phase_start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                recompute_max_error=float(streamed['recompute_max_error'])))
            if reference is None:
                reference=local_call(distributed,lambda:{name:p.grad.detach().cpu().clone()
                    for name,p in actor.named_parameters() if p.grad is not None})
                reference_loss=loss
                comparison=dict(relative_gradient_l2=0.,maximum_coordinate_error=0.)
            else:
                def compare():
                    current={name:p.grad for name,p in actor.named_parameters() if p.grad is not None}
                    if set(current)!=set(reference):raise AssertionError('Condition boundary gradient presence differs')
                    numbers=torch.zeros(3,dtype=torch.float64,device=device)
                    for name,gradient in current.items():
                        expected=reference[name].to(device=device,dtype=torch.float64)
                        difference=gradient.double()-expected
                        numbers[0]+=difference.square().sum();numbers[1]+=expected.square().sum()
                        numbers[2]=torch.maximum(numbers[2],difference.abs().max())
                    delta,scale,maximum=numbers.cpu().tolist()
                    relative=math.sqrt(delta/scale) if scale>0 else 0. if delta==0 else float('inf')
                    return dict(relative_gradient_l2=relative,maximum_coordinate_error=maximum)
                comparisons=distributed.all_gather_object(local_call(distributed,compare))
                comparison={key:max(value[key] for value in comparisons) for key in comparisons[0]}
            variant=dict(condition_chunk_chains=condition_chunk,chains_per_condition_block=
                [1025] if condition_chunk is None else [1024,1],microbatch=128,loss=loss,
                absolute_loss_difference=abs(loss-reference_loss),**comparison,
                recompute_max_error=max(value['recompute_max_error'] for value in measurements),
                wall_seconds=max(value['wall_seconds'] for value in measurements),
                peak_allocated_bytes=max(value['peak_allocated_bytes'] for value in measurements) if device.type=='cuda' else None,
                peak_reserved_bytes=max(value['peak_reserved_bytes'] for value in measurements) if device.type=='cuda' else None,
                measurements_by_rank=measurements)
            variants.append(variant)
            if (not math.isfinite(variant['relative_gradient_l2']) or variant['relative_gradient_l2']>2e-5
                    or not math.isclose(loss,reference_loss,rel_tol=1e-10,abs_tol=1e-10)
                    or variant['recompute_max_error']>1e-4):
                raise AssertionError('Real GPU 1024-chain boundary mismatch: '+repr(variant))
        after=local_call(distributed,lambda:dict(actor=_model_fingerprint(actor),
            rng=exact_state_digest(capture_local_rng())))
        unchanged=distributed.all_gather_object(dict(before=before,after=after,identical=before==after))
        if not all(value['identical'] for value in unchanged):
            raise AssertionError('Condition boundary audit changed Actor parameters/buffers or RNG')
        return dict(enabled=True,passed=True,available_chains_by_rank=available,selected_chains_by_rank=counts,
            global_chains=denominator,trainable_steps=list(policy.trainable_steps),variants=variants,
            relative_gradient_l2_tolerance=2e-5,state_unchanged_by_rank=unchanged,
            wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
            reference='same_records_unbounded_condition_graph_complete_gradient_global_SUM',
            scope='1024_plus_1_real_chain_boundary_not_independent_Normal_or_physical_quality',
            memory_scope='backward_and_SUM_PyTorch_peak_includes_existing_model_cache_excludes_PhysX_and_Adam')
    finally:
        actor.zero_grad(set_to_none=True)
        restore_local_rng(rng)


def argument_parser():
    """仅构造CLI便于CPU核验；解析help或可选开关不会初始化分布式/CUDA。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/stage2_terminal5_executed_server2.yaml')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--learning-rates',type=float,nargs='+',default=[1e-6,3e-6,1e-5])
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--max-optimizer-steps',type=int,help='有限固定数据比较可限制优化步数，保留实际完整链Mini-batch')
    parser.add_argument('--reset-actor-adam',action='store_true',help='模拟新逐步DPPO迁移时重置旧联合目标Adam矩')
    parser.add_argument('--adaptive-kl',action='store_true',help='逐去噪步骤KL回溯；各LR为候选上限而非承诺接受步长')
    parser.add_argument('--audit-condition-chunk-boundary',action='store_true',
        help='一次性用每rank1025条有效真实链验证默认1024+1条件块；样本不足明确失败')
    parser.add_argument('--audit-full-gradient-direction',action='store_true',
        help='每接受步额外计算整批PPO真实梯度和Adam方向，不改变候选接受规则')
    parser.add_argument('--save-probability-records',action='store_true',
        help='保存更新前/每接受步/最终整批轻量概率和固定优势；最终额外只读前向')
    parser.add_argument('--export-evaluation-actors',action='store_true',
        help='仅完整计划更新且生产soft/hard KL通过时导出只供评价的Actor')
    return parser


def main():
    parser=argument_parser()
    args=parser.parse_args()
    if ((args.save_probability_records or args.export_evaluation_actors)
            and len({f'{lr:.8g}' for lr in args.learning_rates})!=len(args.learning_rates)):
        parser.error('概率/Actor新文件要求各学习率生成不同的候选文件名')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8 or rank!=local:raise ValueError('Requires one explicitly allocated eight-GPU node')
    dist.init_process_group('gloo',timeout=timedelta(minutes=30));group=None;cache=None;restore_initial=None
    try:
        check=[None]
        if rank==0:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            check[0]=dict(ok=not busy and not args.output_dir.exists(),busy=busy)
        dist.broadcast_object_list(check,src=0)
        if not check[0]['ok']:raise RuntimeError('Fresh output directory and free GPUs required: '+repr(check[0]))
        torch.cuda.set_device(local);torch.set_num_threads(4)
        group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
        distributed=DistributedCollectives(rank,world,tensor_group=group,device=f'cuda:{local}')
        root_call(distributed,lambda:args.output_dir.mkdir(parents=True,exist_ok=False))
        config=configuration(args.config,genmo_repo=ROOT)
        config['runtime']['genmo_device']=str(distributed.device)
        config['stage9']['bc_batch']=config['stage9']['bc_batch']//world
        config['stage9']['seed']+=1000003*rank
        actor,train,loading=local_call(distributed,lambda:load_actor(config))
        policy=build_policy(actor,config['stage9'],config['stage10']['precision'])
        header=json.loads((args.case/'case.json').read_text())
        learning_path=args.case/'learning_state.pt'
        if file_sha256(learning_path)!=header['learning_state_sha256']:raise ValueError('Behavior Actor file SHA differs')
        learning=torch.load(learning_path,map_location='cpu',weights_only=False,mmap=True)
        actor.load_state_dict(learning['actor'])
        payload=local_call(distributed,lambda:load_rollout(args.case/f'rank{rank:02d}'))
        if payload['kernel']!=policy.kernel_config:
            raise ValueError('Actual current Gaussian behavior kernel differs from the saved real rollout')
        rows,targets,manifest=payload['transitions'],payload['targets'],payload['global_manifest']
        cache=RolloutTensorCache(rows,targets,distributed.device,max_device_bytes=4*1024**3)
        probability=probability_check_local(policy,rows,global_manifest=manifest,distributed=distributed,
            denoising_microbatch=128,tensor_cache=cache,include_frozen=True)
        # 新加入的训练时刻沿用原真实噪声；记录实际sigma，不能把floor当成全部噪声。
        first=[row for row in rows if row.free_mask.any()][:1]
        local_std={str(step):float(cache.get('old_stds',first,[step]).double().mean())
            for step in policy.trainable_steps} if first else {}
        probability['actual_std_first_eligible_chain_by_rank']=distributed.all_gather_object(local_std)
        probability['actual_std_scope']='normalized_Gaussian_sigma_not_floor_not_trajectory_diversity'
        gradients=gradient_validation(policy,rows,cache,distributed)
        condition_boundary=(condition_chunk_boundary_validation(policy,rows,cache,manifest,distributed)
            if args.audit_condition_chunk_boundary else None)
        root_call(distributed,lambda:atomic_json(args.output_dir/'probability_and_gradient.json',dict(
            probability=probability,gradients=gradients,condition_chunk_boundary=condition_boundary)))
        optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=args.learning_rates[0],weight_decay=0.)
        optimizer.load_state_dict(learning['actor_optimizer'])
        if args.reset_actor_adam:optimizer.state.clear()
        controller=BCGradientController(**config['stage10']['bc_gradient'])
        from gem.closedloop.dppo.continuation import DisabledBC
        anchor=(Stage1RetentionAnchor(config,actor,train,source_global_step=loading['source_global_step'])
            if config['stage9'].get('bc_enabled',True) else DisabledBC())
        controller.load_state_dict(learning['controller'])
        rank_state=torch.load(args.case/f'rank{rank:02d}'/'rank_state.pt',map_location='cpu',weights_only=False)
        generators=dict(actor=torch.Generator(),critic=torch.Generator())
        anchor.load_state_dict(rank_state['bc'])
        restore_local_rng(rank_state['rng'],generators)
        def snapshot():return immutable_cpu_snapshot(dict(actor=actor.state_dict(),optimizer=optimizer.state_dict(),
            controller=controller.state_dict(),bc=anchor.state_dict(),rng=capture_local_rng(generators)))
        initial=snapshot();trials=[]
        def restore_initial():
            actor.load_state_dict(initial['actor']);optimizer.load_state_dict(copy.deepcopy(initial['optimizer']))
            controller.load_state_dict(initial['controller']);anchor.load_state_dict(initial['bc'])
            restore_local_rng(initial['rng'],generators)
        case_sha256=file_sha256(args.case/'case.json')
        for lr in args.learning_rates:
            restore_initial()
            audit=compare_recovered_state(initial,snapshot())
            if not audit['identical']:raise AssertionError('Candidate start restore failed: '+repr(audit))
            optimizer.param_groups[0]['lr']=lr
            candidate_stem=f'candidate_lr{lr:.8g}_epoch{args.epochs}'
            writer=None;probability_artifact=None
            if args.save_probability_records:
                def create_writer():
                    nonlocal writer
                    writer=ProbabilityRecordWriter(args.output_dir/f'{candidate_stem}_probability',
                        case_sha256=case_sha256,manifest=manifest,trainable_steps=policy.trainable_steps)
                root_call(distributed,create_writer)
            def observe_probability(stage,records,advantages,sources):
                nonlocal probability_artifact
                probability_artifact=root_call(distributed,lambda:writer.observe(stage,records,advantages,sources))
                return probability_artifact
            started=time.perf_counter();torch.cuda.reset_peak_memory_stats(distributed.device)
            try:
                trust=copy.deepcopy(config['stage9']['trust_region']) if args.adaptive_kl else None
                if trust is not None:trust['maximum_lr']=lr
                report=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,distributed=distributed,
                    anchor=anchor,generator=generators['actor'],ppo_epochs=args.epochs,minibatch_chains=config['stage10']['scale']['actor_minibatch_chains'],
                    sampled_steps=len(policy.trainable_steps),step_strategy=config['stage9']['ppo_step_strategy'],
                    max_optimizer_steps=args.max_optimizer_steps,audit_full_rollout_objective=True,
                    audit_full_gradient_direction=args.audit_full_gradient_direction,
                    full_rollout_observer=observe_probability if args.save_probability_records else None,
                    detailed_diagnostics=True,
                    microbatch=config['stage9']['denoising_microbatch'],clip=config['stage9']['ppo_clip'],soft_kl=config['stage9']['kl_soft_stop_denoising_step'],
                    bc_samples_per_rollout=config['stage9']['bc_samples_per_rollout'],
                    bc_min_global_batch=config['stage9']['bc_min_global_batch'],
                    bc_enabled=config['stage9'].get('bc_enabled',True),
                    max_active_clip=config['stage9']['max_active_clip_fraction'],
                    trust_region=trust,kl_limits=config['stage10']['kl'])
                kl=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=128,clip=config['stage9']['ppo_clip'])
                hard,acceptance=candidate_update_checks(report,kl,config)
                if args.save_probability_records:
                    observe_final_probability(policy,rows,cache,manifest,distributed,
                        config['stage9']['denoising_microbatch'],observe_probability)
                report['accepted_optimizer_steps']=0
                report['diagnostic_optimizer_steps']=report['optimizer_steps']
                trial=dict(learning_rate=lr,epochs=args.epochs,actor=report,kl=kl,acceptance=acceptance,
                    hard_kl_acceptance=hard,formal_optimizer_steps_committed=0,
                    start_restore=audit,wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
                    peak_allocated_bytes=max(distributed.all_gather_object(torch.cuda.max_memory_allocated(distributed.device))))
                if args.export_evaluation_actors:
                    from gem.closedloop.dppo.evaluation import _model_fingerprint
                    fingerprints=distributed.all_gather_object(local_call(distributed,lambda:_model_fingerprint(actor)))
                    if len(set(fingerprints))!=1:raise RuntimeError('Evaluation candidate Actor fingerprints differ across ranks')
                    trial['evaluation_actor']=root_call(distributed,lambda:export_evaluation_actor(
                        args.output_dir/f'{candidate_stem}_evaluation_actor.pt',actor=actor,kernel=policy.kernel_config,
                        source_identity=learning['identity'],case_sha256=case_sha256,report=report,hard=hard,
                        acceptance=acceptance,learning_rate=lr,epochs=args.epochs,clip=config['stage9']['ppo_clip']))
            except (FloatingPointError,RuntimeError) as error:
                message=str(error)
                if not any(word in message.lower() for word in ('ppo','nonfinite','overflow','underflow','streaming')):raise
                trial=dict(learning_rate=lr,epochs=args.epochs,acceptance=dict(passed=False,
                    failures={'numerical':message}),start_restore=audit,
                    hard_kl_acceptance=None,formal_optimizer_steps_committed=0,
                    rejected_probability=rejected_probability_diagnostics(policy,rows,cache,manifest,distributed,
                        config['stage9']['denoising_microbatch']),
                    wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))
                if args.export_evaluation_actors:
                    trial['evaluation_actor']=dict(exported=False,reason='numerical_candidate_rejected')
                if args.save_probability_records:
                    observe_final_probability(policy,rows,cache,manifest,distributed,
                        config['stage9']['denoising_microbatch'],observe_probability)
            if args.save_probability_records:trial['probability_records']=probability_artifact
            trials.append(trial)
            root_call(distributed,lambda:atomic_json(args.output_dir/f'candidate_lr{lr:.8g}_epoch{args.epochs}.json',trial))
            if rank==0:print(json.dumps(dict(lr=lr,kl_passed=trial['acceptance']['passed'],
                completed_planned_steps=(trial.get('actor',{}).get('optimizer_steps',0)==
                    trial.get('actor',{}).get('planned_optimizer_steps',-1)),
                steps=trial.get('actor',{}).get('optimizer_steps'),
                step_kl=trial.get('kl',{}).get('denoising_step_kl'),
                failures=trial['acceptance']['failures']),ensure_ascii=False),flush=True)
        restore_initial()
        restored=compare_recovered_state(initial,snapshot())
        if not restored['identical']:raise AssertionError(restored)
        root_call(distributed,lambda:atomic_json(args.output_dir/'summary.json',dict(status='passed',
            scope='fixed_real_data_math_and_KL_calibration_not_physical_acceptance',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            case=str(args.case),case_sha256=file_sha256(args.case/'case.json'),
            probability=probability,gradient_validation=gradients,condition_chunk_boundary=condition_boundary,
            trials=trials,final_restore=restored)))
        restore_initial=None
    finally:
        try:
            if restore_initial is not None:
                restore_initial()
                restored=compare_recovered_state(initial,snapshot())
                if not restored['identical']:raise AssertionError('Diagnostic final restore failed: '+repr(restored))
        finally:
            if cache is not None:cache.close()
            if group is not None:dist.destroy_process_group(group)
            dist.destroy_process_group()


if __name__=='__main__':main()
