"""只读复现2026-10-28八卡失败参考，并比较旧/新投影的CUDA计算开销。

从用户指定的正式失败checkpoint读取模型及保存真实/想象起点，不训练、不推进物理，
不覆盖原日志。failures模式记录原证书、新SQP双精度残差、转换FP32残差及最终证书；
profile模式用同一起点和固定checkpoint比较H8/H16/H32的完整闭环BPTT，记录CUDA
profiler、耗时、显存及活跃约束统计。此工具不能提供真实跟踪改善或物理方向证据。
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=('failures','profile'),default='failures')
    parser.add_argument('--gpu-case',type=int,default=4)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--horizons',type=int,nargs='+',default=[8,16,32])
    parser.add_argument('--implementations',nargs='+',default=['old','new'])
    parser.add_argument('--kineto-horizon',type=int,default=0,help='可选代表性短轨迹CUDA profiler；完整时域仅CUDA event和墙钟，避免旧SQP海量事件扭曲耗时')
    args=parser.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(root/'source_supermimic'))
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    from gem.closedloop.reference_adapter.application import build_engine,read_tasks
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl,select_state
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.dynamics import frozen_parameters
    def load(gpu):
        path=next((root/f'gpu{gpu}').glob('full*failure*.pt'))
        return path,torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    _,first=load(args.gpu_case)
    cfg=first['config'];engine=build_engine(cfg,'cuda:0')
    bank=VectorBank(read_tasks(cfg['tasks']),engine,'cuda:0');control=VectorControl(engine,bank)
    control.adapter.disable_root_xy=True;control.soft_joint_limit_cost=False;control.sm_soft_termination_only=True
    calibration=json.loads(Path(cfg['alternating']['calibration']).read_text())
    control.precision=PrecisionObjective(calibration['scales'],**cfg['alternating']['precision'])
    oldpath=root/'source_genmo/gem/closedloop/reference_adapter/reference_constraints.py'
    spec=importlib.util.spec_from_file_location('gem.closedloop.reference_adapter.legacy_constraints',oldpath)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    def projection(kind):
        kw=dict(cfg['alternating']['projection']);kw['constrain_root_angular_acceleration']=True
        if kind=='old':kw.pop('constrain_root_angular_acceleration',None)
        value=(old.ReferenceProjection if kind=='old' else ReferenceProjection)(control.bundle.spec,control.adapter.bounds,**kw).cuda()
        if kind=='old':
            value.acceleration_mask=torch.ones(27,device='cuda:0',dtype=torch.bool)
            value.constrain_root_angular_acceleration=True
            value.last_statistics={}
        return value
    def saved(data):
        extra=data['extra_state'];item=extra.get('imagination_failure')
        if item:
            return item['state'],item['residual'],item['certificate']
        return extra['constraint_state'],extra['constraint_proposal']['residual'],extra['constraint_failure']
    def small(value):
        if isinstance(value,dict):return {k:small(v) for k,v in value.items()}
        if isinstance(value,torch.Tensor):
            z=value.detach().cpu()
            if z.numel()<40:return z.tolist()
            return dict(shape=list(z.shape),min=float(z.float().min()),max=float(z.float().max()),mean=float(z.float().mean()))
        return value
    original_propose=control.propose
    def diagnosed_propose(z,*positional,**keywords):
        try:return original_propose(z,*positional,**keywords)
        except RuntimeError as error:
            if hasattr(error,'bad_mask'):
                with torch.no_grad():
                    raw=control.adapter.forward_packed(control.packed(control.fields(z)))
                    raw[...,:2]=0
                    frames=(z['frame'][:,None]+bank.offset13).clamp_min(0)
                    rotation=bank.pose(z['song'][:,None],frames)[1]
                    evidence=dict(raw=raw.cpu(),left=z['left'].detach().cpu(),anchor=z['anchor'].detach().cpu(),
                        rotation=rotation.cpu(),bad_mask=error.bad_mask.cpu(),bounds=control.adapter.bounds.cpu(),
                        joint_names=control.bundle.spec.joint_names,
                        diagnostic={k:v.detach().cpu() for k,v in control.projection.last_statistics.items()},
                        state={k:v.detach().cpu() for k,v in z.items()})
                torch.save(evidence,out/f'failed_projection_{time.time_ns()}.pt')
            raise
    control.propose=diagnosed_propose
    if args.mode=='failures':
        reports=[];fixtures=[]
        for gpu in range(8):
            path,data=load(gpu);engine.step.adapter.load_state_dict(data['adapter'])
            state,previous,old_certificate=saved(data)
            bad=(~old_certificate['feasible']).nonzero().flatten()
            z={k:v[bad].to('cuda:0') for k,v in state.items()}
            frames=(z['frame'][:,None]+bank.offset13).clamp_min(0)
            rotation=bank.pose(z['song'][:,None],frames)[1]
            with torch.no_grad():raw=control.adapter.forward_packed(control.packed(control.fields(z)))
            raw[...,:2]=0
            control.projection=projection('new')
            torch.cuda.synchronize();start=time.perf_counter()
            try:
                with torch.no_grad():result=control.projection(raw,z['left'],z['anchor'],rotation)
                torch.cuda.synchronize()
                certificate=control.projection.certificate(result,z['left'],z['anchor'],rotation)
                row=dict(gpu=gpu,checkpoint=str(path),failed_indices=bad.tolist(),seconds=time.perf_counter()-start,
                    old_certificate=small({k:v[bad] for k,v in old_certificate.items()}),
                    new_certificate=small(certificate),solver=small(control.projection.last_statistics))
            except Exception as error:
                row=dict(gpu=gpu,checkpoint=str(path),error=repr(error),solver=small(control.projection.last_statistics))
                result=previous[bad].to('cuda:0')
            fixtures.append(dict(gpu=gpu,raw=raw.cpu(),left=z['left'].cpu(),anchor=z['anchor'].cpu(),rotation=rotation.cpu(),
                old_result=previous[bad].cpu(),new_result=result.detach().cpu(),
                solver_solution64=getattr(control.projection,'last_solution64',None),
                state={k:v.cpu() for k,v in z.items()}))
            reports.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            (out/'projection_failures.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False))
        torch.save(dict(cases=fixtures,bounds=control.adapter.bounds.detach().cpu(),joint_names=control.bundle.spec.joint_names),out/'projection_fixtures.pt')
        return
    path,data=load(args.gpu_case);engine.step.adapter.load_state_dict(data['adapter']);engine.dynamics.load_state_dict(data['dynamics'])
    state=data['collector_state']
    z={k:v[:args.batch].to('cuda:0') for k,v in state.items()}
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    from gem.closedloop.reference_adapter.alternating_learning import imagine
    compile_report=compile_kinematics(engine,bank)
    (out/'compiled_fk.json').write_text(json.dumps(compile_report,indent=2))
    from torch.profiler import profile,ProfilerActivity
    rows=[]
    def rollout(horizon):
        result=imagine(control,engine.dynamics,None,z,horizon,terminal_value=False,checkpoint_steps=True)
        with torch.profiler.record_function('stage2.BPTT_backward'):(-result['objective'].mean()).backward()
        return result
    class CudaScope:
        def __init__(self,name,records):
            self.name,self.records=name,records
            self.start=torch.cuda.Event(enable_timing=True);self.end=torch.cuda.Event(enable_timing=True)
        def __enter__(self):self.start.record();return self
        def __exit__(self,*args):self.end.record();self.records.append((self.name,self.start,self.end))
    for kind in args.implementations:
        control.projection=projection(kind)
        # 排除编译及首个CUDA核缓存开销；与正式入口相同的真实起点和checkpoint图。
        with frozen_parameters(engine.dynamics):
            try:rollout(1)
            except RuntimeError:pass
        for horizon in args.horizons:
            engine.step.adapter.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            records=[];original_record=torch.profiler.record_function
            def scope(name,*rest,**kwargs):
                if name.startswith(('stage2.','stage2_','projection.')):return CudaScope(name,records)
                return original_record(name,*rest,**kwargs)
            torch.cuda.synchronize();start=time.perf_counter()
            first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            try:
                torch.profiler.record_function=scope
                first.record()
                with frozen_parameters(engine.dynamics):result=rollout(horizon)
                last.record();torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                finite=bool(torch.stack([p.grad.isfinite().all() for p in control.adapter.parameters() if p.grad is not None]).all())
                norm=float(torch.stack([p.grad.square().sum() for p in control.adapter.parameters() if p.grad is not None]).sum().sqrt())
                timing={}
                for name,a,b in records:
                    row=timing.setdefault(name,dict(name=name,cuda_inclusive_ms=0.,calls=0))
                    row['cuda_inclusive_ms']+=a.elapsed_time(b);row['calls']+=1
                row=dict(implementation=kind,horizon=horizon,batch=len(z['state']),seconds=elapsed,
                         cuda_elapsed_ms=first.elapsed_time(last),gradient_finite=finite,gradient_norm=norm,
                         objective_mean=float(result['objective'].mean()),
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,events=list(timing.values()),
                         projection_statistics=small(result['projection_statistics']),
                         timing_scope='one_formal_BPTT_microbatch_not_1024_sample_update',
                         real_start='same_saved_collector_state',compiled_FK=True,checkpoint_steps=True)
            except Exception as error:row=dict(implementation=kind,horizon=horizon,error=repr(error),
                diagnostic=small(getattr(error,'diagnostic',{})),bad_mask=small(getattr(error,'bad_mask',None)))
            finally:torch.profiler.record_function=original_record
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='projection_statistics'}),flush=True)
            (out/'projection_profile.json').write_text(json.dumps(rows,indent=2))
        if args.kineto_horizon:
            engine.step.adapter.zero_grad(set_to_none=True)
            try:
                with frozen_parameters(engine.dynamics),profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False) as prof:
                    rollout(args.kineto_horizon)
                torch.cuda.synchronize()
                events=[dict(name=e.key,cpu_ms=e.cpu_time_total/1000,cuda_ms=e.device_time_total/1000,calls=e.count)
                        for e in prof.key_averages() if e.key.startswith(('stage2.','stage2_','projection.'))
                            or '_CoupledProjection' in e.key or '_EuclideanProjection' in e.key]
                trace=dict(implementation=kind,horizon=args.kineto_horizon,batch=len(z['state']),events=events,
                           scope='representative_short_CUDA_profiler_not_wall_time_benchmark')
            except Exception as error:trace=dict(implementation=kind,horizon=args.kineto_horizon,error=repr(error))
            (out/f'kineto_{kind}.json').write_text(json.dumps(trace,indent=2))
    return


if __name__=='__main__':main()
