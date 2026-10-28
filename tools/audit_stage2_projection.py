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
        return (old.ReferenceProjection if kind=='old' else ReferenceProjection)(control.bundle.spec,control.adapter.bounds,**kw).cuda()
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
    state,_,_=saved(data)
    z={k:v[:args.batch].to('cuda:0') for k,v in state.items()}
    from torch.profiler import profile,ProfilerActivity,record_function
    rows=[]
    def rollout(horizon):
        current=z;score=z['state'].new_zeros(len(z['state']));stats=[]
        for step in range(horizon):
            with record_function('stage2.Adapter_Projection_Actor'):
                proposal=control.propose(current,checkpoint_actor=True)
            with record_function('stage2.WM'):
                nxt=engine.dynamics.step(current['state'],proposal['action'],current['xi'])
            with record_function('stage2.Reward_and_History'):
                current,reward,_,_,continuation,_,_=control.advance(current,proposal,nxt)
            score=score+.99**step*reward
            if hasattr(control.projection,'last_statistics'):stats.append(dict(control.projection.last_statistics))
        with record_function('stage2.BPTT_backward'):(-score.mean()).backward()
        return stats
    for kind in args.implementations:
        control.projection=projection(kind)
        for horizon in args.horizons:
            engine.step.adapter.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            try:
                with frozen_parameters(engine.dynamics),profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False) as prof:
                    stats=rollout(horizon)
                torch.cuda.synchronize()
                events=[dict(name=e.key,cpu_ms=e.cpu_time_total/1000,cuda_ms=e.device_time_total/1000,calls=e.count)
                        for e in prof.key_averages() if e.key.startswith(('stage2.','projection.'))]
                row=dict(implementation=kind,horizon=horizon,batch=len(z['state']),seconds=time.perf_counter()-start,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,events=events,projection_statistics=[small(v) for v in stats])
            except Exception as error:row=dict(implementation=kind,horizon=horizon,error=repr(error))
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='projection_statistics'}),flush=True)
            (out/'projection_profile.json').write_text(json.dumps(rows,indent=2))
    return


if __name__=='__main__':main()
