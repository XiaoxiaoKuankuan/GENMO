"""在目标GPU比较原实现、前瞻复用和CUDA Graph的完整请求。

使用真实离线历史构造合法在线请求，完整DDIM20/eta0/CFG2.5；每种实现单独预热，逐请求
固定输入/噪声，核对58维前瞻和世界参考数值，再报告P50/P95/P99、吞吐和0.5秒预算。
这里没有并发Newton，不能声称实际控制接管通过；真实墙钟/固定交付闭环由评价CLI执行。
所有结果写入独立新目录，GPU环境、线程数、预热耗时和数值误差均保留。
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--requests',type=int,default=12)
    parser.add_argument('--set',action='append',default=[])
    args = parser.parse_args()
    if args.requests<10:raise ValueError('至少连续10个请求')
    import numpy as np
    import torch
    from gem.closedloop.joint_physics.config import read_config,build_adapter
    cfg = read_config(args.config,args.set)
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    adapter = build_adapter(cfg,cfg['runtime']['device'])
    from gem.closedloop.joint_physics.evaluation import load_actor
    from gem.closedloop.joint_physics.data import SourceCatalog,SidecarStore
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    from gem.closedloop.joint_physics.requests import PhysicsRequestStream
    from gem.closedloop.joint_physics.reference import attach_sidecars
    from gem.closedloop.joint_physics.inference import configure,timed_request
    from gem.closedloop.mb_control.scheduler import Scheduler
    actor,_ = load_actor(cfg,args.checkpoint,cfg['runtime']['device'])
    frozen = adapter.bundle.frozen_snapshot()
    catalog = SourceCatalog(cfg,'train')
    store = SidecarStore(cfg['data']['sidecar'],'train',asset_contract(cfg,adapter),sample_contract=cfg['data']['sample_contract'])
    stream = PhysicsRequestStream(catalog,store,adapter.bundle.spec.joint_names,32,cfg['evaluation']['seed'])
    scheduler = Scheduler(cfg['evaluation']['latency_ticks'])
    requests = []
    for i in range(args.requests):
        batch,meta = stream.next(1,cfg['runtime']['device'])
        _,_,windows,_ = attach_sidecars(batch,store,adapter,actor.stage1.endecoder.codec,32)
        dataset,row = stream.sources[meta[0]['source']]
        music = dataset.reader.load_aligned_sequence(dataset.rows[row])['music'].to(cfg['runtime']['device'])[None]
        state = replace(windows[0].request,music=music,music_end_tick=music.shape[1]*20)
        ticket = scheduler.reserve(state)
        generator = torch.Generator(device=state.state.device).manual_seed(12345+i)
        requests.append((state,ticket,torch.randn(1,60,30,generator=generator,device=state.state.device)))
    output = Path(args.output);output.mkdir(parents=True,exist_ok=False)
    reports,baseline = {},[]
    for backend in ('eager','fast_forecast','cuda_graph'):
        configure(actor,backend)
        begun = time.perf_counter()
        for _ in range(3):timed_request(actor,adapter,*requests[0])
        warmup = time.perf_counter()-begun
        rows=[]
        for i,request in enumerate(requests):
            plan,context,timing = timed_request(actor,adapter,*request)
            if backend=='eager':baseline.append((plan.world_qpos.detach().clone(),context.features.detach().clone(),plan.contact.detach().clone()))
            timing.update(reference_max_abs=float((plan.world_qpos-baseline[i][0]).abs().max()),
                          forecast_max_abs=float((context.features-baseline[i][1]).abs().max()),
                          contact_max_abs=float((plan.contact-baseline[i][2]).abs().max()),
                          prefix_frames=request[1].prefix_frames,
                          forecast_steps=(request[1].first_influence_tick-request[0].tick)//12)
            rows.append(timing)
        quantiles={k:{f'p{p}':float(np.percentile([r[k] for r in rows],p)) for p in (50,95,99)}
                   for k in ('forecast_seconds','generation_decode_seconds','total_seconds')}
        reports[backend]={'rows':rows,'quantiles':quantiles,'warmup_seconds':warmup,
                          'stable_requests_per_second':len(rows)/sum(r['total_seconds'] for r in rows),
                          'within_0_5_seconds':sum(r['total_seconds']<=.5 for r in rows),
                          'parity_passed':all(max(r['reference_max_abs'],r['forecast_max_abs'],r['contact_max_abs'])<=1e-5 for r in rows)}
        (output/'benchmark.json').write_text(json.dumps({'device':str(requests[0][0].state.device),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cpu_threads':torch.get_num_threads(),
            'concurrent_newton':False,'ddim_steps':20,'cfg':2.5,'eta':0.,'precision':'fp32','backends':reports},indent=2))
    adapter.bundle.assert_unchanged(frozen)


if __name__=='__main__':main()
