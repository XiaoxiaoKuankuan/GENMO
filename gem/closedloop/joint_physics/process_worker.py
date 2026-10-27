"""独立推理进程，隔离Newton与生成线程的Python及CUDA上下文同步。

仅用于固定checkpoint的部署评价：子进程读取同一GENMO与冻结SM/WM资产，保留完整
DDIM20/CFG和当前真实快照条件。通过CPU张量传递请求/参考，不传未来真实状态；往返拷贝、
排队和接收耗时计入实际完成时刻。父进程持续推进物理，不能等待生成暂停控制。
启动及数值预热在正式计时前单独记录，并对完整世界参考/前瞻作同输入数值核验。
不用于训练，不与优化器并发改权重；关闭只回收本实例创建的子进程，不触碰其他作业。
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import copy
import multiprocessing as mp
from queue import Empty
import time
import traceback

import torch


def encode_state(state):
    if state.pending or state.teacher_reference is not None:
        raise ValueError('新方法进程推理只接受无pending且无生成教师的真实请求')
    data = {f.name:getattr(state,f.name) for f in fields(state) if f.name!='reference'}
    data = {k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
    ref = state.reference
    data['reference_payload'] = {k:getattr(ref,k).detach().cpu() for k in ('ticks','world_qpos','contact','source_versions')}
    data['reference_payload']['version'] = ref.version
    return data


def decode_state(payload,adapter,device):
    from gem.closedloop.mb_control.contracts import ClosedLoopState
    from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
    data = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in payload.items() if k!='reference_payload'}
    ref = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in payload['reference_payload'].items()}
    data['reference'] = ReferenceBuffer(**ref,kin=adapter.runtime.kin,to_sm=adapter.to_sm)
    return ClosedLoopState(**data)


def _serve(cfg,checkpoint,incoming,outgoing):
    try:
        from .config import build_adapter
        from .evaluation import load_actor
        from .inference import configure,timed_request
        torch.set_num_threads(cfg['runtime']['cpu_threads'])
        device = cfg['runtime']['device']
        adapter = build_adapter(cfg,device)
        actor,_ = load_actor(cfg,checkpoint,device)
        configure(actor,cfg['runtime'].get('inference_backend','eager'))
        frozen = adapter.bundle.frozen_snapshot()
        outgoing.put({'ready':True,'frozen_sha256':adapter.bundle.metadata['sha256']})
        while True:
            message = incoming.get()
            if message is None:break
            snapshot,ticket,noise,warmup = message
            state = decode_state(snapshot,adapter,device)
            noise = noise.to(device)
            began = time.perf_counter()
            for _ in range(3 if warmup else 1):
                plan,context,timing = timed_request(actor,adapter,state,ticket,noise)
            payload = {'world_qpos':plan.world_qpos.cpu(),'contact':plan.contact.cpu(),
                       'features':context.features.cpu(),'valid':context.valid.cpu(),
                       'request_tick':context.request_tick,'forecast_tick':context.forecast_tick,
                       'timings':timing,'child_total_seconds':time.perf_counter()-began}
            outgoing.put(payload)
        adapter.bundle.assert_unchanged(frozen)
    except BaseException:
        outgoing.put({'error':traceback.format_exc()})


class ProcessGenerationWorker:
    def __init__(self,cfg,checkpoint,adapter):
        context = mp.get_context('spawn')
        child_cfg = copy.deepcopy(cfg)
        child_cfg['runtime']['device'] = cfg['runtime'].get('inference_device',cfg['runtime']['device'])
        self.inference_device = child_cfg['runtime']['device']
        self.incoming,self.outgoing=context.Queue(),context.Queue()
        self.process=context.Process(target=_serve,args=(child_cfg,str(checkpoint),self.incoming,self.outgoing),daemon=True)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='genmo-receive')
        self.device,self.future=adapter.runtime.kin.device,None
        begun=time.perf_counter()
        self.process.start()
        ready=self.receive()
        if ready['frozen_sha256']!=adapter.bundle.metadata['sha256']:
            self.close();raise ValueError('进程推理SM/WM资产不一致')
        self.initialization_seconds=time.perf_counter()-begun

    def receive(self):
        while True:
            try:
                message=self.outgoing.get(timeout=1.)
                if 'error' in message:raise RuntimeError(message['error'])
                return message
            except Empty:
                if not self.process.is_alive():raise RuntimeError('推理进程已退出且没有返回有效结果')

    def _result(self,ticket):
        from gem.closedloop.mb_control.contracts import GeneratedPlan,ForecastContext
        reply=self.receive()
        plan=GeneratedPlan(ticket,reply['world_qpos'].to(self.device),reply['contact'].to(self.device))
        context=ForecastContext(reply['features'].to(self.device),reply['valid'].to(self.device),reply['request_tick'],reply['forecast_tick'])
        if plan.world_qpos.is_cuda:torch.cuda.current_stream(self.device).synchronize()
        finished=time.perf_counter()
        timing=reply['timings']
        timing.update(total_seconds=finished-self.started,
                      transport_queue_seconds=max(0.,finished-self.started-reply['child_total_seconds']),
                      worker_backend='separate_process',inference_device=self.inference_device,
                      physics_device=str(self.device))
        return plan,finished,timing,context

    def warmup(self,state,ticket,noise):
        self.started=time.perf_counter()
        self.incoming.put((encode_state(state),ticket,noise.detach().cpu(),True))
        result=self._result(ticket)
        return result,{'initialization_seconds':self.initialization_seconds,'warmup_seconds':time.perf_counter()-self.started,
                       'calls':3,'before_physics_episode':True,'pid':self.process.pid}

    def submit(self,snapshot,ticket,noise,*,request_wall=None):
        if self.future is not None:raise RuntimeError('每环境只有一个在途请求')
        self.started=time.perf_counter() if request_wall is None else request_wall
        self.incoming.put((encode_state(snapshot),ticket,noise.detach().cpu(),False))
        self.future=self.executor.submit(self._result,ticket)

    def poll(self):
        return self.future.result() if self.future is not None and self.future.done() else None

    def clear(self):
        if self.future is None or not self.future.done():raise RuntimeError('未完成的请求不能清除')
        self.future=None

    def close(self):
        if self.future is not None:self.future.result()
        self.incoming.put(None)
        self.process.join(timeout=10.)
        if self.process.is_alive():
            self.process.terminate();self.process.join()
            raise RuntimeError('本实例推理子进程未正常退出，已回收，不能报告正常关闭')
        self.executor.shutdown(wait=True)
        self.incoming.close();self.outgoing.close()
        self.future=None
