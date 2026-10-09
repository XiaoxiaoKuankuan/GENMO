"""GPU多环境训练的隔离评估世界，复用原固定任务、奖励、报告和模型选择逻辑。

训练世界保持原PhysX状态不动，每rank另外启动一个GPU评估世界，按固定验证任务
逐条执行；八张卡仍并行评估各自分片。评估明确使用N=1和训练相同的部署时钟，
不把较小评估batch的耗时当成训练吞吐，不与旧CPU基线混合。Stage1和后续模型共用
这一评估协议、固定起点、种子及明确的延迟预算。新v3固定profile，不使用训练N
重标定延迟；旧v2仍显式使用实测关键路径。世界和环境两层日志都先落盘再
ACK，世界日志等待从critical时间排除。退出时恢复训练backend对象，评估不重置
训练物理状态、音乐游标或预算；原评估器负责恢复模型模式、梯度和所有训练RNG。
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import yaml
import torch
from .parallel_support import local_call
from .run_management import GuardedStepJournal
from .vector_collector import VectorWorldClient, VectorLaneBackend
from gem.runtime.closedloop_protocol import RemoteError


class SingleVectorLaneTransport:
    def __init__(self, world):
        self.world, self.sequence, self.last_call_timing = world, 0, {}

    def call(self, method, **payload):
        if method=='verify_frozen':
            self.last_call_timing={}
            return self.world.call(method,**payload)
        self.sequence += 1
        key = str(self.sequence)
        result = self.world.call('exchange',requests=[dict(env_id=0,method=method,payload=payload,request_id=key)])
        self.last_call_timing = dict(nested_journal_seconds=self.world.last_call_timing['journal_seconds'])
        if result.get('fatal'):raise RuntimeError(str(result['fatal']))
        replies = [r for r in result['replies'] if r['request_id']==key]
        if len(replies)!=1:raise RuntimeError('Isolated N=1 evaluation unexpectedly deferred')
        if not replies[0]['ok']:raise RemoteError(replies[0]['error'])
        return replies[0]['result']

    def close(self):pass


@contextmanager
def isolated_vector_evaluation(c,label):
    from tools.eval.run_closedloop_baseline import Workers
    path = c.session/'phases'/f'vector_eval_backend_{label}'/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True,exist_ok=False)
    config = copy.deepcopy(c.config)
    from .vector_devices import bind_vector_device
    device_environment=bind_vector_device(config,c.distributed.rank)
    config['runtime'].update(num_envs=1,asset_conversion_dir=str(path/'usd_assets'))
    config_path=path/'config.yaml';config_path.write_text(yaml.safe_dump(config,allow_unicode=True))
    workers=Workers(config,path)
    old_backend=c.backend
    world=journal=None
    try:
        socket=Path(workers.temp.name)/'eval.sock'
        client=local_call(c.distributed,lambda:workers.start('gmt',[config['paths']['isaac_python'],'-B',
            str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt_vector.py'),
            '--config',str(config_path),'--socket',str(socket),'--headless'],config['paths']['gmt_repo'],socket,
            strip_distributed=True,environment=device_environment))
        def check_device():
            if workers.entries[0]['identity'].get('gpu_uuid')!=str(torch.cuda.get_device_properties(c.distributed.device).uuid):
                raise RuntimeError('Evaluation world is on a different physical GPU')
        local_call(c.distributed,check_device)
        journal=GuardedStepJournal(path/'world.sqlite',c.guard,format='genmo.execution_journal.ndarray.v2')
        world=VectorWorldClient(client,journal,socket_path=socket)
        c.backend=VectorLaneBackend(SingleVectorLaneTransport(world),None)
        yield
    finally:
        c.backend=old_backend
        def close():
            try:
                if world is not None:
                    from .vector_collector import validate_vector_close
                    result=world.call('close')
                    world.client.close()
                    workers.record_external_close('gmt',result)
                    validate_vector_close(result)
            finally:
                try:
                    workers.close()
                    if any(v.get('close_error') or v.get('forced_shutdown') or v.get('process_exit_code')!=0
                           for v in workers.shutdown.values()):
                        raise RuntimeError('Isolated GPU evaluation worker did not close cleanly')
                finally:
                    if journal is not None:journal.close()
        local_call(c.distributed,close)
