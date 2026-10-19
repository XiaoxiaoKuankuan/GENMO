"""GPU多环境训练的隔离评估世界，复用原固定任务、奖励、报告和模型选择逻辑。

训练世界保持原PhysX状态不动，每rank另外启动一个GPU评估世界，环境数由配置
绑定；任务在该世界内并行，按任务独立噪声和部署时钟推进。可在整个会话常驻，
每次只轮换接收日志，世界启动/关闭耗时单独记录。Stage1和后续模型共用同一
评估协议、固定起点、种子及明确的延迟预算。世界权威日志持久化后才ACK，
环境索引不重复存储完整物理记录。退出时恢复训练backend对象，评估不重置
训练物理状态、音乐游标或预算；原评估器负责恢复模型模式、梯度和所有训练RNG。
评估/MC世界的启动与关闭耗时另记入上下文的非迭代开销，不计入评估自身wall_seconds。
世界进程启动失败（例如Isaac远程地面USD下载瞬断）时换新目录最多重试三次；启动失败发生在
任何物理交互之前，因此重试不改变评估结果，失败目录与日志保留。
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import time
import yaml
import torch
from .parallel_support import local_call
from .run_management import GuardedStepJournal
from .memory_execution import MemoryReceiveJournal
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


class IsolatedWorld:
    """一个独立评估/MC世界的进程、日志与后端；不含collective。"""
    def __init__(self):
        self.workers=self.world=self.journal=self.backend=None
        self.start_seconds=self.close_seconds=0.
        self.num_envs=1
        self.closed=False


def isolated_env_count(c,label):
    """MC可并行更多完整音乐；固定Actor评估仍使用原4槽协议。"""
    value=c.stage['evaluation'].get('envs_per_rank',1)
    if label.startswith('mc_'):
        value=c.stage.get('critic',{}).get('complete_mc',{}).get('envs_per_rank',value)
    return value


def open_isolated_world(c,label,*,attempts=3):
    """启动并核验Isaac世界；进程启动失败（如地面USD远程下载瞬断）时换新目录重试。

    每次尝试使用独立目录并保留失败日志；世界进入物理执行前的启动失败不产生任何交互，
    因此重试不改变评估结果。三次都失败才抛出。不做跨rank同步。
    """
    errors=[]
    for attempt in range(attempts):
        suffix='' if attempt==0 else f'_retry{attempt}'
        try:
            return _open_isolated_world_once(c,label+suffix)
        except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
            time.sleep(5*(attempt+1))
    raise RuntimeError('Isolated evaluation world failed to start: '+' | '.join(errors))


def _open_isolated_world_once(c,label):
    from tools.eval.run_closedloop_baseline import Workers
    started=time.perf_counter();handle=IsolatedWorld()
    path = c.session/'phases'/f'vector_eval_backend_{label}'/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True,exist_ok=False)
    config = copy.deepcopy(c.config)
    from .vector_devices import bind_vector_device
    device_environment=bind_vector_device(config,c.distributed.rank)
    # 独立评估/MC世界不重复首个列式-逐行全量对照；训练世界每会话已执行一次，数值路径相同。
    handle.num_envs=isolated_env_count(c,label)
    config['runtime'].update(num_envs=handle.num_envs,asset_conversion_dir=str(path/'usd_assets'),
        columnar_first_audit=False,evidence_level='training')
    config_path=path/'config.yaml';config_path.write_text(yaml.safe_dump(config,allow_unicode=True))
    handle.workers=Workers(config,path)
    try:
        socket=Path(handle.workers.temp.name)/'eval.sock'
        client=handle.workers.start('gmt',[config['paths']['isaac_python'],'-B',
            str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt_vector.py'),
            '--config',str(config_path),'--socket',str(socket),'--headless'],config['paths']['gmt_repo'],socket,
            strip_distributed=True,environment=device_environment)
        if handle.workers.entries[0]['identity'].get('gpu_uuid')!=str(torch.cuda.get_device_properties(c.distributed.device).uuid):
            raise RuntimeError('Evaluation world is on a different physical GPU')
        handle.journal=MemoryReceiveJournal(path/'world.sqlite')
        handle.world=VectorWorldClient(client,handle.journal,socket_path=socket)
        handle.backend=(VectorLaneBackend(SingleVectorLaneTransport(handle.world),None)
            if handle.num_envs==1 else SimpleNamespace(journal=None))
    except BaseException:
        close_isolated_world(handle)
        raise
    handle.start_seconds=time.perf_counter()-started
    return handle


def close_isolated_world(handle):
    """关闭世界与进程；任何非干净退出都抛错，journal无论如何关闭。"""
    if handle.closed:return
    handle.closed=True
    started=time.perf_counter()
    try:
        if handle.world is not None:
            from .vector_collector import validate_vector_close
            result=handle.world.call('close')
            handle.world.client.close()
            handle.workers.record_external_close('gmt',result)
            validate_vector_close(result)
    finally:
        try:
            if handle.workers is not None:
                handle.workers.close()
                if any(v.get('close_error') or v.get('forced_shutdown') or v.get('process_exit_code')!=0
                       for v in handle.workers.shutdown.values()):
                    raise RuntimeError('Isolated GPU evaluation worker did not close cleanly')
        finally:
            if handle.journal is not None:handle.journal.close()
            handle.close_seconds=time.perf_counter()-started


@contextmanager
def isolated_vector_evaluation(c,label):
    kind = 'mc' if label.startswith('mc_') else 'evaluation'
    overhead = getattr(c, 'overhead', None)
    old_backend=c.backend
    persistent=c.stage['evaluation'].get('persistent_world',False)
    if kind=='mc' and isolated_env_count(c,label)!=c.stage['evaluation'].get('envs_per_rank',1):
        persistent=False
    handle=getattr(c,'evaluation_world',None) if persistent else None
    failure=None;created=handle is None
    try:
        if created:handle=open_isolated_world(c,label)
    except Exception as error:
        failure=f'rank {c.distributed.rank}: {type(error).__name__}: {error}'
    failures=c.distributed.all_gather_object(failure)
    if any(failures):
        # 其他rank启动失败时，本rank已启动的世界也必须关闭后再共同抛错。
        if handle is not None:
            try:close_isolated_world(handle)
            except Exception:pass
        if persistent:c.evaluation_world=None
        raise RuntimeError('; '.join(item for item in failures if item))
    if persistent:c.evaluation_world=handle
    c.active_evaluation_world=handle
    def rotate_journal():
        path=c.session/'phases'/f'vector_eval_journal_{label}'/f'rank{c.distributed.rank:02d}'
        path.mkdir(parents=True,exist_ok=False)
        handle.journal.close()
        handle.journal=MemoryReceiveJournal(path/'world.sqlite')
        handle.world.journal=handle.journal
    if not created:local_call(c.distributed,rotate_journal)
    if overhead is not None and created:
        overhead[kind+'_world_start_seconds']=overhead.get(kind+'_world_start_seconds',0.)+handle.start_seconds
    c.backend=handle.backend
    try:
        yield
    finally:
        c.backend=old_backend
        c.active_evaluation_world=None
        if not persistent:local_call(c.distributed,lambda:close_isolated_world(handle))
        if overhead is not None and not persistent:
            overhead[kind+'_world_close_seconds']=overhead.get(kind+'_world_close_seconds',0.)+handle.close_seconds
