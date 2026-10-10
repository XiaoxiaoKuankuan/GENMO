"""训练通信与异步归档统计的可信度、异步就绪状态和线程安全CPU验收。

真实CPU梯度与SUM通信替身核对算法保持不变，包括本rank缺失梯度由其他rank补齐，
presence/梯度bucket次数及本地tensor bytes。CUDA Event/stream计数替身使用不同于
host wall的毫秒值，验证单位换算、未就绪不报零、只在阶段末等待一次、不逐bucket
同步且不触碰其他设备。归档使用真实封存/journal/SHA及临时文件，核对不可变返回
值、成功/失败/正在执行的区别，以及后台完成记录和主线程drain并发不丢失/重复。
测试只使用pytest tmp_path，禁止正式checkpoint、CUDA/Isaac和长期训练。
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import distributed_runtime as runtime
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.long_run import LongRunMaintenance
from gem.closedloop.dppo.run_management import RunManager
from tests.closedloop.dppo.test_stage10_async_archives import iteration, stage


def test_actor_step_curves_keep_denoising_ratio_clip_and_missing_values(tmp_path):
    import copy
    from gem.closedloop.dppo.periodic_monitor import actor_minibatch_curves, log_metrics
    records=[dict(optimizer_step=1,per_denoising_step=[
        dict(step=0,visits=0,mean_joint_kl=None,mean_ratio=None,clip_fraction=None),
        dict(step=19,visits=3,mean_joint_kl=.02,mean_ratio=1.002,clip_fraction=1/3)])]
    original=copy.deepcopy(records)
    logged=[]
    writer=SimpleNamespace(add_scalar=lambda tag,value,step:logged.append((tag,value,step)))
    values=log_metrics(writer,tmp_path/'curves.jsonl',7,
        dict(actor_minibatches=actor_minibatch_curves(records)),durable=False)
    prefix='actor_minibatches/step1/denoising_steps/'
    assert values[prefix+'step19/mean_ratio']==1.002
    assert values[prefix+'step19/clip_fraction']==1/3
    assert values[prefix+'step19/mean_joint_kl']==.02
    assert values[prefix+'step0/visits']==0
    assert prefix+'step0/mean_ratio' not in values
    assert (prefix+'step19/clip_fraction',1/3,7) in logged
    assert records==original and isinstance(records[0]['per_denoising_step'],list)


def test_cpu_sum_statistics_cover_presence_and_bucket_without_changing_gradients(monkeypatch):
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(torch.ones(2))
    model.remote_only = torch.nn.Parameter(torch.ones(3))
    model.first.grad = torch.tensor([1., 2.])
    collective = DistributedCollectives(0, 2, measure_gradient_communication=True)
    counter = iter(range(6))
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(perf_counter=lambda: float(next(counter))))
    def reduce(tensor, **kwargs):
        if tensor.dtype == torch.int32:
            tensor.copy_(torch.tensor([2, 1], dtype=torch.int32))
        else:
            tensor.add_(5.)
    monkeypatch.setattr(runtime.dist, 'all_reduce', reduce)
    assert collective.sum_gradients(model) is None
    torch.testing.assert_close(model.first.grad, torch.tensor([6., 7.]))
    torch.testing.assert_close(model.remote_only.grad, torch.full((3,), 5.))
    timing = collective.collect_gradient_timings()
    assert timing['completed_call_count'] == 1 and timing['pending_call_count'] == 0
    assert timing['seconds'] == 2.
    summary = timing['by_module']['Module']
    assert summary['collective_count'] == 2 and summary['tensor_bytes'] == 2*4+5*4
    assert summary['call_wall_seconds'] == 5.
    assert [row['kind'] for row in timing['calls'][0]['collectives']] == ['gradient_presence', 'gradient_bucket']
    assert collective.collect_gradient_timings()['seconds'] is None


def test_legacy_collective_does_not_accumulate_timings_without_consumer(monkeypatch):
    model = torch.nn.Linear(1, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    collective = DistributedCollectives(0, 1)
    monkeypatch.setattr(runtime.dist, 'all_reduce', lambda *args, **kwargs: None)
    collective.sum_gradients(model)
    assert collective.collect_gradient_timings()['completed_call_count'] == 0
    collective.enable_gradient_timing()
    collective.sum_gradients(model)
    assert collective.collect_gradient_timings()['completed_call_count'] == 1


def test_cuda_event_statistics_defer_wait_and_measure_gpu_milliseconds(monkeypatch):
    events, waits, streams = [], [], []
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {'enable_timing': True}
            self.index, self.complete = len(events), False
            events.append(self)
        def record(self, stream):
            streams.append(stream.cuda_stream)
        def query(self):
            return self.complete
        def synchronize(self):
            waits.append(self.index)
            for previous in events[:self.index+1]:
                previous.complete = True
        def elapsed_time(self, end):
            assert self.complete and end.complete
            return 8.
    def current_stream(device):
        assert device == torch.device('cuda:3')
        return SimpleNamespace(cuda_stream=333)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', current_stream)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: pytest.fail('Global CUDA synchronize is forbidden'))
    monkeypatch.setattr(runtime.dist, 'all_reduce', lambda *args, **kwargs: None)
    collective = DistributedCollectives(3, 8, device='cuda:3')
    tensor = SimpleNamespace(device=torch.device('cuda:3'), numel=lambda: 8, element_size=lambda: 4)
    record = dict(sequence=1, module='Stage1Actor', call_wall_seconds=.02, collectives=[])
    collective._timed_gradient_reduce(tensor, record, 'gradient_presence')
    collective._timed_gradient_reduce(tensor, record, 'gradient_bucket')
    collective._gradient_timing_pending.append(record)
    assert waits == [] and streams == [333]*4
    pending = collective.collect_gradient_timings()
    assert pending['pending_call_count'] == 1 and pending['seconds'] is None and waits == []
    ready = collective.collect_gradient_timings(synchronize=True)
    assert waits == [3] and ready['pending_call_count'] == 0
    assert ready['seconds'] == pytest.approx(.016)
    assert ready['by_module']['Stage1Actor']['tensor_bytes'] == 64
    assert ready['calls'][0]['collectives'][0]['clock'] == 'cuda_stream_event'


def test_archive_finished_statistics_preserve_manifest_and_report_real_stages(tmp_path):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        assert maintenance.drain_archive_timings()['total_seconds'] is None
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        manifest = maintenance.archive_iteration(directory)
        timing = maintenance.drain_archive_timings()
        assert timing['completed_count'] == timing['successful_count'] == 1
        record = timing['records'][0]
        assert record['iteration'] == 1 and record['status'] == 'passed'
        stages = record['stage_seconds']
        assert {'compression_seconds', 'verification_seconds', 'archive_fsync_seconds', 'publish_seconds', 'reclaim_seconds'} <= stages.keys()
        assert all(value >= 0 for value in stages.values())
        assert sum(stages.values()) <= record['total_seconds']
        assert 'stage_seconds' not in manifest  # 不改不可变原始契约。
        assert maintenance.archive_iteration(directory) == manifest
        retry = maintenance.drain_archive_timings()['records'][0]
        assert 'compression_seconds' not in retry['stage_seconds']  # 恢复重试没有压缩不能伪报0。


def test_archive_failure_statistics_are_not_success_or_missing(tmp_path):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        directory, journal = iteration(manager)
        manager.seal_iteration(directory, 1, closed_journals=[journal])
        (directory/'fixed_targets.pt').write_bytes(b'changed')
        with pytest.raises(ValueError, match='immutable seal'):
            maintenance.archive_iteration(directory)
        timing = maintenance.drain_archive_timings()
        assert timing['completed_count'] == 1 and timing['successful_count'] == 0
        assert timing['successful_seconds'] is None and timing['records'][0]['status'] == 'failed'
        assert timing['records'][0]['error']['type'] == 'ValueError'
        assert (directory/'fixed_targets.pt').exists()


def test_archive_thread_and_consumer_drain_never_lose_or_duplicate_records(tmp_path, monkeypatch):
    with RunManager(tmp_path/'run') as manager:
        maintenance = LongRunMaintenance(manager, stage())
        started, release, stop, consumed = (threading.Event() for _ in range(4))
        actual = maintenance._archive_via_process
        def blocked(directory):
            started.set()
            assert release.wait(5)
            return actual(directory)
        monkeypatch.setattr(maintenance, '_archive_via_process', blocked)
        snapshots = []
        consumer = None
        try:
            for index in range(1, 5):
                directory, journal = iteration(manager, index)
                manager.seal_iteration(directory, index, closed_journals=[journal])
                assert maintenance.enqueue_archive(directory)
            assert started.wait(2)
            pending = maintenance.drain_archive_timings()
            assert pending['completed_count'] == 0 and pending['total_seconds'] is None and pending['pending_count'] == 4
            assert len(pending['enqueues']) == 4
            assert all(row['stage_seconds']['submit_wall_seconds'] >= 0 for row in pending['enqueues'])
            def consume():
                while not stop.wait(.001):
                    snapshots.append(maintenance.drain_archive_timings())
                    consumed.set()
            consumer = threading.Thread(target=consume)
            consumer.start()
            assert consumed.wait(2)
            release.set()
            maintenance.drain()
            stop.set()
            consumer.join(2)
            snapshots.append(maintenance.drain_archive_timings())
            records = [row for snapshot in snapshots for row in snapshot['records']]
            assert len(records) == 4 and sorted(r['iteration'] for r in records) == [1, 2, 3, 4]
            assert all(r['status'] == 'passed' and r['thread'] == 'genmo-execution-archive' for r in records)
        finally:
            release.set()
            stop.set()
            if consumer is not None:
                consumer.join(2)
